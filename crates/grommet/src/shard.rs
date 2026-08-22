//! The per-core shard reactor.
//!
//! One shard owns one core, one scheduler, one processor instance and every key
//! that routes to it. It runs a single-threaded Tokio runtime, so nothing it
//! holds needs to be `Send` and nothing it does needs synchronization.

use crate::clock::Clock;
use crate::error::{Fallout, ProcessError};
use crate::mailbox::Inbox;
use crate::metrics::{ShardHot, ShardStats};
use crate::processor::{KeyOf, PanicPolicy, Processor};
use crate::work::{Envelope, Stamped, Work};
use ahash::AHashSet;
use futures::FutureExt;
use futures::future::Either;
use futures::stream::{FuturesUnordered, StreamExt};
use grommet_core::{Admit, ClassId, Completion, Dispatch, Disposition, Scheduler};
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::Duration;

/// The scheduler as a shard specializes it.
type Book<K, W, S, const CLASSES: usize> = Scheduler<K, Stamped<W>, S, CLASSES>;

/// Request ids currently queued or in flight, scoped by key.
type Live<K, I> = AHashSet<(K, I)>;

/// Per-shard tuning.
///
/// Build one with [`ShardConfig::new`] and adjust the fields you care about.
/// The struct is `#[non_exhaustive]` so that later releases can add a knob
/// without that being a breaking change for anyone who did exactly that.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct ShardConfig<const CLASSES: usize = 2> {
    pub scheduler: grommet_core::Config<CLASSES>,
    /// Maximum already-ready mailbox items admitted after one awaited receive.
    /// This amortizes cross-thread wakeups, which dominate at high rates, while
    /// bounding how long an arrival burst can delay completion harvesting.
    pub admit_batch: usize,
    /// How often gauges are published and eviction is swept.
    pub tick: Duration,
    pub panic_policy: PanicPolicy,
    /// Suppress an item whose [`Work::request_id`] matches one already queued
    /// or in flight for the same key, handing it to
    /// [`Processor::on_coalesced`] instead of running it twice.
    ///
    /// This covers a client retrying while its first attempt is still
    /// outstanding, which is exactly when a retry storm is most expensive. It
    /// is not durable deduplication: once the original completes, its id leaves
    /// the index and a later retry is admitted normally, because answering that
    /// correctly needs the original's recorded outcome from your store.
    pub coalesce_duplicates: bool,
}

impl<const CLASSES: usize> ShardConfig<CLASSES> {
    pub fn new(max_inflight: [usize; CLASSES]) -> Self {
        Self {
            scheduler: grommet_core::Config::new(max_inflight),
            admit_batch: 64,
            tick: Duration::from_secs(1),
            panic_policy: PanicPolicy::default(),
            coalesce_duplicates: false,
        }
    }
}

/// What one entry in a shard's outstanding set resolved to.
///
/// Processing an item and flushing a key are different jobs with different
/// follow-ups, but they are the same thing to the reactor: work it started and
/// must hear back about before it can exit. Sharing one output type lets them
/// share one [`FuturesUnordered`], which is one poll site, one emptiness test
/// and one shutdown condition instead of two of each.
enum Outcome<K, S, I> {
    /// A dispatched item finished, however it finished.
    Ran {
        completion: Completion<K, S>,
        /// Returned so the coalescing index can release it.
        request_id: Option<I>,
        panicked: bool,
        /// How the processor classified its failure, if it failed.
        fallout: Option<Fallout>,
    },
    /// A key's state finished flushing, so the key can be released.
    Flushed(K),
}

/// What a shard's outstanding futures resolve to, for a given processor.
type OutcomeOf<P> =
    Outcome<KeyOf<P>, <P as Processor>::State, <<P as Processor>::Work as Work>::Id>;

/// One dispatched item, start to finish.
///
/// This is a free async fn rather than a direct call through the trait so that
/// the future has one concrete type per processor, rather than one per work
/// variant.
async fn run_one<P: Processor>(
    processor: P,
    dispatch: Dispatch<KeyOf<P>, Stamped<P::Work>, P::State>,
) -> OutcomeOf<P> {
    let Dispatch { key, class, state, payload } = dispatch;
    let request_id = payload.request_id;
    // A panic must not unwind the reactor loop: that would take down every key
    // this shard owns and leave the in-flight accounting permanently wrong. The
    // state moved into the future is gone regardless, so the key reloads.
    let outcome =
        AssertUnwindSafe(processor.process(key, state, payload.work)).catch_unwind().await;
    let (state, panicked, fallout) = match outcome {
        Ok(Ok(state)) => (state, false, None),
        Ok(Err(error)) => {
            // An error means the processor no longer holds state it can trust,
            // so the key reloads. `on_error` sees the classified failure.
            let fallout = error.fallout();
            processor.on_error(key, &error);
            (Disposition::Drop, false, Some(fallout))
        }
        // A panic tells us nothing about what the work managed to do first.
        Err(_) => (Disposition::Drop, true, Some(Fallout::InDoubt)),
    };
    Outcome::Ran { completion: Completion { key, class, state }, request_id, panicked, fallout }
}

/// Flush one key's state. The key stays quiesced until this resolves.
async fn run_flush<P: Processor>(processor: P, key: KeyOf<P>, state: P::State) -> OutcomeOf<P> {
    let _ = AssertUnwindSafe(processor.on_evict(key, state)).catch_unwind().await;
    Outcome::Flushed(key)
}

fn admit_one<P: Processor, const CLASSES: usize>(
    book: &mut Book<KeyOf<P>, P::Work, P::State, CLASSES>,
    live: &mut Live<KeyOf<P>, <P::Work as Work>::Id>,
    processor: &P,
    hot: &ShardHot,
    envelope: Envelope<P::Work>,
    coalesce: bool,
) {
    let Envelope { key, class, request_id, expires_at, enqueued, work } = envelope;

    // Suppress a retry that arrived while its original is still outstanding.
    // The index is keyed by (key, id) so an id reused across different keys —
    // legitimate in some schemes — is never confused for a duplicate.
    let tracked = match request_id {
        Some(id) if coalesce => {
            if !live.insert((key, id.clone())) {
                hot.bump(&hot.coalesced);
                processor.on_coalesced(key, work);
                return;
            }
            Some(id)
        }
        _ => None,
    };

    hot.bump(&hot.started);
    book.admit(Admit {
        key,
        class,
        expires_at,
        payload: Stamped { enqueued, request_id: tracked, work },
    });
}

#[allow(clippy::too_many_arguments)]
fn admit_ready<P: Processor, const CLASSES: usize>(
    book: &mut Book<KeyOf<P>, P::Work, P::State, CLASSES>,
    live: &mut Live<KeyOf<P>, <P::Work as Work>::Id>,
    processor: &P,
    hot: &ShardHot,
    rx: &mut Inbox<Envelope<P::Work>>,
    first: Envelope<P::Work>,
    cfg: &ShardConfig<CLASSES>,
) {
    admit_one::<P, CLASSES>(book, live, processor, hot, first, cfg.coalesce_duplicates);
    for _ in 1..cfg.admit_batch.max(1) {
        if book.is_saturated() {
            break;
        }
        let Ok(envelope) = rx.try_recv() else {
            break;
        };
        admit_one::<P, CLASSES>(book, live, processor, hot, envelope, cfg.coalesce_duplicates);
    }
}

/// Run one shard until its mailbox closes and everything outstanding drains.
///
/// The future is `!Send` by construction. Give it a current-thread runtime on a
/// pinned core; [`crate::runtime`] does that for you.
pub async fn run<P, C, const CLASSES: usize>(
    mut rx: Inbox<Envelope<P::Work>>,
    processor: P,
    clock: C,
    stats: Arc<ShardStats<CLASSES>>,
    cfg: ShardConfig<CLASSES>,
) where
    P: Processor,
    C: Clock,
{
    let hot = ShardHot::default();
    let mut book: Book<KeyOf<P>, P::Work, P::State, CLASSES> = Scheduler::new(cfg.scheduler);
    let mut live: Live<KeyOf<P>, <P::Work as Work>::Id> = Live::default();
    let mut outstanding = FuturesUnordered::new();
    let mut flushing = Vec::new();
    let mut tick = tokio::time::interval(cfg.tick);
    let mut open = true;
    let mut final_flush = true;

    // Carried across turns rather than re-read at the top of each. The reading
    // taken at the end of one turn's bookkeeping and the one that would be taken
    // at the start of the next are separated by nothing but the loop back-edge,
    // and `Instant::now` is a real syscall-shaped cost — around 25ns, against a
    // hot path measured in hundreds.
    let mut now = clock.now();

    loop {
        // Fill every class budget from its own round-robin ring.
        for class in 0..CLASSES {
            while let Some(dispatch) = book.next(class as ClassId, now) {
                hot.add(
                    &hot.queue_wait_nanos,
                    now.saturating_sub(dispatch.payload.enqueued).as_nanos() as u64,
                );
                outstanding.push(Either::Left(run_one(processor.clone(), dispatch)));
            }
        }
        // Work whose deadline has passed never costs a dispatch turn.
        while let Some((key, stamped)) = book.pop_expired() {
            hot.bump(&hot.expired);
            if let Some(id) = stamped.request_id {
                live.remove(&(key, id));
            }
            processor.on_expired(key, stamped.work);
        }
        debug_assert!(book.check_invariants().is_ok());

        // Closed mailbox and nothing outstanding: the dispatch loop above has
        // already drained everything that was still queued.
        if !open && outstanding.is_empty() {
            // Resident state is a write-back cache — between dispatches the
            // scheduler holds the only copy. Exiting without draining it
            // discards writes the processor was told it could keep, so the last
            // thing a shard does is hand every one of them to `on_evict`. By
            // here nothing can arrive and nothing can dispatch, so one pass is
            // enough and the flushes it starts are the last work there is.
            if std::mem::take(&mut final_flush) {
                book.evict_all(&mut flushing);
                for (key, state) in flushing.drain(..) {
                    hot.bump(&hot.evicted);
                    outstanding.push(Either::Right(run_flush(processor.clone(), key, state)));
                }
            }
            if outstanding.is_empty() {
                break;
            }
        }

        tokio::select! {
            // No `biased`: tokio randomizes among ready branches. Admission
            // drains only a bounded burst, so completion harvesting is delayed
            // by at most `admit_batch` rather than by an unbounded producer.

            // Guarded because `FuturesUnordered::next` resolves to `None`
            // immediately when empty, which would busy-spin the loop.
            Some(outcome) = outstanding.next(), if !outstanding.is_empty() => {
                let at = clock.now();
                match outcome {
                    Outcome::Ran {
                        completion,
                        request_id,
                        panicked,
                        fallout
                    } => {
                        hot.bump(&hot.completed);
                        if let Some(fallout) = fallout {
                            hot.bump(&hot.failed);
                            if fallout.is_in_doubt() {
                                hot.bump(&hot.in_doubt);
                            }
                        }
                        if panicked {
                            hot.bump(&hot.panicked);
                            if cfg.panic_policy == PanicPolicy::Abort {
                                stats.publish(&hot, &book.snapshot());
                                std::process::abort();
                            }
                        }
                        if let Some(id) = request_id {
                            live.remove(&(completion.key, id));
                        }
                        book.complete(completion, at);
                    }
                    Outcome::Flushed(key) => book.finish_evict(key, at),
                }
                now = clock.now();
                hot.add(&hot.busy_nanos, now.saturating_sub(at).as_nanos() as u64);
            }

            // Admit only under the pending cap. At the cap this branch is
            // disabled, the bounded mailbox fills, and the router's `send`
            // suspends its caller: end-to-end backpressure.
            envelope = rx.recv(), if open && !book.is_saturated() => {
                match envelope {
                    Some(envelope) => {
                        let at = clock.now();
                        admit_ready::<P, CLASSES>(
                            &mut book,
                            &mut live,
                            &processor,
                            &hot,
                            &mut rx,
                            envelope,
                            &cfg,
                        );
                        now = clock.now();
                        hot.add(&hot.busy_nanos, now.saturating_sub(at).as_nanos() as u64);
                    }
                    // The router is gone. Stop admitting and drain.
                    None => open = false,
                }
            }

            _ = tick.tick(), if open => {
                stats.publish(&hot, &book.snapshot());
                now = clock.now();
                book.evict(now, &mut flushing);
                for (key, state) in flushing.drain(..) {
                    hot.bump(&hot.evicted);
                    outstanding.push(Either::Right(run_flush(processor.clone(), key, state)));
                }
            }
        }
    }

    stats.publish(&hot, &book.snapshot());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::ManualClock;
    use crate::router::Router;
    use crate::work::Work;
    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::work::{COMPUTE, IO};

    #[derive(Debug)]
    struct Item {
        key: u64,
        class: ClassId,
        ttl: Option<Duration>,
        id: Option<u64>,
    }

    impl Item {
        fn new(key: u64) -> Self {
            Self { key, class: IO, ttl: None, id: None }
        }

        fn retry(key: u64, id: u64) -> Self {
            Self { id: Some(id), ..Self::new(key) }
        }
    }

    impl Work for Item {
        type Key = u64;
        type Id = u64;
        fn key(&self) -> u64 {
            self.key
        }
        fn class(&self) -> ClassId {
            self.class
        }
        fn time_to_live(&self) -> Option<Duration> {
            self.ttl
        }
        fn request_id(&self) -> Option<u64> {
            self.id
        }
    }

    /// Everything the shard did, in order, so tests assert on behaviour rather
    /// than on timing.
    #[derive(Default)]
    struct Log {
        processed: Vec<(u64, Option<u64>)>,
        evicted: Vec<(u64, u64)>,
        expired: Vec<u64>,
        failed: Vec<u64>,
        coalesced: Vec<u64>,
    }

    /// An error whose outcome is unknown, so the key must reload.
    #[derive(Debug)]
    struct Fault;

    impl crate::error::ProcessError for Fault {
        fn fallout(&self) -> Fallout {
            Fallout::InDoubt
        }
    }

    #[derive(Clone)]
    struct Recorder {
        log: Rc<RefCell<Log>>,
        panic_on: Option<u64>,
        fail_on: Option<u64>,
    }

    impl Recorder {
        fn new() -> Self {
            Self { log: Rc::new(RefCell::new(Log::default())), panic_on: None, fail_on: None }
        }

        fn panicking_on(key: u64) -> Self {
            Self { panic_on: Some(key), ..Self::new() }
        }

        fn failing_on(key: u64) -> Self {
            Self { fail_on: Some(key), ..Self::new() }
        }
    }

    impl Processor for Recorder {
        type Work = Item;
        type State = u64;
        type Error = Fault;

        async fn process(
            &self,
            key: u64,
            state: Option<u64>,
            _work: Item,
        ) -> Result<Disposition<Self::State>, Fault> {
            self.log.borrow_mut().processed.push((key, state));
            assert_ne!(self.panic_on, Some(key), "deliberate processor panic");
            if self.fail_on == Some(key) {
                return Err(Fault);
            }
            Ok(Disposition::Keep(state.unwrap_or(0) + 1))
        }

        fn on_error(&self, key: u64, _error: &Fault) {
            self.log.borrow_mut().failed.push(key);
        }

        fn on_coalesced(&self, key: u64, _work: Item) {
            self.log.borrow_mut().coalesced.push(key);
        }

        async fn on_evict(&self, key: u64, state: u64) {
            self.log.borrow_mut().evicted.push((key, state));
        }

        fn on_expired(&self, key: u64, _work: Item) {
            self.log.borrow_mut().expired.push(key);
        }
    }

    fn config() -> ShardConfig<2> {
        let mut cfg = ShardConfig::new([4, 4]);
        cfg.tick = Duration::from_millis(1);
        cfg.scheduler.evict_after = Duration::from_secs(3600);
        cfg
    }

    /// Drive a shard with `driver`, then close its mailbox and let it drain.
    async fn drive<F, Fut>(
        processor: Recorder,
        cfg: ShardConfig<2>,
        driver: F,
    ) -> Arc<ShardStats<2>>
    where
        F: FnOnce(Router<Item, ManualClock, 2>, ManualClock) -> Fut,
        Fut: Future<Output = ()>,
    {
        let clock = ManualClock::new();
        let (tx, rx) = crate::mailbox::channel(64);
        let router = Router::<Item, ManualClock, 2>::new(vec![tx], clock.clone());
        let stats = Arc::new(ShardStats::<2>::default());
        let engine = run(rx, processor, clock.clone(), stats.clone(), cfg);
        tokio::join!(engine, driver(router, clock));
        stats
    }

    #[tokio::test(start_paused = true)]
    async fn one_key_is_processed_in_submission_order_with_accumulating_state() {
        let processor = Recorder::new();
        let log = processor.log.clone();
        let stats = drive(processor, config(), |router, _clock| async move {
            for _ in 0..3 {
                router.submit(Item::new(7)).await.unwrap();
            }
        })
        .await;

        assert_eq!(
            log.borrow().processed,
            vec![(7, None), (7, Some(1)), (7, Some(2))],
            "state must follow the key across dispatches, in order"
        );
        assert_eq!(stats.completed.load(std::sync::atomic::Ordering::Relaxed), 3);
        assert_eq!(stats.panicked.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn a_panicking_processor_is_contained_and_the_shard_keeps_serving() {
        let processor = Recorder::panicking_on(1);
        let log = processor.log.clone();
        let stats = drive(processor, config(), |router, _clock| async move {
            router.submit(Item::new(1)).await.unwrap();
            router.submit(Item::new(1)).await.unwrap();
            router.submit(Item::new(2)).await.unwrap();
        })
        .await;

        let log = log.borrow();
        assert_eq!(
            log.processed,
            // Key 2 is dispatched alongside key 1 rather than queued behind its
            // panics, and key 1's second item sees no state because the first
            // panic dropped it.
            vec![(1, None), (2, None), (1, None)],
            "a panic drops only the panicking key's state, and blocks no other key"
        );
        let relaxed = std::sync::atomic::Ordering::Relaxed;
        assert_eq!(stats.panicked.load(relaxed), 2);
        assert_eq!(stats.completed.load(relaxed), 3, "a panicked item still completes its slot");
    }

    #[tokio::test(start_paused = true)]
    async fn work_past_its_deadline_is_shed_at_dispatch_without_being_processed() {
        let processor = Recorder::new();
        let log = processor.log.clone();
        drive(processor, config(), |router, _clock| async move {
            let shed = Item { ttl: Some(Duration::ZERO), ..Item::new(5) };
            router.submit(shed).await.unwrap();
            router.submit(Item::new(6)).await.unwrap();
        })
        .await;

        let log = log.borrow();
        assert_eq!(log.expired, vec![5], "the deadline had already passed at dispatch");
        assert_eq!(log.processed, vec![(6, None)], "shed work never reaches the processor");
    }

    #[tokio::test(start_paused = true)]
    async fn idle_state_is_flushed_through_on_evict_and_reloaded_afterwards() {
        let processor = Recorder::new();
        let log = processor.log.clone();
        let observed = log.clone();
        let mut cfg = config();
        cfg.scheduler.evict_after = Duration::ZERO;

        drive(processor, cfg, |router, _clock| async move {
            router.submit(Item::new(4)).await.unwrap();
            // Everything here shares one thread, so the shard only makes
            // progress while this task is suspended. Sleeping under paused time
            // lets its tick fire and sweep eviction.
            for _ in 0..50 {
                if !observed.borrow().evicted.is_empty() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            router.submit(Item::new(4)).await.unwrap();
        })
        .await;

        let log = log.borrow();
        assert_eq!(
            log.evicted,
            vec![(4, 1), (4, 1)],
            "the idle sweep flushes the first state, and shutdown flushes the second"
        );
        assert_eq!(
            log.processed,
            vec![(4, None), (4, None)],
            "a key that was flushed reloads instead of reusing released state"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn state_still_resident_at_shutdown_is_flushed_before_the_shard_exits() {
        let processor = Recorder::new();
        let log = processor.log.clone();
        // An idle window far longer than the test, so nothing is swept: the
        // only thing that can flush these keys is shutdown itself.
        let cfg = config();

        drive(processor, cfg, |router, _clock| async move {
            router.submit(Item::new(1)).await.unwrap();
            router.submit(Item::new(2)).await.unwrap();
            router.submit(Item::new(1)).await.unwrap();
            drop(router);
        })
        .await;

        let mut evicted = log.borrow().evicted.clone();
        evicted.sort_unstable();
        assert_eq!(
            evicted,
            vec![(1, 2), (2, 1)],
            "a write-back cache must not drop its writes on a clean shutdown"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn an_expired_item_releases_its_request_id_for_a_later_retry() {
        let processor = Recorder::new();
        let log = processor.log.clone();
        let mut cfg = config();
        cfg.coalesce_duplicates = true;

        drive(processor, cfg, |router, _clock| async move {
            // Shed at dispatch, so this id never reaches a completion — the
            // only other place the coalescing index is cleared.
            router.try_submit(Item { ttl: Some(Duration::ZERO), ..Item::retry(3, 77) }).unwrap();
            tokio::time::sleep(Duration::from_millis(5)).await;
            router.submit(Item::retry(3, 77)).await.unwrap();
        })
        .await;

        let log = log.borrow();
        assert_eq!(log.expired, vec![3]);
        assert!(log.coalesced.is_empty(), "the original expired, so nothing was live to match");
        assert_eq!(
            log.processed,
            vec![(3, None)],
            "an expired item must release its id rather than blackhole the key's retries"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn queued_work_is_drained_after_the_mailbox_closes() {
        let processor = Recorder::new();
        let log = processor.log.clone();
        let mut cfg = config();
        // One in-flight item at a time, so most of the batch is still queued
        // when the router disappears.
        cfg.scheduler.max_inflight = [1, 1];

        drive(processor, cfg, |router, _clock| async move {
            for key in 0..16 {
                router.submit(Item::new(key)).await.unwrap();
            }
            drop(router);
        })
        .await;

        assert_eq!(log.borrow().processed.len(), 16, "shutdown must not discard queued work");
    }

    #[tokio::test(start_paused = true)]
    async fn class_budgets_are_published_per_class() {
        let processor = Recorder::new();
        let stats = drive(processor, config(), |router, _clock| async move {
            router.submit(Item { class: COMPUTE, ..Item::new(1) }).await.unwrap();
            router.submit(Item { class: 0, ..Item::new(2) }).await.unwrap();
        })
        .await;

        let relaxed = std::sync::atomic::Ordering::Relaxed;
        assert_eq!(stats.started.load(relaxed), 2);
        assert_eq!(stats.completed.load(relaxed), 2);
        assert_eq!(stats.pending.load(relaxed), 0, "everything drained");
        assert_eq!(stats.resident.load(relaxed), 0, "shutdown released every key it flushed");
    }

    #[tokio::test(start_paused = true)]
    async fn an_in_doubt_failure_drops_state_and_is_counted_apart_from_other_errors() {
        let processor = Recorder::failing_on(1);
        let log = processor.log.clone();
        let stats = drive(processor, config(), |router, _clock| async move {
            router.submit(Item::new(1)).await.unwrap();
            router.submit(Item::new(1)).await.unwrap();
            router.submit(Item::new(2)).await.unwrap();
        })
        .await;

        let log = log.borrow();
        assert_eq!(log.failed, vec![1, 1], "on_error sees every classified failure");
        assert_eq!(
            log.processed,
            vec![(1, None), (2, None), (1, None)],
            "an in-doubt failure discards the key's state, so the retry reloads"
        );

        let relaxed = std::sync::atomic::Ordering::Relaxed;
        assert_eq!(stats.failed.load(relaxed), 2);
        assert_eq!(stats.in_doubt.load(relaxed), 2, "in-doubt is counted apart, for alerting");
        assert_eq!(stats.panicked.load(relaxed), 0, "a returned error is not a panic");
    }

    #[tokio::test(start_paused = true)]
    async fn a_concurrent_retry_is_coalesced_but_a_later_one_is_admitted() {
        let processor = Recorder::new();
        let log = processor.log.clone();
        let mut cfg = config();
        cfg.coalesce_duplicates = true;

        let stats = drive(processor, cfg, |router, _clock| async move {
            // Both reach the mailbox before the shard runs, so the second is a
            // retry of an operation that is still outstanding.
            router.try_submit(Item::retry(3, 77)).unwrap();
            router.try_submit(Item::retry(3, 77)).unwrap();
            tokio::time::sleep(Duration::from_millis(5)).await;

            // The original has completed and released its id, so this one is
            // a fresh operation as far as the runtime can tell — answering it
            // correctly is the store's job, not the scheduler's.
            router.submit(Item::retry(3, 77)).await.unwrap();
        })
        .await;

        let log = log.borrow();
        assert_eq!(log.coalesced, vec![3], "the concurrent retry never ran");
        assert_eq!(
            log.processed,
            vec![(3, None), (3, Some(1))],
            "the original ran once, and the later retry ran against its state"
        );
        assert_eq!(stats.coalesced.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn duplicate_ids_are_ignored_when_coalescing_is_off() {
        let processor = Recorder::new();
        let log = processor.log.clone();
        drive(processor, config(), |router, _clock| async move {
            router.try_submit(Item::retry(4, 9)).unwrap();
            router.try_submit(Item::retry(4, 9)).unwrap();
        })
        .await;

        let log = log.borrow();
        assert!(log.coalesced.is_empty());
        assert_eq!(log.processed.len(), 2, "coalescing is opt-in and off by default");
    }
}
