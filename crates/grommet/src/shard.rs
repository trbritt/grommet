//! The per-core shard reactor.
//!
//! One shard owns one core, one scheduler, one processor instance and every key
//! that routes to it. It runs a single-threaded Tokio runtime, so nothing it
//! holds needs to be `Send` and nothing it does needs synchronization.

use crate::clock::Clock;
use crate::driver::Driver;
use crate::error::{Fallout, ProcessError};
use crate::mailbox::Inbox;
use crate::metrics::{ShardHot, ShardStats};
use crate::outstanding::{Harvest, Outstanding};
use crate::processor::{KeyOf, PanicPolicy, Processor};
use crate::work::{Envelope, Stamped, Work};
use ahash::AHashSet;
use futures::FutureExt;
use futures::future::Either;
use grommet_core::timer::Wheel;
use grommet_core::{Admit, ClassId, Completion, Dispatch, Disposition, Scheduler};
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use std::time::Duration;

/// The scheduler as a shard specializes it.
type Book<K, W, S, const CLASSES: usize> = Scheduler<K, Stamped<W>, S, CLASSES>;

/// Request ids currently queued or in flight, scoped by key.
type Live<K, I> = AHashSet<(K, I)>;

/// What a shard schedules on its own wheel.
///
/// One variant today. It is an enum rather than a bare marker because the wheel
/// is the mechanism dispatch deadlines will use, and they arrive as another
/// variant rather than as a second timer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Timer {
    /// Publish gauges and sweep eviction.
    Tick,
}

/// Timers one shard can hold at once. Only the periodic tick uses the wheel
/// today; per-dispatch deadlines will size this against the in-flight budget.
const TIMERS: usize = 64;

#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct ShardConfig<const CLASSES: usize = 2> {
    pub scheduler: grommet_core::Config<CLASSES>,
    /// Maximum already-ready mailbox items admitted after one awaited receive.
    /// This amortizes cross-thread wakeups, which dominate at high rates, while
    /// bounding how long an arrival burst can delay completion harvesting.
    pub admit_batch: usize,
    /// Maximum completions harvested after one awaited completion.
    ///
    /// The mirror of [`admit_batch`], and for the same reason: completions
    /// arrive in the same numbers admissions do, and taking each one through a
    /// full reactor turn spends more on the turn than on the completion. A
    /// batch shares one clock reading and one pass of the loop's bookkeeping
    /// across everything already finished.
    ///
    /// It is a cap, not a target — harvesting stops as soon as nothing else is
    /// ready. The cap is what keeps a steady stream of completions from
    /// starving the other arms of the loop, exactly as `admit_batch` keeps a
    /// producer from starving them.
    ///
    /// [`admit_batch`]: ShardConfig::admit_batch
    pub complete_batch: usize,
    /// Maximum eviction flushes in flight at once.
    ///
    /// Dispatched work is bounded by the scheduler's in-flight budgets, but
    /// `on_evict` flushes share the shard's outstanding set and are otherwise
    /// bounded only by resident keys. This caps their share of it; candidates
    /// beyond the cap wait in the sweep's own buffer until a flush completes,
    /// which the eviction machinery already tolerates.
    pub flush_slots: usize,
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
            complete_batch: 64,
            flush_slots: 256,
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

/// What every stage of a turn needs and no stage changes.
///
/// Bundled because the alternative is threading five references through six
/// functions: it says plainly that these are the shard's fixed parts, and
/// leaves each stage's own signature to name only the state it actually moves.
struct Turn<'a, P: Processor, C: Clock, const CLASSES: usize> {
    processor: &'a P,
    clock: &'a C,
    hot: &'a ShardHot,
    stats: &'a ShardStats<CLASSES>,
    cfg: &'a ShardConfig<CLASSES>,
}

/// What one pass over the mailbox took in.
struct Admitted {
    count: usize,
    /// Every sender is gone: stop admitting and drain.
    closed: bool,
    /// When the first item of the pass was taken, for the busy accounting.
    at: Duration,
}

/// Fill every class budget from its own round-robin ring.
///
/// `start` builds the future for one dispatch. It is a parameter because an
/// `async fn`'s future has no nameable type, so the concrete element type of
/// the outstanding set can only arrive by inference from the call site.
fn dispatch_ready<P, F, const CLASSES: usize>(
    book: &mut Book<KeyOf<P>, P::Work, P::State, CLASSES>,
    outstanding: &mut Outstanding<F>,
    hot: &ShardHot,
    now: Duration,
    mut start: impl FnMut(Dispatch<KeyOf<P>, Stamped<P::Work>, P::State>) -> F,
) where
    P: Processor,
    F: Future<Output = OutcomeOf<P>>,
{
    for class in 0..CLASSES {
        while let Some(dispatch) = book.next(class as ClassId, now) {
            hot.add(
                &hot.queue_wait_nanos,
                now.saturating_sub(dispatch.payload.enqueued).as_nanos() as u64,
            );
            outstanding.push(start(dispatch));
        }
    }
}

/// Hand back work whose deadline passed before it could be dispatched. It
/// never costs a dispatch turn.
fn shed_expired<P: Processor, const CLASSES: usize>(
    book: &mut Book<KeyOf<P>, P::Work, P::State, CLASSES>,
    live: &mut Live<KeyOf<P>, <P::Work as Work>::Id>,
    processor: &P,
    hot: &ShardHot,
) {
    while let Some((key, stamped)) = book.pop_expired() {
        hot.bump(&hot.expired);
        if let Some(id) = stamped.request_id {
            live.remove(&(key, id));
        }
        processor.on_expired(key, stamped.work);
    }
}

/// Move as many staged eviction candidates into flight as the flush budget
/// allows, oldest first. The rest wait in `staged`; they are already quiesced,
/// so nothing dispatches for them meanwhile.
fn begin_flushes<P, F>(
    staged: &mut Vec<(KeyOf<P>, P::State)>,
    outstanding: &mut Outstanding<F>,
    hot: &ShardHot,
    flushes: &mut usize,
    budget: usize,
    mut start: impl FnMut(KeyOf<P>, P::State) -> F,
) where
    P: Processor,
    F: Future<Output = OutcomeOf<P>>,
{
    let room = budget.saturating_sub(*flushes).min(staged.len());
    for (key, state) in staged.drain(..room) {
        hot.bump(&hot.evicted);
        *flushes += 1;
        outstanding.push(start(key, state));
    }
}

/// Take a bounded burst from the mailbox.
///
/// Popping rather than awaiting a receive: nothing compound is built and
/// nothing is dropped part-polled, so the cancel safety a `select!` would have
/// leaned on is not a question that arises. `waker` is whatever should be
/// notified when the mailbox is empty — the task, or the sleeping thread.
fn drain_inbox<P: Processor, C: Clock, const CLASSES: usize>(
    rx: &mut Inbox<Envelope<P::Work>>,
    book: &mut Book<KeyOf<P>, P::Work, P::State, CLASSES>,
    live: &mut Live<KeyOf<P>, <P::Work as Work>::Id>,
    turn: &Turn<'_, P, C, CLASSES>,
    waker: &Waker,
    now: Duration,
) -> Admitted {
    let Turn { processor, clock, hot, cfg, .. } = turn;
    let mut taken = Admitted { count: 0, closed: false, at: now };
    let mut cx = Context::from_waker(waker);
    while taken.count < cfg.admit_batch.max(1) && !book.is_saturated() {
        match rx.poll_recv(&mut cx) {
            Poll::Ready(Some(envelope)) => {
                if taken.count == 0 {
                    taken.at = clock.now();
                }
                admit_one::<P, CLASSES>(
                    book,
                    live,
                    processor,
                    hot,
                    envelope,
                    cfg.coalesce_duplicates,
                );
                taken.count += 1;
            }
            // The router is gone.
            Poll::Ready(None) => {
                taken.closed = true;
                break;
            }
            // Registered for a wake; nothing queued right now.
            Poll::Pending => break,
        }
    }
    taken
}

/// Give the scheduler back everything that finished.
///
/// One clock reading is shared across the batch. That is deliberate: the
/// scheduler needs its timestamps monotonic rather than individually precise,
/// and a batch spans microseconds against an eviction window of seconds.
/// Returns the harvest report and whether a caught panic demands an abort.
fn harvest_completions<P, F, const CLASSES: usize>(
    outstanding: &mut Outstanding<F>,
    book: &mut Book<KeyOf<P>, P::Work, P::State, CLASSES>,
    live: &mut Live<KeyOf<P>, <P::Work as Work>::Id>,
    hot: &ShardHot,
    flushes: &mut usize,
    cfg: &ShardConfig<CLASSES>,
    at: Duration,
) -> (Harvest, bool)
where
    P: Processor,
    F: Future<Output = OutcomeOf<P>>,
{
    let mut abort = false;
    let report = outstanding.harvest(cfg.complete_batch, |outcome| match outcome {
        Outcome::Ran { completion, request_id, panicked, fallout } => {
            hot.bump(&hot.completed);
            if let Some(fallout) = fallout {
                hot.bump(&hot.failed);
                if fallout.is_in_doubt() {
                    hot.bump(&hot.in_doubt);
                }
            }
            if panicked {
                hot.bump(&hot.panicked);
                abort |= cfg.panic_policy == PanicPolicy::Abort;
            }
            if let Some(id) = request_id {
                live.remove(&(completion.key, id));
            }
            book.complete(completion, at);
        }
        Outcome::Flushed(key) => {
            *flushes -= 1;
            book.finish_evict(key, at);
        }
    });
    (report, abort)
}

/// Run anything the wheel says is due. Returns whether it did.
fn service_timers<P: Processor, C: Clock, const CLASSES: usize>(
    timers: &mut Wheel<Timer>,
    due: &mut Vec<Timer>,
    book: &mut Book<KeyOf<P>, P::Work, P::State, CLASSES>,
    staged: &mut Vec<(KeyOf<P>, P::State)>,
    turn: &Turn<'_, P, C, CLASSES>,
    now: Duration,
) -> bool {
    let Turn { stats, hot, cfg, .. } = turn;
    if !timers.is_due_duration(now) {
        return false;
    }
    timers.advance_duration(now, due).expect("monotonic time fits a u64 of nanoseconds");
    for timer in due.drain(..) {
        match timer {
            Timer::Tick => {
                stats.publish(hot, &book.snapshot());
                // Only sweep once the last batch has been fully handed over,
                // so candidates never pile up here: a key waits on the
                // scheduler's own idle list, which is already exact.
                if staged.is_empty() {
                    book.evict(now, staged);
                }
                timers
                    .try_insert_duration(now.saturating_add(cfg.tick), Timer::Tick)
                    .expect("the tick is the only timer, so its slot is free");
            }
        }
    }
    true
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

    // Bounded by what may be dispatched plus what may be flushing, which is
    // exactly what the two gates above the pushes allow. A push beyond it is an
    // accounting bug, and the set says so rather than growing.
    let capacity: usize =
        cfg.scheduler.max_inflight.iter().sum::<usize>().saturating_add(cfg.flush_slots);
    // The element type is the `Either` of the two futures pushed below — one
    // concrete type per processor, inferred rather than named because an
    // `async fn`'s future has no nameable type. That is what keeps the set a
    // flat slab with no dynamic dispatch anywhere.
    let mut outstanding = Outstanding::with_capacity(capacity);

    // Where a sweep hands its candidates back, reused across sweeps so the
    // steady state allocates nothing. It doubles as the queue for anything the
    // flush budget could not take yet — a sweep may return more candidates than
    // there are slots, and draining from the front keeps them in the
    // least-recently-idle order the scheduler chose. A key waiting here is
    // already quiesced, so nothing dispatches for it meanwhile.
    let mut staged: Vec<(KeyOf<P>, P::State)> = Vec::new();
    let mut timers: Wheel<Timer> = Wheel::with_capacity(TIMERS);
    let mut due: Vec<Timer> = Vec::with_capacity(TIMERS);
    let mut open = true;
    let mut final_flush = true;
    let mut flushes = 0usize;

    let mut driver = crate::driver::Host::new();
    let turn = Turn { processor: &processor, clock: &clock, hot: &hot, stats: &stats, cfg: &cfg };

    // Carried across turns rather than re-read at the top of each. The reading
    // taken at the end of one turn's bookkeeping and the one that would be taken
    // at the start of the next are separated by nothing but the loop back-edge,
    // and `Instant::now` is a real syscall-shaped cost — around 25ns, against a
    // hot path measured in hundreds.
    let mut now = clock.now();

    // Time is the `Clock`'s, not tokio's. The two used to disagree: the interval
    // fired on tokio's clock while eviction judged idleness against this one, so
    // a simulated clock could hold still while sweeps ran anyway. Now a sweep
    // happens exactly when the clock the caller supplied says it should, which is
    // what lets a simulation reproduce.
    timers
        .try_insert_duration(now.saturating_add(cfg.tick), Timer::Tick)
        .expect("a fresh wheel has room for the tick");

    std::future::poll_fn(|cx| {
        // Before anything this turn will rely on when deciding to wait. A wake
        // arriving after this re-polls the loop, so nothing can be lost in the
        // gap between finding no work and going to sleep.
        outstanding.register_owner(cx.waker());

        // Once per wake, not once per turn. Within a turn the reading is
        // carried and refreshed only after work, because `Clock::now` is a real
        // syscall-shaped cost against a hot path measured in hundreds of
        // nanoseconds. But arriving here means something happened — and if that
        // something was the wheel's own deadline, a stale reading would find
        // nothing due and arm the same wait again, forever.
        now = clock.now();

        loop {
            dispatch_ready::<P, _, CLASSES>(&mut book, &mut outstanding, &hot, now, |dispatch| {
                Either::Left(run_one(processor.clone(), dispatch))
            });
            shed_expired(&mut book, &mut live, &processor, &hot);
            begin_flushes::<P, _>(
                &mut staged,
                &mut outstanding,
                &hot,
                &mut flushes,
                cfg.flush_slots,
                |key, state| Either::Right(run_flush(processor.clone(), key, state)),
            );
            debug_assert!(book.check_invariants().is_ok());

            // Closed mailbox and nothing left in flight: everything queued has
            // already been dispatched and harvested.
            if !open && outstanding.is_empty() && staged.is_empty() {
                // Resident state is a write-back cache — between dispatches the
                // scheduler holds the only copy. Exiting without draining it
                // discards writes the processor was told it could keep, so the
                // last thing a shard does is hand every one of them to
                // `on_evict`. By here nothing can arrive and nothing can
                // dispatch, so one pass is enough and the flushes it starts are
                // the last work there is.
                if std::mem::take(&mut final_flush) {
                    book.evict_all(&mut staged);
                    if !staged.is_empty() {
                        continue;
                    }
                }
                return Poll::Ready(());
            }

            let admitted = if open {
                let taken = drain_inbox(&mut rx, &mut book, &mut live, &turn, cx.waker(), now);
                if taken.closed {
                    open = false;
                }
                if taken.count > 0 {
                    now = clock.now();
                    hot.add(&hot.busy_nanos, now.saturating_sub(taken.at).as_nanos() as u64);
                }
                taken.count + usize::from(taken.closed)
            } else {
                0
            };

            let at = clock.now();
            let (harvest, abort) = harvest_completions::<P, _, CLASSES>(
                &mut outstanding,
                &mut book,
                &mut live,
                &hot,
                &mut flushes,
                &cfg,
                at,
            );
            if abort {
                stats.publish(&hot, &book.snapshot());
                std::process::abort();
            }
            if harvest.finished > 0 {
                now = clock.now();
                hot.add(&hot.busy_nanos, now.saturating_sub(at).as_nanos() as u64);
            }

            let fired = service_timers::<P, C, CLASSES>(
                &mut timers,
                &mut due,
                &mut book,
                &mut staged,
                &turn,
                now,
            );

            // A truncated harvest counts as progress: its wakes are already
            // spent, so waiting now would strand the bits it handed back.
            if admitted > 0 || harvest.finished > 0 || fired || harvest.truncated {
                continue;
            }

            // Nothing to do this turn. The wheel says when the next thing is;
            // the driver decides how to wait for it, and whether waiting means
            // suspending this task or sleeping in place.
            if driver.wait(timers.next_wakeup_duration(), now, cx).is_pending() {
                return Poll::Pending;
            }
            // The deadline passed while the shard was working: take another
            // turn rather than sleeping through it.
            now = clock.now();
        }
    })
    .await;

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

        drive(processor, cfg, |router, clock| async move {
            router.submit(Item::new(4)).await.unwrap();
            // The sweep is scheduled on the shard's own clock now, so this
            // drives that clock rather than waiting on the wall. The sleep is
            // still needed for a different reason: everything here shares one
            // thread, and the shard only makes progress while this task is
            // suspended.
            for _ in 0..50 {
                if !observed.borrow().evicted.is_empty() {
                    break;
                }
                clock.advance(Duration::from_millis(1));
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

    /// Harvesting completions in a batch must not change what the shard is
    /// obliged to do, whatever the cap.
    ///
    /// Not that it produces one fixed interleaving: the cap decides how many
    /// slots a pass visits, so it necessarily changes which keys advance
    /// together, and both a rotation of one and a rotation of many are fair.
    /// What may never change is the contract — every item processed exactly
    /// once, and each key's own items in the order they were submitted.
    #[tokio::test(start_paused = true)]
    async fn the_completion_batch_size_changes_no_obligation_of_the_shard() {
        const KEYS: u64 = 12;
        const ROUNDS: u64 = 8;

        async fn run_with(complete_batch: usize) -> (Vec<(u64, Option<u64>)>, u64) {
            let processor = Recorder::new();
            let log = processor.log.clone();
            let mut cfg = config();
            cfg.complete_batch = complete_batch;
            // Enough concurrency that completions genuinely arrive together;
            // with a budget of one there would never be a second to harvest.
            cfg.scheduler.max_inflight = [16, 16];

            let stats = drive(processor, cfg, |router, _clock| async move {
                for _ in 0..ROUNDS {
                    for key in 0..KEYS {
                        router.submit(Item::new(key)).await.unwrap();
                    }
                }
            })
            .await;
            let processed = log.borrow().processed.clone();
            (processed, stats.completed.load(std::sync::atomic::Ordering::Relaxed))
        }

        for batch in [1, 2, 7, 64, 4096] {
            let (processed, completed) = run_with(batch).await;
            assert_eq!(
                processed.len() as u64,
                KEYS * ROUNDS,
                "batch {batch} processed the wrong number of items"
            );
            assert_eq!(completed, KEYS * ROUNDS, "batch {batch} lost a completion");

            for key in 0..KEYS {
                let states: Vec<Option<u64>> = processed
                    .iter()
                    .filter(|(seen, _)| *seen == key)
                    .map(|(_, state)| *state)
                    .collect();
                // A key's state accumulates one per dispatch, so the sequence
                // is its submission order made visible.
                let expected: Vec<Option<u64>> =
                    std::iter::once(None).chain((1..ROUNDS).map(Some)).collect();
                assert_eq!(
                    states, expected,
                    "batch {batch}: key {key} ran out of order or ran twice"
                );
            }
        }
    }

    /// A batch of zero would mean harvesting nothing, which cannot make
    /// A batch of zero would mean harvesting nothing, which cannot make
    /// progress. It is clamped rather than rejected, because the value comes
    /// from a config struct a caller can build by hand.
    #[tokio::test(start_paused = true)]
    async fn a_zero_completion_batch_still_makes_progress() {
        let processor = Recorder::new();
        let log = processor.log.clone();
        let mut cfg = config();
        cfg.complete_batch = 0;

        drive(processor, cfg, |router, _clock| async move {
            for key in 0..4u64 {
                router.submit(Item::new(key)).await.unwrap();
            }
        })
        .await;
        assert_eq!(log.borrow().processed.len(), 4);
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
