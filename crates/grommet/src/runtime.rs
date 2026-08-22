//! Building and owning a set of pinned shards.

use crate::clock::{Clock, SystemClock};
use crate::mailbox;
use crate::metrics::ShardStats;
use crate::processor::Processor;
use crate::router::Router;
use crate::shard::{self, ShardConfig};
use crate::topology::{Bound, PinPolicy, Plan, ShardPlacement, TopologyReport, Workload};
use crate::work::Envelope;
use std::fmt;
use std::sync::Arc;
use std::thread::JoinHandle;

/// What a shard thread knows about itself when it builds its processor.
///
/// This is what makes core-local resources possible: the factory runs on the
/// shard's own thread, after it has been placed, so it can size a connection
/// pool per core and — given [`node`] — pick the offload pool and allocations
/// that are local to the memory it will be touching.
///
/// [`node`]: ShardContext::node
#[derive(Clone, Copy, Debug)]
pub struct ShardContext {
    pub index: usize,
    pub shards: usize,
    /// Where the plan put this shard, if there was one to place it.
    pub placement: Option<ShardPlacement>,
    /// What binding achieved, which is not always what was asked for.
    pub bound: Bound,
}

impl ShardContext {
    /// The memory node this shard should keep its state and its offload work on.
    pub fn node(&self) -> Option<usize> {
        self.placement.map(|placement| placement.node)
    }

    /// The CPU this shard was placed on.
    pub fn cpu(&self) -> Option<usize> {
        self.placement.map(|placement| placement.cpu)
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum BuildError {
    /// `PinPolicy::Require` was set and these shard indices could not be
    /// pinned. Only reachable with the `topology` feature, which is what makes
    /// that variant exist.
    NotPinned(Vec<usize>),
    /// A shard thread died before it reported its placement.
    ShardFailed,
    /// The mailbox is deeper than the scheduler will ever admit, so most of
    /// the queue would sit where the scheduler cannot see it: it is missing
    /// from `pending`, it does not close the admission gate, and it is not
    /// bounded by the limit that appears to bound it.
    ///
    /// Raise `ShardConfig::scheduler.max_pending` to at least the mailbox
    /// depth, or shrink the mailbox.
    MailboxDeeperThanScheduler { mailbox: usize, max_pending: usize },
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPinned(shards) => {
                write!(f, "shards {shards:?} could not be pinned under PinPolicy::Require")
            }
            Self::ShardFailed => f.write_str("a shard thread failed during startup"),
            Self::MailboxDeeperThanScheduler { mailbox, max_pending } => write!(
                f,
                "mailbox depth {mailbox} exceeds max_pending {max_pending}, so {} items \
                 would queue where the scheduler cannot account for them",
                mailbox - max_pending,
            ),
        }
    }
}

impl std::error::Error for BuildError {}

pub struct Builder<P: Processor, C: Clock = SystemClock, const CLASSES: usize = 2> {
    shards: usize,
    mailbox: usize,
    shard_config: ShardConfig<CLASSES>,
    pin: PinPolicy,
    plan: Option<Arc<Plan>>,
    clock: C,
    stamp_arrival: bool,
    _processor: std::marker::PhantomData<fn() -> P>,
}

impl<P: Processor, const CLASSES: usize> Builder<P, SystemClock, CLASSES> {
    /// Configure `shards` reactors with the given per-class in-flight budgets.
    ///
    /// Placement is planned from this machine unless [`plan`] supplies one or
    /// [`PinPolicy::Disabled`] turns it off.
    ///
    /// [`plan`]: Builder::plan
    pub fn new(shards: usize, max_inflight: [usize; CLASSES]) -> Self {
        Self::with_clock(shards, max_inflight, SystemClock::new())
    }

    /// One reactor per shard placement in `plan`.
    ///
    /// This is the usual entry point once the layout matters: the plan already
    /// decided how many reactors the machine can carry, after reserving cores
    /// for the offload pool and for the OS, and after honouring any cgroup
    /// bandwidth limit. Choosing a shard count separately is choosing to
    /// disagree with it.
    pub fn for_plan(plan: Arc<Plan>, max_inflight: [usize; CLASSES]) -> Self {
        Self::new(plan.shards.len().max(1), max_inflight).plan(plan)
    }
}

impl<P: Processor, C: Clock, const CLASSES: usize> Builder<P, C, CLASSES> {
    pub fn with_clock(shards: usize, max_inflight: [usize; CLASSES], clock: C) -> Self {
        assert!(shards > 0, "a runtime needs at least one shard");
        Self {
            shards,
            mailbox: 1024,
            shard_config: ShardConfig::new(max_inflight),
            pin: PinPolicy::default(),
            plan: None,
            clock,
            stamp_arrival: true,
            _processor: std::marker::PhantomData,
        }
    }

    /// Mailbox depth per shard. This is the queue that absorbs bursts before
    /// submitters feel backpressure.
    ///
    /// Depth composes rather than replaces: a shard holds up to `capacity`
    /// items here *plus* whatever its scheduler has already admitted, so the
    /// worst-case queue in front of one shard is
    /// `capacity + ShardConfig::scheduler.max_pending` items, and the
    /// worst-case queue-wait is that many items times the service time. Size
    /// the pair against your latency objective, not either one alone.
    ///
    /// # Panics
    ///
    /// If `capacity` is zero.
    pub fn mailbox(mut self, capacity: usize) -> Self {
        assert!(capacity > 0, "a mailbox needs capacity");
        self.mailbox = capacity;
        self
    }

    /// The worst-case number of items queued in front of one shard: its
    /// mailbox plus everything its scheduler will admit.
    ///
    /// This is the number that sets tail latency, and it is the one worth
    /// watching when tuning either half.
    pub fn queue_depth(&self) -> usize {
        self.mailbox.saturating_add(self.shard_config.scheduler.max_pending)
    }

    pub fn shard_config(mut self, config: ShardConfig<CLASSES>) -> Self {
        self.shard_config = config;
        self
    }

    pub fn pin(mut self, policy: PinPolicy) -> Self {
        self.pin = policy;
        self
    }

    /// Place shards according to `plan`, round-robin if there are more shards
    /// than the plan has placements for.
    ///
    /// The same plan should be given to the offload pools, so that a shard and
    /// the workers it submits to agree about which memory node they are on.
    pub fn plan(mut self, plan: Arc<Plan>) -> Self {
        self.plan = Some(plan);
        self
    }

    /// Suppress a retry whose request id is already queued or in flight for
    /// the same key. See [`ShardConfig::coalesce_duplicates`].
    pub fn coalesce_duplicates(mut self, coalesce: bool) -> Self {
        self.shard_config.coalesce_duplicates = coalesce;
        self
    }

    /// See [`Router::with_options`].
    pub fn stamp_arrival(mut self, stamp: bool) -> Self {
        self.stamp_arrival = stamp;
        self
    }

    /// Start every shard, building one processor per shard on its own thread.
    ///
    /// The factory runs inside the shard's runtime, which is what lets each
    /// shard own core-local resources — connection pools, caches, buffers —
    /// rather than sharing one set across cores.
    pub fn spawn<F>(self, factory: F) -> Result<Runtime<P, C, CLASSES>, BuildError>
    where
        F: Fn(&ShardContext) -> P + Send + Sync + 'static,
    {
        // Cross-validate before anything is started, so a misconfiguration
        // costs nothing and is reported once rather than per shard.
        let max_pending = self.shard_config.scheduler.max_pending;
        if self.mailbox > max_pending {
            return Err(BuildError::MailboxDeeperThanScheduler {
                mailbox: self.mailbox,
                max_pending,
            });
        }

        // Reading the machine is deferred to here rather than done in `new`, so
        // that a runtime which never starts never pays for it, and so a caller
        // who supplies a plan never reads the machine twice.
        let plan = match (self.plan, self.pin) {
            (plan @ Some(_), _) => plan,
            (None, PinPolicy::Disabled) => None,
            (None, _) => crate::topology::detect(&Workload::default()).ok().map(Arc::new),
        };
        let placements: &[ShardPlacement] =
            plan.as_ref().map(|plan| plan.shards.as_slice()).unwrap_or_default();
        let placement_for =
            |index: usize| (!placements.is_empty()).then(|| placements[index % placements.len()]);

        let mut cpus: Vec<usize> =
            (0..self.shards).filter_map(|index| placement_for(index).map(|at| at.cpu)).collect();
        cpus.sort_unstable();
        cpus.dedup();
        let distinct_cores = cpus.len();

        let factory = Arc::new(factory);
        let (report, reports) = std::sync::mpsc::channel();

        let mut senders = Vec::with_capacity(self.shards);
        let mut workers = Vec::with_capacity(self.shards);
        let mut stats = Vec::with_capacity(self.shards);

        for index in 0..self.shards {
            let (tx, rx) = mailbox::channel::<Envelope<P::Work>>(self.mailbox);
            senders.push(tx);
            let shard_stats = Arc::new(ShardStats::<CLASSES>::default());
            stats.push(shard_stats.clone());

            let placement = placement_for(index);
            let context_shards = self.shards;
            let clock = self.clock.clone();
            let config = self.shard_config;
            let policy = self.pin;
            let factory = factory.clone();
            let report = report.clone();
            let plan = plan.clone();

            workers.push(
                std::thread::Builder::new()
                    .name(format!("shard-{index}"))
                    .spawn(move || {
                        // Bind first. Memory binding only governs pages touched
                        // afterwards, and everything this thread allocates from
                        // here on — the runtime, the processor, the key states —
                        // should come from its own node.
                        let bound = match (policy, placement, &plan) {
                            (PinPolicy::Disabled, _, _) | (_, None, _) | (_, _, None) => {
                                Bound::default()
                            }
                            (_, Some(placement), Some(plan)) => plan.bind_shard(&placement),
                        };
                        // Report before blocking forever, so the builder can
                        // fail fast rather than wait on a shard that started.
                        let _ = report.send((index, bound));

                        let runtime = tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                            .expect("shard runtime");
                        runtime.block_on(async move {
                            let context =
                                ShardContext { index, shards: context_shards, placement, bound };
                            let processor = factory(&context);
                            shard::run(rx, processor, clock, shard_stats, config).await;
                        });
                    })
                    .expect("spawn shard thread"),
            );
        }
        drop(report);

        let mut pinned = 0;
        let mut memory_bound = 0;
        let mut unpinned = Vec::new();
        for _ in 0..self.shards {
            let (index, bound) = reports.recv().map_err(|_| BuildError::ShardFailed)?;
            if bound.cpu {
                pinned += 1;
            } else {
                unpinned.push(index);
            }
            if bound.memory {
                memory_bound += 1;
            }
        }

        // Without the `topology` feature there is no `Require` to compare
        // against: a build that cannot bind a thread cannot be asked to insist
        // that it did, and the compiler says so at the call site.
        #[cfg(feature = "topology")]
        if self.pin == PinPolicy::Require && !unpinned.is_empty() {
            unpinned.sort_unstable();
            // Closing every mailbox tells the shards to drain and exit.
            drop(senders);
            for worker in workers {
                let _ = worker.join();
            }
            return Err(BuildError::NotPinned(unpinned));
        }

        Ok(Runtime {
            router: Some(Arc::new(Router::with_options(senders, self.clock, self.stamp_arrival))),
            workers,
            stats,
            report: TopologyReport {
                shards: self.shards,
                distinct_cores,
                pinned,
                memory_bound,
                policy: self.pin,
            },
        })
    }
}

/// A running set of shards. Dropping it closes every mailbox and waits for the
/// shards to drain.
pub struct Runtime<P: Processor, C: Clock = SystemClock, const CLASSES: usize = 2> {
    router: Option<Arc<Router<P::Work, C, CLASSES>>>,
    workers: Vec<JoinHandle<()>>,
    stats: Vec<Arc<ShardStats<CLASSES>>>,
    report: TopologyReport,
}

impl<P: Processor, const CLASSES: usize> Runtime<P, SystemClock, CLASSES> {
    /// Start configuring a runtime on the system clock. Use
    /// [`Builder::with_clock`] directly for a different one.
    pub fn builder(
        shards: usize,
        max_inflight: [usize; CLASSES],
    ) -> Builder<P, SystemClock, CLASSES> {
        Builder::new(shards, max_inflight)
    }

    /// Start configuring a runtime laid out by `plan`, one shard per placement.
    pub fn for_plan(
        plan: Arc<Plan>,
        max_inflight: [usize; CLASSES],
    ) -> Builder<P, SystemClock, CLASSES> {
        Builder::for_plan(plan, max_inflight)
    }
}

impl<P: Processor, C: Clock, const CLASSES: usize> Runtime<P, C, CLASSES> {
    pub fn router(&self) -> &Arc<Router<P::Work, C, CLASSES>> {
        self.router.as_ref().expect("router is present until shutdown")
    }

    pub fn stats(&self) -> &[Arc<ShardStats<CLASSES>>] {
        &self.stats
    }

    pub fn topology(&self) -> &TopologyReport {
        &self.report
    }

    /// Close the mailboxes and wait for every shard to finish draining.
    ///
    /// Shutdown is driven by dropping the router, so any clone of it that you
    /// are still holding will keep the shards alive. Drop those first.
    pub fn shutdown(mut self) {
        self.close();
    }

    fn close(&mut self) {
        drop(self.router.take());
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl<P: Processor, C: Clock, const CLASSES: usize> Drop for Runtime<P, C, CLASSES> {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::respond::Call;
    use crate::work::{IO, Work};
    use grommet_core::{ClassId, Disposition};
    use std::convert::Infallible;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

    #[derive(Debug)]
    struct Item(u64);

    impl Work for Item {
        type Key = u64;
        type Id = u64;
        fn key(&self) -> u64 {
            self.0
        }
        fn class(&self) -> ClassId {
            IO
        }
    }

    /// What the shard threads observed, readable from the test thread.
    ///
    /// A processor is built on its own shard's thread and never crosses back, so
    /// anything a test wants to assert on has to be written through a handle
    /// that is `Send + Sync`. That is the only reason this is not an `Rc` like
    /// the single-threaded processors elsewhere in the crate.
    #[derive(Default)]
    struct Observed {
        contexts: Mutex<Vec<(usize, usize)>>,
        processed: Mutex<Vec<(usize, u64)>>,
        dispatches: AtomicUsize,
    }

    /// Counts dispatches per key and answers with the running total, so a test
    /// can tell a second dispatch of a key from a first.
    #[derive(Clone)]
    struct Counter {
        index: usize,
        observed: Arc<Observed>,
    }

    impl Processor for Counter {
        type Work = Call<Item, u64>;
        type State = u64;
        type Error = Infallible;

        async fn process(
            &self,
            key: u64,
            state: Option<u64>,
            call: Call<Item, u64>,
        ) -> Result<Disposition<u64>, Infallible> {
            let (_, responder) = call.into_parts();
            let count = state.unwrap_or(0) + 1;
            self.observed.processed.lock().unwrap().push((self.index, key));
            self.observed.dispatches.fetch_add(1, Relaxed);
            responder.send(count);
            Ok(Disposition::Keep(count))
        }
    }

    #[test]
    fn a_mailbox_deeper_than_the_scheduler_is_refused_before_anything_starts() {
        let mut config = ShardConfig::new([4, 4]);
        config.scheduler.max_pending = 16;
        let error = Runtime::<Counter>::builder(1, [4, 4])
            .pin(PinPolicy::Disabled)
            .shard_config(config)
            .mailbox(64)
            .spawn(|_| unreachable!("the factory must never run for a rejected configuration"));

        let Err(error) = error else {
            panic!("a mailbox the scheduler cannot account for is a misconfiguration");
        };
        assert!(matches!(
            error,
            BuildError::MailboxDeeperThanScheduler { mailbox: 64, max_pending: 16 }
        ));
        assert!(error.to_string().contains("48"), "the message names the unaccounted depth");
    }

    #[test]
    fn queue_depth_is_the_mailbox_and_the_scheduler_together() {
        let mut config = ShardConfig::<2>::new([4, 4]);
        config.scheduler.max_pending = 512;
        let builder = Runtime::<Counter>::builder(1, [4, 4]).shard_config(config).mailbox(128);
        assert_eq!(
            builder.queue_depth(),
            640,
            "tail latency is set by both queues, so the depth that matters is their sum"
        );
    }

    /// A runtime over `shards` unpinned shard threads, plus the shared record of
    /// what they did. Pinning is off because these tests are about the shard
    /// lifecycle, and a CI runner may not permit binding at all.
    fn runtime(shards: usize) -> (Runtime<Counter>, Arc<Observed>) {
        let observed = Arc::new(Observed::default());
        let factory = observed.clone();
        let runtime = Runtime::<Counter>::builder(shards, [16, 16])
            .pin(PinPolicy::Disabled)
            .spawn(move |context: &ShardContext| {
                factory.contexts.lock().unwrap().push((context.index, context.shards));
                Counter { index: context.index, observed: factory.clone() }
            })
            .expect("an unpinned runtime starts on any machine");
        (runtime, observed)
    }

    #[tokio::test]
    async fn a_key_keeps_its_state_between_dispatches() {
        let (runtime, _observed) = runtime(2);

        // The same key twice: the second dispatch must see what the first kept.
        assert_eq!(runtime.router().call(Item(7)).await.expect("first call"), 1);
        assert_eq!(runtime.router().call(Item(7)).await.expect("second call"), 2);

        // A different key starts from nothing, whichever shard it lands on.
        assert_eq!(runtime.router().call(Item(8)).await.expect("other key"), 1);
    }

    #[tokio::test]
    async fn every_shard_builds_its_own_processor_on_its_own_thread() {
        let (runtime, observed) = runtime(4);

        // A shard reports its binding before it builds its tokio runtime and
        // calls the factory, so `spawn` returning does not mean every processor
        // exists yet — only that every thread got far enough to say where it
        // landed. Shutdown joins the threads, which is the point every factory
        // has certainly run.
        runtime.shutdown();

        let mut contexts = observed.contexts.lock().unwrap().clone();
        contexts.sort_unstable();
        assert_eq!(
            contexts,
            vec![(0, 4), (1, 4), (2, 4), (3, 4)],
            "each shard is built once, and is told how many it is one of"
        );
    }

    #[tokio::test]
    async fn work_is_processed_by_the_shard_that_owns_its_key() {
        let (runtime, observed) = runtime(4);

        for key in 0..16u64 {
            runtime.router().call(Item(key)).await.expect("call");
        }
        // The router's answer for a key and the shard that actually ran it are
        // the same claim; if they ever disagree, key affinity is a fiction.
        for (index, key) in observed.processed.lock().unwrap().iter() {
            assert_eq!(*index, runtime.router().shard_index(*key), "key {key} ran off-shard");
        }
    }

    #[tokio::test]
    async fn a_disabled_pin_policy_plans_nothing_and_reports_nothing_pinned() {
        let (runtime, _observed) = runtime(3);

        let report = runtime.topology();
        assert_eq!(report.shards, 3);
        assert_eq!(report.policy, PinPolicy::Disabled);
        assert_eq!(report.pinned, 0, "nothing was asked to bind");
        assert_eq!(report.memory_bound, 0);
        // No plan means no placements, so no CPU was claimed by any shard.
        assert_eq!(report.distinct_cores, 0);
    }

    #[tokio::test]
    async fn shutdown_drains_work_that_was_already_queued() {
        let (runtime, observed) = runtime(2);

        // Submitted without awaiting a reply, so these are still in flight or
        // queued when shutdown is called. `Call` is not `Debug`, so the result
        // is asserted rather than unwrapped.
        for key in 0..32u64 {
            let (call, _receive) = Call::new(Item(key));
            assert!(runtime.router().submit(call).await.is_ok(), "shard {key} accepted");
        }
        runtime.shutdown();

        assert_eq!(
            observed.dispatches.load(Relaxed),
            32,
            "closing the mailboxes must drain what was queued, not discard it"
        );
    }

    #[tokio::test]
    async fn dropping_a_runtime_drains_it_the_same_way() {
        let (runtime, observed) = runtime(2);

        for key in 0..16u64 {
            let (call, _receive) = Call::new(Item(key));
            assert!(runtime.router().submit(call).await.is_ok(), "shard {key} accepted");
        }
        drop(runtime);

        assert_eq!(observed.dispatches.load(Relaxed), 16);
    }

    #[tokio::test]
    async fn stats_are_reported_per_shard_and_account_for_every_dispatch() {
        let (runtime, _observed) = runtime(3);

        for key in 0..12u64 {
            runtime.router().call(Item(key)).await.expect("call");
        }
        assert_eq!(runtime.stats().len(), 3, "one set of counters per shard");

        // A shard counts into thread-local `Cell`s and publishes them to these
        // atomics on its tick, so a read taken the instant a call returns is
        // racing that tick. Shutdown ends with a final publish, which is the
        // point at which the totals are actually settled — so the handles are
        // kept and read after it.
        let stats: Vec<_> = runtime.stats().to_vec();
        runtime.shutdown();

        let completed: u64 = stats.iter().map(|shard| shard.completed.load(Relaxed)).sum();
        assert_eq!(completed, 12, "every dispatch is counted by exactly one shard");
    }

    #[test]
    fn a_shard_that_was_never_placed_reports_no_cpu_and_no_node() {
        let floating =
            ShardContext { index: 0, shards: 1, placement: None, bound: Bound::default() };
        assert_eq!(floating.cpu(), None);
        assert_eq!(floating.node(), None);

        let placed =
            ShardContext { placement: Some(ShardPlacement { cpu: 5, node: 1 }), ..floating };
        assert_eq!(placed.cpu(), Some(5));
        assert_eq!(placed.node(), Some(1));
    }

    #[test]
    fn a_build_error_names_what_went_wrong() {
        assert_eq!(
            BuildError::NotPinned(vec![1, 3]).to_string(),
            "shards [1, 3] could not be pinned under PinPolicy::Require"
        );
        assert_eq!(BuildError::ShardFailed.to_string(), "a shard thread failed during startup");
    }

    #[test]
    #[should_panic(expected = "a runtime needs at least one shard")]
    fn a_runtime_with_no_shards_is_refused() {
        let _ = Runtime::<Counter>::builder(0, [1, 1]);
    }

    #[test]
    #[should_panic(expected = "a mailbox needs capacity")]
    fn a_mailbox_with_no_capacity_is_refused() {
        let _ = Runtime::<Counter>::builder(1, [1, 1]).mailbox(0);
    }
}
