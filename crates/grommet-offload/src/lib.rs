//! Pinned, bounded Rayon pools for CPU-bound work submitted from shards.
//!
//! Shard cores and compute cores are disjoint on purpose. A shard core spends
//! its time dispatching and awaiting; a compute core spends its time saturated.
//! Running both on one core makes each worse, and the split is what keeps a long
//! computation from adding latency to unrelated keys.
//!
//! There is a pool per memory node rather than one pool for the machine. A shard
//! that ships a closure and its captured data to a worker on another node pays
//! the interconnect on every task, which is exactly the cost that pinning the
//! shard was meant to avoid — so [`OffloadPools`] builds the set a [`Plan`] asks
//! for, and each shard submits to the one local to itself.

#![deny(unsafe_code)]

use grommet::offload::{Offload, OffloadError};
use grommet::topology::{OffloadPool, Plan};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

#[derive(Debug, Default)]
pub struct OffloadStats {
    pub runs: AtomicU64,
    pub panics: AtomicU64,
    pub latency_nanos_sum: AtomicU64,
    pub latency_max_nanos: AtomicU64,
    /// Time spent waiting for a permit, which is what backpressure from a
    /// saturated pool looks like from the shard's side. This is the number
    /// [`calibrate`] reads to decide the pool is too small.
    ///
    /// [`calibrate`]: grommet::topology::calibrate
    pub permit_wait_nanos: AtomicU64,
    /// Workers whose CPU binding took effect. Fewer than [`RayonOffload::workers`]
    /// means some of this pool is running wherever the OS put it.
    pub bound_workers: AtomicU64,
}

impl OffloadStats {
    fn record(&self, elapsed: Duration) {
        let nanos = elapsed.as_nanos() as u64;
        self.runs.fetch_add(1, Relaxed);
        self.latency_nanos_sum.fetch_add(nanos, Relaxed);
        self.latency_max_nanos.fetch_max(nanos, Relaxed);
    }

    /// Mean completed-task latency in nanoseconds, or zero before the first task
    /// completes.
    pub fn mean_latency_nanos(&self) -> u64 {
        let runs = self.runs.load(Relaxed);
        if runs == 0 {
            return 0;
        }
        self.latency_nanos_sum.load(Relaxed) / runs
    }

    /// Summed duration of completed tasks. Workers are saturated for the whole
    /// of a task, so over a window this is compute demand in core-seconds.
    pub fn busy(&self) -> Duration {
        Duration::from_nanos(self.latency_nanos_sum.load(Relaxed))
    }

    pub fn permit_wait(&self) -> Duration {
        Duration::from_nanos(self.permit_wait_nanos.load(Relaxed))
    }
}

/// A Rayon pool whose workers are bound to dedicated CPUs on one memory node,
/// with a bound on how many tasks may be outstanding at once.
#[derive(Clone)]
pub struct RayonOffload {
    pool: Arc<rayon::ThreadPool>,
    permits: Arc<tokio::sync::Semaphore>,
    stats: Arc<OffloadStats>,
    workers: usize,
    node: Option<usize>,
}

impl RayonOffload {
    /// One worker per CPU in `pool`, bound to it, with four outstanding tasks
    /// per worker.
    pub fn for_pool(plan: Arc<Plan>, pool: &OffloadPool) -> Self {
        Self::with_queue_depth(plan, pool, 4)
    }

    /// `queue_depth` outstanding tasks per worker. Deeper keeps workers fed
    /// through bursts; shallower makes a shard feel backpressure sooner, which
    /// is usually what a latency objective wants — and makes
    /// `permit_wait_nanos` a more responsive signal that the pool is too small.
    pub fn with_queue_depth(plan: Arc<Plan>, pool: &OffloadPool, queue_depth: usize) -> Self {
        let workers = pool.cpus.len().max(1);
        let node = pool.node;
        let placement = pool.clone();
        let stats = Arc::new(OffloadStats::default());
        let binding = stats.clone();
        Self::assemble(
            workers,
            queue_depth,
            Some(node),
            stats,
            Some(Arc::new(move |worker: usize| {
                if plan.bind_offload_worker(&placement, worker).cpu {
                    binding.bound_workers.fetch_add(1, Relaxed);
                }
            })),
        )
    }

    /// A pool of `workers` threads that are not bound to anything.
    ///
    /// For tests, for platforms that will not honour binding, and for callers
    /// with no plan. It is a working pool, not a placed one: the OS decides
    /// where these threads run, including on top of the shard reactors.
    pub fn unbound(workers: usize, queue_depth: usize) -> Self {
        Self::assemble(workers.max(1), queue_depth, None, Arc::new(OffloadStats::default()), None)
    }

    fn assemble(
        workers: usize,
        queue_depth: usize,
        node: Option<usize>,
        stats: Arc<OffloadStats>,
        bind: Option<Arc<dyn Fn(usize) + Send + Sync>>,
    ) -> Self {
        assert!(queue_depth > 0, "a compute pool needs a queue depth");
        let panic_stats = stats.clone();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(move |index| match node {
                Some(node) => format!("offload-n{node}-{index}"),
                None => format!("offload-{index}"),
            })
            .start_handler(move |index| {
                if let Some(bind) = &bind {
                    bind(index);
                }
            })
            // Without a handler, Rayon aborts the process when a spawned task
            // panics. A panicking task must only fail its own caller, which it
            // does: the dropped responder surfaces as `WorkerLost`.
            .panic_handler(move |_| {
                panic_stats.panics.fetch_add(1, Relaxed);
            })
            .build()
            .expect("build offload pool");

        Self {
            pool: Arc::new(pool),
            permits: Arc::new(tokio::sync::Semaphore::new(workers * queue_depth)),
            stats,
            workers,
            node,
        }
    }

    pub fn workers(&self) -> usize {
        self.workers
    }

    /// The memory node this pool's workers sit on, if it was placed.
    pub fn node(&self) -> Option<usize> {
        self.node
    }

    pub fn stats(&self) -> &Arc<OffloadStats> {
        &self.stats
    }
}

/// The pools a [`Plan`] calls for: one per memory node, each bound to its own.
#[derive(Clone, Default)]
pub struct OffloadPools {
    pools: Vec<RayonOffload>,
}

impl OffloadPools {
    /// Build every pool in `plan`. Empty when the plan reserved no compute
    /// cores, which is what a `compute_fraction` of zero asks for.
    pub fn build(plan: Arc<Plan>, queue_depth: usize) -> Self {
        let pools = plan
            .offload
            .iter()
            .map(|pool| RayonOffload::with_queue_depth(plan.clone(), pool, queue_depth))
            .collect();
        Self { pools }
    }

    /// The pool local to `node`, falling back to the first — a shard is better
    /// served by a remote pool than by no pool.
    pub fn for_node(&self, node: Option<usize>) -> Option<RayonOffload> {
        node.and_then(|node| self.pools.iter().find(|pool| pool.node == Some(node)))
            .or_else(|| self.pools.first())
            .cloned()
    }

    pub fn is_empty(&self) -> bool {
        self.pools.is_empty()
    }

    pub fn workers(&self) -> usize {
        self.pools.iter().map(RayonOffload::workers).sum()
    }

    /// Workers across every pool whose binding took effect.
    pub fn bound_workers(&self) -> usize {
        self.pools.iter().map(|pool| pool.stats().bound_workers.load(Relaxed) as usize).sum()
    }

    pub fn stats(&self) -> Vec<Arc<OffloadStats>> {
        self.pools.iter().map(|pool| pool.stats().clone()).collect()
    }
}

impl Offload for RayonOffload {
    async fn run<F, T>(&self, task: F) -> Result<T, OffloadError>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let waiting = Instant::now();
        let _permit =
            self.permits.clone().acquire_owned().await.map_err(|_| OffloadError::Closed)?;
        self.stats.permit_wait_nanos.fetch_add(waiting.elapsed().as_nanos() as u64, Relaxed);

        let (respond, receive) = tokio::sync::oneshot::channel();
        let started = Instant::now();
        self.pool.spawn_fifo(move || {
            let _ = respond.send(task());
        });
        let output = receive.await.map_err(|_| OffloadError::WorkerLost)?;
        self.stats.record(started.elapsed());
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grommet::topology::{Workload, detect};

    fn pool() -> RayonOffload {
        RayonOffload::unbound(2, 2)
    }

    #[tokio::test]
    async fn work_runs_off_the_calling_thread_and_returns_its_value() {
        let offload = pool();
        let caller = std::thread::current().id();
        let worker = offload.run(move || (6 * 7, std::thread::current().id())).await.unwrap();
        assert_eq!(worker.0, 42);
        assert_ne!(worker.1, caller, "compute must not run on the shard core");
        assert_eq!(offload.stats().runs.load(Relaxed), 1);
        assert!(offload.stats().mean_latency_nanos() > 0);
        assert!(offload.stats().busy() > Duration::ZERO);
    }

    #[tokio::test]
    async fn a_panicking_task_fails_only_its_caller() {
        let offload = pool();
        let failed: Result<(), _> = offload.run(|| panic!("compute blew up")).await;
        assert_eq!(failed, Err(OffloadError::WorkerLost));

        // The caller learns of the failure when the responder is dropped during
        // unwinding, which happens before Rayon reaches its panic handler, so
        // the counter is not readable the instant `run` returns.
        for _ in 0..1_000 {
            if offload.stats().panics.load(Relaxed) == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert_eq!(offload.stats().panics.load(Relaxed), 1);

        // The pool is still usable, which is the point of catching it.
        assert_eq!(offload.run(|| 1 + 1).await, Ok(2));
    }

    #[tokio::test]
    async fn many_concurrent_tasks_all_complete_under_a_bounded_permit_count() {
        let offload = pool();
        let tasks = (0..64u64).map(|value| {
            let offload = offload.clone();
            tokio::spawn(async move { offload.run(move || value * 2).await })
        });
        let mut total = 0;
        for task in tasks {
            total += task.await.unwrap().unwrap();
        }
        assert_eq!(total, (0..64u64).map(|value| value * 2).sum::<u64>());
        assert_eq!(offload.stats().runs.load(Relaxed), 64);
    }

    #[test]
    fn mean_latency_is_zero_before_anything_runs() {
        assert_eq!(OffloadStats::default().mean_latency_nanos(), 0);
        assert_eq!(OffloadStats::default().busy(), Duration::ZERO);
    }

    #[tokio::test]
    async fn a_plan_produces_one_working_pool_per_memory_node() {
        let plan = Arc::new(detect(&Workload::default()).expect("read this machine"));
        let pools = OffloadPools::build(plan.clone(), 2);
        assert_eq!(pools.workers(), plan.offload_workers());

        for shard in &plan.shards {
            let Some(offload) = pools.for_node(Some(shard.node)) else {
                assert!(pools.is_empty(), "a non-empty set must serve every shard");
                continue;
            };
            // A shard's pool is either on its own node or the fallback, never on
            // some third node picked at random.
            assert!(
                offload.node() == Some(shard.node) || offload.node() == pools.pools[0].node,
                "shard on node {} got a pool on {:?}",
                shard.node,
                offload.node(),
            );
            assert_eq!(offload.run(|| 6 * 7).await, Ok(42));
        }
    }

    #[test]
    fn an_unplaced_pool_is_still_a_pool() {
        let offload = RayonOffload::unbound(0, 1);
        assert_eq!(offload.workers(), 1, "zero workers would be a pool that cannot run anything");
        assert_eq!(offload.node(), None);
        assert_eq!(offload.stats().bound_workers.load(Relaxed), 0);
    }

    #[test]
    fn an_empty_set_of_pools_serves_nobody_rather_than_pretending() {
        // `compute_fraction: 0.0` plans no offload cores. Handing back a silent
        // inline pool would put compute back on the reactors it was moved off.
        let plan = Arc::new(
            detect(&Workload { compute_fraction: 0.0, ..Workload::default() })
                .expect("read this machine"),
        );
        let pools = OffloadPools::build(plan, 2);
        assert!(pools.is_empty());
        assert_eq!(pools.workers(), 0);
        assert!(pools.for_node(Some(0)).is_none());
    }
}
