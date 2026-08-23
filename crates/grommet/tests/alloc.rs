//! What one dispatch costs the allocator once a shard is warm.
//!
//! The interesting number is the *marginal* one. A shard allocates plenty at
//! startup — its scheduler slab, its key map, its outstanding set — and a test
//! that counted from zero would report mostly that. So the same workload runs
//! twice at different sizes and the difference is divided by the difference in
//! items: fixed costs cancel, and what is left is what each additional dispatch
//! actually cost.
//!
//! The number this pins is the one that decided the outstanding set's design.
//! Measured against `futures`' unordered set, the same workload cost 1.008
//! allocations per dispatch — one heap node per pushed future, exactly as that
//! structure documents. The slab reuses its slots, so what is left here is
//! warm-up amortizing away and nothing else.

use grommet::metrics::ShardStats;
use grommet::{Disposition, ManualClock, Processor, Router, ShardConfig, Work, shard};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static COUNTING: AtomicBool = AtomicBool::new(false);

/// Counts allocations while armed, and is otherwise the system allocator.
///
/// Gated rather than always-on so that the harness around the measurement —
/// building the runtime, collecting results — cannot contribute to it.
struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Relaxed) {
            ALLOCATIONS.fetch_add(1, Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Relaxed) {
            ALLOCATIONS.fetch_add(1, Relaxed);
        }
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[derive(Debug)]
struct Item(u64);

impl Work for Item {
    type Key = u64;
    type Id = ();

    fn key(&self) -> u64 {
        self.0
    }

    fn class(&self) -> grommet::ClassId {
        0
    }
}

/// Accumulates per key and never awaits, so the measurement is the reactor's
/// own allocation rather than a processor's.
#[derive(Clone, Copy)]
struct Counter;

impl Processor for Counter {
    type Work = Item;
    type State = u64;
    type Error = std::convert::Infallible;

    async fn process(
        &self,
        _key: u64,
        state: Option<u64>,
        _work: Item,
    ) -> Result<Disposition<u64>, Self::Error> {
        Ok(Disposition::Keep(state.unwrap_or(0) + 1))
    }
}

/// Run `items` submissions through a real shard, counting allocations from the
/// moment the shard is built until it has drained.
fn allocations_for(items: u64) -> usize {
    const KEYS: u64 = 32;

    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    runtime.block_on(async move {
        let clock = ManualClock::new();
        let (tx, rx) = grommet::channel(4096);
        let router = Router::<Item, ManualClock, 2>::new(vec![tx], clock.clone());
        let mut cfg = ShardConfig::new([64, 64]);
        // Reserve the scheduler's slab up front, so its growth is not counted
        // as a per-dispatch cost it is not.
        cfg.scheduler.queue_reserve = cfg.scheduler.max_pending;

        let engine = shard::run(rx, Counter, clock, Arc::new(ShardStats::default()), cfg);
        let driver = async move {
            // Warm every path once before the count starts: the first pass
            // through a slot, a hash bucket or a ring is an allocation that
            // will not recur.
            for key in 0..KEYS {
                router.submit(Item(key)).await.unwrap();
            }
            tokio::task::yield_now().await;

            ALLOCATIONS.store(0, Relaxed);
            COUNTING.store(true, Relaxed);
            for index in 0..items {
                router.submit(Item(index % KEYS)).await.unwrap();
            }
            drop(router);
        };
        tokio::join!(engine, driver);
    });

    COUNTING.store(false, Relaxed);
    ALLOCATIONS.load(Relaxed)
}

#[test]
fn a_warm_shard_allocates_nothing_to_dispatch() {
    // Two sizes an order apart, so the fixed cost of a run divides away.
    let small = allocations_for(2_000);
    let large = allocations_for(20_000);
    let marginal = (large as f64 - small as f64) / 18_000.0;

    println!(
        "{small} allocations for 2,000 items, {large} for 20,000 \
         -> {marginal:.3} per dispatch"
    );

    // Submission itself allocates nothing per item — the mailbox is bounded and
    // pre-sized, and the scheduler's slab is reserved above — so anything left
    // belongs to the set holding the dispatched future. It boxes a slot the
    // first time that slot is used and reuses it through `Pin::set` forever
    // after, which is the claim the whole structure exists to make.
    assert!(
        marginal < 0.05,
        "a warm shard allocated {marginal:.3} times per dispatch; slots must be reused"
    );
}
