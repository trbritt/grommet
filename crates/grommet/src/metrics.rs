//! Metrics shaped for a thread-per-core hot path.
//!
//! [`ShardHot`] is written only by the shard that owns it, so it uses plain
//! `Cell` counters — there are no atomics anywhere on the per-item path. Once
//! per tick the shard publishes a snapshot into [`ShardStats`], which an
//! exporter on another thread can read.
//!
//! These are the runtime's own counters and deliberately do not try to cover
//! your workload. A [`Processor`] is user-owned and single-threaded, so the
//! natural place for domain metrics is a `Cell` inside your own processor.
//!
//! [`Processor`]: crate::processor::Processor

use grommet_core::Snapshot;
use std::cell::Cell;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;

/// Hot, single-threaded, per-shard counters. Cumulative and never reset; an
/// exporter diffs successive published snapshots to get rates.
#[derive(Default, Debug)]
pub struct ShardHot {
    /// Items admitted from the mailbox.
    pub started: Cell<u64>,
    /// Items whose processing finished, panics included.
    pub completed: Cell<u64>,
    /// Processing futures that panicked and were caught.
    pub panicked: Cell<u64>,
    /// Items that returned an error.
    pub failed: Cell<u64>,
    /// Errors classified as `Fallout::InDoubt`: operations whose durable
    /// outcome is unknown. This is the number worth alerting on.
    pub in_doubt: Cell<u64>,
    /// Items suppressed because the same request id was already live for
    /// their key.
    pub coalesced: Cell<u64>,
    /// Items discarded at dispatch because their deadline had passed.
    pub expired: Cell<u64>,
    /// Keys whose state was handed to `on_evict`.
    pub evicted: Cell<u64>,
    /// Time inside the reactor loop doing scheduling bookkeeping — admission,
    /// completion, dispatch. This is the runtime's own overhead. Work itself
    /// shows up as parked time, not here.
    pub busy_nanos: Cell<u64>,
    /// Submission-to-dispatch latency, summed over dispatched items.
    pub queue_wait_nanos: Cell<u64>,
}

impl ShardHot {
    #[inline]
    pub fn add(&self, counter: &Cell<u64>, value: u64) {
        counter.set(counter.get().wrapping_add(value));
    }

    #[inline]
    pub fn bump(&self, counter: &Cell<u64>) {
        counter.set(counter.get().wrapping_add(1));
    }
}

/// The published snapshot an exporter thread reads. The first block is
/// cumulative; the rest are instantaneous gauges.
#[derive(Debug)]
pub struct ShardStats<const CLASSES: usize = 2> {
    pub started: AtomicU64,
    pub completed: AtomicU64,
    pub panicked: AtomicU64,
    pub failed: AtomicU64,
    pub in_doubt: AtomicU64,
    pub coalesced: AtomicU64,
    pub expired: AtomicU64,
    pub evicted: AtomicU64,
    pub busy_nanos: AtomicU64,
    pub queue_wait_nanos: AtomicU64,
    pub inflight: [AtomicU64; CLASSES],
    pub ready: [AtomicU64; CLASSES],
    pub pending: AtomicU64,
    pub resident: AtomicU64,
    pub evicting: AtomicU64,
    pub queue_capacity: AtomicU64,
}

// `[T; N]: Default` only covers `N <= 32`, so the class count cannot rely on it.
impl<const CLASSES: usize> Default for ShardStats<CLASSES> {
    fn default() -> Self {
        Self {
            started: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            panicked: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            in_doubt: AtomicU64::new(0),
            coalesced: AtomicU64::new(0),
            expired: AtomicU64::new(0),
            evicted: AtomicU64::new(0),
            busy_nanos: AtomicU64::new(0),
            queue_wait_nanos: AtomicU64::new(0),
            inflight: std::array::from_fn(|_| AtomicU64::new(0)),
            ready: std::array::from_fn(|_| AtomicU64::new(0)),
            pending: AtomicU64::new(0),
            resident: AtomicU64::new(0),
            evicting: AtomicU64::new(0),
            queue_capacity: AtomicU64::new(0),
        }
    }
}

impl<const CLASSES: usize> ShardStats<CLASSES> {
    pub(crate) fn publish(&self, hot: &ShardHot, snapshot: &Snapshot<CLASSES>) {
        self.started.store(hot.started.get(), Relaxed);
        self.completed.store(hot.completed.get(), Relaxed);
        self.panicked.store(hot.panicked.get(), Relaxed);
        self.failed.store(hot.failed.get(), Relaxed);
        self.in_doubt.store(hot.in_doubt.get(), Relaxed);
        self.coalesced.store(hot.coalesced.get(), Relaxed);
        self.expired.store(hot.expired.get(), Relaxed);
        self.evicted.store(hot.evicted.get(), Relaxed);
        self.busy_nanos.store(hot.busy_nanos.get(), Relaxed);
        self.queue_wait_nanos.store(hot.queue_wait_nanos.get(), Relaxed);
        for class in 0..CLASSES {
            self.inflight[class].store(snapshot.inflight[class] as u64, Relaxed);
            self.ready[class].store(snapshot.ready[class] as u64, Relaxed);
        }
        self.pending.store(snapshot.pending as u64, Relaxed);
        self.resident.store(snapshot.resident as u64, Relaxed);
        self.evicting.store(snapshot.evicting as u64, Relaxed);
        self.queue_capacity.store(snapshot.queue_capacity as u64, Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publishing_copies_hot_counters_and_scheduler_gauges() {
        let hot = ShardHot::default();
        hot.bump(&hot.started);
        hot.bump(&hot.started);
        hot.bump(&hot.panicked);
        hot.add(&hot.busy_nanos, 900);
        hot.add(&hot.queue_wait_nanos, 25);

        let snapshot = Snapshot::<2> {
            inflight: [3, 1],
            ready: [7, 2],
            pending: 13,
            resident: 5,
            evicting: 1,
            queue_capacity: 64,
        };
        let stats = ShardStats::<2>::default();
        stats.publish(&hot, &snapshot);

        assert_eq!(stats.started.load(Relaxed), 2);
        assert_eq!(stats.panicked.load(Relaxed), 1);
        assert_eq!(stats.busy_nanos.load(Relaxed), 900);
        assert_eq!(stats.queue_wait_nanos.load(Relaxed), 25);
        assert_eq!(stats.inflight[0].load(Relaxed), 3);
        assert_eq!(stats.ready[1].load(Relaxed), 2);
        assert_eq!(stats.pending.load(Relaxed), 13);
        assert_eq!(stats.resident.load(Relaxed), 5);
        assert_eq!(stats.evicting.load(Relaxed), 1);
        assert_eq!(stats.queue_capacity.load(Relaxed), 64);
    }
}
