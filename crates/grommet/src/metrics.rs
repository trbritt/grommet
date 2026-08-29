//! Metrics shaped for a thread-per-core hot path.
//!
//! [`ShardHot`] is written only by the shard that owns it, so it uses plain
//! `Cell` counters: there are no atomics anywhere on the per-item path. Once
//! per tick the shard publishes a snapshot into [`ShardStats`], which an
//! exporter on another thread can read.
//!
//! These are the runtime's own counters and deliberately do not try to cover
//! your workload. A [`Processor`] is user-owned and single-threaded, so the
//! natural place for domain metrics is a `Cell` inside your own processor.
//!
//! # Latency
//!
//! Sums give means, and a mean hides exactly the thing anyone chose
//! thread-per-core to control. A runtime that sells a starvation bound and
//! deadline scheduling has to be able to show its own tail, so queue wait and
//! processing duration go into HDR histograms rather than accumulators.
//!
//! Those are `hdrhistogram`'s, not ours. Built through
//! [`Histogram::new_with_bounds`] the counts are sized once and auto-resizing
//! is off, so recording is a bucket increment and never allocates. That is the
//! only property the hot path needs from it, and it is asserted below rather
//! than assumed. Two significant figures put any reported quantile within one
//! percent, for about thirty kilobytes per histogram.
//!
//! Published histograms are the buckets, not pre-computed quantiles, because
//! percentiles do not average: a runtime-wide p99 has to come from merging the
//! shards' distributions, which is [`Histogram::add`].
//!
//! [`Processor`]: crate::processor::Processor

use grommet_core::Snapshot;
pub use hdrhistogram::Histogram;
use parking_lot::Mutex;
use std::cell::{Cell, RefCell};
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;

/// The smallest latency told apart from zero. One nanosecond: a queue wait on
/// an idle shard is genuinely sub-microsecond, and that is worth seeing.
const LOWEST_NANOS: u64 = 1;

/// The largest latency recorded distinctly. A minute, after which everything
/// shares the top bucket. A dispatch that slow is a fault to alert on, and
/// [`ShardStats::inflight_age_max_nanos`] is the gauge for it.
const HIGHEST_NANOS: u64 = 60_000_000_000;

/// Significant figures kept. Two bounds the error on a reported quantile at one
/// percent; three would cost seven times the memory for a digit nobody reads.
const SIGNIFICANT_FIGURES: u8 = 2;

/// An empty histogram with the bounds every shard's uses.
///
/// Public because merging is the only correct way to read a runtime-wide
/// quantile, and a caller needs somewhere to merge into. Anything built here
/// is compatible with every shard's.
pub fn histogram() -> Histogram<u64> {
    Histogram::new_with_bounds(LOWEST_NANOS, HIGHEST_NANOS, SIGNIFICANT_FIGURES)
        .expect("the bounds above are constant and valid")
}

/// Hot, single-threaded, per-shard counters. Cumulative and never reset; an
/// exporter diffs successive published snapshots to get rates.
#[derive(Debug)]
pub struct ShardHot<const CLASSES: usize = 2> {
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
    /// Keys whose state was handed to `on_evict`.
    pub evicted: Cell<u64>,
    /// Time inside the reactor loop doing scheduling bookkeeping: admission,
    /// completion, dispatch. This is the runtime's own overhead. Work itself
    /// shows up as parked time, not here.
    pub busy_nanos: Cell<u64>,
    /// Submission-to-dispatch latency, summed over dispatched items. Kept
    /// alongside the histogram because a sum is what a rate is computed from,
    /// and a quantile is not.
    pub queue_wait_nanos: Cell<u64>,
    /// Items admitted from the mailbox, split by class.
    ///
    /// Only the split is stored. The totals an exporter reads are sums of
    /// these, taken once per publish rather than kept as a second counter that
    /// the hot path has to bump and that could drift from this one.
    pub started_by_class: [Cell<u64>; CLASSES],
    /// Items whose processing finished, panics included, split by class.
    pub completed_by_class: [Cell<u64>; CLASSES],
    /// Items discarded at dispatch because their deadline had passed, split by
    /// class.
    pub expired_by_class: [Cell<u64>; CLASSES],
    /// How long the oldest still-running dispatch has been running, as of the
    /// last time the shard looked. Zero when nothing is in flight.
    ///
    /// This is the gauge that makes a wedged future visible: every other number
    /// here describes work that finished.
    pub inflight_age_max_nanos: Cell<u64>,
    /// Submission-to-dispatch latency.
    queue_wait: RefCell<Histogram<u64>>,
    /// Dispatch-to-completion latency, the processor's own time.
    process: RefCell<Histogram<u64>>,
}

// `[T; N]: Default` only covers `N <= 32`, so the class count cannot rely on it.
impl<const CLASSES: usize> Default for ShardHot<CLASSES> {
    fn default() -> Self {
        Self {
            panicked: Cell::new(0),
            failed: Cell::new(0),
            in_doubt: Cell::new(0),
            coalesced: Cell::new(0),
            evicted: Cell::new(0),
            busy_nanos: Cell::new(0),
            queue_wait_nanos: Cell::new(0),
            started_by_class: std::array::from_fn(|_| Cell::new(0)),
            completed_by_class: std::array::from_fn(|_| Cell::new(0)),
            expired_by_class: std::array::from_fn(|_| Cell::new(0)),
            inflight_age_max_nanos: Cell::new(0),
            queue_wait: RefCell::new(histogram()),
            process: RefCell::new(histogram()),
        }
    }
}

impl<const CLASSES: usize> ShardHot<CLASSES> {
    #[inline]
    pub fn add(&self, counter: &Cell<u64>, value: u64) {
        counter.set(counter.get().wrapping_add(value));
    }

    #[inline]
    pub fn bump(&self, counter: &Cell<u64>) {
        counter.set(counter.get().wrapping_add(1));
    }

    /// Items admitted from the mailbox, all classes.
    #[inline]
    pub fn started(&self) -> u64 {
        Self::total(&self.started_by_class)
    }

    /// Items whose processing finished, all classes.
    #[inline]
    pub fn completed(&self) -> u64 {
        Self::total(&self.completed_by_class)
    }

    /// Items shed at dispatch for a passed deadline, all classes.
    #[inline]
    pub fn expired(&self) -> u64 {
        Self::total(&self.expired_by_class)
    }

    fn total(counters: &[Cell<u64>; CLASSES]) -> u64 {
        counters.iter().fold(0u64, |sum, counter| sum.wrapping_add(counter.get()))
    }

    /// Record how long a dispatched item waited to be dispatched.
    ///
    /// `saturating_record` rather than `record`: a latency past the top of the
    /// range is a fault to be seen in the top bucket, not an error to hand back
    /// to a reactor that has nothing useful to do with one.
    #[inline]
    pub fn record_queue_wait(&self, nanos: u64) {
        self.queue_wait.borrow_mut().saturating_record(nanos);
    }

    /// Record how long a dispatched item took to finish.
    #[inline]
    pub fn record_process(&self, nanos: u64) {
        self.process.borrow_mut().saturating_record(nanos);
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
    /// Resident keys sitting idle, which is the eviction sweep's entire
    /// worklist. It cannot exceed `resident`; an exporter that sees it climb
    /// away from that bound is seeing a leak, which is why it is published
    /// rather than left as an internal detail.
    pub eviction_backlog: AtomicU64,
    pub queue_capacity: AtomicU64,
    /// Items admitted, finished, and shed at dispatch, split by class. The
    /// aggregate counters above cannot show one class starving behind another.
    pub started_by_class: [AtomicU64; CLASSES],
    pub completed_by_class: [AtomicU64; CLASSES],
    pub expired_by_class: [AtomicU64; CLASSES],
    /// How long the oldest still-running dispatch has been running, in
    /// nanoseconds, as of the last publish. Zero when nothing is in flight.
    ///
    /// Every other number here describes work that finished, so this is the
    /// only one that can show work that has not.
    pub inflight_age_max_nanos: AtomicU64,
    /// Submission-to-dispatch latency. Cumulative, like the counters: an
    /// exporter wanting the last interval diffs two snapshots.
    ///
    /// Behind a lock rather than atomics because it is read and written once
    /// per tick and never on the dispatch path. Take it with
    /// [`queue_wait`](ShardStats::queue_wait).
    queue_wait: Mutex<Histogram<u64>>,
    /// Dispatch-to-completion latency: the processor's own time, excluding the
    /// wait to be dispatched.
    process: Mutex<Histogram<u64>>,
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
            eviction_backlog: AtomicU64::new(0),
            queue_capacity: AtomicU64::new(0),
            started_by_class: std::array::from_fn(|_| AtomicU64::new(0)),
            completed_by_class: std::array::from_fn(|_| AtomicU64::new(0)),
            expired_by_class: std::array::from_fn(|_| AtomicU64::new(0)),
            inflight_age_max_nanos: AtomicU64::new(0),
            queue_wait: Mutex::new(histogram()),
            process: Mutex::new(histogram()),
        }
    }
}

impl<const CLASSES: usize> ShardStats<CLASSES> {
    /// This shard's submission-to-dispatch latency at `quantile`, in
    /// nanoseconds.
    ///
    /// Answers for this shard alone. For the runtime's own number, merge every
    /// shard with [`merge_queue_wait_into`] and ask the result once: quantiles
    /// do not average, and averaging them flatters a runtime whose slow shard
    /// is the whole problem.
    ///
    /// [`merge_queue_wait_into`]: ShardStats::merge_queue_wait_into
    pub fn queue_wait_quantile(&self, quantile: f64) -> u64 {
        self.queue_wait.lock().value_at_quantile(quantile)
    }

    /// This shard's dispatch-to-completion latency at `quantile`, in
    /// nanoseconds: the processor's own time, excluding the wait to be
    /// dispatched.
    pub fn process_quantile(&self, quantile: f64) -> u64 {
        self.process.lock().value_at_quantile(quantile)
    }

    /// Add this shard's submission-to-dispatch latencies to `into`.
    ///
    /// ```no_run
    /// # use grommet::metrics::{ShardStats, histogram};
    /// # let shards: Vec<std::sync::Arc<ShardStats<2>>> = Vec::new();
    /// let mut all = histogram();
    /// for shard in &shards {
    ///     shard.merge_queue_wait_into(&mut all);
    /// }
    /// println!("p99 {}ns", all.value_at_quantile(0.99));
    /// ```
    pub fn merge_queue_wait_into(&self, into: &mut Histogram<u64>) {
        Self::merge(&self.queue_wait, into);
    }

    /// Add this shard's dispatch-to-completion latencies to `into`.
    pub fn merge_process_into(&self, into: &mut Histogram<u64>) {
        Self::merge(&self.process, into);
    }

    fn merge(from: &Mutex<Histogram<u64>>, into: &mut Histogram<u64>) {
        into.add(&*from.lock()).expect("the destination must come from `metrics::histogram()`");
    }

    pub(crate) fn publish(&self, hot: &ShardHot<CLASSES>, snapshot: &Snapshot<CLASSES>) {
        self.started.store(hot.started(), Relaxed);
        self.completed.store(hot.completed(), Relaxed);
        self.panicked.store(hot.panicked.get(), Relaxed);
        self.failed.store(hot.failed.get(), Relaxed);
        self.in_doubt.store(hot.in_doubt.get(), Relaxed);
        self.coalesced.store(hot.coalesced.get(), Relaxed);
        self.expired.store(hot.expired(), Relaxed);
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
        self.eviction_backlog.store(snapshot.eviction_backlog as u64, Relaxed);
        self.queue_capacity.store(snapshot.queue_capacity as u64, Relaxed);
        self.inflight_age_max_nanos.store(hot.inflight_age_max_nanos.get(), Relaxed);
        for class in 0..CLASSES {
            self.started_by_class[class].store(hot.started_by_class[class].get(), Relaxed);
            self.completed_by_class[class].store(hot.completed_by_class[class].get(), Relaxed);
            self.expired_by_class[class].store(hot.expired_by_class[class].get(), Relaxed);
        }
        // Replace rather than accumulate: the shard's own copy is cumulative,
        // so this is a snapshot of it and not an addition to a running total.
        // `clear` and `add` reuse the counts in place; neither allocates.
        Self::republish(&self.queue_wait, &hot.queue_wait);
        Self::republish(&self.process, &hot.process);
    }

    fn republish(into: &Mutex<Histogram<u64>>, from: &RefCell<Histogram<u64>>) {
        let mut into = into.lock();
        into.clear();
        into.add(&*from.borrow()).expect("both histograms were built with the same bounds");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert a reported latency is the recorded one to the precision the
    /// histogram promises, rather than to a magic number that would have to be
    /// rewritten whenever the bucketing changed.
    #[track_caller]
    fn close(reported: u64, recorded: u64) {
        let error = (reported as f64 - recorded as f64).abs() / recorded as f64;
        assert!(
            error <= 0.01,
            "reported {reported} for {recorded}, off by {:.2}% and past the one percent \
             the significant figures promise",
            error * 100.0
        );
    }

    fn snapshot() -> Snapshot<2> {
        Snapshot::<2> {
            inflight: [3, 1],
            ready: [7, 2],
            pending: 13,
            resident: 5,
            evicting: 1,
            eviction_backlog: 4,
            queue_capacity: 64,
        }
    }

    #[test]
    fn publishing_copies_hot_counters_and_scheduler_gauges() {
        let hot = ShardHot::<2>::default();
        hot.bump(&hot.started_by_class[0]);
        hot.bump(&hot.started_by_class[0]);
        hot.bump(&hot.panicked);
        hot.add(&hot.busy_nanos, 900);
        hot.add(&hot.queue_wait_nanos, 25);
        hot.inflight_age_max_nanos.set(7_000);

        let stats = ShardStats::<2>::default();
        stats.publish(&hot, &snapshot());

        assert_eq!(stats.started.load(Relaxed), 2);
        assert_eq!(stats.panicked.load(Relaxed), 1);
        assert_eq!(stats.busy_nanos.load(Relaxed), 900);
        assert_eq!(stats.queue_wait_nanos.load(Relaxed), 25);
        assert_eq!(stats.inflight_age_max_nanos.load(Relaxed), 7_000);
        assert_eq!(stats.inflight[0].load(Relaxed), 3);
        assert_eq!(stats.ready[1].load(Relaxed), 2);
        assert_eq!(stats.pending.load(Relaxed), 13);
        assert_eq!(stats.resident.load(Relaxed), 5);
        assert_eq!(stats.evicting.load(Relaxed), 1);
        assert_eq!(stats.eviction_backlog.load(Relaxed), 4);
        assert_eq!(stats.queue_capacity.load(Relaxed), 64);
    }

    #[test]
    fn a_total_is_the_sum_of_its_classes_rather_than_a_second_counter() {
        // The hot path bumps one counter per event. Keeping an aggregate
        // alongside would mean two increments that can disagree, and the
        // disagreement would be invisible: this is what makes it impossible.
        let hot = ShardHot::<2>::default();
        for _ in 0..5 {
            hot.bump(&hot.started_by_class[0]);
        }
        for _ in 0..3 {
            hot.bump(&hot.started_by_class[1]);
        }
        hot.bump(&hot.completed_by_class[1]);
        hot.bump(&hot.expired_by_class[0]);

        assert_eq!(hot.started(), 8);
        assert_eq!(hot.completed(), 1);
        assert_eq!(hot.expired(), 1);

        let stats = ShardStats::<2>::default();
        stats.publish(&hot, &snapshot());
        assert_eq!(stats.started.load(Relaxed), 8);
        assert_eq!(stats.started_by_class[0].load(Relaxed), 5);
        assert_eq!(stats.started_by_class[1].load(Relaxed), 3);
        assert_eq!(
            stats.started.load(Relaxed),
            stats.started_by_class.iter().map(|c| c.load(Relaxed)).sum::<u64>(),
            "the published total must agree with the split it came from"
        );
    }

    #[test]
    fn known_latencies_produce_known_quantiles() {
        // The distribution a mean would lie about: a hundred fast dispatches
        // and one that took a second.
        let hot = ShardHot::<2>::default();
        for _ in 0..99 {
            hot.record_queue_wait(1_000);
        }
        hot.record_queue_wait(1_000_000_000);
        hot.record_process(50_000);

        let stats = ShardStats::<2>::default();
        stats.publish(&hot, &snapshot());

        let mut queue_wait = histogram();
        stats.merge_queue_wait_into(&mut queue_wait);
        assert_eq!(queue_wait.len(), 100);
        // Two significant figures, so a reported value is within one percent.
        close(queue_wait.value_at_quantile(0.5), 1_000);
        close(queue_wait.value_at_quantile(0.99), 1_000);
        // The outlier is the whole reason for keeping a distribution.
        close(queue_wait.value_at_quantile(1.0), 1_000_000_000);

        // A mean over the same data reports about ten milliseconds, which is a
        // wait no submitter here actually had.
        assert!(queue_wait.value_at_quantile(0.5) < 10_000_000 / 1_000);
        close(stats.process_quantile(0.5), 50_000);
    }

    #[test]
    fn shards_merge_into_one_distribution() {
        // Averaging a fast shard's p99 with a slow one's reports neither.
        let (fast, slow) = (ShardHot::<2>::default(), ShardHot::<2>::default());
        for _ in 0..90 {
            fast.record_queue_wait(1_000);
        }
        for _ in 0..10 {
            slow.record_queue_wait(50_000_000);
        }
        let (left, right) = (ShardStats::<2>::default(), ShardStats::<2>::default());
        left.publish(&fast, &snapshot());
        right.publish(&slow, &snapshot());

        let mut all = histogram();
        left.merge_queue_wait_into(&mut all);
        right.merge_queue_wait_into(&mut all);

        assert_eq!(all.len(), 100);
        close(all.value_at_quantile(0.5), 1_000);
        close(all.value_at_quantile(0.95), 50_000_000);
    }

    #[test]
    fn publishing_replaces_rather_than_accumulates() {
        // The shard's own histogram is cumulative; publishing is a copy of it,
        // so an exporter diffing two snapshots sees an interval and not a
        // doubling.
        let hot = ShardHot::<2>::default();
        hot.record_queue_wait(5_000);
        let stats = ShardStats::<2>::default();
        stats.publish(&hot, &snapshot());
        stats.publish(&hot, &snapshot());
        close(stats.queue_wait_quantile(1.0), 5_000);

        hot.record_queue_wait(9_000_000);
        stats.publish(&hot, &snapshot());
        close(stats.queue_wait_quantile(1.0), 9_000_000);
        let mut merged = histogram();
        stats.merge_queue_wait_into(&mut merged);
        assert_eq!(merged.len(), 2, "and only the two that were recorded");
    }

    #[test]
    fn recording_never_grows_the_histogram() {
        // The property the hot path depends on. Bounds are fixed at
        // construction, which turns auto-resizing off, so a latency outside the
        // range lands in an end bucket rather than allocating a new one.
        // `tests/alloc.rs` holds the whole dispatch path to the same standard.
        let hot = ShardHot::<2>::default();
        let cells = hot.queue_wait.borrow().distinct_values();
        for nanos in [0, 1, HIGHEST_NANOS, HIGHEST_NANOS * 1_000, u64::MAX] {
            hot.record_queue_wait(nanos);
        }
        assert_eq!(hot.queue_wait.borrow().distinct_values(), cells);
        assert_eq!(hot.queue_wait.borrow().len(), 5, "every value was still counted");
    }

    #[test]
    fn a_histogram_costs_about_thirty_kilobytes_per_shard() {
        // Stated as a test because it is what the precision above buys, and
        // changing the constants should have to say so out loud.
        let cells = histogram().distinct_values();
        assert_eq!(cells, 3_840);
        assert_eq!(cells * std::mem::size_of::<u64>(), 30_720);
    }
}
