//! The only time source the runtime reads.
//!
//! Every deadline, idle window and latency measurement flows through this
//! trait, so a deterministic test can drive the whole reactor without touching
//! the wall clock. Time is a monotonic `Duration` since an arbitrary origin;
//! all clock instances handed to one runtime must share that origin, which
//! cloning guarantees.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Duration;

pub trait Clock: Clone + Send + Sync + 'static {
    fn now(&self) -> Duration;
}

/// The production clock: a monotonic reading taken against an origin captured
/// at construction.
///
/// Backed by the CPU's cycle counter where the platform has a usable one, and
/// by the operating system's monotonic clock where it does not. That choice is
/// made at construction rather than compile time, so a machine without an
/// invariant counter still gets a correct clock, just a slower one.
///
/// It matters because the reactor reads the clock several times per turn, and
/// on this machine `Instant::now` costs about 28ns against a scheduler
/// operation of about 15. Reading the counter costs about 1.5.
///
/// # Startup cost
///
/// The first clock built in a process spends about 200ms calibrating the
/// counter against the operating system's clock. It is paid once, globally, and
/// [`Builder::new`] builds its clock on the calling thread before any shard
/// starts, so it lands at startup rather than on a pinned thread or a hot path.
/// A process that wants it earlier can construct a `SystemClock` and drop it.
///
/// [`Builder::new`]: crate::scheduler::Builder::new
#[derive(Clone, Debug)]
pub struct SystemClock {
    clock: quanta::Clock,
    origin: quanta::Instant,
}

impl SystemClock {
    pub fn new() -> Self {
        let clock = quanta::Clock::new();
        let origin = clock.now();
        Self { clock, origin }
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SystemClock {
    #[inline]
    fn now(&self) -> Duration {
        // Saturating: `duration_since` yields zero rather than panicking if a
        // reading ever lands before the origin, which a counter that is not
        // perfectly invariant across cores can do. A deadline computed from
        // zero fires immediately, which is the safe direction.
        self.clock.now().duration_since(self.origin)
    }
}

/// A clock that only moves when a test moves it. Clones share one timeline, so
/// a test can hold one handle while the runtime holds another.
#[derive(Clone, Debug, Default)]
pub struct ManualClock(Arc<AtomicU64>);

impl ManualClock {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn advance(&self, by: Duration) {
        self.0.fetch_add(by.as_nanos() as u64, Relaxed);
    }

    pub fn set(&self, to: Duration) {
        self.0.store(to.as_nanos() as u64, Relaxed);
    }
}

impl Clock for ManualClock {
    #[inline]
    fn now(&self) -> Duration {
        Duration::from_nanos(self.0.load(Relaxed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_manual_clock_shares_one_timeline_across_clones() {
        let clock = ManualClock::new();
        let handle = clock.clone();
        assert_eq!(handle.now(), Duration::ZERO);
        clock.advance(Duration::from_millis(250));
        assert_eq!(handle.now(), Duration::from_millis(250));
        clock.set(Duration::from_secs(9));
        assert_eq!(handle.now(), Duration::from_secs(9));
    }

    #[test]
    fn clones_of_a_system_clock_share_one_origin() {
        // Deadlines are compared across handles: the router stamps arrival with
        // its clone and the shard judges expiry with another. Two origins would
        // make that comparison meaningless in a way nothing would report.
        let clock = SystemClock::new();
        let handle = clock.clone();
        let (first, second) = (clock.now(), handle.now());
        let apart = second.saturating_sub(first);
        assert!(
            apart < Duration::from_millis(1),
            "clones disagreed by {apart:?}, so they are not reading one timeline"
        );
    }

    #[test]
    fn the_system_clock_is_monotonic_and_actually_advances_from_its_origin() {
        let clock = SystemClock::default();
        let first = clock.now();
        let second = clock.now();
        assert!(second >= first, "a clock that went backwards would break every deadline");

        // Monotonicity alone is satisfied by a clock stuck at zero, which is
        // indistinguishable from a working one until a deadline has to fire.
        // Spin rather than sleep: this needs a few nanoseconds, not a timer.
        let start = std::time::Instant::now();
        while clock.now() == Duration::ZERO {
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "the system clock never left its origin"
            );
            std::hint::spin_loop();
        }
    }
}
