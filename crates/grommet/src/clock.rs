//! The only time source the runtime reads.
//!
//! Every deadline, idle window and latency measurement flows through this
//! trait, so a deterministic test can drive the whole reactor without touching
//! the wall clock. Time is a monotonic `Duration` since an arbitrary origin;
//! all clock instances handed to one runtime must share that origin, which
//! cloning guarantees.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

pub trait Clock: Clone + Send + Sync + 'static {
    fn now(&self) -> Duration;
}

/// The production clock: a monotonic `Instant` captured at construction.
#[derive(Clone, Copy, Debug)]
pub struct SystemClock {
    origin: Instant,
}

impl SystemClock {
    pub fn new() -> Self {
        Self { origin: Instant::now() }
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
        self.origin.elapsed()
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
