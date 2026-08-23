//! Waiting as a guest of a tokio runtime.
//!
//! This is the compatibility path, and it is meant to be permanent: the
//! ecosystem's database, cache and HTTP clients are written against tokio's IO,
//! and a runtime that refused to host them would be a runtime nobody could
//! adopt incrementally.
//!
//! It suspends rather than sleeps. The shard is one task on the host's
//! current-thread runtime, and that runtime's reactor — the thing that will
//! notice a processor's socket becoming readable — only runs while this task is
//! suspended. Blocking here would stop the IO those futures are waiting on from
//! ever being observed, and the shard would deadlock against itself. It would
//! also starve any co-tenant: tests join a shard with the driver submitting to
//! it, and both are sub-futures of one task on one thread.
//!
//! The timer is tokio's, deliberately and only here. The *schedule* belongs to
//! the shard's wheel and the *clock* to the caller; what tokio supplies is the
//! wakeup, which is a driver's job and not something a guest can provide for
//! itself. An owned driver supplies its own and this file stops being reached.

use super::Driver;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::time::Sleep;

/// Suspends the shard's task until a wake or a deadline.
pub(crate) struct TokioDriver {
    /// One allocation for the life of the shard. `Sleep` is `!Unpin` and has to
    /// stay put to be reset, and resetting is what keeps this off the hot path:
    /// arming a timer registers with the host's timer wheel, and the shard's
    /// schedule moves once per tick rather than once per turn.
    sleep: Pin<Box<Sleep>>,
    /// What the sleep is currently set for, so an unchanged deadline does not
    /// pay to re-register.
    armed: Option<Duration>,
}

impl TokioDriver {
    pub(crate) fn new() -> Self {
        Self { sleep: Box::pin(tokio::time::sleep(Duration::ZERO)), armed: None }
    }
}

impl Driver for TokioDriver {
    fn wait(
        &mut self,
        deadline: Option<Duration>,
        now: Duration,
        cx: &mut Context<'_>,
    ) -> Poll<()> {
        // `is_elapsed` covers the case the cache alone would miss: the same
        // deadline as last time, but that sleep has already fired, so leaving it
        // unarmed would suspend on a timer that can never wake anything.
        if self.armed != deadline || self.sleep.is_elapsed() {
            self.armed = deadline;
            // A shard with nothing scheduled still wakes on work; the far
            // future is simply "no timer worth arming".
            match deadline {
                Some(at) => {
                    let wait = at.saturating_sub(now);
                    self.sleep.as_mut().reset(tokio::time::Instant::now() + wait);
                }
                None => return Poll::Pending,
            }
        }
        if self.armed.is_none() {
            return Poll::Pending;
        }
        // Polled for its side effect: registering this task with the host's
        // timer. `Ready` means the deadline passed while the shard was working,
        // so it should take another turn rather than sleep through it.
        self.sleep.as_mut().poll(cx)
    }
}
