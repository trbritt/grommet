//! Hosting a shard's scheduler on a tokio runtime.
//!
//! Each shard thread builds its own current-thread runtime and runs one
//! scheduler on it. That runtime drives whatever IO the processor performs, so
//! a client keeps its existing database, cache and HTTP libraries unchanged;
//! grommet adds key-affine dispatch and compute offload above them.
//!
//! Waiting here suspends the task rather than blocking the thread. The
//! runtime's reactor is what notices a processor's socket becoming readable,
//! and it only runs while the task is suspended, so blocking would stop the IO
//! those futures wait on from ever being observed. It would also starve any
//! co-tenant: a test that joins a shard with the code submitting to it puts
//! both on one thread.
//!
//! The timer is tokio's, and only here. The schedule belongs to the shard's
//! wheel and the clock to the caller; what the host supplies is the wakeup.

use super::Driver;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::time::Sleep;

/// Suspends the shard's task until a wake or a deadline.
///
/// Selected by the `driver-tokio` feature.
pub(crate) struct TokioDriver {
    /// One allocation for the life of the shard. `Sleep` is `!Unpin` and has to
    /// stay put to be reset. Resetting rather than rebuilding keeps this off
    /// the hot path: arming a timer registers with the host's own timer wheel,
    /// and the shard's schedule moves once per tick rather than once per turn.
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
