//! What the reactor needs from whatever hosts it.
//!
//! A shard decides *when* it next needs to run: its timer wheel computes that
//! from the caller's [`Clock`], and nothing here participates in the decision.
//! What lives behind this seam is only the waiting — and the waiting is the one
//! part of the loop that cannot be written once for every deployment.
//!
//! Today a shard runs as a future on a current-thread tokio runtime, because
//! its processors do their IO there: a PostgreSQL query, a Redis round trip,
//! an outbound HTTP request. That runtime's reactor only runs while the shard's
//! task is suspended, so the shard has to suspend rather than sleep. Once a
//! shard owns its IO driver it owns its wait too, and the same loop blocks in
//! `io_uring_enter` or `epoll_wait` with the wheel's deadline as the timeout.
//!
//! Both shapes fit [`Driver::wait`], which is why the loop above it never has to
//! know which one it has:
//!
//! - **Suspending** ([`Poll::Pending`]) hands the thread back so a host runtime
//!   can drive whatever else is on it. The wake comes from the mailbox, from a
//!   completion, or from the timer the driver armed.
//! - **Blocking** ([`Poll::Ready`]) waits in place and returns for another turn.
//!   Only an owned driver may do this: a future that blocks its thread starves
//!   whatever shares it, which is why the hosted driver cannot.
//!
//! The trait is crate-private, so which driver a build has is never part of the
//! public API — only of the Cargo features that select it.
//!
//! [`Clock`]: crate::clock::Clock

use std::task::{Context, Poll};
use std::time::Duration;

#[cfg(not(feature = "driver-tokio"))]
compile_error!(
    "a shard needs a driver to wait on: enable `driver-tokio`, which is on by default \
     and is what hosts processors doing tokio IO"
);

#[cfg(feature = "driver-tokio")]
pub(crate) mod tokio;

/// The host a shard's reactor waits on.
pub(crate) trait Driver {
    /// Wait until something wakes this shard, or until `deadline` passes.
    ///
    /// `now` is the reactor's own reading of its clock, so a driver that needs a
    /// relative timeout can compute one without taking a second reading that
    /// would disagree with the schedule it was given. `None` means nothing is
    /// scheduled and only a wake will do.
    ///
    /// Returning [`Poll::Ready`] promises the caller may take another turn
    /// immediately; returning [`Poll::Pending`] promises `cx`'s waker has been
    /// registered with everything that could end the wait.
    fn wait(&mut self, deadline: Option<Duration>, now: Duration, cx: &mut Context<'_>)
    -> Poll<()>;
}

/// The driver this build was compiled with.
#[cfg(feature = "driver-tokio")]
pub(crate) type Host = tokio::TokioDriver;
