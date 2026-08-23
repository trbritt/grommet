//! The host runtime a shard's scheduler runs on.
//!
//! Grommet schedules work; it does not drive futures. Something has to own the
//! thread, poll the IO a processor performs, and decide how to wait when there
//! is nothing to do, and that something is an async runtime the caller already
//! uses. A `Driver` is the small surface grommet needs from one.
//!
//! The division is deliberate. A runtime knows how to wake a future when its
//! socket becomes readable; it does not know that two items sharing an affine
//! key must never run at once, or that compute belongs on a different core from
//! the reactor that dispatched it. Grommet supplies that and borrows the rest,
//! which is why adopting it does not mean leaving tokio.
//!
//! One driver is compiled in, chosen by a Cargo feature, and the shard loop is
//! the same across all of them. Each shard thread gets its own host instance,
//! placed on the CPU the topology plan chose for it.
//!
//! The trait is crate-private, so which runtime a build hosts on is expressed
//! by its features rather than by its public API.

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
    /// `now` is the scheduler's own reading of its clock, so a driver needing a
    /// relative timeout can compute one without taking a second reading that
    /// would disagree with the schedule it was given. `None` means nothing is
    /// scheduled and only a wake will do.
    ///
    /// [`Poll::Ready`] promises the caller may take another turn immediately.
    /// [`Poll::Pending`] promises `cx`'s waker is registered with everything
    /// that could end the wait.
    fn wait(&mut self, deadline: Option<Duration>, now: Duration, cx: &mut Context<'_>)
    -> Poll<()>;
}

/// The host this build was compiled against.
#[cfg(feature = "driver-tokio")]
pub(crate) type Host = tokio::TokioDriver;
