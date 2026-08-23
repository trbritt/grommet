//! The boundary between a shard's reactor core and CPU-bound work.
//!
//! A shard core must never run a long computation inline: doing so stalls every
//! other key that shard owns, including latency-sensitive IO work in another
//! class. Instead, hand the computation to an offload pool running on its own
//! cores and await the result, which keeps the reactor free to dispatch.

use std::error::Error;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum OffloadError {
    /// The pool is shutting down and will not accept work.
    Closed,
    /// The worker never produced a result: it panicked, or the pool dropped
    /// the task.
    WorkerLost,
    /// A deterministic test injected this failure.
    Injected,
}

impl fmt::Display for OffloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Closed => "compute pool is closed",
            Self::WorkerLost => "compute worker produced no result",
            Self::Injected => "injected compute failure",
        };
        f.write_str(message)
    }
}

impl Error for OffloadError {}

/// Runs CPU-bound closures away from the shard core that submitted them.
///
/// The closure must be `Send` because it crosses to a worker thread, but the
/// future returned here is awaited on the shard's own core and need not be.
#[allow(async_fn_in_trait)]
pub trait Offload: Clone + 'static {
    async fn run<F, T>(&self, task: F) -> Result<T, OffloadError>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static;
}

/// Runs the closure inline on the calling shard.
///
/// This makes compute deterministic and single-threaded, which is what tests
/// and simulations want. It is emphatically not a production executor: the
/// shard is blocked for the whole computation.
#[derive(Clone, Copy, Debug, Default)]
pub struct InlineOffload;

impl Offload for InlineOffload {
    async fn run<F, T>(&self, task: F) -> Result<T, OffloadError>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        Ok(task())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn inline_offload_runs_the_closure_and_returns_its_value() {
        assert_eq!(InlineOffload.run(|| 6 * 7).await, Ok(42));
    }

    #[test]
    fn offload_errors_describe_themselves() {
        assert_eq!(OffloadError::Closed.to_string(), "compute pool is closed");
        assert_eq!(OffloadError::WorkerLost.to_string(), "compute worker produced no result");
        assert_eq!(OffloadError::Injected.to_string(), "injected compute failure");
    }
}
