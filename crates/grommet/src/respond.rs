//! Request/response on top of a fire-and-forget scheduler.
//!
//! Submission is deliberately one-way: it reports whether work was *accepted*,
//! not what it produced. That is the right primitive for pipelines that reply
//! by writing to a socket, batch their answers, or have no answer at all.
//!
//! When a caller does want the result, wrap the work in a [`Call`] and use
//! [`Router::call`]. The reply channel then travels with the work, which means
//! every path that hands work back — a full mailbox, a downed shard, a passed
//! deadline — hands back the means to answer the caller too.

use crate::clock::Clock;
use crate::router::{Router, SubmitError};
use crate::work::Work;
use grommet_core::ClassId;
use std::error::Error;
use std::fmt;
use std::future::Future;
use std::time::Duration;
use tokio::sync::oneshot;

/// The caller stopped waiting, or the processor finished without answering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cancelled;

impl fmt::Display for Cancelled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the call was dropped without a response")
    }
}

impl Error for Cancelled {}

/// Why a call did not produce a response.
#[derive(Debug, PartialEq, Eq)]
pub enum CallError<W> {
    /// Never accepted. The work is returned so the caller can shed it
    /// deliberately.
    Rejected(SubmitError<W>),
    /// Accepted, but the responder was dropped without an answer — the
    /// processor returned without replying, or its future panicked.
    Cancelled,
}

impl<W> fmt::Display for CallError<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rejected(_) => f.write_str("the call was not accepted"),
            Self::Cancelled => Cancelled.fmt(f),
        }
    }
}

impl<W: fmt::Debug> Error for CallError<W> {}

/// Work paired with the channel its answer travels back on.
pub struct Call<W: Work, R> {
    inner: W,
    respond: oneshot::Sender<R>,
}

impl<W: Work, R> Call<W, R> {
    /// Pair `work` with a fresh reply channel.
    pub fn new(work: W) -> (Self, oneshot::Receiver<R>) {
        let (respond, receive) = oneshot::channel();
        (Self { inner: work, respond }, receive)
    }

    pub fn work(&self) -> &W {
        &self.inner
    }

    /// Split into the work and the means to answer it, so a processor can
    /// consume one and hold the other across awaits.
    pub fn into_parts(self) -> (W, Responder<R>) {
        (self.inner, Responder(self.respond))
    }

    /// Answer and discard the work. Handy from
    /// [`Processor::on_expired`](crate::processor::Processor::on_expired).
    pub fn reply(self, response: R) {
        let _ = self.respond.send(response);
    }

    /// Whether the caller has stopped waiting.
    pub fn is_cancelled(&self) -> bool {
        self.respond.is_closed()
    }
}

impl<W: Work, R: Send + 'static> Work for Call<W, R> {
    type Key = W::Key;
    type Id = W::Id;

    fn key(&self) -> Self::Key {
        self.inner.key()
    }

    fn request_id(&self) -> Option<Self::Id> {
        self.inner.request_id()
    }

    fn class(&self) -> ClassId {
        self.inner.class()
    }

    fn time_to_live(&self) -> Option<Duration> {
        self.inner.time_to_live()
    }
}

/// The answering half of a [`Call`].
pub struct Responder<R>(oneshot::Sender<R>);

impl<R> Responder<R> {
    /// Answer the caller. A caller that has already given up is not an error.
    pub fn send(self, response: R) {
        let _ = self.0.send(response);
    }

    /// Whether the caller has stopped waiting. Worth checking before starting
    /// expensive work, since nobody will read the result.
    pub fn is_cancelled(&self) -> bool {
        self.0.is_closed()
    }
}

impl<W, R, C, const CLASSES: usize> Router<Call<W, R>, C, CLASSES>
where
    W: Work,
    R: Send + 'static,
    C: Clock,
{
    /// Submit and await the response, waiting on backpressure if the target
    /// shard's mailbox is full.
    pub async fn call(&self, work: W) -> Result<R, CallError<W>> {
        let (call, receive) = Call::new(work);
        self.submit(call)
            .await
            .map_err(|error| CallError::Rejected(error.map(|call| call.inner)))?;
        receive.await.map_err(|_| CallError::Cancelled)
    }

    /// Submit without waiting, reporting a full mailbox immediately and
    /// returning the response as a separate future.
    ///
    /// Rejection is synchronous and the answer is not, which is exactly the
    /// shape a load-shedding caller needs: it can give up before committing to
    /// an await.
    pub fn try_call(
        &self,
        work: W,
    ) -> Result<impl Future<Output = Result<R, Cancelled>> + use<W, R, C, CLASSES>, SubmitError<W>>
    {
        let (call, receive) = Call::new(work);
        self.try_submit(call).map_err(|error| error.map(|call| call.inner))?;
        Ok(async move { receive.await.map_err(|_| Cancelled) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::ManualClock;
    use crate::metrics::ShardStats;
    use crate::processor::Processor;
    use crate::shard::{self, ShardConfig};
    use grommet_core::Disposition;
    use std::sync::Arc;
    use tokio::sync::mpsc;

    #[derive(Debug, PartialEq, Eq)]
    struct Job {
        key: u64,
        ttl: Option<Duration>,
    }

    impl Work for Job {
        type Key = u64;
        type Id = ();
        fn key(&self) -> u64 {
            self.key
        }
        fn class(&self) -> ClassId {
            0
        }
        fn time_to_live(&self) -> Option<Duration> {
            self.ttl
        }
    }

    /// Counts per key and answers with the running total, unless told to stay
    /// silent so the cancellation path can be observed.
    #[derive(Clone, Copy)]
    struct Counter {
        answer: bool,
    }

    impl Processor for Counter {
        type Work = Call<Job, u64>;
        type State = u64;
        type Error = std::convert::Infallible;

        async fn process(
            &self,
            _key: u64,
            state: Option<u64>,
            call: Call<Job, u64>,
        ) -> Result<Disposition<u64>, Self::Error> {
            let total = state.unwrap_or(0) + 1;
            let (_job, responder) = call.into_parts();
            if self.answer {
                responder.send(total);
            }
            Ok(Disposition::Keep(total))
        }

        fn on_expired(&self, _key: u64, call: Call<Job, u64>) {
            call.reply(u64::MAX);
        }
    }

    async fn with_shard<F, Fut>(processor: Counter, mailbox: usize, driver: F)
    where
        F: FnOnce(Arc<Router<Call<Job, u64>, ManualClock, 2>>) -> Fut,
        Fut: Future<Output = ()>,
    {
        let clock = ManualClock::new();
        let (tx, rx) = mpsc::channel(mailbox);
        let router = Arc::new(Router::new(vec![tx], clock.clone()));
        let stats = Arc::new(ShardStats::<2>::default());
        let mut cfg = ShardConfig::new([4, 4]);
        cfg.tick = Duration::from_millis(1);
        let engine = shard::run(rx, processor, clock, stats, cfg);
        tokio::join!(engine, driver(router));
    }

    #[tokio::test(start_paused = true)]
    async fn a_call_carries_its_response_back_to_the_caller() {
        with_shard(Counter { answer: true }, 8, |router| async move {
            assert_eq!(router.call(Job { key: 3, ttl: None }).await, Ok(1));
            assert_eq!(router.call(Job { key: 3, ttl: None }).await, Ok(2));
            assert_eq!(router.call(Job { key: 4, ttl: None }).await, Ok(1), "state is per key");
        })
        .await;
    }

    #[tokio::test(start_paused = true)]
    async fn a_processor_that_never_answers_cancels_rather_than_hangs() {
        with_shard(Counter { answer: false }, 8, |router| async move {
            assert_eq!(router.call(Job { key: 1, ttl: None }).await, Err(CallError::Cancelled));
        })
        .await;
    }

    #[tokio::test(start_paused = true)]
    async fn shed_work_is_returned_unwrapped_so_the_caller_can_answer_it() {
        let clock = ManualClock::new();
        let (tx, _rx) = mpsc::channel(1);
        let router = Router::<Call<Job, u64>, ManualClock, 2>::new(vec![tx], clock);

        // Fill the one mailbox slot; nothing is consuming it.
        assert!(router.try_call(Job { key: 1, ttl: None }).is_ok(), "the first call fits");
        let Err(error) = router.try_call(Job { key: 2, ttl: None }) else {
            panic!("a full mailbox must shed");
        };

        assert!(matches!(error, SubmitError::Full(_)));
        assert_eq!(error.into_work().key, 2, "the caller gets its own work back, not a wrapper");
    }

    #[tokio::test(start_paused = true)]
    async fn expired_work_is_answered_from_the_expiry_hook() {
        with_shard(Counter { answer: true }, 8, |router| async move {
            let response = router.call(Job { key: 9, ttl: Some(Duration::ZERO) }).await;
            assert_eq!(response, Ok(u64::MAX), "the deadline path still answers the caller");
        })
        .await;
    }

    #[tokio::test(start_paused = true)]
    async fn a_responder_notices_that_its_caller_gave_up() {
        let (call, receive) = Call::<Job, u64>::new(Job { key: 1, ttl: None });
        assert!(!call.is_cancelled());
        let (_job, responder) = call.into_parts();
        assert!(!responder.is_cancelled());
        drop(receive);
        assert!(responder.is_cancelled(), "an abandoned call is worth detecting before working");
    }
}
