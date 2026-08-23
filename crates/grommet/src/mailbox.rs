//! The queue between a submitter and the shard that owns the key.
//!
//! This is deliberately a *seam* rather than a re-export. The mailbox is the
//! component most likely to be replaced — a shard-owned MPSC ring with its own
//! doorbell is the direction this runtime is going — and a public API that
//! named the channel it happens to use today would make that swap a breaking
//! change for everyone downstream.
//!
//! So the surface here is exactly the four operations the router and the shard
//! perform, and nothing else. Each one forwards to the underlying channel with
//! no branch or allocation of its own, and the types are `#[repr(transparent)]`
//! wrappers, so the insulation costs nothing at runtime.
//!
//! Depth composes: a shard will hold up to `capacity` items here *plus*
//! whatever its scheduler has already admitted, so the number of items in
//! flight for one shard is bounded by `capacity + Config::max_pending`, not by
//! either alone. See [`Builder::mailbox`](crate::runtime::Builder::mailbox).

use tokio::sync::mpsc;

/// Create one shard's mailbox, returning the submitting and receiving halves.
///
/// `capacity` is how much burst the mailbox absorbs before a submitter feels
/// backpressure. It must be greater than zero: a zero-capacity mailbox would
/// make every submission wait for a receive, which is a rendezvous rather than
/// a queue.
///
/// # Panics
///
/// If `capacity` is zero.
pub fn channel<W>(capacity: usize) -> (Mailbox<W>, Inbox<W>) {
    assert!(capacity > 0, "a mailbox needs capacity");
    let (sender, receiver) = mpsc::channel(capacity);
    (Mailbox { inner: sender }, Inbox { inner: receiver })
}

/// The submitting half of one shard's mailbox. Cheap to clone, and every clone
/// feeds the same shard.
#[repr(transparent)]
pub struct Mailbox<W> {
    inner: mpsc::Sender<W>,
}

// A manual impl: the channel handle is cloneable whatever `W` is, and a derive
// would demand `W: Clone` for no reason.
impl<W> Clone for Mailbox<W> {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

impl<W> std::fmt::Debug for Mailbox<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mailbox").field("capacity", &self.inner.max_capacity()).finish()
    }
}

impl<W> Mailbox<W> {
    /// Submit, waiting while the mailbox is full.
    ///
    /// The wait is the backpressure: a saturated shard stops admitting, its
    /// mailbox fills, and this suspends the caller rather than letting a queue
    /// grow without bound.
    #[inline]
    pub async fn send(&self, item: W) -> Result<(), Closed<W>> {
        self.inner.send(item).await.map_err(|error| Closed(error.0))
    }

    /// Submit without ever waiting, reporting a full mailbox instead.
    #[inline]
    pub fn try_send(&self, item: W) -> Result<(), TrySendError<W>> {
        match self.inner.try_send(item) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(item)) => Err(TrySendError::Full(item)),
            Err(mpsc::error::TrySendError::Closed(item)) => Err(TrySendError::Closed(item)),
        }
    }
}

/// The receiving half of one shard's mailbox. A shard owns exactly one, which
/// is why this is not cloneable.
#[repr(transparent)]
pub struct Inbox<W> {
    inner: mpsc::Receiver<W>,
}

impl<W> std::fmt::Debug for Inbox<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inbox").finish_non_exhaustive()
    }
}

impl<W> Inbox<W> {
    /// Wait for the next item, or `None` once every [`Mailbox`] has been
    /// dropped and the queue is drained — which is how a shard is told to
    /// finish draining and exit.
    #[inline]
    pub async fn recv(&mut self) -> Option<W> {
        self.inner.recv().await
    }

    /// Poll for the next item, registering the task's waker when none is
    /// queued.
    ///
    /// Crate-private: this is the reactor's drain primitive, and the types in
    /// its signature are `std::task`'s, so nothing of the substrate shows
    /// through it. `Ready(None)` means every [`Mailbox`] is gone and the queue
    /// is drained — the signal to stop admitting.
    #[inline]
    pub(crate) fn poll_recv(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<W>> {
        self.inner.poll_recv(cx)
    }

    /// Take an item that is already queued, without waiting.
    #[inline]
    pub fn try_recv(&mut self) -> Result<W, TryRecvError> {
        self.inner.try_recv().map_err(|error| match error {
            mpsc::error::TryRecvError::Empty => TryRecvError::Empty,
            mpsc::error::TryRecvError::Disconnected => TryRecvError::Closed,
        })
    }
}

/// Every receiver is gone. The item is handed back rather than dropped, so the
/// caller can answer whoever submitted it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Closed<W>(pub W);

impl<W> Closed<W> {
    pub fn into_inner(self) -> W {
        self.0
    }
}

/// Why an immediate submission did not go through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TrySendError<W> {
    /// The mailbox is at capacity. Waiting, or shedding, are both reasonable
    /// answers; the runtime never picks one for you.
    Full(W),
    /// The receiving shard is gone.
    Closed(W),
}

impl<W> TrySendError<W> {
    pub fn into_inner(self) -> W {
        match self {
            Self::Full(item) | Self::Closed(item) => item,
        }
    }
}

/// Why an immediate receive produced nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TryRecvError {
    /// Nothing is queued right now, but senders remain.
    Empty,
    /// Every sender is gone and the queue is drained.
    Closed,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn items_arrive_in_submission_order() {
        let (mailbox, mut inbox) = channel(4);
        for item in 0..3 {
            mailbox.send(item).await.unwrap();
        }
        assert_eq!(inbox.recv().await, Some(0));
        assert_eq!(inbox.recv().await, Some(1));
        assert_eq!(inbox.recv().await, Some(2));
    }

    #[tokio::test]
    async fn a_full_mailbox_hands_the_item_back_rather_than_dropping_it() {
        let (mailbox, mut inbox) = channel(1);
        mailbox.try_send(1).unwrap();
        assert_eq!(
            mailbox.try_send(2),
            Err(TrySendError::Full(2)),
            "the caller gets its work back"
        );

        // Draining makes room again, so a shed submitter can retry.
        assert_eq!(inbox.recv().await, Some(1));
        mailbox.try_send(2).unwrap();
        assert_eq!(inbox.recv().await, Some(2));
    }

    #[tokio::test]
    async fn a_departed_shard_is_reported_by_both_submission_paths() {
        let (mailbox, inbox) = channel::<u8>(4);
        drop(inbox);
        assert_eq!(mailbox.try_send(1), Err(TrySendError::Closed(1)));
        assert_eq!(mailbox.send(2).await, Err(Closed(2)));
        assert_eq!(Closed(3).into_inner(), 3);
        assert_eq!(TrySendError::Full(4).into_inner(), 4);
    }

    #[tokio::test]
    async fn an_immediate_receive_separates_an_empty_queue_from_a_closed_one() {
        let (mailbox, mut inbox) = channel(4);
        assert_eq!(inbox.try_recv(), Err(TryRecvError::Empty), "senders remain, so this is a lull");

        mailbox.send(9).await.unwrap();
        assert_eq!(inbox.try_recv(), Ok(9));

        // Closing is what tells a shard to stop admitting and drain, so it must
        // not be confused with a momentarily empty queue.
        drop(mailbox);
        assert_eq!(inbox.try_recv(), Err(TryRecvError::Closed));
        assert_eq!(inbox.recv().await, None);
    }

    #[tokio::test]
    async fn a_drained_mailbox_still_delivers_what_was_already_queued() {
        let (mailbox, mut inbox) = channel(4);
        mailbox.send(1).await.unwrap();
        mailbox.send(2).await.unwrap();
        drop(mailbox);

        // A shard drains before it exits, so closing must not discard work
        // that was already accepted.
        assert_eq!(inbox.try_recv(), Ok(1));
        assert_eq!(inbox.recv().await, Some(2));
        assert_eq!(inbox.recv().await, None);
    }

    #[test]
    #[should_panic(expected = "a mailbox needs capacity")]
    fn a_zero_capacity_mailbox_is_refused() {
        let _ = channel::<u8>(0);
    }

    #[test]
    fn the_halves_are_the_size_of_the_channel_they_wrap() {
        use std::mem::size_of;
        assert_eq!(size_of::<Mailbox<u64>>(), size_of::<mpsc::Sender<u64>>());
        assert_eq!(size_of::<Inbox<u64>>(), size_of::<mpsc::Receiver<u64>>());
    }
}
