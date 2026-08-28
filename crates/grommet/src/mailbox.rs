//! The queue between a submitter and the shard that owns the key.
//!
//! Four operations, and underneath them a bounded ring the shard owns, a
//! doorbell that wakes it, and the queue of senders parked on a full ring. The
//! surface is deliberately narrow — it is exactly what the router and the shard
//! do and nothing else — because nothing downstream should be able to name the
//! substrate, and because the substrate is the part still moving.
//!
//! Depth composes: a shard will hold up to `capacity` items here *plus*
//! whatever its scheduler has already admitted, so the number of items in
//! flight for one shard is bounded by `capacity + Config::max_pending`, not by
//! either alone. See [`Builder::mailbox`](crate::scheduler::Builder::mailbox).
//!
//! # One theorem, applied in both directions
//!
//! Both halves of this can go to sleep, and each is woken by the other, so the
//! same lost-wakeup problem appears twice and is solved the same way twice.
//!
//! A shard with an empty ring parks on a doorbell; a sender with a full ring
//! parks on a wait list. In both
//! cases the sleeper must **announce itself before the check it would sleep
//! on**, and the waker must **publish before it looks for a sleeper**:
//!
//! | | announces | then checks | woken by |
//! |---|---|---|---|
//! | shard | registers on the doorbell | ring for an item | a sender that pushed |
//! | sender | parks on the wait list | ring for room | the shard, having popped |
//!
//! Getting that order right on each side is necessary but not sufficient,
//! because the two threads are storing one location and loading another, in
//! opposite orders. That is the store-buffer shape, and acquire and release do
//! not forbid both threads from missing each other — only a sequentially
//! consistent fence on each side does. So each side fences between publishing
//! and looking, and those fences are what make "the shard is asleep" and "the
//! ring is empty" impossible to observe together.
//!
//! # What the fences cost, and where they are not paid
//!
//! A sender fences once per push, and only then reads whether the shard is
//! parked, so the doorbell itself is untouched while the shard is running. The
//! shard fences once per *drain*, not once per item: it counts the slots a
//! burst freed and hands them all back at the end, which is the same batching
//! the admission loop already does. A drain that frees nothing, or that finds
//! nobody parked, does not fence at all.
//!
//! For contrast, the thing this replaces returns its capacity through a
//! semaphore whose wait list is a mutex, and takes that mutex on *every*
//! receive whether or not any sender is waiting.

use crate::doorbell::Doorbell;
use crate::waiters::{Ticket, Waiters};
use grommet_core::ring;
use std::sync::Arc;
use std::task::{Context, Poll};

#[cfg(loom)]
use loom::sync::atomic::{AtomicBool, AtomicUsize, Ordering, fence};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering, fence};

/// Everything both halves share.
struct Shared<W> {
    /// Producers push through this; the [`Inbox`] holds the other end.
    ring: ring::Producer<W>,
    /// Wakes the shard when it is parked on an empty ring.
    bell: Doorbell,
    /// Senders parked on a full ring, in arrival order.
    waiters: Waiters,
    /// Whether the shard is parked. Read by a sender after every push, so that
    /// a running shard is never rung.
    parked: AtomicBool,
    /// Live [`Mailbox`] handles. Reaching zero is what tells the shard to drain
    /// and stop.
    producers: AtomicUsize,
    /// Whether the [`Inbox`] is gone, which is what makes a send fail rather
    /// than queue into a ring nobody will read.
    departed: AtomicBool,
}

/// Create one shard's mailbox, returning the submitting and receiving halves.
///
/// `capacity` is how much burst the mailbox absorbs before a submitter feels
/// backpressure, and it is exact rather than rounded. It must be greater than
/// zero: a zero-capacity mailbox would make every submission wait for a
/// receive, which is a rendezvous rather than a queue.
///
/// # Panics
///
/// If `capacity` is zero.
pub fn channel<W>(capacity: usize) -> (Mailbox<W>, Inbox<W>) {
    assert!(capacity > 0, "a mailbox needs capacity");
    let (producer, consumer) = ring::bounded(capacity);
    let shared = Arc::new(Shared {
        ring: producer,
        bell: Doorbell::new(),
        waiters: Waiters::new(),
        parked: AtomicBool::new(false),
        producers: AtomicUsize::new(1),
        departed: AtomicBool::new(false),
    });
    (Mailbox { shared: Arc::clone(&shared) }, Inbox { ring: consumer, shared, freed: 0 })
}

/// The submitting half of one shard's mailbox. Cheap to clone, and every clone
/// feeds the same shard.
pub struct Mailbox<W> {
    shared: Arc<Shared<W>>,
}

impl<W> Clone for Mailbox<W> {
    fn clone(&self) -> Self {
        // Relaxed: this handle already keeps the mailbox open, so the increment
        // publishes nothing a reader could act on that it cannot already see.
        self.shared.producers.fetch_add(1, Ordering::Relaxed);
        Self { shared: Arc::clone(&self.shared) }
    }
}

impl<W> Drop for Mailbox<W> {
    fn drop(&mut self) {
        // Release, so a shard that observes zero has also observed everything
        // every departing sender pushed before leaving.
        if self.shared.producers.fetch_sub(1, Ordering::Release) == 1 {
            // The last one out. A parked shard has to learn that there will be
            // nothing more, so that it can drain and exit rather than sleep.
            self.shared.bell.ring();
        }
    }
}

impl<W> std::fmt::Debug for Mailbox<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mailbox").field("capacity", &self.shared.ring.capacity()).finish()
    }
}

impl<W> Mailbox<W> {
    /// Submit, waiting while the mailbox is full.
    ///
    /// The wait is the backpressure: a saturated shard stops admitting, its
    /// mailbox fills, and this suspends the caller rather than letting a queue
    /// grow without bound. Waiting senders are admitted in the order they
    /// arrived, so a steady stream of new submitters cannot starve one that has
    /// been waiting.
    ///
    /// Cancel-safe: dropping this future before it completes hands the item
    /// back by dropping it, removes the caller from the queue, and — if a slot
    /// had already been set aside — passes that slot to the next sender in line
    /// rather than stranding them.
    pub async fn send(&self, item: W) -> Result<(), Closed<W>> {
        let full = match self.try_send(item) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Closed(item)) => return Err(Closed(item)),
            Err(TrySendError::Full(item)) => item,
        };

        let shared = &*self.shared;
        let mut item = Some(full);
        let mut registration = Registration { shared, ticket: None };

        std::future::poll_fn(move |cx| {
            let value = item.take().expect("the item is put back on every pending path");

            // Announce before retrying. A slot freed between the retry and the
            // registration would otherwise be handed to nobody.
            match registration.ticket {
                Some(ticket) => shared.waiters.refresh(ticket, cx.waker()),
                None => match shared.waiters.park(cx.waker()) {
                    Ok(ticket) => registration.ticket = Some(ticket),
                    Err(_) => return Poll::Ready(Err(Closed(value))),
                },
            }
            // The other half of the store-buffer pair; the shard fences after
            // freeing a slot and before looking for someone to give it to.
            fence(Ordering::SeqCst);

            match self.try_send(value) {
                Ok(()) => {
                    registration.finish();
                    Poll::Ready(Ok(()))
                }
                Err(TrySendError::Closed(value)) => {
                    registration.finish();
                    Poll::Ready(Err(Closed(value)))
                }
                Err(TrySendError::Full(value)) => {
                    item = Some(value);
                    Poll::Pending
                }
            }
        })
        .await
    }

    /// Submit without ever waiting, reporting a full mailbox instead.
    #[inline]
    pub fn try_send(&self, item: W) -> Result<(), TrySendError<W>> {
        let shared = &*self.shared;
        if shared.departed.load(Ordering::Acquire) {
            return Err(TrySendError::Closed(item));
        }
        match shared.ring.try_push(item) {
            Ok(()) => {
                shared.wake_shard();
                Ok(())
            }
            // Re-checked, so that a shard that went away while this was pushing
            // is reported as gone rather than as momentarily full: a caller
            // told "full" would retry forever.
            Err(item) if shared.departed.load(Ordering::Acquire) => Err(TrySendError::Closed(item)),
            Err(item) => Err(TrySendError::Full(item)),
        }
    }
}

impl<W> Shared<W> {
    /// Ring the doorbell, but only if the shard is actually asleep.
    ///
    /// The fence is what makes the check safe to believe: without it this
    /// thread's push and the shard's park could each fail to see the other, and
    /// the shard would sleep on an item already sitting in the ring.
    #[inline]
    fn wake_shard(&self) {
        fence(Ordering::SeqCst);
        if self.parked.load(Ordering::Relaxed) {
            self.bell.ring();
        }
    }
}

/// A sender's place in the queue, given up whichever way its future ends.
struct Registration<'a, W> {
    shared: &'a Shared<W>,
    ticket: Option<Ticket>,
}

impl<W> Registration<'_, W> {
    /// Give up the place because the item went through. Any wake this sender
    /// received was spent on that push, so there is nothing to pass on.
    fn finish(&mut self) {
        if let Some(ticket) = self.ticket.take() {
            self.shared.waiters.cancel(ticket);
        }
    }
}

impl<W> Drop for Registration<'_, W> {
    fn drop(&mut self) {
        let Some(ticket) = self.ticket.take() else { return };
        // `cancel` reporting the registration already over means a slot was set
        // aside for a sender that is now walking away. Handing it to the next
        // in line is the difference between a cancelled `select!` branch and a
        // starved submitter.
        if !self.shared.waiters.cancel(ticket) {
            self.shared.waiters.wake_one();
        }
    }
}

/// The receiving half of one shard's mailbox. A shard owns exactly one, which
/// is why this is not cloneable.
pub struct Inbox<W> {
    ring: ring::Consumer<W>,
    shared: Arc<Shared<W>>,
    /// Slots this drain has freed and not yet handed back. Counted rather than
    /// released one at a time so that a burst fences once instead of per item.
    freed: usize,
}

impl<W> std::fmt::Debug for Inbox<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inbox").finish_non_exhaustive()
    }
}

impl<W> Drop for Inbox<W> {
    fn drop(&mut self) {
        // Order matters: senders must see the mailbox as gone before they are
        // woken, or a woken sender would park again on a queue nobody will
        // ever drain.
        self.shared.departed.store(true, Ordering::Release);
        self.shared.waiters.close();
        self.shared.bell.close();
    }
}

impl<W> Inbox<W> {
    /// Wait for the next item, or `None` once every [`Mailbox`] has been
    /// dropped and the queue is drained, which is how a shard is told to
    /// finish draining and exit.
    #[inline]
    pub async fn recv(&mut self) -> Option<W> {
        std::future::poll_fn(|cx| self.poll_recv(cx)).await
    }

    /// Poll for the next item, registering the task's waker when none is
    /// queued.
    ///
    /// Crate-private: this is the reactor's drain primitive, and the types in
    /// its signature are `std::task`'s, so nothing of the substrate shows
    /// through it. `Ready(None)` means every [`Mailbox`] is gone and the queue
    /// is drained: the signal to stop admitting.
    pub(crate) fn poll_recv(&mut self, cx: &mut Context<'_>) -> Poll<Option<W>> {
        if let Some(item) = self.take() {
            return Poll::Ready(Some(item));
        }

        // The burst is over: hand back what it freed before going to sleep on
        // the assumption that nothing more is coming.
        self.release();

        // Read before the last look at the ring, not after. Zero producers
        // means every sender had already dropped, and each one published
        // whatever it pushed before it did, so a ring that then reads empty
        // really is drained rather than momentarily behind.
        let closed = self.shared.producers.load(Ordering::Acquire) == 0;

        self.shared.bell.register(cx.waker());
        self.shared.parked.store(true, Ordering::Relaxed);
        fence(Ordering::SeqCst);

        if let Some(item) = self.take() {
            self.unpark();
            self.release();
            return Poll::Ready(Some(item));
        }
        if closed {
            self.unpark();
            return Poll::Ready(None);
        }
        Poll::Pending
    }

    /// Take an item that is already queued, without waiting.
    pub fn try_recv(&mut self) -> Result<W, TryRecvError> {
        // As in `poll_recv`: closed is read first so that an empty ring
        // afterwards is conclusive.
        let closed = self.shared.producers.load(Ordering::Acquire) == 0;
        let item = self.take();
        self.release();
        match item {
            Some(item) => Ok(item),
            None if closed => Err(TryRecvError::Closed),
            None => Err(TryRecvError::Empty),
        }
    }

    /// Pop one item, counting the slot it freed.
    #[inline]
    fn take(&mut self) -> Option<W> {
        let item = self.ring.pop()?;
        self.unpark();
        self.freed += 1;
        Some(item)
    }

    /// Hand this burst's freed slots to the senders waiting for them.
    ///
    /// The fence pairs with the one a parking sender performs, and is why a
    /// sender cannot come to rest while the room it wanted is already there.
    /// Nothing was freed means nothing to hand back and no fence to pay for.
    fn release(&mut self) {
        if self.freed == 0 {
            return;
        }
        let freed = std::mem::take(&mut self.freed);
        fence(Ordering::SeqCst);
        if !self.shared.waiters.any() {
            return;
        }
        for _ in 0..freed {
            if !self.shared.waiters.wake_one() {
                break;
            }
        }
    }

    /// Note that the shard is running, so senders stop ringing the doorbell.
    ///
    /// Loaded before storing because every sender reads this line on every
    /// push: writing it unconditionally would invalidate their copy each time
    /// the shard took an item.
    #[inline]
    fn unpark(&self) {
        if self.shared.parked.load(Ordering::Relaxed) {
            self.shared.parked.store(false, Ordering::Relaxed);
        }
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
    fn a_submitting_handle_is_one_pointer_wide() {
        // The router holds one of these per shard and clones them freely, so
        // the handle carries a pointer and the state lives behind it.
        assert_eq!(std::mem::size_of::<Mailbox<u64>>(), std::mem::size_of::<usize>());
    }
}

#[cfg(all(test, not(loom)))]
mod backpressure_tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;
    use std::time::Duration;

    /// Turns a lost wakeup into a failure rather than a hung test run.
    async fn before_long<F: Future>(future: F) -> F::Output {
        tokio::time::timeout(Duration::from_secs(5), future)
            .await
            .expect("a sender was never woken")
    }

    /// Drive a future once, so that it registers and suspends.
    async fn poll_once<F: Future>(future: &mut Pin<Box<F>>) {
        std::future::poll_fn(|cx| {
            let _ = future.as_mut().poll(cx);
            Poll::Ready(())
        })
        .await;
    }

    #[tokio::test]
    async fn a_full_mailbox_suspends_the_sender_until_the_shard_drains() {
        let (mailbox, mut inbox) = channel(1);
        mailbox.send(1).await.unwrap();

        let sender = mailbox.clone();
        let parked = tokio::spawn(async move { sender.send(2).await });
        tokio::task::yield_now().await;

        // Taking the queued item is what admits the waiting sender.
        assert_eq!(inbox.try_recv(), Ok(1));
        before_long(parked).await.unwrap().unwrap();
        assert_eq!(inbox.recv().await, Some(2));
    }

    #[tokio::test]
    async fn waiting_senders_are_admitted_in_the_order_they_arrived() {
        // The fairness the wait list exists for: a submitter that has waited
        // longest goes first, so a busy mailbox cannot starve it.
        let (mailbox, mut inbox) = channel(1);
        mailbox.send(0).await.unwrap();

        let mut senders = Vec::new();
        for item in 1..=3 {
            let mailbox = mailbox.clone();
            senders.push(tokio::spawn(async move { mailbox.send(item).await }));
            // One at a time, so the arrival order under test is the one set up.
            tokio::task::yield_now().await;
        }

        let mut seen = Vec::new();
        while seen.len() < 4 {
            seen.push(before_long(inbox.recv()).await.expect("senders remain"));
        }
        for sender in senders {
            before_long(sender).await.unwrap().unwrap();
        }
        assert_eq!(seen, [0, 1, 2, 3]);
    }

    #[tokio::test]
    async fn abandoning_a_waiting_sender_leaves_the_queue_behind_it_intact() {
        let (mailbox, mut inbox) = channel(1);
        mailbox.send(0).await.unwrap();

        let abandoned = mailbox.clone();
        let mut abandoned = Box::pin(abandoned.send(1));
        poll_once(&mut abandoned).await;

        let next = mailbox.clone();
        let waiting = tokio::spawn(async move { next.send(2).await });
        tokio::task::yield_now().await;

        // A `select!` branch losing its race, in effect.
        drop(abandoned);

        assert_eq!(inbox.try_recv(), Ok(0));
        before_long(waiting).await.unwrap().unwrap();
        assert_eq!(inbox.recv().await, Some(2));
    }

    #[tokio::test]
    async fn a_sender_dropped_after_being_admitted_passes_its_slot_on() {
        // The subtle half of cancel safety. The slot freed below is offered to
        // the first sender, which then goes away without using it. If that wake
        // were simply discarded, the second sender would wait forever with room
        // sitting in front of it.
        let (mailbox, mut inbox) = channel(1);
        mailbox.send(0).await.unwrap();

        let first = mailbox.clone();
        let mut first = Box::pin(first.send(1));
        poll_once(&mut first).await;

        let second = mailbox.clone();
        let waiting = tokio::spawn(async move { second.send(2).await });
        tokio::task::yield_now().await;

        assert_eq!(inbox.try_recv(), Ok(0), "this is the slot the first sender is offered");
        drop(first);

        before_long(waiting).await.unwrap().unwrap();
        assert_eq!(inbox.recv().await, Some(2));
    }

    #[tokio::test]
    async fn a_departing_shard_hands_every_waiting_sender_its_item_back() {
        let (mailbox, inbox) = channel(1);
        mailbox.send(0).await.unwrap();

        let senders: Vec<_> = (1..=3)
            .map(|item| {
                let mailbox = mailbox.clone();
                tokio::spawn(async move { mailbox.send(item).await })
            })
            .collect();
        tokio::task::yield_now().await;

        drop(inbox);
        for (expected, sender) in (1..=3).zip(senders) {
            let returned = before_long(sender).await.unwrap();
            assert!(
                matches!(returned, Err(Closed(item)) if item == expected) || returned == Ok(()),
                "a parked sender neither sent nor got its item back: {returned:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_sender_that_arrives_after_the_shard_has_gone_is_told_so() {
        let (mailbox, inbox) = channel::<u8>(1);
        drop(inbox);
        assert_eq!(mailbox.send(1).await, Err(Closed(1)));
    }
}

/// Exhaustive interleavings of the two directions a wake travels.
///
/// The unit tests above can show that a sender is admitted and that a shard
/// receives; they cannot show that no schedule exists in which one of them
/// sleeps while the thing it was waiting for is already there. That is what
/// these are for, and it is why the fences in this module are sequentially
/// consistent rather than merely release and acquire — under acquire and
/// release both of these models fail.
#[cfg(all(test, loom))]
mod loom_tests {
    use super::*;
    use loom::sync::atomic::AtomicBool;
    use std::sync::Arc as StdArc;
    use std::task::{Wake, Waker};

    struct Flag(AtomicBool);

    impl Flag {
        fn waker() -> (StdArc<Self>, Waker) {
            let flag = StdArc::new(Self(AtomicBool::new(false)));
            (StdArc::clone(&flag), Waker::from(flag))
        }

        fn woken(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
    }

    impl Wake for Flag {
        fn wake(self: StdArc<Self>) {
            self.0.store(true, Ordering::Release);
        }

        fn wake_by_ref(self: &StdArc<Self>) {
            self.0.store(true, Ordering::Release);
        }
    }

    /// A shard that finds nothing and parks must be rung by the sender that
    /// pushed. Whichever order the two interleave in, "the shard is asleep" and
    /// "there is an item in the ring" cannot both be true at the end.
    #[test]
    fn loom_a_shard_never_parks_on_an_item_already_pushed() {
        loom::model(|| {
            let (mailbox, mut inbox) = channel::<u32>(1);
            let (flag, waker) = Flag::waker();

            let sender = mailbox.clone();
            let producer = loom::thread::spawn(move || sender.try_send(1).is_ok());

            // The shard's turn: look, and park if there is nothing.
            let polled = inbox.poll_recv(&mut Context::from_waker(&waker));
            assert!(producer.join().unwrap(), "an empty mailbox refused a push");

            match polled {
                Poll::Ready(Some(item)) => assert_eq!(item, 1),
                Poll::Pending => {
                    assert!(flag.woken(), "the shard parked on an item that was already there");
                }
                Poll::Ready(None) => panic!("a live sender was reported as gone"),
            }
            // The original handle is held to the end, so nothing above can be
            // explained by the mailbox closing.
            drop(mailbox);
        });
    }

    /// The same theorem in the other direction. A sender that finds the ring
    /// full and parks must be woken by the shard that drained it.
    ///
    /// Both halves of the arrangement here are load-bearing. The sender parks
    /// on this thread so that its registration is still live when the assertion
    /// runs — a sender that has gone away is owed nothing. And the drainer
    /// hands the inbox back rather than dropping it, because dropping it closes
    /// the mailbox and wakes every parked sender, which would satisfy the
    /// assertion for the wrong reason.
    #[test]
    fn loom_a_sender_never_parks_on_room_already_freed() {
        loom::model(|| {
            let (mailbox, mut inbox) = channel::<u32>(1);
            assert!(mailbox.try_send(1).is_ok(), "an empty mailbox refused a push");
            let (flag, waker) = Flag::waker();

            let drainer = loom::thread::spawn(move || {
                let taken = inbox.try_recv();
                (inbox, taken)
            });

            let mut send = Box::pin(mailbox.send(2));
            let polled = send.as_mut().poll(&mut Context::from_waker(&waker));

            let (inbox, taken) = drainer.join().unwrap();
            assert_eq!(taken, Ok(1), "the queued item was not drained");
            if polled.is_pending() {
                assert!(flag.woken(), "a sender parked on room that was already free");
            }
            drop(send);
            drop(inbox);
        });
    }
}
