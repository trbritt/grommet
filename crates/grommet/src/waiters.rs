//! The queue of submitters parked on a full mailbox.
//!
//! Backpressure is the point of a bounded mailbox, so something has to hold the
//! senders that arrived while it was full and let them back in, in order, as it
//! drains. That is all this is.
//!
//! # Off the hot path, by construction
//!
//! A mailbox that is not full never touches this structure, and neither does
//! `try_send`. The lock below is taken on exactly two occasions: a sender
//! parking because the ring was full, and the shard handing a freed slot back
//! to whoever has waited longest. The shard's own check is an atomic load of
//! [`Waiters::any`], so a shard draining a mailbox nobody is waiting on never
//! reaches the lock at all.
//!
//! That is worth saying plainly because the thing being replaced does not have
//! this property: `tokio::sync::mpsc` returns its permit through a semaphore
//! whose wait list is a mutex, and it takes that mutex on *every* receive
//! whether or not anyone is waiting.
//!
//! # Why a lock and not a lock-free list
//!
//! Because senders can be cancelled. A `submit` inside a `select!` may be
//! dropped at any point, including while it is parked, and its registration has
//! to come out of the queue when it does; an entry left behind is a wake
//! delivered to nobody, which is a slot freed that no live sender is told
//! about, which is a starved submitter.
//!
//! Removing a node from a lock-free singly-linked queue is the problem hazard
//! pointers exist for, and it is why tokio, futures-intrusive and everything
//! else in this space guards its wait list with a mutex too. Taking one here is
//! not the compromise; taking one on the *drain* path would be.
//!
//! # Tickets, not indices
//!
//! A parked sender holds a [`Ticket`], and slots are recycled, so a bare index
//! would let a cancelled sender cancel whoever inherited its slot. Each
//! registration therefore gets a generation, and every operation checks it,
//! the same discipline the timer wheel uses for its handles. A ticket whose
//! generation has moved on refers to a registration that is already over, and
//! every operation on one is a no-op.
//!
//! Cancellation frees the slot but leaves the queue entry behind as a
//! tombstone, because removing from the middle of a `VecDeque` is linear.
//! Waking skips tombstones, and the queue is compacted when they come to
//! outnumber the live entries, so the memory a burst of cancellations costs is
//! given back rather than held for the life of the mailbox.

use std::collections::VecDeque;
use std::task::Waker;

#[cfg(loom)]
use loom::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicUsize, Ordering};

// Under loom the lock is loom's, so that the protocol built on it is explored
// rather than assumed. That is a weaker substitution than it would be for the
// waker cell, and deliberately so: there the algorithm was the thing under
// test, so a model of it proved nothing. Here mutual exclusion is the contract
// and loom's mutex implements that contract faithfully; what these models check
// is the protocol around the lock, not the lock.
#[cfg(loom)]
use loom::sync::Mutex;
#[cfg(not(loom))]
use parking_lot::Mutex;

#[cfg(loom)]
type Guard<'a, T> = loom::sync::MutexGuard<'a, T>;
#[cfg(not(loom))]
type Guard<'a, T> = parking_lot::MutexGuard<'a, T>;

/// A parked sender's claim on its registration.
///
/// Copy, because a sender holds one across polls and hands it to at most one of
/// [`Waiters::refresh`] or [`Waiters::cancel`]; the generation is what makes a
/// second use harmless rather than dangerous.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ticket {
    slot: u32,
    generation: u64,
}

/// Why a sender could not park.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Closed;

struct Slot {
    /// `None` once the registration is over, whether it was woken or cancelled.
    waker: Option<Waker>,
    /// Bumped every time the slot is handed out, so a ticket from a previous
    /// occupant is recognisable.
    generation: u64,
}

struct Inner {
    slots: Vec<Slot>,
    free: Vec<u32>,
    /// Registration order. Holds the generation alongside the slot so that an
    /// entry outlived by its registration is recognised and skipped.
    order: VecDeque<(u32, u64)>,
    next_generation: u64,
    closed: bool,
}

impl Inner {
    /// Whether `ticket` still names a live registration.
    fn live(&self, ticket: Ticket) -> bool {
        self.slots
            .get(ticket.slot as usize)
            .is_some_and(|slot| slot.generation == ticket.generation && slot.waker.is_some())
    }

    /// End the registration in `slot`, returning its waker if it was live.
    fn retire(&mut self, slot: u32) -> Option<Waker> {
        let entry = &mut self.slots[slot as usize];
        let waker = entry.waker.take();
        if waker.is_some() {
            // Only recycled once, however many stale tickets name it.
            entry.generation = entry.generation.wrapping_add(1);
            self.free.push(slot);
        }
        waker
    }

    /// Drop queue entries whose registrations are over, once they outnumber the
    /// ones that are not.
    fn compact(&mut self) {
        if self.order.len() <= 8 || self.order.len() < self.free.len() * 2 {
            return;
        }
        let slots = &self.slots;
        self.order.retain(|&(slot, generation)| {
            slots[slot as usize].generation == generation && slots[slot as usize].waker.is_some()
        });
    }
}

/// The senders parked on a full mailbox, in the order they arrived.
pub(crate) struct Waiters {
    inner: Mutex<Inner>,
    /// Live registrations. Published so that the shard can decide whether to
    /// take the lock without taking it.
    waiting: AtomicUsize,
}

impl Waiters {
    pub(crate) fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                slots: Vec::new(),
                free: Vec::new(),
                order: VecDeque::new(),
                next_generation: 0,
                closed: false,
            }),
            waiting: AtomicUsize::new(0),
        }
    }

    /// Whether anyone is parked.
    ///
    /// The shard's gate. This is a bare load on purpose: the ordering that
    /// makes it safe to act on lives at the call site, which has to fence
    /// between freeing a slot and reading this. See `mailbox`.
    #[inline]
    pub(crate) fn any(&self) -> bool {
        self.waiting.load(Ordering::Relaxed) != 0
    }

    /// Park `waker`, or report that the mailbox is gone.
    pub(crate) fn park(&self, waker: &Waker) -> Result<Ticket, Closed> {
        let mut inner = self.lock();
        if inner.closed {
            return Err(Closed);
        }

        let generation = inner.next_generation;
        inner.next_generation = generation.wrapping_add(1);

        let slot = match inner.free.pop() {
            Some(slot) => {
                let entry = &mut inner.slots[slot as usize];
                entry.waker = Some(waker.clone());
                entry.generation = generation;
                slot
            }
            None => {
                let slot = u32::try_from(inner.slots.len())
                    .expect("a mailbox parks fewer than 2^32 senders at once");
                inner.slots.push(Slot { waker: Some(waker.clone()), generation });
                slot
            }
        };
        inner.order.push_back((slot, generation));
        inner.compact();
        // Published while the lock is held, so a shard that observes it and
        // then takes the lock cannot see a half-built registration.
        self.publish(&inner);
        Ok(Ticket { slot, generation })
    }

    /// Point an existing registration at a new waker, as a re-polled future
    /// must.
    ///
    /// A ticket whose registration is over is ignored: the sender was already
    /// woken, and its next poll will find the room that woke it.
    pub(crate) fn refresh(&self, ticket: Ticket, waker: &Waker) {
        let mut inner = self.lock();
        if !inner.live(ticket) {
            return;
        }
        let entry = &mut inner.slots[ticket.slot as usize];
        match &entry.waker {
            // The steady state for a future polled repeatedly on one task.
            Some(existing) if existing.will_wake(waker) => {}
            _ => entry.waker = Some(waker.clone()),
        }
    }

    /// End a registration because the sender went away.
    ///
    /// Returns whether it was still live, which is what tells a cancelled
    /// sender it never received the wake it is giving up.
    pub(crate) fn cancel(&self, ticket: Ticket) -> bool {
        let mut inner = self.lock();
        if !inner.live(ticket) {
            return false;
        }
        drop(inner.retire(ticket.slot));
        inner.compact();
        self.publish(&inner);
        true
    }

    /// Hand one freed slot to whoever has waited longest.
    ///
    /// Returns whether anyone was woken, so a caller releasing several slots
    /// can stop once the queue is empty.
    pub(crate) fn wake_one(&self) -> bool {
        let waker = {
            let mut inner = self.lock();
            let woken = loop {
                let Some((slot, generation)) = inner.order.pop_front() else { break None };
                // A tombstone: this registration ended before its turn came up.
                if inner.slots[slot as usize].generation != generation {
                    continue;
                }
                if let Some(waker) = inner.retire(slot) {
                    break Some(waker);
                }
            };
            self.publish(&inner);
            woken
        };
        // Outside the lock: waking runs a scheduler, which must not be able to
        // re-enter this structure while it is held.
        match waker {
            Some(waker) => {
                waker.wake();
                true
            }
            None => false,
        }
    }

    /// Retire the queue: nobody may park again, and everyone parked is woken to
    /// discover it.
    pub(crate) fn close(&self) {
        let woken = {
            let mut inner = self.lock();
            inner.closed = true;
            let mut woken = Vec::new();
            while let Some((slot, generation)) = inner.order.pop_front() {
                if inner.slots[slot as usize].generation != generation {
                    continue;
                }
                woken.extend(inner.retire(slot));
            }
            self.publish(&inner);
            woken
        };
        for waker in woken {
            waker.wake();
        }
    }

    /// Republish the live count from state the lock protects.
    fn publish(&self, inner: &Inner) {
        let live = inner.slots.len() - inner.free.len();
        self.waiting.store(live, Ordering::Release);
    }

    fn lock(&self) -> Guard<'_, Inner> {
        #[cfg(not(loom))]
        {
            self.inner.lock()
        }
        #[cfg(loom)]
        {
            self.inner.lock().unwrap()
        }
    }
}

impl std::fmt::Debug for Waiters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Waiters").field("waiting", &self.waiting.load(Ordering::Relaxed)).finish()
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use std::sync::Arc;

    use std::task::Wake;

    /// Identifies itself when woken, so a test can assert on *which* sender was
    /// let back in rather than merely that someone was.
    struct Sender {
        id: usize,
        woken: Arc<std::sync::Mutex<Vec<usize>>>,
    }

    impl Wake for Sender {
        fn wake(self: Arc<Self>) {
            self.woken.lock().unwrap().push(self.id);
        }
    }

    /// A recording set of senders, handing out one waker per identity.
    struct Log(Arc<std::sync::Mutex<Vec<usize>>>);

    impl Log {
        fn new() -> Self {
            Self(Arc::new(std::sync::Mutex::new(Vec::new())))
        }

        fn waker(&self, id: usize) -> Waker {
            Waker::from(Arc::new(Sender { id, woken: Arc::clone(&self.0) }))
        }

        fn woken(&self) -> Vec<usize> {
            self.0.lock().unwrap().clone()
        }
    }

    #[test]
    fn senders_are_let_back_in_in_the_order_they_arrived() {
        // The fairness property the whole structure exists for: a submitter
        // that has waited longest goes first, so a busy mailbox cannot starve
        // whoever got there earliest.
        let waiters = Waiters::new();
        let log = Log::new();
        for id in 0..4 {
            waiters.park(&log.waker(id)).expect("the mailbox is open");
        }
        assert!(waiters.any());

        for _ in 0..4 {
            assert!(waiters.wake_one());
        }
        assert_eq!(log.woken(), [0, 1, 2, 3]);
        assert!(!waiters.any(), "waking every sender empties the queue");
        assert!(!waiters.wake_one(), "an empty queue has nobody to hand a slot to");
    }

    #[test]
    fn a_woken_sender_is_not_woken_a_second_time() {
        let waiters = Waiters::new();
        let log = Log::new();
        let first = waiters.park(&log.waker(0)).unwrap();
        waiters.park(&log.waker(1)).unwrap();

        waiters.wake_one();
        // Its registration is over, so the ticket it still holds does nothing.
        assert!(!waiters.cancel(first), "a woken registration is already over");
        waiters.wake_one();
        assert_eq!(log.woken(), [0, 1]);
    }

    #[test]
    fn a_cancelled_sender_is_skipped_and_the_rest_keep_their_order() {
        // A `submit` dropped inside a `select!`. Its slot must not absorb a
        // wake meant for someone still waiting.
        let waiters = Waiters::new();
        let log = Log::new();
        let first = waiters.park(&log.waker(0)).unwrap();
        let second = waiters.park(&log.waker(1)).unwrap();
        waiters.park(&log.waker(2)).unwrap();

        assert!(waiters.cancel(second), "the registration was live");
        assert!(!waiters.cancel(second), "cancelling twice is harmless");

        assert!(waiters.wake_one());
        assert!(waiters.wake_one());
        assert!(!waiters.wake_one());
        assert_eq!(log.woken(), [0, 2], "the cancelled sender absorbed no wake");
        assert!(!waiters.cancel(first));
    }

    #[test]
    fn a_stale_ticket_cannot_cancel_whoever_inherited_its_slot() {
        // Slots are recycled, so this is the failure a bare index would allow:
        // a departed sender silently unregistering a live one.
        let waiters = Waiters::new();
        let log = Log::new();
        let first = waiters.park(&log.waker(0)).unwrap();
        waiters.cancel(first);

        let second = waiters.park(&log.waker(1)).unwrap();
        assert_eq!(first.slot, second.slot, "the test is only meaningful if the slot is reused");
        assert_ne!(first.generation, second.generation);

        assert!(!waiters.cancel(first), "the stale ticket must do nothing");
        assert!(waiters.wake_one());
        assert_eq!(log.woken(), [1], "the live sender survived the stale cancel");
    }

    #[test]
    fn re_polling_a_parked_sender_points_it_at_the_new_waker() {
        let waiters = Waiters::new();
        let log = Log::new();
        let ticket = waiters.park(&log.waker(0)).unwrap();

        // A future may be polled by a different task than the one that parked
        // it, and the wake has to follow it.
        waiters.refresh(ticket, &log.waker(1));
        waiters.wake_one();
        assert_eq!(log.woken(), [1]);

        // Refreshing a registration that is over must not resurrect it.
        waiters.refresh(ticket, &log.waker(2));
        assert!(!waiters.wake_one());
        assert_eq!(log.woken(), [1]);
    }

    #[test]
    fn closing_wakes_everyone_and_refuses_new_arrivals() {
        let waiters = Waiters::new();
        let log = Log::new();
        for id in 0..3 {
            waiters.park(&log.waker(id)).unwrap();
        }

        waiters.close();
        assert_eq!(log.woken(), [0, 1, 2], "every parked sender is told to look again");
        assert!(!waiters.any());
        assert_eq!(waiters.park(&log.waker(9)), Err(Closed), "a closed mailbox parks nobody");
        assert_eq!(log.woken(), [0, 1, 2]);
    }

    #[test]
    fn slots_are_recycled_so_the_queue_tracks_peak_waiters_not_throughput() {
        let waiters = Waiters::new();
        let log = Log::new();
        for round in 0..1_000 {
            waiters.park(&log.waker(round)).unwrap();
            assert!(waiters.wake_one());
        }
        assert_eq!(waiters.inner.lock().slots.len(), 1, "one sender at a time needs one slot");
        assert_eq!(log.woken().len(), 1_000);
    }

    #[test]
    fn a_burst_of_cancellations_does_not_grow_the_queue_without_bound() {
        // Cancellation leaves a tombstone rather than paying a linear removal,
        // so the compaction that bounds them is load-bearing.
        let waiters = Waiters::new();
        let log = Log::new();
        let held = waiters.park(&log.waker(0)).unwrap();
        for round in 1..10_000 {
            let ticket = waiters.park(&log.waker(round)).unwrap();
            assert!(waiters.cancel(ticket));
        }
        assert!(
            waiters.inner.lock().order.len() < 64,
            "tombstones accumulated: {}",
            waiters.inner.lock().order.len()
        );

        // And the sender that waited through all of it is still first in line.
        assert!(waiters.wake_one());
        assert_eq!(log.woken(), [0]);
        assert!(!waiters.cancel(held));
    }

    #[test]
    fn the_debug_rendering_reports_how_many_are_parked() {
        let waiters = Waiters::new();
        let log = Log::new();
        assert!(format!("{waiters:?}").contains("waiting: 0"));
        waiters.park(&log.waker(0)).unwrap();
        assert!(format!("{waiters:?}").contains("waiting: 1"));
    }
}
