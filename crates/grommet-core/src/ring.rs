//! A bounded MPSC ring specialized for a reactor consumer.
//!
//! Many threads submit; exactly one shard drains. That asymmetry is the whole
//! design. A general-purpose MPMC queue has to let any thread pop, so its
//! consumer pays a compare-exchange to claim a position and has to spin when it
//! finds a slot whose producer has not published yet. Neither cost buys
//! anything here: the shard is the only consumer, so its read position is a
//! plain field it owns, and it has a reactor turn's worth of other work to do
//! rather than a reason to wait.
//!
//! # The protocol
//!
//! Positions are Vyukov stamps: a lap in the high bits and a slot index in the
//! low bits, packed into one `usize`. Each slot carries the stamp of the
//! position that may next act on it, which is what lets a producer decide
//! whether a slot is free without consulting the consumer, and the consumer
//! decide whether a slot is published without consulting the producers. Nothing
//! reads anyone else's cursor on the hot path.
//!
//! A producer claims a position by compare-exchanging the shared tail forward,
//! writes into the slot it claimed, and then releases the slot's stamp. That
//! release is the publication: the value is visible to the consumer exactly
//! when the stamp is.
//!
//! The consumer looks at one slot. If its stamp says published, it takes the
//! value, releases the slot for the next lap, and advances its own position.
//! Otherwise it returns `None`, and that is where this differs from a general
//! queue: "not published" covers both an empty ring and a slot some producer
//! has claimed but not yet filled. A linearizable `pop` has to tell those apart
//! and so has to wait for the producer. A reactor does not. It treats both as
//! nothing right now, does its other work, and looks again next turn.
//!
//! # Where spinning is and is not allowed
//!
//! The consumer never spins. A shard's reactor holds timers, completions and
//! dispatch behind its drain, so a backoff loop there does not cost throughput,
//! it costs latency on everything else the turn was going to do. `pop` looks at
//! one slot and returns.
//!
//! Producers race each other for a position, and there the opposite holds: they
//! must back off. Two paths retry, losing the compare-exchange and finding that
//! someone advanced the tail mid-read, and the second is the common one under
//! load because every producer reads the tail before acting on it. Going
//! straight back around on either turns ordinary contention into a storm on the
//! one line they are all chasing. Backing off on both is worth an order of
//! magnitude at eight producers: it is the difference between a push cost flat
//! in the number of producers and one that degrades linearly with it.
//!
//! Neither path waits on a *particular* thread. A producer that cannot get in
//! reports a full ring rather than blocking behind whoever holds the claim.
//!
//! # Why there is no `was_empty`
//!
//! It is tempting to have `try_push` report whether the ring had been empty, so
//! the layer above can skip a wake when it was not. That signal is wrong here,
//! and quietly so. The consumer can pop the last published item, find the next
//! slot claimed but unpublished, report empty and park, while the ring is not
//! empty at all. The producer holding that claim then publishes, sees a
//! non-empty ring, skips the wake, and the shard sleeps on work sitting in
//! front of it.
//!
//! So the signal is not offered. Deciding when to wake belongs to the layer
//! that owns the parking, and the correct form of that decision is a flag the
//! consumer publishes before parking and the producer reads after publishing.
//! That is a store-buffer pattern, and it needs sequential consistency on both
//! sides: acquire and release alone permit both threads to miss each other.
//!
//! # Safety invariant
//!
//! > A slot's value cell is accessed only by the thread that observed the
//! > slot's stamp naming its own position, and only before that thread stores
//! > the next stamp.
//!
//! A producer reaches a slot only by winning the compare-exchange that claims
//! that exact position, and at most one thread can win a given position. The
//! consumer reaches a slot only when the stamp says the value is published,
//! which happens after the producer's last access to it. Positions are handed
//! out in order and a lap's worth apart, so no two live accesses name the same
//! slot.
//!
//! Values live in `Option<T>` rather than `MaybeUninit<T>`. That costs a
//! discriminant per slot and removes an entire invariant: no initialization
//! state to track, no `assume_init`, and no hand-written `Drop` to leak or
//! double-free, because dropping the buffer drops whatever is still in it. The
//! exclusion above is then the only thing this module has to be right about,
//! and it is what loom checks directly.
#![allow(unsafe_code)]

use crate::cell::UnsafeCell;
use crossbeam_utils::{Backoff, CachePadded};
use std::sync::Arc;

#[cfg(loom)]
use loom::sync::atomic::AtomicUsize;
#[cfg(not(loom))]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};

/// One position's worth of storage and its publication stamp.
struct Slot<T> {
    /// The position that may next act on this slot. Equal to a position means
    /// "free, claim me"; one past it means "published, take me".
    stamp: AtomicUsize,
    value: UnsafeCell<Option<T>>,
}

struct Inner<T> {
    /// The next position to be claimed. The only cell every producer touches,
    /// and therefore the only one worth padding.
    tail: CachePadded<AtomicUsize>,
    slots: Box<[Slot<T>]>,
    capacity: usize,
    /// A stamp of `{ lap: 1, index: 0 }`: the smallest power of two greater
    /// than the capacity. Making it a power of two is what lets a position be
    /// split into lap and index with a mask rather than a division, while the
    /// capacity itself stays exactly what the caller asked for.
    one_lap: usize,
}

// SAFETY: `Inner` shares `T` between threads and nothing else. The stamp
// protocol serializes every access to a value cell, so the only requirement is
// that `T` may cross a thread boundary at all.
unsafe impl<T: Send> Send for Inner<T> {}
unsafe impl<T: Send> Sync for Inner<T> {}

impl<T> Inner<T> {
    #[inline]
    fn index(&self, position: usize) -> usize {
        position & (self.one_lap - 1)
    }

    /// The position after `position`, wrapping the index and carrying the lap.
    #[inline]
    fn advance(&self, position: usize) -> usize {
        if self.index(position) + 1 < self.capacity {
            position + 1
        } else {
            (position & !(self.one_lap - 1)).wrapping_add(self.one_lap)
        }
    }
}

/// The largest ring the lap encoding can address. A lap has to fit above the
/// capacity in the same word, so half the position space is the ceiling. Named
/// and checked so that an absurd capacity says what is wrong rather than
/// failing as an arithmetic overflow inside the constructor.
const MAX_CAPACITY: usize = usize::MAX >> 1;

/// Create a ring of exactly `capacity` slots.
///
/// # Panics
///
/// If `capacity` is zero, which would be a rendezvous rather than a queue, or
/// if it exceeds what the lap encoding can address, which is half the
/// position space.
pub fn bounded<T>(capacity: usize) -> (Producer<T>, Consumer<T>) {
    assert!(capacity > 0, "a ring needs capacity");
    assert!(capacity <= MAX_CAPACITY, "a ring holds at most {MAX_CAPACITY} items");
    let one_lap = (capacity + 1).next_power_of_two();
    let inner = Arc::new(Inner {
        tail: CachePadded::new(AtomicUsize::new(0)),
        // Slot `i` starts free for lap zero: the stamp `{ lap: 0, index: i }`,
        // which is the position that will first claim it.
        slots: (0..capacity)
            .map(|index| Slot { stamp: AtomicUsize::new(index), value: UnsafeCell::new(None) })
            .collect(),
        capacity,
        one_lap,
    });
    (Producer { inner: Arc::clone(&inner) }, Consumer { inner, head: 0 })
}

/// The submitting half. Cheap to clone, and every clone feeds the same ring.
pub struct Producer<T> {
    inner: Arc<Inner<T>>,
}

// A manual impl: the handle clones whatever `T` is, and a derive would demand
// `T: Clone` for no reason.
impl<T> Clone for Producer<T> {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

impl<T> Producer<T> {
    /// Slots in the ring, which is what the caller asked for and not a rounded
    /// version of it.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    /// Push, or hand the value back if there is no room.
    ///
    /// Never waits, and never waits on another producer: a claimed-but-
    /// unpublished slot means every position is spoken for, which is a full
    /// ring and is reported as one.
    ///
    /// `Err` means the ring was full at some instant during the call. Like any
    /// lock-free bounded queue, that instant may already have passed by the
    /// time the caller reads the result.
    pub fn try_push(&self, value: T) -> Result<(), T> {
        let inner = &*self.inner;
        let mut tail = inner.tail.load(Relaxed);
        // Only ever used on the contended path below, and cheap to construct.
        let backoff = Backoff::new();

        loop {
            let index = inner.index(tail);
            debug_assert!(index < inner.slots.len(), "position outside the ring: {tail}");
            // SAFETY: a position's index is masked into `0..one_lap` and only
            // ever takes values `advance` produces, which stay below the
            // capacity. The `debug_assert!` above checks that in every test,
            // simulation and Miri run.
            let slot = unsafe { inner.slots.get_unchecked(index) };

            // Acquire, so that a slot the consumer released is seen released
            // along with everything the consumer did before releasing it.
            let stamp = slot.stamp.load(Acquire);

            if stamp == tail {
                // Acquire on success is enough: it keeps the write below from
                // moving above the claim, and the stamp's own release is what
                // publishes the value. The tail itself carries no data, so
                // nothing needs to be released with it.
                match inner.tail.compare_exchange_weak(tail, inner.advance(tail), Acquire, Relaxed)
                {
                    Ok(_) => {
                        // SAFETY: winning this exchange claims exactly this
                        // position, and only one thread can win it. The
                        // consumer will not touch the slot until the stamp
                        // below says it may.
                        // `write` rather than an assignment: assigning through
                        // a raw pointer drops what was there, which means
                        // reading the slot before writing it. The slot is
                        // always `None` here, so there is nothing to drop.
                        unsafe { slot.value.with_mut(|cell| cell.write(Some(value))) };
                        // The publication. Release, so the write above is
                        // visible to whoever acquires this stamp.
                        slot.stamp.store(tail.wrapping_add(1), Release);
                        return Ok(());
                    }
                    Err(current) => {
                        // Lost the claim to another producer. Back off before
                        // trying again: on a load-linked/store-conditional
                        // machine an immediate retry makes every participant
                        // lose its reservation rather than letting one of them
                        // win. See the module's note on where spinning is
                        // allowed.
                        tail = current;
                        backoff.spin();
                        continue;
                    }
                }
            }

            // The slot still holds the previous lap's value, so the ring is
            // full. This test reads nothing but the slot, which is what keeps a
            // saturated ring from turning into a storm on the tail: producers
            // that cannot get in stop touching the one line everybody else is
            // trying to advance.
            if stamp.wrapping_add(inner.one_lap) == tail.wrapping_add(1) {
                return Err(value);
            }

            // Neither free nor visibly full, which leaves two possibilities: a
            // producer claimed this position and has not published yet, or this
            // thread is looking at a stale tail. Relaxed, because the tail is
            // only a hint about which slot to try. The compare-exchange is what
            // claims one, and the stamp is what synchronizes.
            let current = inner.tail.load(Relaxed);
            if current == tail {
                // The view is current, so the position really is claimed:
                // every slot is spoken for, which is a full ring.
                return Err(value);
            }
            // Someone else advanced the tail while this thread was reading it.
            // Every producer reads the tail before acting on it, so every
            // producer lands here whenever another gets in first: this is the
            // busiest path under load, and the one whose backoff matters most.
            // Without it, push cost degrades linearly in the number of
            // producers instead of staying flat.
            tail = current;
            backoff.spin();
        }
    }
}

impl<T> std::fmt::Debug for Producer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Producer").field("capacity", &self.capacity()).finish_non_exhaustive()
    }
}

/// The draining half. Not cloneable: the single consumer is the premise the
/// whole structure is built on, so it is a property of the type rather than a
/// rule in the documentation.
pub struct Consumer<T> {
    inner: Arc<Inner<T>>,
    /// The next position to read. Owned outright by this handle, which is why
    /// draining costs no atomic read-modify-write at all.
    head: usize,
}

impl<T> Consumer<T> {
    /// Slots in the ring.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    /// Take the next published value, if there is one.
    ///
    /// `None` means *nothing right now*, which covers an empty ring and a slot
    /// a producer has claimed but not yet filled. The caller is expected to
    /// have other work and to look again later; this never spins.
    pub fn pop(&mut self) -> Option<T> {
        let inner = &*self.inner;
        let index = inner.index(self.head);
        debug_assert!(index < inner.slots.len(), "position outside the ring: {}", self.head);
        // SAFETY: as in `try_push`. The index is masked, and `advance` never
        // produces one at or beyond the capacity.
        let slot = unsafe { inner.slots.get_unchecked(index) };

        // Acquire, so that taking the value below sees the producer's write.
        if slot.stamp.load(Acquire) != self.head.wrapping_add(1) {
            return None;
        }

        // SAFETY: the stamp says a producer published this position and is done
        // with the slot, and this is the only consumer, so no other access to
        // this cell is live.
        let value = unsafe { slot.value.with_mut(|cell| (*cell).take()) };
        debug_assert!(value.is_some(), "a published slot always holds a value");

        // Release the slot for the next lap. Release, so a producer that
        // acquires this stamp also sees the cell emptied.
        slot.stamp.store(self.head.wrapping_add(inner.one_lap), Release);
        self.head = inner.advance(self.head);
        value
    }

    /// Whether a [`pop`] right now would return `None`.
    ///
    /// Carries the same meaning as `pop`'s `None`: it is a statement about this
    /// instant, and a claimed-but-unpublished slot counts as nothing yet.
    ///
    /// [`pop`]: Consumer::pop
    #[inline]
    pub fn is_empty(&self) -> bool {
        let inner = &*self.inner;
        let index = inner.index(self.head);
        debug_assert!(index < inner.slots.len(), "position outside the ring: {}", self.head);
        // SAFETY: as above.
        let slot = unsafe { inner.slots.get_unchecked(index) };
        slot.stamp.load(Acquire) != self.head.wrapping_add(1)
    }
}

impl<T> std::fmt::Debug for Consumer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Consumer").field("capacity", &self.capacity()).finish_non_exhaustive()
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn values_come_out_in_the_order_one_producer_put_them_in() {
        let (producer, mut consumer) = bounded(4);
        for value in 0..4 {
            producer.try_push(value).unwrap();
        }
        assert_eq!(
            (0..4).map(|_| consumer.pop()).collect::<Vec<_>>(),
            [Some(0), Some(1), Some(2), Some(3)]
        );
        assert_eq!(consumer.pop(), None);
    }

    #[test]
    fn a_full_ring_hands_the_value_back_rather_than_dropping_it() {
        let (producer, mut consumer) = bounded(2);
        producer.try_push(1).unwrap();
        producer.try_push(2).unwrap();
        assert_eq!(producer.try_push(3), Err(3), "the caller gets its value back");

        // Draining makes room again, and the freed slot is reusable.
        assert_eq!(consumer.pop(), Some(1));
        producer.try_push(3).unwrap();
        assert_eq!(consumer.pop(), Some(2));
        assert_eq!(consumer.pop(), Some(3));
        assert_eq!(consumer.pop(), None);
    }

    #[test]
    fn capacity_is_exactly_what_was_asked_for_even_when_it_is_not_a_power_of_two() {
        // The lap encoding rounds up internally; the capacity the caller sees
        // must not, or a mailbox would silently absorb more burst than it was
        // configured for.
        for capacity in 1..=9 {
            let (producer, mut consumer) = bounded(capacity);
            assert_eq!(producer.capacity(), capacity);
            assert_eq!(consumer.capacity(), capacity);

            for value in 0..capacity {
                producer.try_push(value).unwrap_or_else(|_| panic!("{value} fits in {capacity}"));
            }
            assert_eq!(producer.try_push(usize::MAX), Err(usize::MAX), "capacity {capacity}");
            for value in 0..capacity {
                assert_eq!(consumer.pop(), Some(value));
            }
            assert_eq!(consumer.pop(), None);
        }
    }

    #[test]
    fn positions_wrap_through_many_laps_without_losing_their_ordering() {
        // Exercises the lap carry: with a capacity that is not a power of two,
        // the index wraps before the lap bits do, which is where an off-by-one
        // in `advance` would show up.
        let (producer, mut consumer) = bounded(3);
        for round in 0..10_000 {
            producer.try_push(round).unwrap();
            producer.try_push(round + 1).unwrap();
            assert_eq!(consumer.pop(), Some(round));
            assert_eq!(consumer.pop(), Some(round + 1));
            assert!(consumer.is_empty());
        }
    }

    #[test]
    fn an_empty_ring_reports_itself_empty_and_a_filled_one_does_not() {
        let (producer, mut consumer) = bounded(2);
        assert!(consumer.is_empty());
        producer.try_push(1).unwrap();
        assert!(!consumer.is_empty());
        assert_eq!(consumer.pop(), Some(1));
        assert!(consumer.is_empty());
    }

    /// Counts its own drops, so a leak or a double free is visible rather than
    /// merely absent from Miri's output.
    struct Tracked<'a>(&'a AtomicUsize);

    impl Drop for Tracked<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    #[test]
    fn dropping_the_ring_drops_what_was_still_in_it_exactly_once() {
        let drops = AtomicUsize::new(0);
        {
            let (producer, mut consumer) = bounded(4);
            for _ in 0..4 {
                assert!(producer.try_push(Tracked(&drops)).is_ok());
            }
            // One taken out and dropped by the caller, three left behind for
            // the ring to dispose of.
            drop(consumer.pop().expect("a value was queued"));
            assert_eq!(drops.load(std::sync::atomic::Ordering::Relaxed), 1);
        }
        assert_eq!(
            drops.load(std::sync::atomic::Ordering::Relaxed),
            4,
            "values left in a dropped ring must be dropped, and only once"
        );
    }

    #[test]
    fn a_ring_outlives_the_handle_that_was_dropped_first() {
        let (producer, mut consumer) = bounded(2);
        producer.try_push(7).unwrap();
        drop(producer);
        assert_eq!(consumer.pop(), Some(7), "the queued value survives its producer");

        let (producer, consumer) = bounded(2);
        producer.try_push(8).unwrap();
        drop(consumer);
        assert_eq!(producer.try_push(9), Ok(()), "a departed consumer is not this layer's concern");
    }

    #[test]
    #[should_panic(expected = "a ring needs capacity")]
    fn a_zero_capacity_ring_is_refused() {
        let _ = bounded::<u8>(0);
    }

    #[test]
    #[should_panic(expected = "a ring holds at most")]
    fn a_capacity_the_lap_encoding_cannot_address_is_refused() {
        // The failure to avoid is an arithmetic overflow deep in the
        // constructor, which would say nothing about what the caller did wrong.
        let _ = bounded::<u8>(MAX_CAPACITY + 1);
    }

    /// Drives real producers against a real consumer and checks the two
    /// properties the structure exists to provide: nothing is lost or
    /// duplicated, and each producer's own values stay in order.
    ///
    /// Miri runs this, which is what makes the exclusion argument above worth
    /// anything: an access that escaped the stamp protocol is a data race here
    /// rather than a silent one in production.
    #[test]
    fn concurrent_producers_lose_nothing_and_keep_their_own_order() {
        let (producers, per_producer, capacity) =
            if cfg!(miri) { (2, 16, 4) } else { (4, 4_000, 8) };
        let total = producers * per_producer;

        let (producer, mut consumer) = bounded::<(usize, usize)>(capacity);
        let threads: Vec<_> = (0..producers)
            .map(|id| {
                let producer = producer.clone();
                std::thread::spawn(move || {
                    for sequence in 0..per_producer {
                        let mut value = (id, sequence);
                        // A deliberately small ring, so the full path is the
                        // common one rather than an edge case.
                        while let Err(returned) = producer.try_push(value) {
                            value = returned;
                            std::thread::yield_now();
                        }
                    }
                })
            })
            .collect();

        let mut seen: Vec<VecDeque<usize>> = (0..producers).map(|_| VecDeque::new()).collect();
        let mut taken = 0;
        while taken < total {
            match consumer.pop() {
                Some((id, sequence)) => {
                    seen[id].push_back(sequence);
                    taken += 1;
                }
                None => std::thread::yield_now(),
            }
        }

        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(consumer.pop(), None, "everything pushed was accounted for");
        for (id, sequences) in seen.iter().enumerate() {
            assert_eq!(sequences.len(), per_producer, "producer {id} lost or duplicated values");
            assert!(
                sequences.iter().copied().eq(0..per_producer),
                "producer {id}'s values were reordered"
            );
        }
    }
}

/// Exhaustive interleavings of the claim-write-publish handoff.
///
/// Every model here also gets the exclusion check for free: the value cells are
/// loom's `UnsafeCell` under `--cfg loom`, so two overlapping accesses fail the
/// model rather than being undefined behaviour that happens not to bite.
#[cfg(all(test, loom))]
mod loom_tests {
    use super::*;

    /// The core property. Two producers and a consumer running concurrently
    /// against a ring with no spare room: every value pushed comes out exactly
    /// once, whatever order the three threads interleave in.
    #[test]
    fn loom_concurrent_producers_and_a_consumer_lose_no_value() {
        loom::model(|| {
            let (producer, mut consumer) = bounded(2);

            let left = {
                let producer = producer.clone();
                loom::thread::spawn(move || producer.try_push(1).is_ok())
            };
            let right = {
                let producer = producer.clone();
                loom::thread::spawn(move || producer.try_push(2).is_ok())
            };

            // Bounded attempts, so the model stays finite while still racing a
            // drain against both pushes.
            let mut taken = Vec::new();
            for _ in 0..2 {
                if let Some(value) = consumer.pop() {
                    taken.push(value);
                }
            }

            assert!(left.join().unwrap(), "a ring with two free slots refused a push");
            assert!(right.join().unwrap(), "a ring with two free slots refused a push");
            while let Some(value) = consumer.pop() {
                taken.push(value);
            }

            taken.sort_unstable();
            assert_eq!(taken, [1, 2], "a value was lost or duplicated");
        });
    }

    /// Capacity is a real bound, not an approximate one. With one slot and two
    /// producers and nobody draining, exactly one of them may win, and the
    /// loser must get its value back rather than have it dropped.
    #[test]
    fn loom_a_full_ring_admits_exactly_one_of_two_racing_producers() {
        loom::model(|| {
            let (producer, mut consumer) = bounded(1);

            let left = {
                let producer = producer.clone();
                loom::thread::spawn(move || producer.try_push(1))
            };
            let right = {
                let producer = producer.clone();
                loom::thread::spawn(move || producer.try_push(2))
            };

            let left = left.join().unwrap();
            let right = right.join().unwrap();
            let admitted = usize::from(left.is_ok()) + usize::from(right.is_ok());
            assert_eq!(admitted, 1, "a one-slot ring admitted {admitted} values");

            let queued = consumer.pop().expect("the admitted value is queued");
            let handed_back = left.err().or(right.err()).expect("the loser gets its value back");
            assert_ne!(queued, handed_back, "the same value was both queued and rejected");
            assert_eq!(consumer.pop(), None);
        });
    }

    /// The lap boundary, which is the ordering that is easiest to get wrong: a
    /// consumer releasing the only slot while a producer is trying to claim it
    /// for the next lap. The producer must either see the release and take the
    /// slot, or not see it and report full. What it must never do is write over
    /// a value the consumer has not taken.
    #[test]
    fn loom_a_slot_released_by_the_consumer_is_safely_reclaimed() {
        loom::model(|| {
            let (producer, mut consumer) = bounded(1);
            producer.try_push(1).expect("an empty ring accepts a value");

            let refill = {
                let producer = producer.clone();
                loom::thread::spawn(move || producer.try_push(2).is_ok())
            };

            let first = consumer.pop();
            let refilled = refill.join().unwrap();

            assert_eq!(first, Some(1), "the queued value was overwritten or skipped");
            let second = consumer.pop();
            if refilled {
                assert_eq!(second, Some(2), "an accepted value never arrived");
            } else {
                assert_eq!(second, None, "a rejected value arrived anyway");
            }
        });
    }
}
