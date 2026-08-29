//! A single-registrar waker cell that loom can see inside.
//!
//! This is a `futures::task::AtomicWaker` in shape and in algorithm, and it
//! exists for one reason: the futures implementation is built on
//! `core::sync::atomic`, which loom cannot instrument. A model that used it
//! would be exploring interleavings of everything *except* the handoff the
//! model is about, and would report success from a state space that never
//! contained the interesting orderings. Loom's own `AtomicWaker` is explicitly
//! a mock — a mutex around an `Option<Waker>` — so verifying against that would
//! prove a property of the mock rather than of the thing that ships.
//!
//! Owning it makes the cell's atomics loom's atomics under `--cfg loom` and
//! `std`'s otherwise, so `grommet`'s doorbell models check the same code
//! the shard runs.
//!
//! # The protocol
//!
//! One `AtomicUsize` guards one `UnsafeCell<Option<Waker>>` as a two-bit lock.
//! `REGISTERING` is held by a thread storing a waker; `WAKING` by a thread
//! taking one. Whoever moves the state out of `WAITING` has exclusive access to
//! the cell until it moves it back, and anyone who finds a bit already set
//! declines rather than waits — there is no blocking anywhere in here.
//!
//! The subtle case is a wake arriving while a registration holds the lock. The
//! waker is mid-store, so the waking thread cannot take it; instead it leaves
//! `WAKING` set and returns. The registering thread then fails its release
//! compare-exchange, sees `REGISTERING | WAKING`, and performs the wake on the
//! waker's behalf before releasing. The wake is deferred, never dropped.
//!
//! # Safety invariant
//!
//! > The `Option<Waker>` cell is accessed only by a thread that observed the
//! > state transition into `REGISTERING` or `WAKING` and has not yet
//! > transitioned it back.
//!
//! Those transitions are compare-exchange and fetch-or on a single atomic, so
//! at most one thread holds either bit at a time, and the two bits are taken
//! from the same `WAITING` origin. Every access below is inside such a window
//! and is commented with which bit it holds.
//!
//! # Single registrar
//!
//! Grommet has exactly one thread that registers on a given cell: the shard
//! that owns it, from the top of its own reactor turn. Concurrent registration
//! is therefore a bug, and `register` `debug_assert!`s against it. It stays
//! *memory-safe* if it happens anyway — the loser simply declines the lock and
//! its waker is dropped, exactly as in the futures implementation — because
//! soundness should not rest on a discipline that is merely intended.
#![allow(unsafe_code)]

use crate::cell::UnsafeCell;
use std::task::Waker;

#[cfg(loom)]
use loom::sync::atomic::AtomicUsize;
#[cfg(not(loom))]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{AcqRel, Acquire, Release};

/// Nobody holds the cell.
const WAITING: usize = 0;
/// A thread is storing a waker.
const REGISTERING: usize = 0b01;
/// A thread is taking the waker to wake it.
const WAKING: usize = 0b10;

/// Where the owner's waker lives between turns.
pub struct WakerSlot {
    state: AtomicUsize,
    waker: UnsafeCell<Option<Waker>>,
}

// SAFETY: the state word serializes every access to `waker`, so the cell is
// never touched by two threads at once. `Waker` is itself `Send + Sync`.
unsafe impl Send for WakerSlot {}
unsafe impl Sync for WakerSlot {}

impl WakerSlot {
    pub fn new() -> Self {
        Self { state: AtomicUsize::new(WAITING), waker: UnsafeCell::new(None) }
    }

    /// Store `waker` as the one to notify, replacing whoever was there.
    ///
    /// Must be called before the check that decides to park; see `grommet`'s
    /// `doorbell` module for why that order is the one that cannot lose a
    /// wake.
    pub fn register(&self, waker: &Waker) {
        match self.state.compare_exchange(WAITING, REGISTERING, Acquire, Acquire) {
            Ok(_) => {}
            Err(WAKING) => {
                // A wake is in flight and cannot reach us, because the waker it
                // wanted is the one being replaced. Waking the incoming waker
                // directly delivers it instead: the owner is scheduled, and its
                // next turn re-registers.
                waker.wake_by_ref();
                return;
            }
            Err(_actual) => {
                // Only reachable if a second thread is registering. Grommet has
                // one registrar per slot, so this is a bug rather than a race
                // to resolve — but it must stay sound, so the loser declines
                // the lock and delivers its wake directly rather than touching
                // the cell.
                debug_assert!(
                    false,
                    "concurrent registration: a waker slot has exactly one registrar"
                );
                waker.wake_by_ref();
                return;
            }
        }

        // SAFETY: the compare-exchange above put this thread, and only this
        // thread, into `REGISTERING`; the bit is still held here.
        let previous = unsafe {
            self.waker.with_mut(|slot| {
                let previous = (*slot).take();
                // `will_wake` avoids a clone when the shard re-registers the
                // same task, which is every turn in the steady state.
                match previous {
                    Some(old) if old.will_wake(waker) => {
                        *slot = Some(old);
                        None
                    }
                    other => {
                        *slot = Some(waker.clone());
                        other
                    }
                }
            })
        };

        match self.state.compare_exchange(REGISTERING, WAITING, AcqRel, Acquire) {
            Ok(_) => drop(previous),
            Err(actual) => {
                // A wake landed while the cell was held. It could not take the
                // waker, so it left `WAKING` set for us to honour.
                debug_assert_eq!(actual, REGISTERING | WAKING);
                // SAFETY: both bits are held by this thread — `REGISTERING`
                // from the exchange above, and `WAKING` set by a thread that
                // has already returned without touching the cell.
                let pending = unsafe { self.waker.with_mut(|slot| (*slot).take()) };
                self.state.swap(WAITING, AcqRel);
                drop(previous);
                if let Some(pending) = pending {
                    pending.wake();
                }
            }
        }
    }

    /// Take the registered waker, if this thread can get to it.
    ///
    /// `None` means either that nothing is registered or that another thread is
    /// already inside the cell, in which case that thread delivers the wake.
    pub fn take(&self) -> Option<Waker> {
        match self.state.fetch_or(WAKING, AcqRel) {
            WAITING => {
                // SAFETY: this thread moved the state from `WAITING` to
                // `WAKING` and so holds the cell until the store below.
                let waker = unsafe { self.waker.with_mut(|slot| (*slot).take()) };
                self.state.fetch_and(!WAKING, Release);
                waker
            }
            actual => {
                // `REGISTERING`: the registrar will see our `WAKING` bit when
                // it releases and will wake on our behalf. `WAKING`: another
                // waker is already delivering. Either way this call is done.
                debug_assert!(
                    actual == REGISTERING || actual == WAKING || actual == REGISTERING | WAKING
                );
                None
            }
        }
    }

    /// Wake whoever is registered, if anyone is and if this thread can reach
    /// them.
    #[inline]
    pub fn wake(&self) {
        if let Some(waker) = self.take() {
            waker.wake();
        }
    }
}

impl Default for WakerSlot {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for WakerSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WakerSlot").finish_non_exhaustive()
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::Wake;

    struct Counter(AtomicUsize);

    impl Counter {
        fn waker() -> (Arc<Self>, Waker) {
            let counter = Arc::new(Self(AtomicUsize::new(0)));
            (counter.clone(), Waker::from(counter))
        }

        fn count(&self) -> usize {
            self.0.load(Ordering::Relaxed)
        }
    }

    impl Wake for Counter {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn a_registered_waker_is_woken_once_and_then_the_slot_is_empty() {
        let slot = WakerSlot::new();
        let (counter, waker) = Counter::waker();
        slot.register(&waker);

        slot.wake();
        assert_eq!(counter.count(), 1);

        // Waking consumes the registration: the owner is already scheduled and
        // will re-register on its next turn, so a second wake has nothing to
        // deliver to. This is what makes repeated notifications free.
        slot.wake();
        assert_eq!(counter.count(), 1);
    }

    #[test]
    fn waking_an_empty_slot_does_nothing() {
        let slot = WakerSlot::new();
        assert!(slot.take().is_none());
        slot.wake();
    }

    #[test]
    fn registering_replaces_the_previous_waker_and_drops_it() {
        let slot = WakerSlot::new();
        let (first, first_waker) = Counter::waker();
        let (second, second_waker) = Counter::waker();

        slot.register(&first_waker);
        assert_eq!(Arc::strong_count(&first), 3, "the Arc, the Waker, and the slot's clone");
        slot.register(&second_waker);
        assert_eq!(Arc::strong_count(&first), 2, "the replaced clone must be dropped");

        slot.wake();
        assert_eq!(first.count(), 0);
        assert_eq!(second.count(), 1);
    }

    #[test]
    fn re_registering_the_same_waker_does_not_clone_it() {
        let slot = WakerSlot::new();
        let (counter, waker) = Counter::waker();

        slot.register(&waker);
        let after_first = Arc::strong_count(&counter);
        // The shard re-registers the same task at the top of every turn, so
        // this is the steady-state path and must not pay for a clone.
        for _ in 0..8 {
            slot.register(&waker);
        }
        assert_eq!(Arc::strong_count(&counter), after_first);

        slot.wake();
        assert_eq!(counter.count(), 1);
    }

    #[test]
    fn taking_hands_the_waker_to_the_caller_rather_than_waking_it() {
        let slot = WakerSlot::new();
        let (counter, waker) = Counter::waker();
        slot.register(&waker);

        let taken = slot.take().expect("a waker was registered");
        assert_eq!(counter.count(), 0, "take must not wake on the caller's behalf");
        assert!(slot.take().is_none(), "the slot is empty after a take");

        taken.wake();
        assert_eq!(counter.count(), 1);
    }

    /// Drives the register/wake race across real threads. This is the test Miri
    /// runs: every access to the cell below is behind the state word, so a
    /// mistake in which bit guards which access shows up here as a data race
    /// rather than as a silent one in production.
    #[test]
    fn concurrent_wakes_racing_registration_are_never_lost() {
        let rounds = if cfg!(miri) { 24 } else { 2_000 };

        for _ in 0..rounds {
            let slot = Arc::new(WakerSlot::new());
            let work = Arc::new(AtomicUsize::new(0));
            let (counter, waker) = Counter::waker();

            let producer = {
                let slot = Arc::clone(&slot);
                let work = Arc::clone(&work);
                std::thread::spawn(move || {
                    // Publish before ringing, which is the contract the wake
                    // protocol is built on.
                    work.store(1, Ordering::Release);
                    slot.wake();
                })
            };

            // Register before the check, which is the other half of it.
            slot.register(&waker);
            let observed = work.load(Ordering::Acquire) == 1;
            producer.join().unwrap();

            assert!(
                observed || counter.count() > 0,
                "the owner neither observed the work nor was woken for it"
            );
        }
    }

    #[test]
    fn the_debug_rendering_does_not_reach_into_the_cell() {
        let slot = WakerSlot::new();
        assert!(format!("{slot:?}").contains("WakerSlot"));
    }
}

/// Exhaustive interleavings of the cell's two-bit lock.
///
/// These are the reason this type exists rather than `futures::task::
/// AtomicWaker`: the atomics and the `UnsafeCell` below are loom's under
/// `--cfg loom`, so loom explores the handoff itself and checks that no two
/// threads are ever inside the cell at once.
#[cfg(all(test, loom))]
mod loom_tests {
    use super::*;
    use loom::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::task::Wake;

    struct Flag(AtomicBool);

    impl Flag {
        fn waker() -> (Arc<Self>, Waker) {
            let flag = Arc::new(Self(AtomicBool::new(false)));
            (flag.clone(), Waker::from(flag))
        }

        fn woken(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
    }

    impl Wake for Flag {
        fn wake(self: Arc<Self>) {
            self.0.store(true, Ordering::Release);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.store(true, Ordering::Release);
        }
    }

    /// The end-to-end property callers actually depend on: under every
    /// interleaving the owner either sees the work or is woken for it.
    ///
    /// Note what this does *not* isolate. When a wake lands mid-registration,
    /// the registrar's failed release exchange acquires the notifier's `WAKING`
    /// write, which orders the notifier's earlier publish before the check
    /// below — so the check finds the work and the model passes even if the
    /// deferred wake were dropped entirely. That path is only observable when a
    /// second registration replaces the waker the notifier was reaching for,
    /// which is what the third model here drives.
    #[test]
    fn loom_a_wake_racing_a_registration_is_deferred_not_lost() {
        loom::model(|| {
            let slot = Arc::new(WakerSlot::new());
            let work = Arc::new(AtomicBool::new(false));
            let (flag, waker) = Flag::waker();

            let producer = {
                let slot = Arc::clone(&slot);
                let work = Arc::clone(&work);
                loom::thread::spawn(move || {
                    work.store(true, Ordering::Release);
                    slot.wake();
                })
            };

            slot.register(&waker);
            let observed = work.load(Ordering::Acquire);
            producer.join().unwrap();

            assert!(observed || flag.woken(), "a wake was lost across a registration");
        });
    }

    /// Two notifiers on one registration. One of them gets the cell and the
    /// other finds `WAKING` already set and declines; the owner is still
    /// scheduled exactly because the winner delivers.
    #[test]
    fn loom_concurrent_wakes_deliver_exactly_one_of_them() {
        loom::model(|| {
            let slot = Arc::new(WakerSlot::new());
            let (flag, waker) = Flag::waker();
            slot.register(&waker);

            let left = {
                let slot = Arc::clone(&slot);
                loom::thread::spawn(move || slot.take().is_some())
            };
            let right = {
                let slot = Arc::clone(&slot);
                loom::thread::spawn(move || slot.take().is_some())
            };

            let took = usize::from(left.join().unwrap()) + usize::from(right.join().unwrap());
            assert_eq!(took, 1, "a registered waker must be handed out exactly once");
            assert!(!flag.woken(), "take must not wake on the caller's behalf");
        });
    }

    /// The model that pins the deferred wake down. A notifier races the
    /// replacement of the waker it wanted: it finds `REGISTERING` set, declines
    /// the cell and returns without waking, so the only thing that can still
    /// deliver is the registrar honouring the `WAKING` bit on its way out.
    /// Which of the two tasks gets scheduled is not specified; that one of them
    /// does is.
    #[test]
    fn loom_a_wake_racing_a_replacement_still_schedules_somebody() {
        loom::model(|| {
            let slot = Arc::new(WakerSlot::new());
            let (first, first_waker) = Flag::waker();
            let (second, second_waker) = Flag::waker();
            slot.register(&first_waker);

            let notifier = {
                let slot = Arc::clone(&slot);
                loom::thread::spawn(move || slot.wake())
            };
            slot.register(&second_waker);
            notifier.join().unwrap();

            assert!(first.woken() || second.woken(), "a wake vanished between two registrations");
        });
    }
}
