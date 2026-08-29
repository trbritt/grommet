//! One wake, delivered once, never after the receiver is gone.
//!
//! Every component a shard parks on faces the same three problems, and none of
//! them is the interesting part of that component. Another thread makes work
//! visible and has to schedule the owner. The owner has to decide it has
//! nothing to do and sleep, without racing the notification that would have
//! told it otherwise. And the arrangement has to survive the owner going away,
//! because a stale notifier outliving its receiver is how a wake lands on a
//! task that no longer exists.
//!
//! A `Doorbell` is that protocol and nothing else. It holds no readiness state:
//! what counts as "there is work" belongs to whoever rings, because only they
//! can coalesce it cheaply. [`Outstanding`] rings on a ready bit's `0 -> 1`
//! transition, the mailbox on a push into an empty ring. Both want the same
//! answer to *how the owner is woken*.
//!
//! # Registering before the check
//!
//! The owner must [`register`] before testing the predicate it would park on.
//! The other order leaves a window where a notification arrives between the two
//! and finds no waker, and the owner sleeps on work already visible.
//! Registering first closes it: a notification in that window either sees the
//! waker and schedules the owner, or is ordered before the test and the test
//! finds the work. The shard's reactor does this at the top of every turn.
//!
//! # Closing
//!
//! [`close`] is one-way. Afterwards no [`ring`] delivers a wake and the
//! registered waker has been dropped, which is what lets a receiver be
//! destroyed while notifiers it handed out are still alive and still firing.
//! Those late rings are not errors but the ordinary consequence of a waker
//! outliving what it points at, and they are discarded.
//!
//! A ring racing a close may still deliver. The two are ordered against each
//! other by the atomic, so the wake happens fully before the close or is
//! suppressed by it, and one delivered just before schedules a task that then
//! finds nothing to do. What cannot happen is a wake delivered *through* a
//! dropped waker, which is the only outcome that matters.
//!
//! [`Outstanding`]: crate::outstanding
//! [`register`]: Doorbell::register
//! [`close`]: Doorbell::close
//! [`ring`]: Doorbell::ring

use grommet_core::waker_slot::WakerSlot;
use std::task::Waker;

#[cfg(loom)]
use loom::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicBool, Ordering};

/// The owner's wake-up, shared with everything that may need to deliver it.
///
/// Cheap to share: one `AtomicWaker` and one flag, both of which stay quiet
/// while the owner is running and nothing is parked.
pub(crate) struct Doorbell {
    /// Whoever is to be woken. [`WakerSlot`] is the piece that makes a register
    /// racing a wake safe in either order, which is the property the
    /// register-before-check discipline above is built on.
    owner: WakerSlot,
    /// Set once, by the receiver, on its way out. Monotonic: nothing ever
    /// re-opens a doorbell, so a load that sees `false` was accurate at some
    /// point in the ring's own past, which is all a notifier needs.
    closed: AtomicBool,
}

impl Doorbell {
    pub(crate) fn new() -> Self {
        Self { owner: WakerSlot::new(), closed: AtomicBool::new(false) }
    }

    /// Name the task to wake, replacing whoever was named before.
    ///
    /// Call this before the readiness check that decides to park. See the
    /// module documentation for why the other order loses wakes.
    #[inline]
    pub(crate) fn register(&self, waker: &Waker) {
        self.owner.register(waker);
    }

    /// Wake the registered owner, unless the receiver has already gone.
    #[inline]
    pub(crate) fn ring(&self) {
        if self.is_closed() {
            return;
        }
        self.owner.wake();
    }

    /// Whether the receiver is gone.
    ///
    /// Notifiers use this to skip work they would otherwise do before ringing:
    /// [`ring`] checks again, so this is an optimization rather than part of
    /// the protocol.
    ///
    /// [`ring`]: Doorbell::ring
    #[inline]
    pub(crate) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// Retire the doorbell: no further ring delivers, and the registered waker
    /// is released now rather than whenever the last notifier happens to drop.
    ///
    /// Releasing eagerly is the point. A notifier is typically an `Arc` handed
    /// to something outside this crate's control, and a waker kept alive inside
    /// one keeps the owner task's allocation alive with it.
    pub(crate) fn close(&self) {
        // Ordered before the take, so a notifier that observes the flag is
        // already past the point where it could have used the waker, and one
        // that does not observe it wakes a waker that is still there.
        self.closed.store(true, Ordering::Release);
        drop(self.owner.take());
    }
}

impl Default for Doorbell {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Doorbell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Doorbell").field("closed", &self.is_closed()).finish_non_exhaustive()
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::task::Wake;

    /// Counts deliveries, so a test can assert on coalescing rather than on
    /// merely having been woken at some point.
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
    fn a_ring_wakes_whoever_registered_last() {
        let bell = Doorbell::new();
        let (first, first_waker) = Counter::waker();
        let (second, second_waker) = Counter::waker();

        bell.register(&first_waker);
        bell.register(&second_waker);
        bell.ring();

        // Re-registering replaces rather than accumulates: a shard re-registers
        // on every turn, and a doorbell that kept each one would wake every task
        // that ever owned it.
        assert_eq!(first.count(), 0, "a replaced owner must not be woken");
        assert_eq!(second.count(), 1);
    }

    #[test]
    fn a_ring_consumes_the_registration_so_repeats_coalesce() {
        let bell = Doorbell::new();
        let (counter, waker) = Counter::waker();
        bell.register(&waker);

        // The owner is already scheduled after the first; further rings before
        // it runs and re-registers have nothing left to deliver to. This is why
        // a notifier may ring freely without checking whether it is the first.
        bell.ring();
        bell.ring();
        bell.ring();
        assert_eq!(counter.count(), 1);

        bell.register(&waker);
        bell.ring();
        assert_eq!(counter.count(), 2, "re-registering re-arms it");
    }

    #[test]
    fn ringing_an_unregistered_doorbell_is_a_no_op() {
        let bell = Doorbell::new();
        bell.ring();
        assert!(!bell.is_closed());
    }

    #[test]
    fn closing_silences_later_rings_and_releases_the_waker() {
        let bell = Doorbell::new();
        let (counter, waker) = Counter::waker();
        bell.register(&waker);
        assert_eq!(Arc::strong_count(&counter), 3, "the Arc, the Waker, and the doorbell");

        bell.close();
        assert!(bell.is_closed());
        assert_eq!(
            Arc::strong_count(&counter),
            2,
            "closing must drop the registration, not merely ignore it"
        );

        // Notifiers outlive the receiver by design; their rings are discarded
        // rather than being an error.
        bell.ring();
        bell.ring();
        assert_eq!(counter.count(), 0);
    }

    #[test]
    fn a_registration_after_close_still_never_rings() {
        let bell = Doorbell::new();
        let (counter, waker) = Counter::waker();
        bell.close();

        // Nothing forbids a late registration: a notifier and a receiver shut
        // down independently. So the closed flag, rather than the absence of a
        // waker, has to be what stops the wake.
        bell.register(&waker);
        bell.ring();
        assert_eq!(counter.count(), 0);
    }

    #[test]
    fn closing_twice_is_harmless() {
        let bell = Doorbell::new();
        bell.close();
        bell.close();
        assert!(bell.is_closed());
    }

    #[test]
    fn the_debug_rendering_names_the_state_that_matters() {
        let bell = Doorbell::default();
        assert!(format!("{bell:?}").contains("closed: false"));
        bell.close();
        assert!(format!("{bell:?}").contains("closed: true"));
    }
}

/// Exhaustive interleavings of the two races the protocol exists to settle.
///
/// One caveat, and it is why these models are shaped the way they are:
/// `AtomicWaker` is built on `core::sync::atomic` and has no loom support, so
/// loom cannot see inside it or explore its interior orderings. What these
/// models do explore, with the real [`Doorbell`] code running, is everything
/// around it: the closed flag, the register-then-check order, and the predicate
/// a notifier publishes before ringing. Wake delivery is observed through a
/// loom atomic in the waker itself, so the assertions are on state loom
/// tracks.
#[cfg(all(test, loom))]
mod loom_tests {
    use super::*;
    use loom::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use std::task::Wake;

    /// Records delivery in a loom-visible cell.
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

    /// The theorem the parked reactor rests on: an owner that registers before
    /// testing its predicate cannot sleep through work published after the
    /// test. Either the test observes the work, or the ring that followed it
    /// found the registration and scheduled the owner.
    #[test]
    fn loom_register_before_check_cannot_strand_a_ring() {
        loom::model(|| {
            let bell = Arc::new(Doorbell::new());
            let work = Arc::new(AtomicBool::new(false));
            let (flag, waker) = Flag::waker();

            let producer = {
                let bell = bell.clone();
                let work = work.clone();
                loom::thread::spawn(move || {
                    work.store(true, Ordering::Release);
                    bell.ring();
                })
            };

            // The owner's turn: register first, then decide whether to park.
            bell.register(&waker);
            let observed = work.load(Ordering::Acquire);

            producer.join().unwrap();
            assert!(
                observed || flag.woken(),
                "the owner neither saw the work nor was woken for it"
            );
        });
    }

    /// A notifier and a receiver shut down independently, so every ring races a
    /// close. Whichever order they land in, a ring that begins after the close
    /// has completed must deliver nothing.
    #[test]
    fn loom_a_ring_racing_close_never_delivers_afterwards() {
        loom::model(|| {
            let bell = Arc::new(Doorbell::new());
            let (flag, waker) = Flag::waker();
            bell.register(&waker);

            let notifier = {
                let bell = bell.clone();
                loom::thread::spawn(move || bell.ring())
            };
            bell.close();
            notifier.join().unwrap();

            // The racing ring may legitimately have landed first. What must not
            // happen is a delivery once the close is known to be complete.
            let after_close = flag.woken();
            bell.register(&waker);
            bell.ring();
            assert_eq!(flag.woken(), after_close, "a ring after close delivered a wake");
        });
    }

    /// Concurrent notifiers coalesce onto one registration rather than losing
    /// it: whoever gets there first delivers, and the owner is scheduled once.
    #[test]
    fn loom_concurrent_rings_still_schedule_the_owner() {
        loom::model(|| {
            let bell = Arc::new(Doorbell::new());
            let (flag, waker) = Flag::waker();
            bell.register(&waker);

            let left = {
                let bell = bell.clone();
                loom::thread::spawn(move || bell.ring())
            };
            let right = {
                let bell = bell.clone();
                loom::thread::spawn(move || bell.ring())
            };
            left.join().unwrap();
            right.join().unwrap();

            assert!(flag.woken(), "two rings on an open doorbell woke nobody");
        });
    }
}
