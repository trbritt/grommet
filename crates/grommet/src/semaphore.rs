//! A counted permit, with the same waiting protocol the mailbox uses.
//!
//! Compute offload bounds how much work may be outstanding at once, which is
//! the same shape as a bounded mailbox with the queue taken out: a count, a
//! queue of callers waiting for it to be non-zero, and the discipline that
//! stops a caller sleeping on a permit that is already free.
//!
//! It is here rather than in the crate that uses it because the discipline is
//! the interesting part, and that lives in this crate's wait list alongside the
//! mailbox's. Owning it also keeps the offload pool free of any runtime: a
//! `tokio::sync::Semaphore` would put tokio in the graph of a crate that only
//! runs Rayon tasks, which the driver seam exists to avoid.
//!
//! # Fairness
//!
//! Permits go to whoever has waited longest, because the queue underneath is
//! ordered. A caller that finds a permit free takes it without joining the
//! queue, so an uncontended acquire touches nothing but the counter.

use crate::waiters::{Attempt, Waiters};
use std::sync::Arc;

#[cfg(loom)]
use loom::sync::atomic::{AtomicBool, AtomicUsize, Ordering, fence};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering, fence};

/// The semaphore is gone, so no further permit will ever be issued.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Closed;

impl std::fmt::Display for Closed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the semaphore was closed")
    }
}

impl std::error::Error for Closed {}

/// A count of permits, handed out in arrival order.
#[derive(Debug)]
pub struct Semaphore {
    permits: AtomicUsize,
    waiters: Waiters,
    closed: AtomicBool,
}

impl Semaphore {
    pub fn new(permits: usize) -> Self {
        Self {
            permits: AtomicUsize::new(permits),
            waiters: Waiters::new(),
            closed: AtomicBool::new(false),
        }
    }

    /// Take a permit if one is free, without joining the queue.
    ///
    /// Note that this jumps the queue: it is for a caller that would rather
    /// shed than wait, and one that intends to wait should use [`acquire`] so
    /// that it takes its place.
    ///
    /// [`acquire`]: Semaphore::acquire
    pub fn try_acquire(self: &Arc<Self>) -> Option<Permit> {
        self.take().then(|| Permit { semaphore: Arc::clone(self) })
    }

    /// Take a permit, waiting behind whoever is already waiting.
    pub async fn acquire(self: &Arc<Self>) -> Result<Permit, Closed> {
        self.waiters
            .park_until(|| match self.try_acquire() {
                Some(permit) => Attempt::Ready(permit),
                None if self.closed.load(Ordering::Acquire) => Attempt::Closed,
                None => Attempt::Retry,
            })
            .await
            .map_err(|_| Closed)
    }

    /// Permits free right now. A hint: it can change before the caller acts.
    pub fn available(&self) -> usize {
        self.permits.load(Ordering::Relaxed)
    }

    /// Stop issuing permits and wake everyone waiting for one.
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.waiters.close();
    }

    /// Decrement if there is anything to decrement.
    fn take(&self) -> bool {
        let mut available = self.permits.load(Ordering::Relaxed);
        loop {
            if available == 0 {
                return false;
            }
            match self.permits.compare_exchange_weak(
                available,
                available - 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(current) => available = current,
            }
        }
    }

    /// Give a permit back and hand it to whoever has waited longest.
    ///
    /// The fence is the other half of the pair the wait list's `park_until`
    /// performs. Without it this thread's release and a caller's park could
    /// each fail to see the other, and the caller would sleep on a permit that
    /// is already free.
    fn release(&self) {
        self.permits.fetch_add(1, Ordering::Release);
        fence(Ordering::SeqCst);
        if self.waiters.any() {
            self.waiters.wake_one();
        }
    }
}

/// A held permit. Returns it on drop, whether the work finished or panicked.
#[derive(Debug)]
pub struct Permit {
    semaphore: Arc<Semaphore>,
}

impl Drop for Permit {
    fn drop(&mut self) {
        self.semaphore.release();
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;
    use std::task::Poll;
    use std::time::Duration;

    async fn before_long<F: Future>(future: F) -> F::Output {
        tokio::time::timeout(Duration::from_secs(5), future)
            .await
            .expect("a waiting caller was never given a permit")
    }

    async fn poll_once<F: Future>(future: &mut Pin<Box<F>>) {
        std::future::poll_fn(|cx| {
            let _ = future.as_mut().poll(cx);
            Poll::Ready(())
        })
        .await;
    }

    #[tokio::test]
    async fn permits_are_finite_and_come_back_when_dropped() {
        let semaphore = Arc::new(Semaphore::new(2));
        let first = semaphore.try_acquire().expect("two are free");
        let second = semaphore.try_acquire().expect("one is free");
        assert_eq!(semaphore.available(), 0);
        assert!(semaphore.try_acquire().is_none(), "none are free");

        drop(first);
        assert_eq!(semaphore.available(), 1);
        drop(second);
        assert_eq!(semaphore.available(), 2);
    }

    #[tokio::test]
    async fn a_waiting_caller_is_admitted_when_a_permit_comes_back() {
        let semaphore = Arc::new(Semaphore::new(1));
        let held = semaphore.acquire().await.unwrap();

        let waiting = tokio::spawn({
            let semaphore = Arc::clone(&semaphore);
            async move { semaphore.acquire().await.map(drop) }
        });
        tokio::task::yield_now().await;

        drop(held);
        before_long(waiting).await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn permits_go_to_whoever_waited_longest() {
        // The fairness the queue underneath provides. A pool that handed
        // permits to whoever happened to poll first would starve the caller
        // that has already waited longest, which is the shard feeling
        // backpressure hardest.
        let semaphore = Arc::new(Semaphore::new(1));
        let held = semaphore.acquire().await.unwrap();

        let order = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let mut waiting = Vec::new();
        for id in 0..3 {
            let semaphore = Arc::clone(&semaphore);
            let order = Arc::clone(&order);
            waiting.push(tokio::spawn(async move {
                let permit = semaphore.acquire().await.unwrap();
                order.lock().push(id);
                drop(permit);
            }));
            tokio::task::yield_now().await;
        }

        drop(held);
        for task in waiting {
            before_long(task).await.unwrap();
        }
        assert_eq!(*order.lock(), [0, 1, 2]);
    }

    #[tokio::test]
    async fn abandoning_a_waiting_caller_does_not_strand_the_next_one() {
        let semaphore = Arc::new(Semaphore::new(1));
        let held = semaphore.acquire().await.unwrap();

        let mut abandoned = Box::pin({
            let semaphore = Arc::clone(&semaphore);
            async move { semaphore.acquire().await.map(drop) }
        });
        poll_once(&mut abandoned).await;

        let waiting = tokio::spawn({
            let semaphore = Arc::clone(&semaphore);
            async move { semaphore.acquire().await.map(drop) }
        });
        tokio::task::yield_now().await;

        // The permit is offered to the first caller, which then goes away
        // without using it. Discarding that wake would leave the second one
        // waiting on a permit sitting free.
        drop(held);
        drop(abandoned);
        before_long(waiting).await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn closing_wakes_everyone_waiting_and_refuses_new_callers() {
        let semaphore = Arc::new(Semaphore::new(1));
        let _held = semaphore.acquire().await.unwrap();

        let waiting: Vec<_> = (0..3)
            .map(|_| {
                let semaphore = Arc::clone(&semaphore);
                tokio::spawn(async move { semaphore.acquire().await.map(drop) })
            })
            .collect();
        tokio::task::yield_now().await;

        semaphore.close();
        for task in waiting {
            assert_eq!(before_long(task).await.unwrap(), Err(Closed));
        }
        assert_eq!(semaphore.acquire().await.map(drop), Err(Closed));
    }

    #[tokio::test]
    async fn a_permit_returns_even_when_its_holder_panics() {
        // The offload pool's whole reason for holding one: a Rayon task that
        // unwinds must not permanently consume capacity.
        let semaphore = Arc::new(Semaphore::new(1));
        let taken = Arc::clone(&semaphore);
        let panicked = std::thread::spawn(move || {
            let _permit = taken.try_acquire().expect("one is free");
            panic!("the task blew up");
        })
        .join();

        assert!(panicked.is_err(), "the thread really did unwind");
        assert_eq!(semaphore.available(), 1, "the permit came back during unwinding");
    }
}

/// Exhaustive interleavings of the release side.
///
/// The parking half belongs to the wait list and is modelled with the mailbox.
/// What is specific here is the other half of the pair: the permit counter, and
/// the fence between giving one back and looking for somebody to give it to.
#[cfg(all(test, loom))]
mod loom_tests {
    use super::*;
    use loom::sync::atomic::AtomicBool;
    use std::sync::Arc as StdArc;
    use std::task::{Context, Poll, Wake, Waker};

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

    /// A caller that finds no permit and parks must be woken by whoever
    /// returned one. Neither "the caller is asleep" nor "a permit is free" may
    /// be observable together at the end.
    ///
    /// The caller parks on this thread so its registration is still live when
    /// the assertion runs, and the returner hands the semaphore back rather
    /// than closing it, which would wake everyone for the wrong reason.
    #[test]
    fn loom_a_caller_never_parks_on_a_permit_already_returned() {
        loom::model(|| {
            let semaphore = Arc::new(Semaphore::new(1));
            let held = semaphore.try_acquire().expect("the only permit is free");
            let (flag, waker) = Flag::waker();

            let returner = loom::thread::spawn(move || drop(held));

            let mut acquire = Box::pin(semaphore.acquire());
            let polled = acquire.as_mut().poll(&mut Context::from_waker(&waker));

            returner.join().unwrap();
            if polled.is_pending() {
                assert!(flag.woken(), "a caller parked on a permit that was already free");
            }
            drop(acquire);
        });
    }

    /// Two callers and one permit: exactly one gets in.
    ///
    /// The permits are carried back out of the threads rather than dropped
    /// inside them. Dropping one at the end of the expression that took it
    /// returns it, and two callers that never overlapped could then both
    /// succeed. That is correct behaviour, and it would make this prove
    /// nothing.
    #[test]
    fn loom_one_permit_admits_exactly_one_of_two_racing_callers() {
        loom::model(|| {
            let semaphore = Arc::new(Semaphore::new(1));

            let left = {
                let semaphore = Arc::clone(&semaphore);
                loom::thread::spawn(move || semaphore.try_acquire())
            };
            let right = {
                let semaphore = Arc::clone(&semaphore);
                loom::thread::spawn(move || semaphore.try_acquire())
            };

            let held = (left.join().unwrap(), right.join().unwrap());
            let admitted = usize::from(held.0.is_some()) + usize::from(held.1.is_some());
            assert_eq!(admitted, 1, "one permit admitted {admitted} callers at once");

            drop(held);
            assert_eq!(semaphore.available(), 1, "the permit came back on drop");
        });
    }
}
