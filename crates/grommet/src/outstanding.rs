//! The shard's outstanding set: dispatched work, held until it completes.
//!
//! This replaces `FuturesUnordered`, and the honest way to say why is that
//! under the shard's constraints there is no `FuturesUnordered` left to need.
//! That structure earns its complexity by holding any mix of future types,
//! growing without bound, accepting pushes from other threads, and surviving
//! panics mid-poll. The shard has one concrete future type per processor, a
//! population bounded by its in-flight budgets, a single thread that both
//! pushes and polls, and panics already caught inside the future by
//! [`run_one`]'s `catch_unwind`. Each constraint deletes a hard part; what is
//! left is a flat slab, a ready bitmap, and one wake cell.
//!
//! # What it is worth, measured
//!
//! Both were built and benchmarked against each other through the same loop,
//! because the argument above is only a reason to expect a win, not evidence of
//! one. On the reactor benchmark, against `FuturesUnordered`:
//!
//! | in flight | time | allocations per dispatch |
//! |---|---|---|
//! | 8 | −9% | 1.008 -> 0.008 |
//! | 64 | −9% | 1.008 -> 0.008 |
//! | 1 | +5% | 1.008 -> 0.008 |
//!
//! A set sized for its budgets is scanned as a bitmap, and scanning one is
//! not free when only one slot is ever occupied. There, an intrusive list
//! that walks straight to the single ready node wins. It is the concurrency
//! this runtime is built for that pays for the bitmap, and a shard configured
//! to keep one item in flight has given up the thing that makes thread-per-core
//! worth doing.
//!
//! `tests/alloc.rs` holds the allocation figures to a marginal cost per
//! dispatch, so a change that reintroduced one would fail rather than drift.
//!
//! # Storage: boxed once, reused forever
//!
//! Each slot holds a `Pin<Box<F>>`, allocated the first time the slot is used
//! and reused for every later occupant through [`Pin::set`], which drops the
//! finished future in place and writes the new one into the same allocation,
//! in safe code. Steady state performs no allocation at all; warm-up performs
//! exactly one per slot actually reached, the same arrangement the scheduler's
//! `queue_reserve` makes for its own slab.
//!
//! # Wakes: one bit, one cell
//!
//! The only state shared with other threads is the ready bitmap and one
//! [`AtomicWaker`]. A wake (from an offload worker, a oneshot sender, a
//! timer) sets its slot's bit with `Release` and wakes the owner; the
//! harvest takes whole words with a `swap(0, Acquire)` and polls only slots
//! that are both marked and occupied. The loop's side of the bargain is to
//! register its waker *before* any check it will rely on for the decision to
//! park; any wake after registration re-polls the loop, so nothing can land in
//! the gap between "found nothing" and "went to sleep".
//!
//! The ordering discipline here is loom-checked (`just loom`). The
//! [`AtomicWaker`] cell itself is delegated, not rewritten: modelling its
//! internals is loom's limit, and hand-rolling the one genuinely subtle
//! concurrent object in the neighbourhood is exactly what this module exists
//! to avoid.
//!
//! # Stale wakes
//!
//! A wake can arrive after its future completed, or after the slot was
//! refilled. The `occupied` check answers both: a stale bit for a free slot
//! is skipped — which is also what makes re-polling a completed `async fn`
//! (which panics) unreachable — and a stale bit for a refilled slot costs one
//! spurious poll, which every future must tolerate by contract. No generation
//! counter is needed at this layer.

use futures::task::AtomicWaker;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

#[cfg(loom)]
use loom::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicU64, Ordering};

/// Slots per ready word: the width of the `AtomicU64` that carries them.
const WORD: usize = 64;

/// The per-slot ready bits, shared with every thread that may wake a slot.
///
/// This is the loom-checked half of the wake protocol. Publishing uses
/// `Release` and harvesting uses `Acquire`, so a bit observed by the harvest
/// happens, after everything the waker did before setting it.
pub(crate) struct ReadySet {
    words: Box<[AtomicU64]>,
}

impl ReadySet {
    pub(crate) fn new(slots: usize) -> Self {
        Self { words: (0..slots.div_ceil(WORD)).map(|_| AtomicU64::new(0)).collect() }
    }

    /// Mark one slot ready. Callable from any thread.
    #[inline]
    pub(crate) fn mark(&self, slot: usize) {
        self.words[slot / WORD].fetch_or(1 << (slot % WORD), Ordering::Release);
    }

    /// Take a whole word of ready bits, clearing it.
    ///
    /// A relaxed load as a fast path for empty words was tried and reverted: it
    /// cost nothing where the set is busy and made a lightly loaded shard
    /// measurably slower, because the word that matters is usually the one that
    /// is occupied, and checking it twice is worse than swapping it once.
    #[inline]
    pub(crate) fn take(&self, word: usize) -> u64 {
        self.words[word].swap(0, Ordering::Acquire)
    }

    /// Put back bits that were taken but not acted on — the harvest cap was
    /// reached. Merging rather than storing, because a waker may have set new
    /// bits in the meantime and those must not be lost.
    #[inline]
    pub(crate) fn restore(&self, word: usize, bits: u64) {
        self.words[word].fetch_or(bits, Ordering::Release);
    }

    pub(crate) fn words(&self) -> usize {
        self.words.len()
    }
}

/// What one thread shares with everyone who can wake it.
struct Shared {
    ready: ReadySet,
    owner: AtomicWaker,
}

/// The waker handed to a slot's future. Cloning it clones an `Arc`; waking it
/// is one atomic OR and one `AtomicWaker` wake. Built once per slot at
/// startup, so the hot path never constructs one.
struct SlotWaker {
    shared: Arc<Shared>,
    slot: usize,
}

impl Wake for SlotWaker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        // Publish the bit before the wake, so the harvest that runs because of
        // this wake is guaranteed to see it.
        self.shared.ready.mark(self.slot);
        self.shared.owner.wake();
    }
}

/// What a harvest pass reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Harvest {
    /// Futures that completed and had their output handed to the callback.
    pub(crate) finished: usize,
    /// The cap stopped the pass with marked slots still unvisited. The caller
    /// must treat this as progress and take another turn rather than park —
    /// the wakes for those bits have already been consumed.
    pub(crate) truncated: bool,
}

/// A bounded set of one concrete future type, polled in place.
///
/// Single-threaded by construction: one thread pushes and harvests. Only
/// wakes cross threads, and they touch nothing but [`ReadySet`] and the owner
/// cell.
pub(crate) struct Outstanding<F> {
    slots: Vec<Option<Pin<Box<F>>>>,
    /// Whether a slot holds a live, incomplete future. A freed slot keeps its
    /// box for reuse, so `slots[i].is_some()` cannot answer this.
    occupied: Vec<bool>,
    wakers: Vec<Waker>,
    free: Vec<u32>,
    shared: Arc<Shared>,
    live: usize,
    /// Where the next harvest starts scanning.
    ///
    /// Always resuming at slot zero would let one slot monopolize a capped
    /// harvest: the free list hands back the slot just vacated, so a key with
    /// a queue would be re-dispatched into it and polled again ahead of
    /// everything else, forever. Resuming where the last pass stopped makes
    /// the scan a rotation, which bounds how long a ready slot waits by one
    /// trip around the set.
    cursor: usize,
}

impl<F: Future> Outstanding<F> {
    /// A set with room for `capacity` futures. The capacity is a hard bound,
    /// sized by the caller from the budgets that gate its pushes.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let shared = Arc::new(Shared { ready: ReadySet::new(capacity), owner: AtomicWaker::new() });
        Self {
            slots: (0..capacity).map(|_| None).collect(),
            occupied: vec![false; capacity],
            wakers: (0..capacity)
                .map(|slot| Waker::from(Arc::new(SlotWaker { shared: shared.clone(), slot })))
                .collect(),
            free: (0..capacity as u32).rev().collect(),
            shared,
            live: 0,
            cursor: 0,
        }
    }

    /// Whether anything is still outstanding — the shutdown condition.
    pub(crate) fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Register the owning task before checking anything the decision to park
    /// will rely on. Wakes that arrive after this re-poll the owner, which is
    /// what closes the found-nothing-then-slept window.
    pub(crate) fn register_owner(&self, waker: &Waker) {
        self.shared.owner.register(waker);
    }

    /// Add a future to the set. It is marked ready immediately: a new future
    /// must be polled at least once, and the next harvest is where that
    /// happens.
    ///
    /// # Panics
    ///
    /// If the set is full. Both call sites are gated — dispatch by the
    /// scheduler's in-flight budgets, flushes by `flush_slots` — so a full set
    /// is an accounting bug, not a load condition.
    pub(crate) fn push(&mut self, future: F) {
        let index = self
            .free
            .pop()
            .expect("outstanding set overflow: capacity must cover budgets plus flush slots")
            as usize;
        match &mut self.slots[index] {
            // Reuse: drop the previous occupant in place and write the new
            // future into the same allocation. Safe, and allocation-free.
            Some(slot) => slot.as_mut().set(future),
            // Warm-up: this slot's one and only allocation.
            empty @ None => *empty = Some(Box::pin(future)),
        }
        self.occupied[index] = true;
        self.live += 1;
        self.shared.ready.mark(index);
    }

    /// Poll every marked, occupied slot, handing each completion to `out`.
    /// Stops after `cap` polls; see [`Harvest::truncated`].
    pub(crate) fn harvest(&mut self, cap: usize, mut out: impl FnMut(F::Output)) -> Harvest {
        let cap = cap.max(1);
        let words = self.shared.ready.words();
        let mut polled = 0;
        let mut report = Harvest { finished: 0, truncated: false };

        // One rotation of the set, beginning where the last pass stopped.
        'rotation: for step in 0..words {
            let word = (self.cursor / WORD + step) % words;
            // Within the first word, start at the cursor's own bit. Rotating
            // puts that bit at position zero, so the scan runs from there and
            // wraps to the bits behind it — which are ready too, just later in
            // this rotation.
            let start = if step == 0 { self.cursor % WORD } else { 0 } as u32;
            let mut bits = self.shared.ready.take(word).rotate_right(start);

            while bits != 0 {
                if polled == cap {
                    // Hand the unvisited bits back and say so: their wakes are
                    // spent, so parking now would strand them.
                    self.shared.ready.restore(word, bits.rotate_left(start));
                    report.truncated = true;
                    break 'rotation;
                }
                let offset = bits.trailing_zeros();
                bits &= bits - 1;
                let bit = ((start + offset) % WORD as u32) as usize;
                let index = word * WORD + bit;
                // Resume after this slot, so a capped pass leaves the next one
                // pointing at work it has not seen.
                self.cursor = (index + 1) % (words * WORD);

                // A stale bit for a freed slot: the wake raced the completion.
                // Skipping it here is also what keeps a completed future from
                // ever being polled again.
                if !self.occupied[index] {
                    continue;
                }
                polled += 1;
                let future = self.slots[index].as_mut().expect("occupied slots hold a future");
                let mut cx = Context::from_waker(&self.wakers[index]);
                if let Poll::Ready(output) = future.as_mut().poll(&mut cx) {
                    self.occupied[index] = false;
                    self.free.push(index as u32);
                    self.live -= 1;
                    out(output);
                    report.finished += 1;
                }
            }
        }
        report
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    /// Pends `remaining` times, waking itself each time, then yields its
    /// payload — so completion depends on the waker actually working.
    struct Countdown {
        remaining: u32,
        payload: u64,
    }

    impl Future for Countdown {
        type Output = u64;
        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<u64> {
            if self.remaining == 0 {
                return Poll::Ready(self.payload);
            }
            self.remaining -= 1;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }

    /// An `async fn`, so the slab holds a compiler-generated `!Unpin` future
    /// rather than the conveniently movable struct above — which is the case
    /// the storage has to handle and the only one worth testing.
    async fn job(remaining: u32, payload: u64) -> u64 {
        Countdown { remaining, payload }.await
    }

    fn drain(set: &mut Outstanding<impl Future<Output = u64>>) -> Vec<u64> {
        let mut done = Vec::new();
        while !set.is_empty() {
            set.harvest(usize::MAX, |payload| done.push(payload));
        }
        done
    }

    #[test]
    fn every_future_completes_and_slots_recycle() {
        let mut set = Outstanding::with_capacity(4);
        for round in 0..64u64 {
            set.push(job(round as u32 % 3, round));
            set.push(job((round as u32 + 1) % 3, round + 1_000));
            let mut done = drain(&mut set);
            done.sort_unstable();
            assert_eq!(done, vec![round, round + 1_000]);
        }
        assert!(set.is_empty(), "sixteen times the capacity passed through four slots");
    }

    #[test]
    fn a_self_wake_during_poll_is_not_lost() {
        let mut set = Outstanding::with_capacity(1);
        set.push(job(5, 7));
        // Each harvest polls once (the future re-marks itself), so completion
        // takes exactly `remaining + 1` passes — if any self-wake were lost,
        // this would hang instead.
        let mut passes = 0;
        let mut done = Vec::new();
        while set.harvest(usize::MAX, |payload| done.push(payload)).finished == 0 {
            passes += 1;
            assert!(passes < 10, "a self-wake was dropped");
        }
        assert_eq!(done, vec![7]);
    }

    #[test]
    fn the_cap_truncates_and_nothing_is_stranded() {
        let mut set = Outstanding::with_capacity(8);
        for payload in 0..8u64 {
            set.push(job(0, payload));
        }
        let mut done = Vec::new();
        let first = set.harvest(3, |payload| done.push(payload));
        assert_eq!(first.finished, 3);
        assert!(first.truncated, "five marked slots were left unvisited");

        // The caller's contract: truncated means take another turn. The bits
        // were restored, so the rest complete without any new wake.
        while !set.is_empty() {
            set.harvest(3, |payload| done.push(payload));
        }
        done.sort_unstable();
        assert_eq!(done, (0..8).collect::<Vec<_>>());
    }

    /// A capped harvest must not serve the same slot forever.
    ///
    /// This is the shape that made it necessary: the free list hands back the
    /// slot just vacated, so a slot that keeps being refilled would keep
    /// landing at the same index. A scan that always began at zero would poll
    /// it and only it, and everything else would wait for a cap it never
    /// reached.
    #[test]
    fn a_capped_harvest_rotates_rather_than_favouring_low_slots() {
        let mut set = Outstanding::with_capacity(4);
        for payload in 0..4u64 {
            // Each pends once before finishing, so all four stay resident and
            // ready across several capped passes.
            set.push(job(1, payload));
        }

        let mut served = Vec::new();
        // One poll per pass, four passes: every slot must get exactly one.
        for _ in 0..4 {
            set.harvest(1, |payload| served.push(payload));
        }
        assert!(served.is_empty(), "one poll each only advances them past their pend");

        for _ in 0..4 {
            set.harvest(1, |payload| served.push(payload));
        }
        served.sort_unstable();
        assert_eq!(
            served,
            vec![0, 1, 2, 3],
            "a second poll each finishes all four; a fixed scan order would have \
             finished one repeatedly and starved the rest"
        );
    }

    #[test]
    fn a_slot_refilled_every_pass_cannot_monopolize_a_capped_harvest() {
        let mut set = Outstanding::with_capacity(2);
        set.push(job(0, 100)); // completes on its first poll, freeing its slot
        set.push(job(3, 200)); // needs four polls

        let mut served = Vec::new();
        for round in 0..8u64 {
            set.harvest(1, |payload| served.push(payload));
            // Refill greedily, exactly as the reactor re-dispatches a key with
            // more queued work. The freed slot is handed straight back.
            if !set.free.is_empty() {
                set.push(job(0, round));
            }
        }
        assert!(served.contains(&200), "the long-running slot never got polled: {served:?}");
    }

    #[test]
    fn a_stale_wake_for_a_freed_slot_is_skipped() {
        let mut set = Outstanding::with_capacity(2);
        set.push(job(0, 1));
        let waker = set.wakers[0].clone();
        assert_eq!(set.harvest(usize::MAX, |_| {}).finished, 1);

        // The wake arrives after completion: the classic ABA shape. It must
        // neither panic (re-polling a finished async fn does) nor invent work.
        waker.wake();
        let report = set.harvest(usize::MAX, |_| panic!("nothing is live"));
        assert_eq!(report, Harvest { finished: 0, truncated: false });

        // And a refill after the stale wake runs normally. A double poll of
        // the finished occupant would have panicked above; a double poll of
        // the refill would panic here the same way.
        set.push(job(0, 42));
        let mut done = Vec::new();
        assert_eq!(set.harvest(usize::MAX, |payload| done.push(payload)).finished, 1);
        assert_eq!(done, vec![42]);
    }

    #[test]
    #[should_panic(expected = "outstanding set overflow")]
    fn exceeding_capacity_is_an_accounting_bug_and_says_so() {
        let mut set = Outstanding::with_capacity(1);
        set.push(job(1, 1));
        set.push(job(1, 2));
    }
}

/// Exhaustive interleaving checks on the wake protocol.
///
/// These model the [`AtomicWaker`] as a plain flag with the contract its
/// documentation states — a wake after a registration re-polls the owner —
/// because loom cannot see inside a foreign crate's atomics. What is proved is
/// our side: publish-then-wake against register-then-check can never strand a
/// ready bit, however the threads interleave.
#[cfg(all(test, loom))]
mod loom_tests {
    use super::ReadySet;
    use loom::sync::Arc;
    use loom::sync::atomic::{AtomicBool, Ordering};
    use loom::thread;

    /// A wake is never observable before the bit it is announcing.
    ///
    /// This is the property the parked loop rests on. The loop registers, finds
    /// nothing, and sleeps; some thread then marks a slot and wakes it. When it
    /// runs again it harvests — and must find work. If a wake could ever become
    /// visible while its bit was not, the loop would wake, harvest nothing, and
    /// park again with the work stranded.
    ///
    /// The `woken` flag models the observable effect of `AtomicWaker::wake` —
    /// the owning task being scheduled. The cell's own register-versus-wake race
    /// is its documented contract and is not modelled here; what is checked is
    /// the ordering this module is responsible for, the `Release` in
    /// [`ReadySet::mark`] against the `Acquire` in [`ReadySet::take`].
    #[test]
    fn loom_a_wake_is_never_visible_before_its_bit() {
        loom::model(|| {
            let ready = Arc::new(ReadySet::new(64));
            let woken = Arc::new(AtomicBool::new(false));

            let producer = {
                let ready = ready.clone();
                let woken = woken.clone();
                // `SlotWaker::wake_by_ref`, verbatim: publish, then wake.
                thread::spawn(move || {
                    ready.mark(3);
                    woken.store(true, Ordering::Release);
                })
            };

            // The woken loop's next turn: notice it was woken, then harvest.
            let saw_wake = woken.load(Ordering::Acquire);
            let bits = ready.take(0);
            if saw_wake {
                assert_ne!(
                    bits, 0,
                    "woken while the bit was still invisible: the loop would \
                     harvest nothing and park again, stranding the work"
                );
            }

            producer.join().unwrap();
            // Whatever the interleaving, one harvest after the wake finds it.
            assert_ne!(bits | ready.take(0), 0, "the bit was lost entirely");
        });
    }

    #[test]
    fn loom_concurrent_wakes_for_one_slot_collapse_to_one_bit() {
        loom::model(|| {
            let ready = Arc::new(ReadySet::new(64));
            let threads: Vec<_> = (0..2)
                .map(|_| {
                    let ready = ready.clone();
                    thread::spawn(move || ready.mark(5))
                })
                .collect();
            for handle in threads {
                handle.join().unwrap();
            }
            assert_eq!(ready.take(0), 1 << 5, "idempotent marks, exactly one bit");
            assert_eq!(ready.take(0), 0);
        });
    }

    #[test]
    fn loom_restore_merges_with_a_concurrent_wake_rather_than_clobbering_it() {
        loom::model(|| {
            let ready = Arc::new(ReadySet::new(64));
            ready.mark(1);
            ready.mark(2);
            let taken = ready.take(0);

            let producer = {
                let ready = ready.clone();
                thread::spawn(move || ready.mark(9))
            };
            // The truncated-harvest path: hand back what was not visited.
            ready.restore(0, taken & !(1 << 1));
            producer.join().unwrap();

            let word = ready.take(0);
            assert_ne!(word & (1 << 9), 0, "restore erased a wake that raced it");
            assert_ne!(word & (1 << 2), 0, "restore lost a bit it was handing back");
        });
    }
}
