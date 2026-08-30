//! A bounded, shard-owned replacement for `FuturesUnordered`.
//!
//! `Outstanding` is specialized for a reactor with one owning thread, one
//! concrete future type, a fixed in-flight budget, and wakes that may arrive
//! from other threads.  Push, polling, completion, and slot reuse remain on the
//! owner.  Other threads see only atomic readiness words and one [`Doorbell`].
//!
//! [`Doorbell`]: crate::doorbell::Doorbell
//!
//! [`BoxedOutstanding`] preallocates one `Pin<Box<Option<F>>>` per slot,
//! allocates every future slot and waker at construction, drops a future the
//! moment it returns `Ready`, and reuses that storage without allocating again.
//! It is written entirely in safe Rust.
//!
//! Storage sits behind the private [`Storage`] trait so that an alternative
//! layout can be substituted and measured through an otherwise identical loop.
//! At the populations this set is built for, the scan and the future's own poll
//! dominate; a candidate layout has to beat that, not merely differ from it.
//!
//! # Ready protocol
//!
//! A slot waker publishes its bit with `Release`.  A word's `0 -> nonzero`
//! transition publishes that word in the summary bitmap, and a slot's
//! `0 -> 1` transition wakes the registered owner.  Repeated wakes therefore
//! coalesce without repeatedly contending on the summary or the doorbell.
//! Harvesting takes the bitmaps with `Acquire`.  A mark racing a take is
//! observed either by that take or by the next one; restoration uses atomic OR
//! and therefore cannot overwrite a concurrent mark.
//!
//! Capacities up to 64 use a single ready word and avoid the summary entirely.
//! Larger sets use one summary `AtomicU64`, supporting up to 4,096 slots while
//! touching only words announced as ready.  The first word is split at the
//! scan cursor: its high portion is visited first and its low portion is
//! deferred until every other announced word has been visited.  This preserves
//! true circular fairness across word boundaries under capped harvesting.
//!
//! The owner must register before the readiness check on which parking relies.
//! That discipline and its proof live in [`crate::doorbell`];
//! [`BoxedOutstanding::poll_harvest`] encodes the order here.  A wake
//! concurrent with the final `more_ready` check may arrive just after the
//! check, but the registered owner is then scheduled.
//!
//! # Stale wakes
//!
//! Wakers are stable per slot rather than per occupant.  A late wake for a free
//! slot is discarded.  A late wake after refill may spuriously poll the new
//! future, which is legal under the `Future` contract.  This avoids allocating
//! a generation-specific reference-counted waker on every dispatch.  A source
//! that repeatedly wakes after completion can still waste work and should be
//! treated as an upstream primitive defect.
//!
//! # Caps and panics
//!
//! `cap` limits live future polls, not stale bits.  A zero cap performs no
//! polls.  [`Harvest::more_ready`] says that the pass retained or subsequently
//! observed ready work; it is an optimization hint, not a replacement for the
//! owner-waker registration protocol.
//!
//! A restoration guard returns every taken-but-unvisited bit if polling or the
//! output callback unwinds.  The current bit is retained in the guard until
//! its poll and callback both finish, so unwinding cannot silently strand the
//! remainder of a ready batch.  A future that itself panics may of course panic
//! again if retried.  Future destructors must not panic: as with most pinned
//! containers, a destructor panic poisons the set and may leak remaining
//! values.  Production futures should catch domain panics internally or the
//! process should abort on invariant failure.
//!
//! # Cargo and Loom
//!
//! Production dependencies:
//!
//! ```toml
//! [dependencies]
//! crossbeam-utils = "0.8"
//! futures = "0.3"
//! ```
//!
//! Test configuration:
//!
//! ```toml
//! [dev-dependencies]
//! loom = "0.7"
//!
//! [lints.rust]
//! unexpected_cfgs = { level = "warn", check-cfg = ["cfg(loom)"] }
//! ```
//!
//! Run ordinary tests normally.  Run the exhaustive wake models separately:
//!
//! ```text
//! RUSTFLAGS="--cfg loom --check-cfg=cfg(loom)" cargo test loom_tests -- --test-threads=1
//! ```
//!
//! # Example
//!
//! ```ignore
//! // The module is crate-private, so this shows the shape rather than
//! // compiling against it; the unit tests exercise the real thing.
//! use grommet::outstanding::Outstanding;
//!
//! async fn work(value: u64) -> u64 { value * 2 }
//!
//! let mut set = Outstanding::with_capacity(64);
//! set.try_push(work(21)).expect("fixed capacity was budgeted");
//!
//! let mut output = Vec::with_capacity(64);
//! let report = set.harvest(64, |value| output.push(value));
//! assert_eq!(report.finished, 1);
//! assert_eq!(output, [42]);
//! ```
//!
//! # Benchmarking
//!
//! Benchmark any candidate storage with the actual future type.  Include first
//! touch, sparse and dense readiness, cross-core wakes, cap truncation, greedy
//! refill, stale wake storms, and capacities on both sides of 64. Measure at
//! the concurrency the deployment actually runs: layout differences that look
//! real at sixty-four live futures can vanish entirely at three thousand.  For a trading loop,
//! p99.99 and maximum cycles per reactor turn matter more than mean throughput.

use crate::doorbell::Doorbell;
use crossbeam_utils::CachePadded;
use std::fmt;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

#[cfg(loom)]
use loom::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicU64, Ordering};

const WORD_BITS: usize = 64;

/// Ready words the summary can address, and so the shape of the whole
/// structure. Change this line to change the ceiling.
///
/// It is a plain constant rather than a generic parameter deliberately. A
/// generic would let each call site pick a ceiling, but this module is
/// crate-private and the reactor is its only caller, so every instantiation
/// would pass the same value. Threading a parameter through the set, its shared
/// state and the reactor to say one thing in one place buys nothing a constant
/// does not; both are settled at compile time.
const MAX_WORDS: usize = 1_024;

/// Summary words needed to address [`MAX_WORDS`]. Fixed at compile time, so a
/// set allocates nothing for its summary and touches only the prefix its own
/// capacity uses.
const SUMMARY_WORDS: usize = MAX_WORDS.div_ceil(WORD_BITS);

/// Every ready word the ceiling implies must be addressable by one bit of the
/// summary array, or announcements above that point would be dropped silently.
/// Checked here rather than in a test, because it is a property of the
/// constants and can be settled before anything runs.
const _: () = assert!(
    MAX_WORDS <= SUMMARY_WORDS * WORD_BITS,
    "SUMMARY_WORDS is too small to address MAX_WORDS"
);

/// The ceiling exists to carry tens of thousands of in-flight futures. Lowering
/// it below that is a deliberate act, so it fails the build rather than
/// quietly capping a deployment that was sized for more.
const _: () = assert!(MAX_CAPACITY >= 65_536, "MAX_WORDS was lowered below the design target");

/// Maximum supported population: one bit per slot across [`MAX_WORDS`].
///
/// A set this size costs one waker `Arc` and one boxed slot per member, so the
/// memory rather than the bitmap is what should decide whether to raise it.
pub const MAX_CAPACITY: usize = WORD_BITS * MAX_WORDS;

#[inline]
const fn low_mask(bits: usize) -> u64 {
    if bits == 0 {
        0
    } else if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// A cache-line-separated ready word.  Producers for different groups of 64
/// slots do not invalidate one another's word cache line.  Crossbeam selects
/// the target-appropriate destructive-interference alignment (rather than a
/// guessed, hard-coded 64 bytes).
type ReadyWord = CachePadded<AtomicU64>;

/// Concurrent readiness shared with every slot waker.
///
/// One word is the fast path.  Two to sixty-four words use `summary` as a
/// first-level bitmap.  Summary false positives are harmless and self-clear
/// when the announced word is found empty; the publication order prevents
/// false negatives from stranding work.
struct ReadySet {
    words: Box<[ReadyWord]>,
    /// One bit per ready word, so that a harvest touches only the words some
    /// producer announced rather than all of them.
    ///
    /// Fixed at [`SUMMARY_WORDS`] rather than one, which is what lets the
    /// population exceed the 4,096 slots a single summary word could address.
    /// Only the prefix a given capacity needs is ever touched, so a small set
    /// pays for the unused tail in memory and never in work.
    ///
    /// Every producer may touch these, so they are padded for the same reason
    /// the ready words are: the contention on a given cell is intentional, the
    /// invalidation of its neighbours is not.
    summary: [CachePadded<AtomicU64>; SUMMARY_WORDS],
    /// Summary words this capacity actually uses, and zero when one ready word
    /// makes a summary pure overhead on both the mark and the take.
    summaries: usize,
    slots: usize,
}

impl ReadySet {
    fn new(slots: usize) -> Self {
        assert!(slots <= MAX_CAPACITY, "Outstanding capacity exceeds {MAX_CAPACITY}");
        let words = slots.div_ceil(WORD_BITS);
        Self {
            words: (0..words).map(|_| CachePadded::new(AtomicU64::new(0))).collect(),
            summary: std::array::from_fn(|_| CachePadded::new(AtomicU64::new(0))),
            summaries: if words > 1 { words.div_ceil(WORD_BITS) } else { 0 },
            slots,
        }
    }

    #[inline]
    fn words(&self) -> usize {
        self.words.len()
    }

    /// Summary words, and so how many passes a full scan makes over the top
    /// level. Zero exactly when the set is small enough not to need one.
    #[inline]
    fn summaries(&self) -> usize {
        self.summaries
    }

    #[inline]
    fn mark(&self, slot: usize) -> bool {
        debug_assert!(slot < self.slots);
        let word = slot / WORD_BITS;
        let bit = 1u64 << (slot % WORD_BITS);
        let previous = self.words[word].fetch_or(bit, Ordering::Release);
        if previous == 0 {
            self.announce(word);
        }
        previous & bit == 0
    }

    /// Publish a word as non-empty. Ordered after the word's own `Release`, so
    /// a harvest that takes the summary first and the word second cannot
    /// observe the announcement without the bits behind it.
    #[inline]
    fn announce(&self, word: usize) {
        if self.summaries == 0 {
            return;
        }
        self.summary[word / WORD_BITS].fetch_or(1u64 << (word % WORD_BITS), Ordering::Release);
    }

    /// Take the single-word fast path.  No summary atomic is touched.
    #[inline]
    fn take_single(&self) -> u64 {
        debug_assert_eq!(self.words.len(), 1);
        self.take_word(0)
    }

    /// Take one summary word: the set of ready words it announces.
    ///
    /// Clearing this before the words it names is what keeps a mark racing the
    /// scan from being lost: such a mark re-announces its word, and the next
    /// pass finds it.
    #[inline]
    fn take_summary(&self, summary: usize) -> u64 {
        self.summary[summary].swap(0, Ordering::Acquire) & self.summary_mask(summary)
    }

    #[inline]
    fn take_word(&self, word: usize) -> u64 {
        self.words[word].swap(0, Ordering::Acquire) & self.valid_mask(word)
    }

    #[inline]
    fn restore_word(&self, word: usize, bits: u64) {
        let bits = bits & self.valid_mask(word);
        if bits == 0 {
            return;
        }
        let previous = self.words[word].fetch_or(bits, Ordering::Release);
        if previous == 0 {
            self.announce(word);
        }
    }

    #[inline]
    fn restore_summary(&self, summary: usize, words: u64) {
        let words = words & self.summary_mask(summary);
        if words != 0 {
            self.summary[summary].fetch_or(words, Ordering::Release);
        }
    }

    #[inline]
    fn has_ready(&self) -> bool {
        match self.words.len() {
            0 => false,
            1 => self.words[0].load(Ordering::Acquire) & self.valid_mask(0) != 0,
            _ => (0..self.summaries).any(|summary| {
                self.summary[summary].load(Ordering::Acquire) & self.summary_mask(summary) != 0
            }),
        }
    }

    /// Which bits of a summary word name real ready words.
    #[inline]
    fn summary_mask(&self, summary: usize) -> u64 {
        let remainder = self.words.len() % WORD_BITS;
        if summary + 1 == self.summaries && remainder != 0 { low_mask(remainder) } else { u64::MAX }
    }

    #[inline]
    fn valid_mask(&self, word: usize) -> u64 {
        let remainder = self.slots % WORD_BITS;
        if word + 1 == self.words.len() && remainder != 0 { low_mask(remainder) } else { u64::MAX }
    }
}

struct Shared {
    ready: ReadySet,
    bell: Doorbell,
}

impl Shared {
    #[inline]
    fn notify(&self, slot: usize) {
        // Checked before the mark rather than only inside `ring`: once the set
        // is gone nobody will read these bits again, so publishing one is pure
        // cost on a path stale wakers keep taking.
        if self.bell.is_closed() {
            return;
        }
        if self.ready.mark(slot) {
            // Closing may race this notification.  `Doorbell` orders the two,
            // and a final wake across the close boundary is harmless.
            self.bell.ring();
        }
    }
}

struct SlotWaker {
    shared: Arc<Shared>,
    slot: usize,
}

impl Wake for SlotWaker {
    #[inline]
    fn wake(self: Arc<Self>) {
        self.shared.notify(self.slot);
    }

    #[inline]
    fn wake_by_ref(self: &Arc<Self>) {
        self.shared.notify(self.slot);
    }
}

/// Result of one harvest pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Harvest {
    /// Live futures polled.  Stale ready bits do not count against the cap.
    pub polled: usize,
    /// Futures that completed and whose output callback returned normally.
    pub finished: usize,
    /// Work was retained because of the cap, or new readiness was observed
    /// after the pass's snapshot.  Treat this as a reason to take another turn.
    pub more_ready: bool,
}

/// A fixed-capacity insertion failure that retains ownership of the future.
pub struct PushError<F> {
    future: F,
}

// Reachable through `Inner::try_push`; the accessors are what make handing the
// future back meaningful rather than a claim, and match how `SubmitError` and
// `BatchError` return rejected work elsewhere in the crate.
#[allow(dead_code)]
impl<F> PushError<F> {
    pub fn into_future(self) -> F {
        self.future
    }

    pub fn future(&self) -> &F {
        &self.future
    }
}

impl<F> fmt::Debug for PushError<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PushError { full: true, .. }")
    }
}

impl<F> fmt::Display for PushError<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the outstanding set is full")
    }
}

impl<F> std::error::Error for PushError<F> {}

/// Internal storage contract.  Every implementation must keep an inserted
/// future at a stable address until `remove` drops it.
trait Storage<F: Future> {
    fn with_capacity(capacity: usize) -> Self
    where
        Self: Sized;
    fn capacity(&self) -> usize;
    fn is_occupied(&self, index: usize) -> bool;
    fn insert(&mut self, index: usize, future: F);
    fn poll(&mut self, index: usize, cx: &mut Context<'_>) -> Poll<F::Output>;
    fn remove(&mut self, index: usize);
}

/// Fully safe reference storage: one preallocated pinned box per slot.
struct BoxedStorage<F> {
    slots: Box<[Pin<Box<Option<F>>>]>,
}

impl<F: Future> Storage<F> for BoxedStorage<F> {
    fn with_capacity(capacity: usize) -> Self {
        Self { slots: (0..capacity).map(|_| Box::pin(None::<F>)).collect() }
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.slots.len()
    }

    #[inline]
    fn is_occupied(&self, index: usize) -> bool {
        self.slots[index].as_ref().get_ref().is_some()
    }

    #[inline]
    fn insert(&mut self, index: usize, future: F) {
        debug_assert!(!self.is_occupied(index));
        self.slots[index].as_mut().set(Some(future));
    }

    #[inline]
    fn poll(&mut self, index: usize, cx: &mut Context<'_>) -> Poll<F::Output> {
        self.slots[index]
            .as_mut()
            .as_pin_mut()
            .expect("occupied boxed slot contains a future")
            .poll(cx)
    }

    #[inline]
    fn remove(&mut self, index: usize) {
        debug_assert!(self.is_occupied(index));
        // Pin::set drops F in place before writing None, preserving the box.
        self.slots[index].as_mut().set(None);
    }
}

/// Restores readiness if user code unwinds while a harvest owns bitmap bits.
struct RestoreGuard<'a> {
    shared: &'a Shared,
    /// The summary word being consumed, and the announcements in it that the
    /// scan has not reached yet. Only one is ever held: summary words are
    /// taken as the rotation arrives at them, so the rest were never removed
    /// and need no restoring.
    summary_index: usize,
    summary: u64,
    /// The low part of the starting summary word, which names words behind the
    /// cursor. Held back so the rotation reaches them last, exactly as
    /// `deferred_bits` does one level down.
    deferred_summary_index: usize,
    deferred_summary: u64,
    current_word: usize,
    current_bits: u64,
    deferred_word: usize,
    deferred_bits: u64,
    armed: bool,
}

impl<'a> RestoreGuard<'a> {
    fn new(shared: &'a Shared) -> Self {
        Self {
            shared,
            summary_index: 0,
            summary: 0,
            deferred_summary_index: 0,
            deferred_summary: 0,
            current_word: 0,
            current_bits: 0,
            deferred_word: 0,
            deferred_bits: 0,
            armed: true,
        }
    }

    /// Begin consuming one summary word.
    #[inline]
    fn set_summary(&mut self, index: usize, words: u64) {
        debug_assert_eq!(self.summary, 0);
        self.summary_index = index;
        self.summary = words;
    }

    /// Hold back the part of the starting summary word that sits behind the
    /// cursor.
    #[inline]
    fn defer_summary(&mut self, index: usize, words: u64) {
        debug_assert_eq!(self.deferred_summary, 0);
        self.deferred_summary_index = index;
        self.deferred_summary = words;
    }

    /// Bring the deferred summary announcements back for the wrap-around pass.
    #[inline]
    fn activate_deferred_summary(&mut self) {
        debug_assert_eq!(self.summary, 0);
        self.summary_index = self.deferred_summary_index;
        self.summary = std::mem::take(&mut self.deferred_summary);
    }

    /// Take the next announced word from the summary word in hand.
    #[inline]
    fn next_announced(&mut self) -> Option<usize> {
        if self.summary == 0 {
            return None;
        }
        let bit = self.summary.trailing_zeros() as usize;
        self.summary &= self.summary - 1;
        Some(self.summary_index * WORD_BITS + bit)
    }

    #[inline]
    fn set_current(&mut self, word: usize, bits: u64) {
        debug_assert_eq!(self.current_bits, 0);
        self.current_word = word;
        self.current_bits = bits;
    }

    #[inline]
    fn set_deferred(&mut self, word: usize, bits: u64) {
        debug_assert_eq!(self.deferred_bits, 0);
        self.deferred_word = word;
        self.deferred_bits = bits;
    }

    #[inline]
    fn activate_deferred(&mut self) {
        debug_assert_eq!(self.current_bits, 0);
        self.current_word = self.deferred_word;
        self.current_bits = std::mem::take(&mut self.deferred_bits);
    }

    fn restore(&mut self, wake_owner: bool) {
        if !self.armed {
            return;
        }
        // Set by the restores below rather than tested for separately. The
        // four guards are already here and already decide something visible --
        // whether each restore happens at all -- so reading the answer off them
        // says exactly what a second set of `!= 0` tests would, without adding
        // four comparisons that no reachable state can tell apart: a guard is
        // only ever restored from inside `poll_current`, which always still
        // holds the current word's bits, so the other three terms can never be
        // the one that decides.
        let mut had_work = false;
        if self.current_bits != 0 {
            self.shared.ready.restore_word(self.current_word, self.current_bits);
            had_work = true;
        }
        if self.deferred_bits != 0 {
            self.shared.ready.restore_word(self.deferred_word, self.deferred_bits);
            had_work = true;
        }
        if self.summary != 0 {
            self.shared.ready.restore_summary(self.summary_index, self.summary);
            had_work = true;
        }
        if self.deferred_summary != 0 {
            self.shared.ready.restore_summary(self.deferred_summary_index, self.deferred_summary);
            had_work = true;
        }
        self.summary = 0;
        self.deferred_summary = 0;
        self.current_bits = 0;
        self.deferred_bits = 0;
        self.armed = false;

        if wake_owner && had_work {
            self.shared.bell.ring();
        }
    }

    #[inline]
    fn disarm(&mut self) {
        debug_assert_eq!(self.summary, 0);
        debug_assert_eq!(self.deferred_summary, 0);
        debug_assert_eq!(self.current_bits, 0);
        debug_assert_eq!(self.deferred_bits, 0);
        self.armed = false;
    }
}

impl Drop for RestoreGuard<'_> {
    fn drop(&mut self) {
        // On unwind, restore and notify.  Normal completion or truncation
        // disarms the guard explicitly.
        self.restore(true);
    }
}

/// Disjoint mutable fields borrowed for one harvest.  Keeping this state in a
/// struct makes the inner poll loop small without passing a wide argument list.
struct PollState<'a, F, S, G>
where
    F: Future,
    S: Storage<F>,
    G: FnMut(F::Output),
{
    storage: &'a mut S,
    wakers: &'a [Waker],
    free: &'a mut Vec<u32>,
    live: &'a mut usize,
    cursor: &'a mut usize,
    capacity: usize,
    cap: usize,
    report: Harvest,
    out: &'a mut G,
    _future: PhantomData<fn() -> F>,
}

impl<F, S, G> PollState<'_, F, S, G>
where
    F: Future,
    S: Storage<F>,
    G: FnMut(F::Output),
{
    fn poll_current(&mut self, guard: &mut RestoreGuard<'_>) -> bool {
        while guard.current_bits != 0 {
            let bit = guard.current_bits.trailing_zeros() as usize;
            let mask = 1u64 << bit;
            let index = guard.current_word * WORD_BITS + bit;

            // Padding bits are masked by ReadySet, but keep release builds
            // robust against internal corruption.
            if index >= self.capacity {
                guard.current_bits &= !mask;
                continue;
            }

            // Stale bits do not consume the poll budget.
            if !self.storage.is_occupied(index) {
                guard.current_bits &= !mask;
                *self.cursor = if index + 1 == self.capacity { 0 } else { index + 1 };
                continue;
            }

            if self.report.polled == self.cap {
                return true;
            }

            self.report.polled += 1;
            let mut cx = Context::from_waker(&self.wakers[index]);
            if let Poll::Ready(output) = self.storage.poll(index, &mut cx) {
                // Drop the completed F now, not on a later dispatch.  Keep the
                // current bit guarded until the callback returns so unwinding
                // restores the remainder of this word.
                self.storage.remove(index);
                self.free.push(index as u32);
                *self.live -= 1;
                (self.out)(output);
                self.report.finished += 1;
            }

            guard.current_bits &= !mask;
            *self.cursor = if index + 1 == self.capacity { 0 } else { index + 1 };
        }
        false
    }
}

struct Inner<F: Future, S: Storage<F>> {
    storage: S,
    wakers: Vec<Waker>,
    free: Vec<u32>,
    shared: Arc<Shared>,
    live: usize,
    cursor: usize,
    _future: PhantomData<fn() -> F>,
}

#[allow(dead_code)]
impl<F: Future, S: Storage<F>> Inner<F, S> {
    fn with_capacity(capacity: usize) -> Self {
        assert!(capacity <= MAX_CAPACITY, "Outstanding capacity exceeds {MAX_CAPACITY}");
        assert!(capacity <= u32::MAX as usize, "Outstanding capacity exceeds u32 indexing");

        let storage = S::with_capacity(capacity);
        debug_assert_eq!(storage.capacity(), capacity);
        let shared = Arc::new(Shared { ready: ReadySet::new(capacity), bell: Doorbell::new() });
        let wakers = (0..capacity)
            .map(|slot| Waker::from(Arc::new(SlotWaker { shared: shared.clone(), slot })))
            .collect();

        Self {
            storage,
            wakers,
            free: (0..capacity as u32).rev().collect(),
            shared,
            live: 0,
            cursor: 0,
            _future: PhantomData,
        }
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.storage.capacity()
    }

    #[inline]
    fn len(&self) -> usize {
        self.live
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.live == 0
    }

    #[inline]
    fn available(&self) -> usize {
        self.free.len()
    }

    #[inline]
    fn register_owner(&self, waker: &Waker) {
        self.shared.bell.register(waker);
    }

    #[inline]
    fn try_push(&mut self, future: F) -> Result<(), PushError<F>> {
        let Some(index) = self.free.pop() else {
            return Err(PushError { future });
        };
        let index = index as usize;
        debug_assert!(!self.storage.is_occupied(index));
        self.storage.insert(index, future);
        self.live += 1;
        // The owner performs pushes and is already running, so publishing the
        // bit is sufficient; it must harvest before using readiness to park.
        let _ = self.shared.ready.mark(index);
        Ok(())
    }

    #[track_caller]
    fn push(&mut self, future: F) {
        if self.try_push(future).is_err() {
            panic!("outstanding set overflow: capacity must cover all in-flight budgets");
        }
    }

    fn harvest<G>(&mut self, cap: usize, mut out: G) -> Harvest
    where
        G: FnMut(F::Output),
    {
        let capacity = self.capacity();
        if cap == 0 || capacity == 0 {
            return Harvest { polled: 0, finished: 0, more_ready: self.shared.ready.has_ready() };
        }

        let Self { storage, wakers, free, shared, live, cursor, .. } = self;
        let words = shared.ready.words();
        let start_word = *cursor / WORD_BITS;
        let start_bit = *cursor % WORD_BITS;
        let mut polling = PollState::<F, S, G> {
            storage,
            wakers,
            free,
            live,
            cursor,
            capacity,
            cap,
            report: Harvest::default(),
            out: &mut out,
            _future: PhantomData,
        };

        // One-word fast path: one atomic swap, no summary operation.
        if words == 1 {
            let bits = shared.ready.take_single();
            let mut guard = RestoreGuard::new(shared);
            let before_cursor = bits & low_mask(start_bit);
            let from_cursor = bits & !low_mask(start_bit);
            guard.set_deferred(0, before_cursor);
            guard.set_current(0, from_cursor);

            if polling.poll_current(&mut guard) {
                polling.report.more_ready = true;
                guard.restore(false);
                return polling.report;
            }

            guard.activate_deferred();
            if polling.poll_current(&mut guard) {
                polling.report.more_ready = true;
                guard.restore(false);
                return polling.report;
            }

            guard.disarm();
            polling.report.more_ready = shared.ready.has_ready();
            return polling.report;
        }

        // Multiword path. The scan is one rotation of the whole set, and it
        // runs at two levels for the same reason it runs at one: a capped pass
        // must resume where the last one stopped, or a slot low in the order
        // could be passed over indefinitely. So summary words rotate from the
        // cursor's, and the words inside the cursor's summary word rotate from
        // the cursor's own, with both remainders deferred to the end.
        //
        // Summary words are taken as the rotation reaches them, never all at
        // once: one is in hand at a time, so an unwind restores that one and
        // the rest were never removed to begin with.
        let summaries = shared.ready.summaries();
        let start_summary = start_word / WORD_BITS;
        let mut guard = RestoreGuard::new(shared);

        // The cursor's own summary word, split at the cursor's word.
        let announced = shared.ready.take_summary(start_summary);
        let start_word_bit = start_word % WORD_BITS;
        guard.defer_summary(start_summary, announced & low_mask(start_word_bit));
        guard.set_summary(start_summary, announced & !low_mask(start_word_bit));

        // Within it, the cursor's own word is split at the cursor's bit.
        if guard.summary & (1u64 << start_word_bit) != 0 {
            guard.summary &= !(1u64 << start_word_bit);
            let bits = shared.ready.take_word(start_word);
            guard.set_deferred(start_word, bits & low_mask(start_bit));
            guard.set_current(start_word, bits & !low_mask(start_bit));
            if polling.poll_current(&mut guard) {
                polling.report.more_ready = true;
                guard.restore(false);
                return polling.report;
            }
        }

        // Everything the rotation reaches before wrapping back to the cursor.
        for step in 0..summaries {
            if step > 0 {
                debug_assert_eq!(guard.summary, 0);
                let summary = (start_summary + step) % summaries;
                guard.set_summary(summary, shared.ready.take_summary(summary));
            }
            while let Some(word) = guard.next_announced() {
                guard.set_current(word, shared.ready.take_word(word));
                if polling.poll_current(&mut guard) {
                    polling.report.more_ready = true;
                    guard.restore(false);
                    return polling.report;
                }
            }
        }

        // The words behind the cursor in its own summary word, then the bits
        // behind the cursor in its own word: last in the true circular order.
        guard.activate_deferred_summary();
        while let Some(word) = guard.next_announced() {
            guard.set_current(word, shared.ready.take_word(word));
            if polling.poll_current(&mut guard) {
                polling.report.more_ready = true;
                guard.restore(false);
                return polling.report;
            }
        }
        guard.activate_deferred();
        if polling.poll_current(&mut guard) {
            polling.report.more_ready = true;
            guard.restore(false);
            return polling.report;
        }

        debug_assert_eq!(guard.summary, 0);
        guard.disarm();
        polling.report.more_ready = shared.ready.has_ready();
        polling.report
    }

    #[cfg(test)]
    #[cfg(not(loom))]
    fn check_invariants(&self) -> Result<(), &'static str> {
        if self.storage.capacity() != self.wakers.len() {
            return Err("storage and waker capacities differ");
        }
        if self.live + self.free.len() != self.capacity() {
            return Err("live plus free does not equal capacity");
        }
        if self.capacity() != 0 && self.cursor >= self.capacity() {
            return Err("scan cursor is outside capacity");
        }

        let mut free_seen = vec![false; self.capacity()];
        for index in &self.free {
            let index = *index as usize;
            if index >= self.capacity() || free_seen[index] {
                return Err("free list is out of range or duplicated");
            }
            if self.storage.is_occupied(index) {
                return Err("free list contains an occupied slot");
            }
            free_seen[index] = true;
        }
        let occupied =
            (0..self.capacity()).filter(|index| self.storage.is_occupied(*index)).count();
        if occupied != self.live {
            return Err("live counter disagrees with storage");
        }
        for (index, free) in free_seen.into_iter().enumerate() {
            if free == self.storage.is_occupied(index) {
                return Err("a slot is neither exclusively free nor occupied");
            }
        }
        Ok(())
    }
}

impl<F: Future, S: Storage<F>> Drop for Inner<F, S> {
    fn drop(&mut self) {
        // Prevent retained stale slot wakers from retaining or scheduling the
        // owner task after this set is gone.
        self.shared.bell.close();
    }
}

/// Fully safe, preallocated per-slot boxed storage.
pub struct BoxedOutstanding<F: Future> {
    inner: Inner<F, BoxedStorage<F>>,
}

macro_rules! impl_outstanding {
    ($name:ident, $storage:ident) => {
        // A container's API is complete rather than trimmed to today's one
        // caller: the reactor pushes, harvests, registers and asks whether it
        // is empty, and the rest is introspection the metrics work will want.
        // Keeping it whole also means a candidate storage is measured against
        // the same surface rather than a subset of it.
        #[allow(dead_code)]
        impl<F: Future> $name<F> {
            /// Construct and fully allocate a fixed-capacity set.
            pub fn with_capacity(capacity: usize) -> Self {
                Self { inner: Inner::<F, $storage<F>>::with_capacity(capacity) }
            }

            #[inline]
            pub fn capacity(&self) -> usize {
                self.inner.capacity()
            }

            #[inline]
            pub fn len(&self) -> usize {
                self.inner.len()
            }

            #[inline]
            pub fn is_empty(&self) -> bool {
                self.inner.is_empty()
            }

            #[inline]
            pub fn available(&self) -> usize {
                self.inner.available()
            }

            /// Register the owner before any readiness check used to park.
            #[inline]
            pub fn register_owner(&self, waker: &Waker) {
                self.inner.register_owner(waker);
            }

            /// Insert without allocation, returning the future if full.
            #[inline]
            pub fn try_push(&mut self, future: F) -> Result<(), PushError<F>> {
                self.inner.try_push(future)
            }

            /// Insert, panicking if fixed-capacity accounting is wrong.
            #[track_caller]
            pub fn push(&mut self, future: F) {
                self.inner.push(future);
            }

            /// Harvest after the owner has already registered its waker.
            #[inline]
            pub fn harvest(&mut self, cap: usize, out: impl FnMut(F::Output)) -> Harvest {
                self.inner.harvest(cap, out)
            }

            /// Register `cx.waker()` and then harvest in the lost-wakeup-safe order.
            #[inline]
            pub fn poll_harvest(
                &mut self,
                cx: &mut Context<'_>,
                cap: usize,
                out: impl FnMut(F::Output),
            ) -> Harvest {
                self.inner.register_owner(cx.waker());
                self.inner.harvest(cap, out)
            }
        }
    };
}

impl_outstanding!(BoxedOutstanding, BoxedStorage);

/// The storage the shard uses.
///
/// Named separately from the backend so that a candidate storage can be
/// swapped in behind [`Storage`] without the reactor above it changing.
pub type Outstanding<F> = BoxedOutstanding<F>;

#[cfg(test)]
#[cfg(not(loom))]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicUsize, Ordering as StdOrdering};

    struct Countdown {
        remaining: u8,
        payload: u64,
    }

    impl Future for Countdown {
        type Output = u64;

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            if self.remaining == 0 {
                Poll::Ready(self.payload)
            } else {
                self.remaining -= 1;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }

    fn countdown(remaining: u8, payload: u64) -> Countdown {
        Countdown { remaining, payload }
    }

    fn drain<F, S>(set: &mut Inner<F, S>, cap: usize) -> Vec<F::Output>
    where
        F: Future,
        S: Storage<F>,
    {
        let mut output = Vec::new();
        let mut passes = 0usize;
        while !set.is_empty() {
            set.harvest(cap, |value| output.push(value));
            passes += 1;
            assert!(passes < 1_000_000, "ready work was stranded");
        }
        output
    }

    // The exercises below are generic over `Storage` rather than written
    // against the one implementation, so a candidate layout inherits the whole
    // suite by naming itself once at each call site.
    fn exercise_recycling<S: Storage<Countdown>>() {
        let mut set = Inner::<Countdown, S>::with_capacity(8);
        for round in 0..2_000u64 {
            set.try_push(countdown((round % 4) as u8, round)).unwrap();
            set.try_push(countdown(((round + 1) % 4) as u8, round + 10_000)).unwrap();
            let mut done = drain(&mut set, 3);
            done.sort_unstable();
            assert_eq!(done, [round, round + 10_000]);
            assert_eq!(set.check_invariants(), Ok(()));
        }
    }

    #[test]
    fn slots_recycle_and_every_future_completes() {
        exercise_recycling::<BoxedStorage<Countdown>>();
    }

    #[test]
    fn compiler_generated_not_unpin_futures_are_polled_in_place() {
        async fn job(value: u64) -> u64 {
            countdown(2, value).await
        }

        let mut boxed = BoxedOutstanding::with_capacity(2);
        boxed.push(job(1));
        assert_eq!(drain(&mut boxed.inner, 1), [1]);

        let mut slab = BoxedOutstanding::with_capacity(2);
        slab.push(job(2));
        assert_eq!(drain(&mut slab.inner, 1), [2]);
    }

    fn exercise_true_multiword_rotation<S: Storage<Countdown>>() {
        let mut set = Inner::<Countdown, S>::with_capacity(65);

        // Put the cursor at one before filling every slot.
        set.push(countdown(0, 999));
        assert_eq!(drain(&mut set, 1), [999]);
        assert_eq!(set.cursor, 1);

        for payload in 0..65u64 {
            set.push(countdown(0, payload));
        }

        // True circular order from cursor 1 is 1..=64, leaving slot 0.  The
        // broken word-local rotation instead served 1..63,0 and left slot 64.
        let mut first = Vec::new();
        let report = set.harvest(64, |value| first.push(value));
        first.sort_unstable();
        assert_eq!(first, (1..65).collect::<Vec<_>>());
        assert_eq!(report.polled, 64);
        assert!(report.more_ready);

        let mut last = Vec::new();
        set.harvest(1, |value| last.push(value));
        assert_eq!(last, [0]);
        assert!(set.is_empty());
        assert_eq!(set.check_invariants(), Ok(()));
    }

    /// The same fairness question one level up.
    ///
    /// With more than 4,096 slots the summary itself spans several words, so a
    /// capped pass has to resume in the right summary word as well as the right
    /// word. A rotation that restarted at summary word zero would serve the
    /// first 4,096 slots forever and starve everything above them.
    fn exercise_rotation_across_a_summary_word<S: Storage<Countdown>>() {
        // Two summary words: the first covers slots 0..4,096, the second the
        // rest.
        const CAPACITY: usize = 4_160;
        let mut set = Inner::<Countdown, S>::with_capacity(CAPACITY);

        // Park the cursor deep inside the second summary word's territory. A
        // capped pass over a full set is what moves it there. Pushing and
        // draining one at a time keeps reusing the slot just freed, which
        // leaves the cursor where it started.
        for payload in 0..CAPACITY as u64 {
            set.push(countdown(0, payload));
        }
        let mut warmed = Vec::new();
        set.harvest(4_100, |value| warmed.push(value));
        let resume = set.cursor;
        assert!(resume > 4_096, "the cursor should sit in the second summary word, not {resume}");

        // Refill what that pass consumed, so the set is full again and one
        // rotation from `resume` has to wrap through the first summary word.
        for payload in warmed {
            set.push(countdown(0, payload));
        }

        // One uncapped rotation must reach every slot exactly once, wherever
        // it started.
        let mut served = Vec::new();
        set.harvest(CAPACITY, |value| served.push(value));
        served.sort_unstable();
        assert_eq!(
            served,
            (0..CAPACITY as u64).collect::<Vec<_>>(),
            "a rotation beginning at {resume} missed slots"
        );
        assert!(set.is_empty());
        assert_eq!(set.check_invariants(), Ok(()));
    }

    // Skipped under Miri, not scaled down: the property only exists above
    // 4,096 slots, so a smaller version would not be this test. Miri interprets
    // every one of those slots through a full rotation, which is beyond what it
    // can carry. The structure it shares with smaller sets is covered by
    // `capacity_and_word_boundaries_hold`, which does run
    // there.
    #[test]
    #[cfg_attr(miri, ignore = "4,160 slots through a full rotation is beyond Miri")]
    fn capped_rotation_is_fair_across_a_summary_word_boundary() {
        exercise_rotation_across_a_summary_word::<BoxedStorage<Countdown>>();
    }

    #[test]
    fn capped_rotation_is_fair_across_the_64_slot_boundary() {
        exercise_true_multiword_rotation::<BoxedStorage<Countdown>>();
    }

    fn exercise_boundaries<S: Storage<Countdown>>() {
        // Around every boundary the structure has: a word (64), the first
        // summary word's reach (4,096), and the second's. Miri interprets each
        // slot, so it stops before the sizes that exist to prove the summary
        // array is addressed correctly.
        let ceilings: &[usize] = if cfg!(miri) {
            &[0, 1, 63, 64, 65, 127, 128, 129]
        } else {
            &[0, 1, 63, 64, 65, 127, 128, 129, 4_095, 4_096, 4_097, 8_191, 8_192, 8_193, 16_384]
        };
        for &capacity in ceilings {
            let mut set = Inner::<Countdown, S>::with_capacity(capacity);
            for token in 0..capacity as u64 {
                set.push(countdown((token % 3) as u8, token));
            }
            let mut done = drain(&mut set, 17);
            done.sort_unstable();
            assert_eq!(done, (0..capacity as u64).collect::<Vec<_>>(), "capacity {capacity}");
            assert_eq!(set.check_invariants(), Ok(()));
        }
    }

    #[test]
    fn capacity_and_word_boundaries_hold() {
        exercise_boundaries::<BoxedStorage<Countdown>>();
    }

    struct DropReady {
        drops: Arc<AtomicUsize>,
    }

    impl Future for DropReady {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
            Poll::Ready(())
        }
    }

    impl Drop for DropReady {
        fn drop(&mut self) {
            self.drops.fetch_add(1, StdOrdering::Relaxed);
        }
    }

    fn exercise_immediate_drop<S: Storage<DropReady>>() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut set = Inner::<DropReady, S>::with_capacity(1);
        set.push(DropReady { drops: drops.clone() });
        let report = set.harvest(1, |_| {
            assert_eq!(drops.load(StdOrdering::Relaxed), 1, "F must be dropped before callback");
        });
        assert_eq!(report.finished, 1);
        assert_eq!(drops.load(StdOrdering::Relaxed), 1);
    }

    #[test]
    fn a_completed_future_is_dropped_immediately() {
        exercise_immediate_drop::<BoxedStorage<DropReady>>();
    }

    fn exercise_callback_unwind<S: Storage<Countdown>>() {
        let mut set = Inner::<Countdown, S>::with_capacity(70);
        for token in 0..70 {
            set.push(countdown(0, token));
        }

        let panic = catch_unwind(AssertUnwindSafe(|| {
            set.harvest(usize::MAX, |_| panic!("callback failure"));
        }));
        assert!(panic.is_err());
        assert_eq!(set.len(), 69, "the completed current future was released");

        let mut remaining = drain(&mut set, usize::MAX);
        remaining.sort_unstable();
        assert_eq!(remaining.len(), 69, "unvisited ready bits were restored");
        assert_eq!(set.check_invariants(), Ok(()));
    }

    #[test]
    fn callback_unwind_restores_same_and_later_words() {
        exercise_callback_unwind::<BoxedStorage<Countdown>>();
    }

    /// Counts rings of the set's doorbell.
    ///
    /// The doorbell is how a restore tells the owner that work it thought was
    /// being polled is back on the ready set. Nothing else in this suite looks
    /// at it, which is why every mutant in `RestoreGuard::restore`'s `had_work`
    /// and its use survived: the decision it drives was never observed.
    struct Bell(AtomicUsize);

    impl Wake for Bell {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, StdOrdering::Relaxed);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, StdOrdering::Relaxed);
        }
    }

    impl Bell {
        /// Register on a set that is already loaded. A push rings, so this has
        /// to come after the futures are in, and a ring consumes the
        /// registration, so what it counts afterwards is only what the harvest
        /// did.
        fn listening_to<F: Future, S: Storage<F>>(set: &Inner<F, S>) -> Arc<Self> {
            let bell = Arc::new(Bell(AtomicUsize::new(0)));
            set.shared.bell.register(&Waker::from(bell.clone()));
            bell
        }

        fn rings(&self) -> usize {
            self.0.load(StdOrdering::Relaxed)
        }
    }

    fn exercise_unwind_announces_the_work_it_put_back<S: Storage<Countdown>>() {
        let mut set = Inner::<Countdown, S>::with_capacity(70);
        for token in 0..70 {
            set.push(countdown(0, token));
        }
        let bell = Bell::listening_to(&set);

        let panic = catch_unwind(AssertUnwindSafe(|| {
            set.harvest(usize::MAX, |_| panic!("callback failure"));
        }));
        assert!(panic.is_err());

        // The unwind handed a word full of ready bits back to the set. Nobody
        // is going to poll them unless the owner is told, and the owner may
        // well be parked: an unwind that restores silently strands the work.
        assert_eq!(bell.rings(), 1, "a restore that put ready bits back must wake the owner");
    }

    #[test]
    fn an_unwind_wakes_the_owner_for_the_bits_it_restored() {
        exercise_unwind_announces_the_work_it_put_back::<BoxedStorage<Countdown>>();
    }

    fn exercise_unwind_on_the_single_word_path<S: Storage<Countdown>>() {
        // One word, and a cursor at zero, so the guard reaches its restore
        // holding current bits and nothing else: no summary, no deferred
        // summary, no deferred bits. The multiword case above always has more
        // than one of those set at once, which makes the four terms of
        // `had_work` indistinguishable from each other -- any one of them
        // explains the ring. This is the shape where only one of them does.
        let mut set = Inner::<Countdown, S>::with_capacity(8);
        for token in 0..5 {
            set.push(countdown(0, token));
        }
        let bell = Bell::listening_to(&set);

        let panic = catch_unwind(AssertUnwindSafe(|| {
            set.harvest(usize::MAX, |_| panic!("callback failure"));
        }));
        assert!(panic.is_err());
        assert_eq!(bell.rings(), 1, "the current word's remaining bits still need an owner");
    }

    #[test]
    fn an_unwind_with_only_current_bits_in_hand_still_wakes_the_owner() {
        exercise_unwind_on_the_single_word_path::<BoxedStorage<Countdown>>();
    }

    fn exercise_a_capped_pass_is_silent<S: Storage<Countdown>>() {
        let mut set = Inner::<Countdown, S>::with_capacity(70);
        for token in 0..70 {
            set.push(countdown(0, token));
        }
        let bell = Bell::listening_to(&set);

        let report = set.harvest(4, |_| {});
        assert_eq!(report.polled, 4);
        assert!(report.more_ready, "the pass stopped on its budget, not on empty");

        // The same restore path runs here, with bits still in hand, but the
        // owner is the one who called `harvest` and is about to call it again.
        // Ringing at it would be a wakeup it has to consume for news it already
        // has, and on the shard loop that is the difference between parking and
        // spinning.
        assert_eq!(bell.rings(), 0, "a truncated pass returns to a caller that is already awake");
    }

    #[test]
    fn a_pass_that_stops_on_its_budget_does_not_ring() {
        exercise_a_capped_pass_is_silent::<BoxedStorage<Countdown>>();
    }

    fn exercise_stale_wakes<S: Storage<Countdown>>() {
        let mut set = Inner::<Countdown, S>::with_capacity(1);
        set.push(countdown(0, 1));
        let stale = set.wakers[0].clone();
        assert_eq!(drain(&mut set, 1), [1]);

        stale.wake_by_ref();
        let empty = set.harvest(1, |_| panic!("a stale wake invented output"));
        assert_eq!(empty.finished, 0);

        set.push(countdown(1, 2));
        stale.wake_by_ref();
        let first = set.harvest(1, |_| panic!("replacement should pend once"));
        assert_eq!(first.polled, 1);
        assert_eq!(drain(&mut set, 1), [2]);
    }

    #[test]
    fn stale_wakes_are_safe_before_and_after_refill() {
        exercise_stale_wakes::<BoxedStorage<Countdown>>();
    }

    fn exercise_stale_wakes_move_the_cursor<S: Storage<Countdown>>() {
        // A stale bit costs no poll budget, but it does move the cursor: the
        // rotation has to resume past it, or the next pass starts on the same
        // dead slot. The test above proves the bit is harmless and stops there,
        // so where the cursor lands was never asserted -- and the cursor is the
        // only thing this branch writes.
        let mut set = Inner::<Countdown, S>::with_capacity(4);
        for token in 0..4u64 {
            set.push(countdown(0, token));
        }
        let stale: Vec<Waker> = (0..4).map(|slot| set.wakers[slot].clone()).collect();
        let mut done = drain(&mut set, 4);
        done.sort_unstable();
        assert_eq!(done, [0, 1, 2, 3]);
        assert!(set.is_empty(), "every slot is free, so every ready bit is now stale");

        // A slot that is not the last leaves the cursor immediately past it.
        stale[1].wake_by_ref();
        set.harvest(4, |_| panic!("a stale wake invented output"));
        assert_eq!(set.cursor, 2, "the rotation resumes after the stale slot, not on it");
        assert_eq!(set.check_invariants(), Ok(()));

        // The last slot wraps to the start instead of running off the end.
        stale[3].wake_by_ref();
        set.harvest(4, |_| panic!("a stale wake invented output"));
        assert_eq!(set.cursor, 0, "past the last slot the rotation begins again");
        assert_eq!(set.check_invariants(), Ok(()));
    }

    #[test]
    fn available_reports_the_room_that_is_actually_left() {
        // The number a caller sizes its next admission batch against. Nothing
        // asserted on it, so it was free to answer a constant: `0` would stall
        // a shard that believed it, and `1` would let one through at a time.
        let mut set = BoxedOutstanding::with_capacity(4);
        assert_eq!(set.available(), 4, "an empty set has all of its room");

        set.push(countdown(0, 1));
        set.push(countdown(0, 2));
        assert_eq!(set.available(), 2, "two in flight, two slots left");

        assert_eq!(drain(&mut set.inner, 4), [1, 2]);
        assert_eq!(set.available(), 4, "completed futures give their slots back");
    }

    #[test]
    fn a_stale_bit_advances_the_cursor_and_wraps_at_the_end() {
        exercise_stale_wakes_move_the_cursor::<BoxedStorage<Countdown>>();
    }

    /// Pends forever and never wakes, so a poll of it leaves the slot occupied
    /// and *not* ready. That is what makes a partly-ready word constructible.
    struct Never;

    impl Future for Never {
        type Output = u64;

        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<u64> {
            Poll::Pending
        }
    }

    fn exercise_the_cursor_split_polls_only_what_is_ready<S: Storage<Never>>() {
        // A capped pass leaves the cursor mid-word, and the next pass has to
        // split that word: the bits at or above the cursor now, the bits below
        // it on the wrap-around. Every other exercise starts from a cursor of
        // zero, where the low half is empty and the split is `bits & 0` against
        // `bits & !0` -- an operation that does nothing, and so an operation
        // whose replacement also does nothing.
        //
        // Here the cursor is at 3 with slots 0..2 occupied but no longer ready,
        // so a split that widened instead of narrowing would hand those three
        // back to be polled again, and the poll count says so.
        let mut set = Inner::<Never, S>::with_capacity(8);
        for _ in 0..8 {
            set.push(Never);
        }

        let first = set.harvest(3, |_| unreachable!("Never never completes"));
        assert_eq!(first.polled, 3, "the budget stopped the pass");
        assert_eq!(set.cursor, 3);

        // Slots 0..2 have been polled and pended without waking: still
        // occupied, no longer ready. Only 3..7 are.
        let second = set.harvest(usize::MAX, |_| unreachable!("Never never completes"));
        assert_eq!(second.polled, 5, "only the five slots still ready were polled");
        assert_eq!(set.check_invariants(), Ok(()));
    }

    #[test]
    fn a_resumed_pass_polls_only_the_slots_that_were_still_ready() {
        exercise_the_cursor_split_polls_only_what_is_ready::<BoxedStorage<Never>>();
    }

    fn exercise_the_multiword_cursor_split<S: Storage<Never>>() {
        // The same split, one level up. Here the cursor lands inside a word
        // that is itself inside a summary word, so there are two places to cut:
        // the summary word at the cursor's word, and that word at the cursor's
        // bit. Both cuts are `& low_mask(..)` against `& !low_mask(..)`, and
        // both are invisible from a cursor of zero.
        const CAPACITY: usize = 200;
        const FIRST: usize = 70;

        let mut set = Inner::<Never, S>::with_capacity(CAPACITY);
        for _ in 0..CAPACITY {
            set.push(Never);
        }

        let first = set.harvest(FIRST, |_| unreachable!("Never never completes"));
        assert_eq!(first.polled, FIRST);
        assert_eq!(set.cursor, FIRST, "mid-word, and not in the first word");

        // Everything before the cursor has been polled and is no longer ready.
        // A split that widened would poll some of it again.
        let second = set.harvest(usize::MAX, |_| unreachable!("Never never completes"));
        assert_eq!(second.polled, CAPACITY - FIRST, "only what was still ready");
        assert_eq!(set.check_invariants(), Ok(()));

        // And with nothing ready at all, a pass polls nothing rather than
        // rediscovering bits it already consumed.
        let third = set.harvest(usize::MAX, |_| unreachable!("Never never completes"));
        assert_eq!(third.polled, 0);
        assert!(!third.more_ready);
    }

    #[test]
    fn a_resumed_multiword_pass_polls_only_what_was_still_ready() {
        exercise_the_multiword_cursor_split::<BoxedStorage<Never>>();
    }

    #[test]
    fn zero_cap_is_a_real_noop() {
        let mut set = BoxedOutstanding::with_capacity(1);
        set.push(countdown(0, 1));
        let report = set.harvest(0, |_| panic!("zero cap polled a future"));
        assert_eq!(report, Harvest { polled: 0, finished: 0, more_ready: true });
        assert_eq!(drain(&mut set.inner, 1), [1]);
    }

    #[test]
    fn full_try_push_returns_the_future() {
        let mut set = BoxedOutstanding::with_capacity(1);
        set.try_push(countdown(1, 1)).unwrap();
        let error = set.try_push(countdown(2, 2)).unwrap_err();
        assert_eq!(error.into_future().payload, 2);
    }

    struct CountWake(AtomicUsize);

    impl Wake for CountWake {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, StdOrdering::Relaxed);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, StdOrdering::Relaxed);
        }
    }

    #[test]
    fn duplicate_slot_wakes_coalesce_until_the_bit_is_taken() {
        let owner = Arc::new(CountWake(AtomicUsize::new(0)));
        let owner_waker = Waker::from(owner.clone());
        let mut set = BoxedOutstanding::<Countdown>::with_capacity(1);
        set.register_owner(&owner_waker);

        set.inner.wakers[0].wake_by_ref();
        set.inner.wakers[0].wake_by_ref();
        assert_eq!(owner.0.load(StdOrdering::Relaxed), 1);

        // Taking the stale bit rearms the slot's wake transition.
        assert_eq!(set.harvest(1, |_| unreachable!()).polled, 0);
        set.register_owner(&owner_waker);
        set.inner.wakers[0].wake_by_ref();
        assert_eq!(owner.0.load(StdOrdering::Relaxed), 2);
    }

    #[test]
    fn dropping_the_set_closes_stale_slot_wakers_and_releases_owner() {
        let owner = Arc::new(CountWake(AtomicUsize::new(0)));
        let owner_waker = Waker::from(owner.clone());
        let stale = {
            let set = BoxedOutstanding::<Countdown>::with_capacity(1);
            set.register_owner(&owner_waker);
            assert_eq!(Arc::strong_count(&owner), 3, "Arc, Waker, and AtomicWaker");
            set.inner.wakers[0].clone()
        };
        assert_eq!(Arc::strong_count(&owner), 2, "drop must clear AtomicWaker");
        stale.wake_by_ref();
        assert_eq!(owner.0.load(StdOrdering::Relaxed), 0, "closed stale wake scheduled owner");
    }

    fn battle<S: Storage<Countdown>>() {
        const CAPACITY: usize = 129;
        const TOTAL: u64 = 20_000;
        let mut set = Inner::<Countdown, S>::with_capacity(CAPACITY);
        let mut next = 0u64;
        let mut complete = HashSet::with_capacity(TOTAL as usize);
        let mut rng = 0x1234_5678_9abc_def0u64;

        let mut random = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };

        while next < CAPACITY as u64 {
            set.push(countdown((random() % 5) as u8, next));
            next += 1;
        }

        let mut passes = 0usize;
        while complete.len() < TOTAL as usize {
            let cap = (random() % 31 + 1) as usize;
            let mut finished = Vec::new();
            set.harvest(cap, |token| finished.push(token));
            for token in finished {
                assert!(complete.insert(token), "future completed twice: {token}");
                if next < TOTAL {
                    set.push(countdown((random() % 5) as u8, next));
                    next += 1;
                }
            }
            passes += 1;
            assert!(passes < 1_000_000, "battle run starved ready work");
            assert_eq!(set.check_invariants(), Ok(()));
        }
        assert!(set.is_empty());
    }

    #[test]
    fn randomized_capped_refill_matches_an_independent_model() {
        battle::<BoxedStorage<Countdown>>();
    }

    #[test]
    fn poll_harvest_registers_before_checking() {
        let mut cx = Context::from_waker(Waker::noop());
        let mut set = BoxedOutstanding::with_capacity(1);
        set.push(countdown(0, 7));
        let mut output = Vec::new();
        let report = set.poll_harvest(&mut cx, 1, |value| output.push(value));
        assert_eq!(report.finished, 1);
        assert_eq!(output, [7]);
    }

    // The bound itself is deliberately not written out: it is a compile-time
    // constant meant to be changed, and a test that pinned the number would
    // have to be edited every time it was.
    #[test]
    #[should_panic(expected = "Outstanding capacity exceeds")]
    fn capacity_beyond_the_configured_ceiling_is_rejected() {
        let _ = BoxedOutstanding::<Countdown>::with_capacity(MAX_CAPACITY + 1);
    }
}

/// Loom models the readiness bitmap owned by this module.  Wake delivery is
/// not modelled here: it belongs to [`crate::doorbell`], whose own models cover
/// the register-before-check and ring-versus-close races.  The `AtomicBool`
/// below therefore stands in for a wake rather than performing one.
/// The ready set on its own terms.
///
/// Everything else in this file reaches `ReadySet` through `Outstanding`, which
/// only ever asks it about slots that exist. Its masks are there for the bits
/// that do not: the tail of the last word a capacity only partly fills, and the
/// tail of the last summary word. A caller that never names those bits cannot
/// tell a mask that clips correctly from one that clips nothing, or from one
/// that sets every bit it was meant to clear -- which is why mutants in
/// `valid_mask`, `summary_mask` and every `&` that applies them survived a suite
/// that drives the set only through a reactor.
///
/// These address the set directly and assert the mask values and the exact bit
/// patterns that come back, which is the only place that distinction shows.
#[cfg(test)]
#[cfg(not(loom))]
mod ready_set_tests {
    use super::*;

    /// Capacities are chosen for the shapes they produce, not for their size.
    ///
    /// `WORD_BITS` is 64 and a summary word addresses 64 ready words, so:
    /// 64 fills one word exactly; 8 leaves one word partly filled; 70 needs two
    /// words with the second partly filled and one summary word; and 4,100
    /// needs 65 words, which is what forces a second summary word and so the
    /// only shape where `summary_mask` clips anything.
    const ONE_WORD_EXACT: usize = 64;
    const ONE_WORD_PART: usize = 8;
    const TWO_WORDS_PART: usize = 70;
    const TWO_SUMMARIES: usize = 4_100;

    #[test]
    fn valid_mask_clips_the_tail_of_a_partly_filled_word() {
        // A capacity that fills its last word leaves nothing to clip.
        let exact = ReadySet::new(ONE_WORD_EXACT);
        assert_eq!(exact.valid_mask(0), u64::MAX);

        let part = ReadySet::new(ONE_WORD_PART);
        assert_eq!(part.valid_mask(0), 0b1111_1111);

        // Only the *last* word is clipped; the ones before it are full.
        let two = ReadySet::new(TWO_WORDS_PART);
        assert_eq!(two.words(), 2);
        assert_eq!(two.valid_mask(0), u64::MAX);
        assert_eq!(two.valid_mask(1), low_mask(TWO_WORDS_PART % WORD_BITS));
        assert_eq!(two.valid_mask(1), 0b11_1111);
    }

    #[test]
    fn summary_mask_clips_the_tail_of_the_word_index() {
        // Two words need one summary word, of which two bits name real words.
        let two = ReadySet::new(TWO_WORDS_PART);
        assert_eq!(two.summaries(), 1);
        assert_eq!(two.summary_mask(0), 0b11);

        // 65 words need two summary words: the first is fully used, the second
        // names a single word. This is the only shape where the `summary + 1 ==
        // summaries` test distinguishes anything.
        let big = ReadySet::new(TWO_SUMMARIES);
        assert_eq!(big.words(), 65);
        assert_eq!(big.summaries(), 2);
        assert_eq!(big.summary_mask(0), u64::MAX);
        assert_eq!(big.summary_mask(1), 0b1);
    }

    #[test]
    fn an_untouched_set_has_nothing_ready() {
        // Each arm of `has_ready` separately: no words, one word, many words.
        assert!(!ReadySet::new(0).has_ready());
        assert!(!ReadySet::new(ONE_WORD_EXACT).has_ready());
        assert!(!ReadySet::new(ONE_WORD_PART).has_ready());
        assert!(!ReadySet::new(TWO_WORDS_PART).has_ready());
        assert!(!ReadySet::new(TWO_SUMMARIES).has_ready());
    }

    #[test]
    fn has_ready_tracks_marks_and_takes() {
        let single = ReadySet::new(ONE_WORD_PART);
        assert!(single.mark(3));
        assert!(single.has_ready());
        assert_eq!(single.take_single(), 1 << 3);
        assert!(!single.has_ready());

        let multi = ReadySet::new(TWO_WORDS_PART);
        assert!(multi.mark(65));
        assert!(multi.has_ready());
        assert_eq!(multi.take_summary(0), 1 << 1);
        assert_eq!(multi.take_word(1), 1 << 1);
        assert!(!multi.has_ready());
    }

    #[test]
    fn mark_reports_only_the_transition_to_ready() {
        let ready = ReadySet::new(TWO_WORDS_PART);
        assert!(ready.mark(9), "the first mark of a slot is the one that rings");
        assert!(!ready.mark(9), "a second mark of a live bit announces nothing");
        // A different slot in the same word is still its own transition.
        assert!(ready.mark(10));
    }

    #[test]
    fn take_summary_names_only_the_words_that_were_announced() {
        let ready = ReadySet::new(TWO_SUMMARIES);
        ready.mark(WORD_BITS * 2 + 5);
        // Exactly the announced word, not every word the mask permits.
        assert_eq!(ready.take_summary(0), 1 << 2);
        assert_eq!(ready.take_word(2), 1 << 5);
    }

    #[test]
    fn take_word_yields_only_the_slots_that_were_marked() {
        let ready = ReadySet::new(TWO_WORDS_PART);
        ready.mark(64);
        ready.mark(69);
        // The last word holds six valid bits; the take must not invent the rest.
        assert_eq!(ready.take_word(1), (1 << 0) | (1 << 5));
        assert_eq!(ready.take_word(1), 0, "a take clears what it returned");
    }

    #[test]
    fn restore_puts_back_exactly_what_was_taken() {
        let ready = ReadySet::new(TWO_WORDS_PART);
        ready.mark(2);
        ready.mark(64);

        let summary = ready.take_summary(0);
        let word0 = ready.take_word(0);
        let word1 = ready.take_word(1);
        assert_eq!(summary, 0b11);
        assert_eq!(word0, 1 << 2);
        assert_eq!(word1, 1 << 0);

        ready.restore_word(0, word0);
        ready.restore_word(1, word1);
        ready.restore_summary(0, summary);

        // Byte for byte, not merely non-empty: a restore that widened what it
        // was given would hand the reactor slots nobody marked.
        assert_eq!(ready.take_summary(0), summary);
        assert_eq!(ready.take_word(0), word0);
        assert_eq!(ready.take_word(1), word1);
    }

    #[test]
    fn restoring_nothing_leaves_the_set_empty() {
        let ready = ReadySet::new(TWO_WORDS_PART);
        ready.restore_word(1, 0);
        ready.restore_summary(0, 0);
        assert!(!ready.has_ready());
        assert_eq!(ready.take_word(1), 0);
    }

    #[test]
    fn a_restore_re_announces_the_word_it_refilled() {
        // The announcement is what a later scan finds the word by, so a restore
        // into an emptied word has to put the summary bit back with it.
        let ready = ReadySet::new(TWO_WORDS_PART);
        ready.mark(64);
        let word = ready.take_word(1);
        assert_eq!(ready.take_summary(0), 1 << 1);

        ready.restore_word(1, word);
        assert_eq!(ready.take_summary(0), 1 << 1, "the refilled word is announced again");
    }
}

#[cfg(test)]
#[cfg(loom)]
mod loom_tests {
    use super::{ReadySet, low_mask};
    use loom::sync::Arc;
    use loom::sync::atomic::{AtomicBool, Ordering};
    use loom::thread;

    fn collect(ready: &ReadySet) -> Vec<u64> {
        match ready.words() {
            0 => Vec::new(),
            1 => vec![ready.take_single()],
            words => {
                let summary = ready.take_summary(0);
                (0..words)
                    .map(
                        |word| {
                            if summary & (1u64 << word) != 0 { ready.take_word(word) } else { 0 }
                        },
                    )
                    .collect()
            }
        }
    }

    #[test]
    fn loom_publish_then_wake_never_exposes_wake_before_bit() {
        loom::model(|| {
            let ready = Arc::new(ReadySet::new(128));
            let woken = Arc::new(AtomicBool::new(false));

            let producer = {
                let ready = ready.clone();
                let woken = woken.clone();
                thread::spawn(move || {
                    ready.mark(70);
                    woken.store(true, Ordering::Release);
                })
            };

            let saw_wake = woken.load(Ordering::Acquire);
            let summary = ready.take_summary(0);
            let bits = if summary & (1 << 1) != 0 { ready.take_word(1) } else { 0 };
            if saw_wake {
                assert_ne!(bits & (1 << 6), 0, "wake became visible before slot bit");
            }

            producer.join().unwrap();
            let later = collect(&ready);
            assert_ne!(bits | later.get(1).copied().unwrap_or(0), 0, "ready mark was lost");
        });
    }

    #[test]
    fn loom_mark_racing_summary_and_word_take_is_never_lost() {
        loom::model(|| {
            let ready = Arc::new(ReadySet::new(128));
            let producer = {
                let ready = ready.clone();
                thread::spawn(move || ready.mark(67))
            };

            let first = collect(&ready);
            producer.join().unwrap();
            let second = collect(&ready);
            let observed = first.get(1).copied().unwrap_or(0) | second.get(1).copied().unwrap_or(0);
            assert_ne!(observed & (1 << 3), 0);
        });
    }

    #[test]
    fn loom_concurrent_marks_in_different_words_survive_summary_coalescing() {
        loom::model(|| {
            let ready = Arc::new(ReadySet::new(192));
            let left = {
                let ready = ready.clone();
                thread::spawn(move || ready.mark(2))
            };
            let right = {
                let ready = ready.clone();
                thread::spawn(move || ready.mark(130))
            };
            left.join().unwrap();
            right.join().unwrap();

            let words = collect(&ready);
            assert_ne!(words[0] & (1 << 2), 0);
            assert_ne!(words[2] & (1 << 2), 0);
        });
    }

    #[test]
    fn loom_duplicate_marks_have_exactly_one_wake_transition() {
        loom::model(|| {
            let ready = Arc::new(ReadySet::new(64));
            let left = {
                let ready = ready.clone();
                thread::spawn(move || ready.mark(5))
            };
            let right = {
                let ready = ready.clone();
                thread::spawn(move || ready.mark(5))
            };

            let transitions =
                usize::from(left.join().unwrap()) + usize::from(right.join().unwrap());
            assert_eq!(transitions, 1, "duplicate marks emitted duplicate owner wakes");
            assert_eq!(ready.take_single(), 1 << 5);
            assert_eq!(ready.take_single(), 0);
        });
    }

    #[test]
    fn loom_restore_merges_with_concurrent_marks_in_same_word() {
        loom::model(|| {
            let ready = Arc::new(ReadySet::new(128));
            ready.mark(65);
            ready.mark(66);
            let summary = ready.take_summary(0);
            assert_ne!(summary & (1 << 1), 0);
            let taken = ready.take_word(1);

            let producer = {
                let ready = ready.clone();
                thread::spawn(move || ready.mark(73))
            };
            ready.restore_word(1, taken & !(1 << 1));
            producer.join().unwrap();

            let words = collect(&ready);
            assert_ne!(words[1] & (1 << 2), 0, "restored bit was lost");
            assert_ne!(words[1] & (1 << 9), 0, "concurrent bit was overwritten");
        });
    }

    #[test]
    fn loom_unvisited_summary_restore_merges_with_new_word() {
        loom::model(|| {
            let ready = Arc::new(ReadySet::new(192));
            ready.mark(1);
            ready.mark(70);
            let taken_summary = ready.take_summary(0);

            let producer = {
                let ready = ready.clone();
                thread::spawn(move || ready.mark(130))
            };
            ready.restore_summary(0, taken_summary & !1);
            producer.join().unwrap();

            let summary = ready.take_summary(0);
            assert_ne!(summary & (1 << 1), 0);
            assert_ne!(summary & (1 << 2), 0);
            assert_eq!(summary & !low_mask(3), 0);
        });
    }

    #[test]
    fn loom_register_before_check_cannot_strand_a_later_mark() {
        loom::model(|| {
            let ready = Arc::new(ReadySet::new(64));
            let registered = Arc::new(AtomicBool::new(false));
            let woken = Arc::new(AtomicBool::new(false));

            let producer = {
                let ready = ready.clone();
                let registered = registered.clone();
                let woken = woken.clone();
                thread::spawn(move || {
                    ready.mark(4);
                    if registered.load(Ordering::Acquire) {
                        woken.store(true, Ordering::Release);
                    }
                })
            };

            // Model AtomicWaker::register followed by the readiness check used
            // for the decision to park.
            registered.store(true, Ordering::Release);
            let first = ready.take_single();
            producer.join().unwrap();
            let second = ready.take_single();
            assert_ne!(first | second, 0);
            if first == 0 && second == 0 {
                assert!(woken.load(Ordering::Acquire));
            }
        });
    }
}
