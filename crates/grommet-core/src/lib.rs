//! Key-affine fair scheduling as a pure data structure.
//!
//! There is no async, no clock, no IO, no thread and no allocation policy here
//! beyond the queues this module owns. Time is an explicit monotonic
//! [`Duration`] supplied by the caller, and work items are opaque payloads.
//!
//! # Fairness model
//!
//! Each key has at most one in-flight item, so a key with a million queued
//! items occupies exactly one dispatch slot, the same as a key with one. Given
//! that, fairness reduces to *which ready key is dispatched next?*
//!
//! A completed key is never greedily re-dispatched. Both new arrivals and
//! completions place a key at the BACK of a round-robin ready ring, and
//! dispatch always pops from the FRONT. A key sitting at position `k` is
//! therefore dispatched within `k` dispatches — a strict bound, hence
//! starvation-free. One key cannot monopolize a shard.
//!
//! There is one ring per work class, each with its own in-flight budget, so a
//! flood of one class cannot starve another. A key routes to a ring based on
//! the class of its CURRENT queue head, so a key with mixed classes moves
//! between rings in FIFO order as it drains.
//!
//! # Trusting the caller
//!
//! Everything this module knows about a payload — its affine key, its work
//! class and its optional deadline — is stamped once by the caller at
//! admission and never recomputed. A caller-supplied trait implementation that
//! answered inconsistently on a second call would otherwise desynchronize the
//! ready rings from the per-key queues, so the opportunity is removed rather
//! than documented away.
//!
//! Unsafe code is confined to the queue slab, which documents and
//! debug-asserts the one invariant it relies on.

#![deny(unsafe_code)]

use ahash::AHashMap;
#[cfg(not(coverage))]
use grommet_macros::always;
use std::collections::VecDeque;
use std::hash::Hash;
use std::time::Duration;

mod queue;
use queue::{List, Slab};

/// Index of a work class, in `0..CLASSES`.
pub type ClassId = u8;

/// Per-shard scheduling limits.
///
/// `max_inflight` is indexed by [`ClassId`]: each class dispatches into its own
/// budget, so saturating one class never blocks another.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config<const CLASSES: usize = 2> {
    /// Maximum simultaneously dispatched items per class.
    pub max_inflight: [usize; CLASSES],
    /// Maximum queued plus in-flight items. Admission above this is the
    /// caller's responsibility to refuse; the value is exposed so the caller
    /// can gate its own mailbox against it.
    pub max_pending: usize,
    /// Soft cap on resident keys. Eviction is bounded work per sweep, so the
    /// cap can be exceeded transiently when every candidate is busy.
    pub max_resident: Option<usize>,
    /// How long a key must sit idle before it becomes an eviction candidate.
    pub evict_after: Duration,
    /// Maximum eviction candidates examined per sweep.
    pub evict_iters: usize,
    /// Queue slab entries reserved up front. The slab never exceeds the peak
    /// number of simultaneously queued items, which `max_pending` bounds, so
    /// reserving that much makes the steady state allocation-free at the cost
    /// of the memory up front.
    pub queue_reserve: usize,
}

impl<const CLASSES: usize> Config<CLASSES> {
    /// A configuration with per-class in-flight budgets and defaults elsewhere.
    pub fn new(max_inflight: [usize; CLASSES]) -> Self {
        Self {
            max_inflight,
            max_pending: 8192,
            max_resident: None,
            evict_after: Duration::from_secs(60),
            evict_iters: 256,
            queue_reserve: 1024,
        }
    }
}

/// A work item being admitted, with its scheduling metadata already stamped.
pub struct Admit<K, P> {
    pub key: K,
    pub class: ClassId,
    /// Monotonic time after which this item is dropped instead of dispatched.
    pub expires_at: Option<Duration>,
    pub payload: P,
}

/// A work item that the scheduler has granted exclusive ownership of its key.
pub struct Dispatch<K, P, S> {
    pub key: K,
    pub class: ClassId,
    /// Resident state, moved out of the scheduler for the duration of the work.
    /// `None` means the key has no state resident and the caller must load it.
    pub state: Option<S>,
    pub payload: P,
}

/// What became of a key's state once its work finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Disposition<S> {
    /// Keep this state resident for the next dispatch of the key.
    Keep(S),
    /// Discard any resident state: the next dispatch must reload it. This is
    /// the correct answer whenever an operation's outcome is unknown, since a
    /// stale in-memory value is worse than no value at all.
    Drop,
}

impl<S> Disposition<S> {
    pub fn into_option(self) -> Option<S> {
        match self {
            Self::Keep(state) => Some(state),
            Self::Drop => None,
        }
    }
}

/// The result of a dispatched item, returned to the scheduler.
pub struct Completion<K, S> {
    pub key: K,
    pub class: ClassId,
    pub state: Disposition<S>,
}

/// Instantaneous scheduler gauges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Snapshot<const CLASSES: usize = 2> {
    pub inflight: [usize; CLASSES],
    pub ready: [usize; CLASSES],
    pub pending: usize,
    pub resident: usize,
    pub evicting: usize,
    /// High-water mark of simultaneously queued items, in slab entries. The
    /// slab never shrinks, so this is what `Config::queue_reserve` should be
    /// set to for an allocation-free steady state.
    pub queue_capacity: usize,
}

// `[T; N]: Default` only covers `N <= 32`, so the class count cannot rely on it.
impl<const CLASSES: usize> Default for Snapshot<CLASSES> {
    fn default() -> Self {
        Self {
            inflight: [0; CLASSES],
            ready: [0; CLASSES],
            pending: 0,
            resident: 0,
            evicting: 0,
            queue_capacity: 0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Presence {
    Idle,
    Ready(ClassId),
    InFlight,
    /// State has been handed to the caller for flushing. The key is quiesced:
    /// arriving work queues behind the flush rather than racing it.
    Evicting,
}

struct Item<P> {
    class: ClassId,
    expires_at: Option<Duration>,
    payload: P,
}

/// Per-key bookkeeping. Its size is independent of the work item type, because
/// queued items live in the shard-wide slab rather than in the slot, which
/// keeps the key map compact when a shard holds many keys.
struct Slot<S> {
    resident: Option<S>,
    queue: List,
    presence: Presence,
    last_touch: Duration,
}

impl<S> Slot<S> {
    fn cold(now: Duration) -> Self {
        Self { resident: None, queue: List::default(), presence: Presence::Idle, last_touch: now }
    }
}

/// A key-affine, class-fair scheduler over opaque payloads `P` and per-key
/// state `S`.
pub struct Scheduler<K, P, S, const CLASSES: usize = 2> {
    cfg: Config<CLASSES>,
    keys: AHashMap<K, Slot<S>>,
    slab: Slab<Item<P>>,
    ready: [VecDeque<K>; CLASSES],
    eviction: VecDeque<(K, Duration)>,
    expired: VecDeque<(K, P)>,
    inflight: [usize; CLASSES],
    pending: usize,
    evicting: usize,
}

impl<K, P, S, const CLASSES: usize> Scheduler<K, P, S, CLASSES>
where
    K: Copy + Eq + Hash,
{
    pub fn new(cfg: Config<CLASSES>) -> Self {
        Self {
            cfg,
            keys: AHashMap::new(),
            slab: Slab::with_capacity(cfg.queue_reserve),
            ready: std::array::from_fn(|_| VecDeque::new()),
            eviction: VecDeque::new(),
            expired: VecDeque::new(),
            inflight: [0; CLASSES],
            pending: 0,
            evicting: 0,
        }
    }

    pub fn config(&self) -> &Config<CLASSES> {
        &self.cfg
    }

    pub fn max_pending(&self) -> usize {
        self.cfg.max_pending
    }

    /// Queued plus in-flight items across every class.
    pub fn pending(&self) -> usize {
        self.pending
    }

    pub fn is_saturated(&self) -> bool {
        self.pending >= self.cfg.max_pending
    }

    /// Queue an item behind its key. The caller is responsible for refusing
    /// admission above [`Config::max_pending`]; this is where backpressure is
    /// applied, and the scheduler deliberately does not decide the policy.
    pub fn admit(&mut self, item: Admit<K, P>, now: Duration) {
        let Admit { key, class, expires_at, payload } = item;
        debug_assert!((class as usize) < CLASSES, "class {class} is outside 0..{CLASSES}");
        self.pending += 1;
        let slot = self.keys.entry(key).or_insert_with(|| Slot::cold(now));
        slot.last_touch = now;
        self.slab.push_back(&mut slot.queue, Item { class, expires_at, payload });
        // A key that is in-flight, already ready, or quiescing for eviction
        // keeps its position; only an idle key joins a ring, and its queue was
        // empty, so the item just pushed is the head whose class decides.
        let joins = match slot.presence {
            Presence::Idle => {
                slot.presence = Presence::Ready(class);
                true
            }
            Presence::Ready(_) | Presence::InFlight | Presence::Evicting => false,
        };
        if joins {
            self.ready[class as usize].push_back(key);
        }
    }

    /// Dispatch the next ready item of `class`, taking exclusive ownership of
    /// its key. Items whose deadline has passed are discarded on the way and
    /// can be collected with [`Scheduler::pop_expired`].
    pub fn next(&mut self, class: ClassId, now: Duration) -> Option<Dispatch<K, P, S>> {
        let index = class as usize;
        if self.inflight[index] >= self.cfg.max_inflight[index] {
            return None;
        }
        loop {
            let key = self.ready[index].pop_front()?;
            let slot = self.keys.get_mut(&key).expect("ready key has a slot");

            // Drop expired heads. Disjoint field borrows keep this legal while
            // `slot` is live, so no scratch buffer is needed.
            let mut taken = None;
            while let Some(head) = self.slab.front(&slot.queue) {
                if head.class != class {
                    break;
                }
                if head.expires_at.is_some_and(|deadline| deadline <= now) {
                    let item =
                        self.slab.pop_front(&mut slot.queue).expect("front was just observed");
                    self.pending -= 1;
                    self.expired.push_back((key, item.payload));
                    continue;
                }
                taken = self.slab.pop_front(&mut slot.queue);
                break;
            }

            match taken {
                Some(item) => {
                    slot.presence = Presence::InFlight;
                    slot.last_touch = now;
                    let state = slot.resident.take();
                    self.inflight[index] += 1;
                    return Some(Dispatch { key, class, state, payload: item.payload });
                }
                None => {
                    // Everything of this class expired. The key now heads a
                    // different ring, or has nothing left at all. Either way it
                    // cannot rejoin the ring being drained, so this terminates.
                    let target = Self::settle(&self.slab, slot);
                    debug_assert!(target != Some(class));
                    self.place(key, target, now);
                }
            }
        }
    }

    /// Take an item that was discarded at dispatch because its deadline had
    /// passed. The caller owns telling whoever submitted it.
    pub fn pop_expired(&mut self) -> Option<(K, P)> {
        self.expired.pop_front()
    }

    /// Return a key's state and re-place it in its ring, or retire it.
    pub fn complete(&mut self, completion: Completion<K, S>, now: Duration) {
        let Completion { key, class, state } = completion;
        self.inflight[class as usize] -= 1;
        self.pending -= 1;
        let slot = self.keys.get_mut(&key).expect("completed key has a slot");
        debug_assert_eq!(slot.presence, Presence::InFlight);
        slot.resident = state.into_option();
        slot.last_touch = now;
        let target = Self::settle(&self.slab, slot);
        self.place(key, target, now);
    }

    /// Collect keys whose resident state should be flushed and released,
    /// appending `(key, state)` pairs to `out`. Each collected key is quiesced
    /// until [`Scheduler::finish_evict`] is called for it: work may queue
    /// behind the flush, but nothing dispatches, so a write-back flush cannot
    /// race a reload of the same key.
    pub fn evict(&mut self, now: Duration, out: &mut Vec<(K, S)>) {
        for _ in 0..self.cfg.evict_iters {
            let Some(&(key, touched)) = self.eviction.front() else {
                break;
            };
            let idle_long_enough = now.saturating_sub(touched) >= self.cfg.evict_after;
            // A quiescing key still occupies a map entry but its state is
            // already being flushed, so it must not count against the cap or a
            // single sweep would evict far past it.
            let resident = self.keys.len() - self.evicting;
            let over_capacity = self.cfg.max_resident.is_some_and(|max| resident > max);
            if !idle_long_enough && !over_capacity {
                break;
            }
            self.eviction.pop_front();
            self.release(key, touched, out);
        }
    }

    /// Flush and release every resident key at once, whatever its idle time and
    /// whatever the capacity cap says, ignoring [`Config::evict_iters`].
    ///
    /// This is the shutdown path. Resident state is a write-back cache: the
    /// scheduler holds the only copy between dispatches, so a shard that stops
    /// without draining it discards writes the processor was told it could keep.
    /// The keys are quiesced exactly as [`Scheduler::evict`] quiesces them, so
    /// each one still needs its [`Scheduler::finish_evict`].
    pub fn evict_all(&mut self, out: &mut Vec<(K, S)>) {
        while let Some((key, touched)) = self.eviction.pop_front() {
            self.release(key, touched, out);
        }
    }

    /// Quiesce one eviction candidate and hand back its state to flush, or drop
    /// the key outright when it has none.
    fn release(&mut self, key: K, touched: Duration, out: &mut Vec<(K, S)>) {
        let Some(slot) = self.keys.get_mut(&key) else {
            return;
        };
        // A stale candidate: the key was touched after this entry was recorded,
        // so a later entry covers it.
        if slot.presence != Presence::Idle || slot.last_touch > touched {
            return;
        }
        match slot.resident.take() {
            Some(state) => {
                slot.presence = Presence::Evicting;
                self.evicting += 1;
                out.push((key, state));
            }
            None => {
                self.keys.remove(&key);
            }
        }
    }

    /// Release a key quiesced by [`Scheduler::evict`] once its flush finished.
    pub fn finish_evict(&mut self, key: K, now: Duration) {
        self.evicting -= 1;
        let Some(slot) = self.keys.get_mut(&key) else {
            debug_assert!(false, "finished eviction for an unknown key");
            return;
        };
        debug_assert_eq!(slot.presence, Presence::Evicting);
        if slot.queue.is_empty() {
            self.keys.remove(&key);
            return;
        }
        // Work arrived during the flush. The state is gone, so the next
        // dispatch reloads it.
        let target = Self::settle(&self.slab, slot);
        self.place(key, target, now);
    }

    pub fn snapshot(&self) -> Snapshot<CLASSES> {
        Snapshot {
            inflight: self.inflight,
            ready: std::array::from_fn(|class| self.ready[class].len()),
            pending: self.pending,
            resident: self.keys.len(),
            evicting: self.evicting,
            queue_capacity: self.slab.capacity(),
        }
    }

    /// Set a slot's presence from its queue head, returning the ring it should
    /// join, or `None` when it has become idle.
    fn settle(slab: &Slab<Item<P>>, slot: &mut Slot<S>) -> Option<ClassId> {
        match slab.front(&slot.queue) {
            Some(head) => {
                slot.presence = Presence::Ready(head.class);
                Some(head.class)
            }
            None => {
                slot.presence = Presence::Idle;
                None
            }
        }
    }

    fn place(&mut self, key: K, target: Option<ClassId>, now: Duration) {
        match target {
            Some(class) => self.ready[class as usize].push_back(key),
            None => self.eviction.push_back((key, now)),
        }
    }

    #[cfg(coverage)]
    pub fn check_invariants(&self) -> Result<(), &'static str> {
        Ok(())
    }

    /// Verify every structural invariant. Linear in resident keys plus total
    /// ring length, with no allocation, so it is affordable under a debug
    /// assertion on every reactor turn — which is where it earns its keep.
    #[cfg(not(coverage))]
    pub fn check_invariants(&self) -> Result<(), &'static str> {
        let queued: usize = self.keys.values().map(|slot| slot.queue.len()).sum();
        let inflight: usize = self.inflight.iter().sum();
        if !always!(self.pending == queued + inflight) {
            return Err("pending != queued + in-flight");
        }
        let owned = self.keys.values().filter(|slot| slot.presence == Presence::InFlight).count();
        if !always!(owned == inflight) {
            return Err("in-flight counters disagree with key ownership");
        }
        let quiesced =
            self.keys.values().filter(|slot| slot.presence == Presence::Evicting).count();
        if !always!(quiesced == self.evicting) {
            return Err("evicting counter disagrees with key ownership");
        }
        // Ring membership is checked outward from the rings and then reconciled
        // by count, rather than by asking each key which rings hold it. The
        // latter reads every ring once per key — quadratic, and at a few
        // thousand resident keys it costs more per item than the scheduling it
        // is guarding, which put a ceiling on how large a simulation could run.
        let mut listed = 0;
        for (class, ring) in self.ready.iter().enumerate() {
            let class = class as ClassId;
            for key in ring {
                listed += 1;
                let Some(slot) = self.keys.get(key) else {
                    return Err("a ready key has no slot");
                };
                if slot.presence != Presence::Ready(class) {
                    return Err("ready key is inconsistent with the ring holding it");
                }
                // Covers an empty queue too: it has no head to disagree.
                if self.slab.front(&slot.queue).map(|item| item.class) != Some(class) {
                    return Err("ready key is inconsistent with its queue head");
                }
            }
        }
        // A key listed in two rings, or twice in one, would need a second
        // `Ready` slot to balance this — and a slot has one presence.
        let ready =
            self.keys.values().filter(|slot| matches!(slot.presence, Presence::Ready(_))).count();
        if !always!(listed == ready) {
            return Err("ready rings disagree with key presence");
        }

        for slot in self.keys.values() {
            match slot.presence {
                Presence::Idle if !slot.queue.is_empty() => {
                    return Err("idle key is queued");
                }
                Presence::InFlight | Presence::Evicting if slot.resident.is_some() => {
                    return Err("a key that gave up its state still holds it");
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IO: ClassId = 0;
    const CPU: ClassId = 1;

    type Book = Scheduler<u64, &'static str, u64, 2>;

    fn config() -> Config<2> {
        Config {
            max_inflight: [1, 1],
            max_pending: 32,
            max_resident: None,
            evict_after: Duration::from_secs(10),
            evict_iters: 32,
            queue_reserve: 8,
        }
    }

    fn item(key: u64, class: ClassId) -> Admit<u64, &'static str> {
        Admit { key, class, expires_at: None, payload: "work" }
    }

    fn expiring(key: u64, class: ClassId, at: Duration) -> Admit<u64, &'static str> {
        Admit { key, class, expires_at: Some(at), payload: "work" }
    }

    fn finish(dispatch: Dispatch<u64, &'static str, u64>) -> Completion<u64, u64> {
        Completion {
            key: dispatch.key,
            class: dispatch.class,
            state: Disposition::Keep(dispatch.state.unwrap_or_default()),
        }
    }

    #[test]
    fn a_backlogged_key_rotates_behind_every_other_ready_key() {
        let mut book = Book::new(config());
        let now = Duration::ZERO;
        book.admit(item(1, IO), now);
        book.admit(item(1, IO), now);
        book.admit(item(2, IO), now);

        let whale = book.next(IO, now).unwrap();
        assert_eq!(whale.key, 1);
        book.complete(finish(whale), now);
        assert_eq!(book.next(IO, now).unwrap().key, 2, "the backlog must not be served twice");
        assert_eq!(book.check_invariants(), Ok(()));
    }

    #[test]
    fn dispatch_position_bounds_starvation_under_sustained_load() {
        let mut book = Book::new(config());
        let now = Duration::ZERO;
        for key in 0..8 {
            book.admit(item(key, IO), now);
        }
        // The hot key keeps arriving; the strict bound says every other key is
        // still served within one rotation of the ring.
        let mut seen = [false; 8];
        for _ in 0..8 {
            book.admit(item(0, IO), now);
            let dispatch = book.next(IO, now).unwrap();
            seen[dispatch.key as usize] = true;
            book.complete(finish(dispatch), now);
        }
        assert!(seen.into_iter().all(|served| served), "a key starved behind the hot key");
    }

    #[test]
    fn class_budgets_are_independent_and_a_key_serializes_across_them() {
        let mut book = Book::new(config());
        let now = Duration::ZERO;
        book.admit(item(1, IO), now);
        book.admit(item(1, IO), now);
        book.admit(item(2, CPU), now);

        let io = book.next(IO, now).unwrap();
        let cpu = book.next(CPU, now).unwrap();
        assert!(book.next(IO, now).is_none(), "key 1 already owns its single in-flight slot");
        assert!(book.next(CPU, now).is_none(), "the compute budget is saturated");
        assert_eq!(book.check_invariants(), Ok(()));

        book.complete(finish(io), now);
        assert!(book.next(IO, now).is_some());
        assert!(book.next(CPU, now).is_none(), "completing IO must not free compute budget");
        book.complete(finish(cpu), now);
    }

    #[test]
    fn a_mixed_key_moves_between_rings_in_fifo_order() {
        let mut book = Book::new(config());
        let now = Duration::ZERO;
        book.admit(item(9, IO), now);
        book.admit(item(9, CPU), now);
        assert!(book.next(CPU, now).is_none(), "the compute item is behind the IO item");

        let first = book.next(IO, now).unwrap();
        book.complete(finish(first), now);
        assert_eq!(book.next(CPU, now).unwrap().key, 9);
    }

    #[test]
    fn state_ownership_transfers_to_exactly_one_dispatch() {
        let mut book = Book::new(config());
        let now = Duration::ZERO;
        book.admit(item(3, IO), now);
        let first = book.next(IO, now).unwrap();
        assert_eq!(first.state, None, "a cold key carries no state");
        book.complete(Completion { key: 3, class: IO, state: Disposition::Keep(77) }, now);

        book.admit(item(3, IO), now);
        let second = book.next(IO, now).unwrap();
        assert_eq!(second.state, Some(77), "resident state follows the key");
        book.complete(Completion { key: 3, class: IO, state: Disposition::Drop }, now);

        book.admit(item(3, IO), now);
        let third = book.next(IO, now).unwrap();
        assert_eq!(third.state, None, "a dropped disposition forces a reload");
        book.complete(finish(third), now);
    }

    #[test]
    fn expired_items_are_discarded_at_dispatch_and_handed_back() {
        let mut book = Book::new(config());
        let deadline = Duration::from_secs(1);
        book.admit(expiring(4, IO, deadline), Duration::ZERO);
        book.admit(expiring(5, IO, deadline), Duration::ZERO);
        book.admit(item(6, IO), Duration::ZERO);

        let now = Duration::from_secs(2);
        let dispatch = book.next(IO, now).expect("the item without a deadline survives");
        assert_eq!(dispatch.key, 6);
        assert_eq!(book.pop_expired().map(|(key, _)| key), Some(4));
        assert_eq!(book.pop_expired().map(|(key, _)| key), Some(5));
        assert_eq!(book.pop_expired().map(|(key, _)| key), None);
        assert_eq!(book.pending(), 1, "expired items leave the pending count");
        book.complete(finish(dispatch), now);
        assert_eq!(book.check_invariants(), Ok(()));
    }

    #[test]
    fn expiring_a_ring_head_re_places_the_key_on_its_next_class() {
        let mut book = Book::new(config());
        let deadline = Duration::from_secs(1);
        book.admit(expiring(7, IO, deadline), Duration::ZERO);
        book.admit(item(7, CPU), Duration::ZERO);

        let now = Duration::from_secs(2);
        assert!(book.next(IO, now).is_none(), "the only IO item expired");
        assert_eq!(book.pop_expired().map(|(key, _)| key), Some(7));
        assert_eq!(book.next(CPU, now).unwrap().key, 7, "the key moved to the compute ring");
        assert_eq!(book.check_invariants(), Ok(()));
    }

    #[test]
    fn idle_keys_are_evicted_after_their_ttl_and_flushed_once() {
        let mut book = Book::new(config());
        book.admit(item(5, IO), Duration::ZERO);
        let _dispatch = book.next(IO, Duration::ZERO).unwrap();
        book.complete(
            Completion { key: 5, class: IO, state: Disposition::Keep(42) },
            Duration::ZERO,
        );

        let mut flushed = Vec::new();
        book.evict(Duration::from_secs(5), &mut flushed);
        assert!(flushed.is_empty(), "the key is still inside its idle window");

        book.evict(Duration::from_secs(11), &mut flushed);
        assert_eq!(flushed, vec![(5, 42)]);
        assert_eq!(book.snapshot().evicting, 1);
        assert_eq!(book.check_invariants(), Ok(()));

        book.finish_evict(5, Duration::from_secs(11));
        assert_eq!(book.snapshot().resident, 0);
    }

    #[test]
    fn work_arriving_during_a_flush_waits_and_then_reloads() {
        let mut book = Book::new(config());
        book.admit(item(8, IO), Duration::ZERO);
        let _dispatch = book.next(IO, Duration::ZERO).unwrap();
        book.complete(
            Completion { key: 8, class: IO, state: Disposition::Keep(11) },
            Duration::ZERO,
        );

        let mut flushed = Vec::new();
        let now = Duration::from_secs(11);
        book.evict(now, &mut flushed);
        assert_eq!(flushed, vec![(8, 11)]);

        // The key is quiesced: nothing dispatches while the flush is running.
        book.admit(item(8, IO), now);
        assert!(book.next(IO, now).is_none(), "a quiesced key must not dispatch");
        assert_eq!(book.check_invariants(), Ok(()));

        book.finish_evict(8, now);
        let after = book.next(IO, now).expect("the key resumes once the flush completes");
        assert_eq!(after.state, None, "flushed state is never silently reused");
        book.complete(finish(after), now);
    }

    #[test]
    fn evict_all_flushes_every_resident_key_regardless_of_idle_time() {
        let mut book = Book::new(Config { evict_iters: 1, ..config() });
        for key in 0..4 {
            book.admit(item(key, IO), Duration::ZERO);
            let _dispatch = book.next(IO, Duration::ZERO).expect("the key dispatches");
            book.complete(
                Completion { key, class: IO, state: Disposition::Keep(key * 10) },
                Duration::ZERO,
            );
        }

        let mut flushed = Vec::new();
        // Well inside the idle window, and `evict_iters` would cap a sweep at
        // one key, so neither limit may apply on the shutdown path.
        book.evict_all(&mut flushed);
        assert_eq!(flushed, vec![(0, 0), (1, 10), (2, 20), (3, 30)]);
        assert_eq!(book.snapshot().evicting, 4);
        assert_eq!(book.check_invariants(), Ok(()));

        for (key, _) in flushed {
            book.finish_evict(key, Duration::ZERO);
        }
        assert_eq!(book.snapshot().resident, 0, "every key is released once its flush lands");
        assert_eq!(book.check_invariants(), Ok(()));
    }

    #[test]
    fn evict_all_skips_keys_that_are_not_idle() {
        let mut book = Book::new(config());
        // Key 2 goes resident and idle; key 1 is left holding its state.
        book.admit(item(2, IO), Duration::ZERO);
        let _dispatch = book.next(IO, Duration::ZERO).expect("key 2 dispatches");
        book.complete(
            Completion { key: 2, class: IO, state: Disposition::Keep(9) },
            Duration::ZERO,
        );
        book.admit(item(1, IO), Duration::ZERO);
        let inflight = book.next(IO, Duration::ZERO).expect("key 1 dispatches");

        let mut flushed = Vec::new();
        book.evict_all(&mut flushed);
        assert_eq!(flushed, vec![(2, 9)], "an in-flight key still owns its state");
        assert_eq!(book.check_invariants(), Ok(()));
        book.complete(finish(inflight), Duration::ZERO);
    }

    #[test]
    fn a_reactivated_key_survives_its_stale_eviction_candidate() {
        let mut book = Book::new(config());
        book.admit(item(5, IO), Duration::ZERO);
        let first = book.next(IO, Duration::ZERO).unwrap();
        book.complete(finish(first), Duration::ZERO);

        let later = Duration::from_secs(5);
        book.admit(item(5, IO), later);
        let second = book.next(IO, later).unwrap();
        book.complete(finish(second), later);

        let mut flushed = Vec::new();
        book.evict(Duration::from_secs(11), &mut flushed);
        assert!(flushed.is_empty(), "the stale candidate must not evict a reactivated key");
        assert_eq!(book.snapshot().resident, 1);

        book.evict(Duration::from_secs(16), &mut flushed);
        assert_eq!(flushed.len(), 1);
    }

    #[test]
    fn capacity_pressure_evicts_before_the_idle_window_elapses() {
        let mut book = Book::new(Config { max_resident: Some(1), ..config() });
        for key in 0..3 {
            book.admit(item(key, IO), Duration::ZERO);
            let _dispatch = book.next(IO, Duration::ZERO).unwrap();
            book.complete(
                Completion { key, class: IO, state: Disposition::Keep(key) },
                Duration::ZERO,
            );
        }
        assert_eq!(book.snapshot().resident, 3);

        let mut flushed = Vec::new();
        book.evict(Duration::ZERO, &mut flushed);
        assert_eq!(flushed, vec![(0, 0), (1, 1)], "the oldest idle keys go first");
        for (key, _) in flushed {
            book.finish_evict(key, Duration::ZERO);
        }
        assert_eq!(book.snapshot().resident, 1);
        assert_eq!(book.check_invariants(), Ok(()));
    }

    #[test]
    fn snapshot_and_saturation_report_the_scheduler_state() {
        let mut book = Book::new(Config { max_pending: 2, ..config() });
        assert_eq!(book.snapshot(), Snapshot::default());
        assert_eq!(book.max_pending(), 2);
        assert_eq!(book.config().max_inflight, [1, 1]);

        book.admit(item(1, IO), Duration::ZERO);
        book.admit(item(2, CPU), Duration::ZERO);
        assert!(book.is_saturated());
        let dispatch = book.next(IO, Duration::ZERO).unwrap();
        assert_eq!(
            book.snapshot(),
            Snapshot {
                inflight: [1, 0],
                ready: [0, 1],
                pending: 2,
                resident: 2,
                evicting: 0,
                queue_capacity: 2,
            }
        );
        book.complete(finish(dispatch), Duration::ZERO);
    }

    #[test]
    fn three_classes_keep_separate_budgets() {
        let mut book: Scheduler<u64, &'static str, u64, 3> = Scheduler::new(Config {
            max_inflight: [1, 1, 1],
            max_pending: 8,
            max_resident: None,
            evict_after: Duration::from_secs(10),
            evict_iters: 8,
            queue_reserve: 8,
        });
        let now = Duration::ZERO;
        for class in 0..3u8 {
            book.admit(Admit { key: u64::from(class), class, expires_at: None, payload: "w" }, now);
        }
        for class in 0..3u8 {
            let dispatch = book.next(class, now).expect("each class has its own budget");
            assert_eq!(dispatch.key, u64::from(class));
        }
        assert_eq!(book.snapshot().inflight, [1, 1, 1]);
        assert_eq!(book.check_invariants(), Ok(()));
    }
}
