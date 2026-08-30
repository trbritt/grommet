//! A predictable, shard-local hierarchical timer wheel.
//!
//! This module is designed for a reactor, matching engine, or eviction shard
//! that is owned by exactly one thread.  No operation on a constructed wheel
//! performs atomic synchronization.  Construction uses one relaxed atomic to
//! give every wheel a process-unique identity; that makes a handle from one
//! wheel harmless when it is accidentally presented to another.
//!
//! # Performance shape
//!
//! * Eight levels of sixty-four slots use 2 KiB of slot heads and eight `u64`
//!   occupancy masks.
//! * `is_due_at` is one integer comparison when the caller already has a tick.
//! * Insert, cancel, and reschedule touch a bounded number of words.
//! * Empty tick ranges are skipped; `advance_to` visits occupied slot events,
//!   not every elapsed tick.
//! * Storage is fixed, initialized, and touched at construction.  Inserting
//!   cannot invoke the allocator and reports `Full` instead.
//! * There is no overflow list.  A deadline outside the supported horizon is
//!   rejected rather than introducing an unbounded rescan into the hot path.
//!
//! A cascade can still move every entry in one coarse bucket at its boundary.
//! That is intrinsic to this style of hierarchical wheel and must be included
//! in tail-latency benchmarks using the production deadline distribution.
//!
//! # Time and precision
//!
//! The core API consumes absolute `u64` ticks.  Keeping clock acquisition and
//! `Duration` conversion outside the hottest path is intentional.  `SHIFT`
//! defines one tick as `2^SHIFT` nanoseconds and may be in `0..=29`; the
//! default is 1,024 ns.
//!
//! Duration deadlines are rounded **up**, while observations of current time
//! are rounded **down**.  A timer may consequently fire less than one tick
//! late, but never early.  This is normally the safe policy for expiry and
//! eviction.  Use the tick-native API when the caller already maintains the
//! reactor clock in this representation.
//!
//! Eight levels address `2^48` ticks.  The maximum accepted delay is one tick
//! less than that: roughly 9.14 years with the default 1,024 ns tick.  Adding
//! another level would increase the horizon by 64 and cost 256 bytes of heads,
//! but eight keeps all wheel metadata comfortably small.
//!
//! # Handles
//!
//! `TimerId` contains a wheel identity, a slab index, and a 64-bit generation.
//! A released slab position increments its generation before reuse, so an old
//! handle cannot cancel or reschedule a later occupant.  A position is retired
//! instead of wrapping its generation.  `Option<TimerId>` has the same size as
//! `TimerId` through `NonZeroU128`'s niche.
//!
//! The robust handle is 16 bytes.  That is a deliberate correctness trade-off.
//! If handle footprint is proven material, a deployment can bit-pack smaller
//! fields only after establishing explicit bounds for wheel count, capacity,
//! process lifetime, and per-position reuse.
//!
//! # Example
//!
//! ```
//! use std::time::Duration;
//! use grommet_core::timer::Wheel;
//!
//! let mut wheel: Wheel<&'static str> = Wheel::with_capacity(128);
//! wheel
//!     .try_insert_duration(Duration::from_millis(50), "sweep")
//!     .expect("capacity and horizon were configured at startup");
//!
//! assert!(!wheel.is_due_duration(Duration::from_millis(10)));
//!
//! let mut due = Vec::with_capacity(128);
//! wheel
//!     .advance_duration(Duration::from_millis(49), &mut due)
//!     .expect("the duration is representable");
//! assert!(due.is_empty());
//! wheel
//!     .advance_duration(Duration::from_millis(50), &mut due)
//!     .expect("the duration is representable");
//! assert!(due.is_empty(), "rounding up prevents early expiry");
//! wheel
//!     .advance_duration(Duration::from_millis(51), &mut due)
//!     .expect("the duration is representable");
//! assert_eq!(due, ["sweep"]);
//! ```
//!
//! # Reactor integration
//!
//! Call `is_due_at(now_tick)` every loop and call `advance_to` only when it is
//! true.  `next_wakeup_tick` is the next time the wheel itself needs attention;
//! it can precede the earliest user deadline because a coarse slot sometimes
//! needs cascading first.  Reserve the output vector for the largest permitted
//! firing burst, or use `advance_to_with` to consume tokens directly without an
//! intermediate collection.  The callback must not panic.

use std::fmt;
use std::num::NonZeroU128;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

/// Hierarchy depth. Eight levels of six bits address 48 bits of relative time.
const LEVELS: usize = 8;
const SLOT_BITS: u32 = 6;
const SLOTS: usize = 1 << SLOT_BITS;
const SLOT_MASK: u64 = SLOTS as u64 - 1;
const WHEEL_BITS: u32 = SLOT_BITS * LEVELS as u32;
const SPAN_TICKS: u64 = 1u64 << WHEEL_BITS;

/// Maximum forward distance accepted by the wheel.
pub const MAX_DELAY_TICKS: u64 = SPAN_TICKS - 1;

/// `u64::MAX` is the empty next-event sentinel, so it is not a valid clock tick.
pub const MAX_TICK: u64 = u64::MAX - 1;

/// End of a slot list and of the free list.
const NIL: u32 = u32::MAX;
const FREE_LEVEL: u8 = u8::MAX - 1;
const RETIRED_LEVEL: u8 = u8::MAX;
const MAX_SHIFT: u32 = 29;

static NEXT_WHEEL_ID: AtomicU32 = AtomicU32::new(1);

fn allocate_wheel_id() -> u32 {
    NEXT_WHEEL_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| current.checked_add(1))
        .expect("the process constructed more than u32::MAX - 1 timer wheels")
}

/// A cancellation and rescheduling handle.
///
/// Handles are wheel-specific and generation-checked.  Passing a stale handle,
/// or a handle created by another wheel, is harmless: cancellation returns
/// `None` and rescheduling returns [`RescheduleError::StaleId`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimerId(NonZeroU128);

impl TimerId {
    fn new(wheel: u32, generation: u64, index: u32) -> Self {
        debug_assert_ne!(wheel, 0);
        debug_assert_ne!(generation, 0);
        debug_assert_ne!(index, NIL);
        let raw =
            (u128::from(wheel) << 96) | (u128::from(generation) << 32) | u128::from(index + 1);
        Self(NonZeroU128::new(raw).expect("wheel id, generation, and index are non-zero encoded"))
    }

    /// The opaque integer representation, useful for logging and storage.
    pub const fn get(self) -> u128 {
        self.0.get()
    }

    /// Reconstruct a previously stored handle.
    ///
    /// Returns `None` for encodings that cannot have been produced by a wheel.
    pub fn from_raw(raw: u128) -> Option<Self> {
        let raw = NonZeroU128::new(raw)?;
        let id = Self(raw);
        // `parts` is the whole of the validation. It rejects a zero wheel,
        // generation or encoded index, and the index it hands back is the
        // encoded one minus a value it has already proved non-zero, so that
        // index cannot be `NIL` either. Re-testing any of the three here would
        // be a condition no input can make false.
        id.parts()?;
        Some(id)
    }

    fn parts(self) -> Option<(u32, u64, u32)> {
        let raw = self.0.get();
        let wheel = (raw >> 96) as u32;
        let generation = (raw >> 32) as u64;
        let encoded_index = raw as u32;
        if wheel == 0 || generation == 0 || encoded_index == 0 {
            return None;
        }
        Some((wheel, generation, encoded_index - 1))
    }
}

impl fmt::Debug for TimerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("TimerId").field(&format_args!("{:#034x}", self.get())).finish()
    }
}

/// Why an insertion was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InsertErrorKind {
    /// Every configured slab position is live or permanently retired.
    Full,
    /// The deadline is more than [`MAX_DELAY_TICKS`] ahead of the cursor.
    DeadlineTooFar,
    /// The supplied deadline cannot be represented by the tick clock.
    TimeOutOfRange,
}

/// A failed insertion, retaining ownership of the token.
pub struct InsertError<T> {
    kind: InsertErrorKind,
    token: T,
}

impl<T> InsertError<T> {
    fn new(kind: InsertErrorKind, token: T) -> Self {
        Self { kind, token }
    }

    pub const fn kind(&self) -> InsertErrorKind {
        self.kind
    }

    pub fn token(&self) -> &T {
        &self.token
    }

    pub fn into_token(self) -> T {
        self.token
    }

    pub fn into_parts(self) -> (InsertErrorKind, T) {
        (self.kind, self.token)
    }
}

impl<T> fmt::Debug for InsertError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InsertError").field("kind", &self.kind).finish_non_exhaustive()
    }
}

impl<T> fmt::Display for InsertError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            InsertErrorKind::Full => f.write_str("the timer wheel is full"),
            InsertErrorKind::DeadlineTooFar => {
                f.write_str("the timer deadline is outside the wheel horizon")
            }
            InsertErrorKind::TimeOutOfRange => {
                f.write_str("the timer deadline is outside the representable clock range")
            }
        }
    }
}

impl<T> std::error::Error for InsertError<T> {}

/// Why a reschedule was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RescheduleError {
    /// The handle is stale, malformed, or belongs to another wheel.
    StaleId,
    /// The new deadline is outside the wheel's supported horizon.
    DeadlineTooFar,
    /// The supplied deadline cannot be represented by the tick clock.
    TimeOutOfRange,
}

impl fmt::Display for RescheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StaleId => f.write_str("the timer id is stale or belongs to another wheel"),
            Self::DeadlineTooFar => f.write_str("the timer deadline is outside the wheel horizon"),
            Self::TimeOutOfRange => {
                f.write_str("the timer deadline is outside the representable clock range")
            }
        }
    }
}

impl std::error::Error for RescheduleError {}

/// A `Duration` could not be represented by the wheel's tick clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeOutOfRange;

impl fmt::Display for TimeOutOfRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("time is outside the representable timer-wheel clock range")
    }
}

impl std::error::Error for TimeOutOfRange {}

struct Entry<T> {
    deadline: u64,
    generation: u64,
    token: Option<T>,
    next: u32,
    prev: u32,
    level: u8,
    slot: u8,
}

/// A fixed-capacity hierarchical timer wheel for one owning thread.
///
/// `SHIFT` is the base-2 logarithm of the tick in nanoseconds.  Construction
/// panics for `SHIFT > 29`, for capacities above `u32::MAX`, or when the
/// initial tick is the reserved empty sentinel.  Those are configuration
/// errors expected to be resolved before entering a reactor loop.
pub struct Wheel<T, const SHIFT: u32 = 10> {
    entries: Vec<Entry<T>>,
    free: u32,
    live: usize,
    retired: usize,
    heads: [[u32; SLOTS]; LEVELS],
    occupied: [u64; LEVELS],
    cursor: u64,
    next_event: u64,
    wheel_id: u32,
}

impl<T, const SHIFT: u32> Wheel<T, SHIFT> {
    const VALID_SHIFT: () = assert!(SHIFT <= MAX_SHIFT, "timer-wheel SHIFT must be in 0..=29");

    /// Construct a fixed-capacity wheel at tick zero.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_at_tick(capacity, 0)
    }

    /// Construct a fixed-capacity wheel whose cursor starts at `now`.
    ///
    /// Every slab entry is initialized here.  Besides fixing the capacity,
    /// this first-touches its metadata before the hot loop starts.  Construct
    /// the wheel on the CPU and NUMA node that will own it.
    pub fn with_capacity_at_tick(capacity: usize, now: u64) -> Self {
        const { Self::VALID_SHIFT };
        assert!(capacity <= NIL as usize, "timer-wheel capacity exceeds u32 indexing");
        assert!(now <= MAX_TICK, "u64::MAX is reserved as the empty-event sentinel");

        let mut entries = Vec::with_capacity(capacity);
        for index in 0..capacity {
            let next = if index + 1 < capacity { (index + 1) as u32 } else { NIL };
            entries.push(Entry {
                deadline: 0,
                generation: 0,
                token: None,
                next,
                prev: NIL,
                level: FREE_LEVEL,
                slot: 0,
            });
        }

        Self {
            entries,
            free: if capacity == 0 { NIL } else { 0 },
            live: 0,
            retired: 0,
            heads: [[NIL; SLOTS]; LEVELS],
            occupied: [0; LEVELS],
            cursor: now,
            next_event: u64::MAX,
            wheel_id: allocate_wheel_id(),
        }
    }

    /// Construct a fixed-capacity wheel at a `Duration` clock observation.
    pub fn with_capacity_at_duration(
        capacity: usize,
        now: Duration,
    ) -> Result<Self, TimeOutOfRange> {
        let now = Self::duration_to_tick_floor(now).ok_or(TimeOutOfRange)?;
        Ok(Self::with_capacity_at_tick(capacity, now))
    }

    /// One wheel tick.
    pub const fn granularity() -> Duration {
        assert!(SHIFT <= MAX_SHIFT, "timer-wheel SHIFT must be in 0..=29");
        Duration::from_nanos(1u64 << SHIFT)
    }

    /// The maximum forward interval accepted by the wheel.
    pub const fn max_delay() -> Duration {
        assert!(SHIFT <= MAX_SHIFT, "timer-wheel SHIFT must be in 0..=29");
        let nanos = (MAX_DELAY_TICKS as u128) << SHIFT;
        Duration::new((nanos / 1_000_000_000) as u64, (nanos % 1_000_000_000) as u32)
    }

    /// Convert current time to ticks by rounding down.
    #[inline]
    pub fn duration_to_tick_floor(at: Duration) -> Option<u64> {
        const { Self::VALID_SHIFT };
        let nanos = u128::from(at.as_secs()) * 1_000_000_000 + u128::from(at.subsec_nanos());
        let ticks = nanos >> SHIFT;
        (ticks <= u128::from(MAX_TICK)).then_some(ticks as u64)
    }

    /// Convert a deadline to ticks by rounding up.
    #[inline]
    pub fn duration_to_tick_ceil(at: Duration) -> Option<u64> {
        const { Self::VALID_SHIFT };
        let nanos = u128::from(at.as_secs()) * 1_000_000_000 + u128::from(at.subsec_nanos());
        let mask = (1u128 << SHIFT) - 1;
        let ticks = (nanos >> SHIFT) + u128::from(nanos & mask != 0);
        (ticks <= u128::from(MAX_TICK)).then_some(ticks as u64)
    }

    /// Convert an absolute tick to `Duration`.
    pub fn tick_to_duration(tick: u64) -> Duration {
        const { Self::VALID_SHIFT };
        assert!(tick <= MAX_TICK, "u64::MAX is not a representable timer tick");
        let nanos = u128::from(tick) << SHIFT;
        Duration::new((nanos / 1_000_000_000) as u64, (nanos % 1_000_000_000) as u32)
    }

    /// Current cursor position.
    #[inline]
    pub const fn cursor(&self) -> u64 {
        self.cursor
    }

    /// Fixed number of slab positions, including live, free, and retired ones.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.entries.len()
    }

    /// Number of scheduled timers.
    #[inline]
    pub const fn len(&self) -> usize {
        self.live
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Positions immediately available for insertion.
    #[inline]
    pub fn available(&self) -> usize {
        self.entries.len() - self.live - self.retired
    }

    /// Positions retired after exhausting their 64-bit generation.
    ///
    /// This should remain zero for the lifetime of any realistic process.
    #[inline]
    pub const fn retired(&self) -> usize {
        self.retired
    }

    /// Whether advancing to `now` would perform wheel maintenance or fire.
    ///
    /// This is the intended busy-loop hot path: one comparison after the
    /// caller has acquired and converted its monotonic clock.
    #[inline]
    pub fn is_due_at(&self, now: u64) -> bool {
        debug_assert!(now <= MAX_TICK);
        now >= self.next_event
    }

    /// Duration wrapper for [`Wheel::is_due_at`].
    #[inline]
    pub fn is_due_duration(&self, now: Duration) -> bool {
        match Self::duration_to_tick_floor(now) {
            Some(now) => self.is_due_at(now),
            None => !self.is_empty(),
        }
    }

    /// Next tick at which the wheel needs attention.
    ///
    /// This can be earlier than any timer deadline because a coarse bucket may
    /// need cascading.  It is always safe as the reactor's park deadline.
    #[inline]
    pub fn next_wakeup_tick(&self) -> Option<u64> {
        (self.next_event != u64::MAX).then_some(self.next_event)
    }

    /// Duration form of [`Wheel::next_wakeup_tick`].
    pub fn next_wakeup_duration(&self) -> Option<Duration> {
        self.next_wakeup_tick().map(Self::tick_to_duration)
    }

    /// Schedule `token` at an absolute tick.
    ///
    /// Deadlines at or behind the cursor are accepted and fire on the next
    /// advance at or beyond the cursor.  Failure retains the token inside the
    /// returned [`InsertError`].
    #[inline]
    pub fn try_insert_at(&mut self, deadline: u64, token: T) -> Result<TimerId, InsertError<T>> {
        if deadline > MAX_TICK {
            return Err(InsertError::new(InsertErrorKind::TimeOutOfRange, token));
        }
        if !self.deadline_fits(deadline) {
            return Err(InsertError::new(InsertErrorKind::DeadlineTooFar, token));
        }
        if self.free == NIL {
            return Err(InsertError::new(InsertErrorKind::Full, token));
        }

        let index = self.allocate(deadline, token);
        self.file(index, deadline);
        let generation = self.entries[index as usize].generation;
        Ok(TimerId::new(self.wheel_id, generation, index))
    }

    /// Schedule a `Duration` deadline, rounding it up to avoid early firing.
    pub fn try_insert_duration(
        &mut self,
        deadline: Duration,
        token: T,
    ) -> Result<TimerId, InsertError<T>> {
        let Some(deadline) = Self::duration_to_tick_ceil(deadline) else {
            return Err(InsertError::new(InsertErrorKind::TimeOutOfRange, token));
        };
        self.try_insert_at(deadline, token)
    }

    /// Return whether `id` currently names a timer in this wheel.
    #[inline]
    pub fn contains(&self, id: TimerId) -> bool {
        self.live_index(id).is_some()
    }

    /// Cancel a timer and return its token.
    ///
    /// Stale, malformed, already-fired, already-cancelled, and foreign handles
    /// all return `None` without modifying the wheel.
    #[inline]
    pub fn cancel(&mut self, id: TimerId) -> Option<T> {
        let index = self.live_index(id)?;
        let (level, slot, singleton) = {
            let entry = &self.entries[index as usize];
            (entry.level as usize, entry.slot as usize, entry.prev == NIL && entry.next == NIL)
        };
        let removed_earliest = singleton && self.slot_wakeup(level, slot) == self.next_event;

        self.unlink(index);
        let token = self.entries[index as usize]
            .token
            .take()
            .expect("a generation-checked live entry holds a token");
        self.release(index);

        if removed_earliest {
            self.refresh();
        }
        Some(token)
    }

    /// Move an existing timer to a new absolute tick without changing its ID.
    #[inline]
    pub fn reschedule_at(&mut self, id: TimerId, deadline: u64) -> Result<(), RescheduleError> {
        if deadline > MAX_TICK {
            return Err(RescheduleError::TimeOutOfRange);
        }
        if !self.deadline_fits(deadline) {
            return Err(RescheduleError::DeadlineTooFar);
        }
        let index = self.live_index(id).ok_or(RescheduleError::StaleId)?;
        let (old_level, old_slot, singleton) = {
            let entry = &self.entries[index as usize];
            (entry.level as usize, entry.slot as usize, entry.prev == NIL && entry.next == NIL)
        };
        let removed_earliest =
            singleton && self.slot_wakeup(old_level, old_slot) == self.next_event;

        self.unlink(index);
        self.entries[index as usize].deadline = deadline;
        self.file(index, deadline);

        if removed_earliest {
            self.refresh();
        }
        Ok(())
    }

    /// Duration wrapper for [`Wheel::reschedule_at`], rounding the deadline up.
    pub fn reschedule_duration(
        &mut self,
        id: TimerId,
        deadline: Duration,
    ) -> Result<(), RescheduleError> {
        let deadline =
            Self::duration_to_tick_ceil(deadline).ok_or(RescheduleError::TimeOutOfRange)?;
        self.reschedule_at(id, deadline)
    }

    /// Advance to `target`, appending due tokens to `out`.
    ///
    /// Returns the number appended.  A backwards target is a harmless no-op.
    /// Tokens are emitted in nondecreasing tick order; order within one tick is
    /// deliberately unspecified.  Pre-reserve `out` to keep this path free of
    /// allocation.
    pub fn advance_to(&mut self, target: u64, out: &mut Vec<T>) -> usize {
        self.advance_to_with(target, |token| out.push(token))
    }

    /// Advance and pass each due token directly to an inlined callback.
    ///
    /// The callback must not panic.  It cannot re-enter this wheel while the
    /// mutable borrow is active.
    #[inline]
    pub fn advance_to_with<F>(&mut self, target: u64, mut emit: F) -> usize
    where
        F: FnMut(T),
    {
        debug_assert!(target <= MAX_TICK);
        if target > MAX_TICK || target < self.cursor {
            return 0;
        }

        if target < self.next_event {
            self.cursor = target;
            return 0;
        }

        let mut fired = 0usize;
        loop {
            let event = self.next_event;
            if event == u64::MAX || event > target {
                self.cursor = target;
                return fired;
            }

            debug_assert!(event >= self.cursor, "cached event moved behind the cursor");
            self.cursor = event;
            self.cascade();
            fired += self.fire(&mut emit);
            self.refresh();
        }
    }

    /// Duration wrapper for [`Wheel::advance_to`].
    pub fn advance_duration(
        &mut self,
        now: Duration,
        out: &mut Vec<T>,
    ) -> Result<usize, TimeOutOfRange> {
        let now = Self::duration_to_tick_floor(now).ok_or(TimeOutOfRange)?;
        Ok(self.advance_to(now, out))
    }

    fn deadline_fits(&self, deadline: u64) -> bool {
        deadline <= self.cursor || deadline - self.cursor <= MAX_DELAY_TICKS
    }

    fn live_index(&self, id: TimerId) -> Option<u32> {
        let (wheel, generation, index) = id.parts()?;
        if wheel != self.wheel_id {
            return None;
        }
        let entry = self.entries.get(index as usize)?;
        (entry.generation == generation && entry.token.is_some()).then_some(index)
    }

    // ---- Placement -----------------------------------------------------

    fn placement(&self, deadline: u64) -> (usize, usize) {
        if deadline <= self.cursor {
            return (0, (self.cursor & SLOT_MASK) as usize);
        }

        debug_assert!(self.deadline_fits(deadline));
        let differing = u64::BITS - 1 - (self.cursor ^ deadline).leading_zeros();
        let level = ((differing / SLOT_BITS) as usize).min(LEVELS - 1);
        let slot = ((deadline >> (level as u32 * SLOT_BITS)) & SLOT_MASK) as usize;
        debug_assert!(self.slot_wakeup(level, slot) <= deadline);
        (level, slot)
    }

    fn file(&mut self, index: u32, deadline: u64) {
        let (level, slot) = self.placement(deadline);
        let head = self.heads[level][slot];
        {
            let entry = &mut self.entries[index as usize];
            entry.deadline = deadline;
            entry.next = head;
            entry.prev = NIL;
            entry.level = level as u8;
            entry.slot = slot as u8;
        }
        if head != NIL {
            self.entries[head as usize].prev = index;
        }
        self.heads[level][slot] = index;
        self.occupied[level] |= 1u64 << slot;

        let wakeup = self.slot_wakeup(level, slot);
        if wakeup < self.next_event {
            self.next_event = wakeup;
        }
    }

    /// Absolute start of the next occurrence of `(level, slot)`.
    fn slot_wakeup(&self, level: usize, slot: usize) -> u64 {
        let shift = level as u32 * SLOT_BITS;
        let width = 1u64 << shift;
        let rotation = width << SLOT_BITS;
        let within_rotation = self.cursor & (rotation - 1);
        let slot_start = slot as u64 * width;
        let distance = if slot_start >= within_rotation {
            slot_start - within_rotation
        } else {
            rotation - within_rotation + slot_start
        };
        self.cursor.saturating_add(distance)
    }

    fn compute_next_event(&self) -> Option<u64> {
        let mut soonest = u64::MAX;
        for level in 0..LEVELS {
            let occupied = self.occupied[level];
            if occupied == 0 {
                continue;
            }

            let shift = level as u32 * SLOT_BITS;
            let here = ((self.cursor >> shift) & SLOT_MASK) as u32;
            let other = occupied & !(1u64 << here);
            if other != 0 {
                let ahead = other.rotate_right(here).trailing_zeros() as usize;
                let slot = (here as usize + ahead) & (SLOTS - 1);
                soonest = soonest.min(self.slot_wakeup(level, slot));
            }
            if occupied & (1u64 << here) != 0 {
                soonest = soonest.min(self.slot_wakeup(level, here as usize));
            }
        }
        (soonest != u64::MAX).then_some(soonest)
    }

    fn refresh(&mut self) {
        self.next_event = self.compute_next_event().unwrap_or(u64::MAX);
    }

    // ---- Sweeping ------------------------------------------------------

    fn cascade(&mut self) {
        for level in (1..LEVELS).rev() {
            let slot = ((self.cursor >> (level as u32 * SLOT_BITS)) & SLOT_MASK) as usize;
            if self.occupied[level] & (1u64 << slot) == 0 {
                continue;
            }

            let mut index = self.take(level, slot);
            while index != NIL {
                let next = self.entries[index as usize].next;
                let deadline = self.entries[index as usize].deadline;
                self.file(index, deadline);
                index = next;
            }
        }
    }

    fn fire<F>(&mut self, emit: &mut F) -> usize
    where
        F: FnMut(T),
    {
        let slot = (self.cursor & SLOT_MASK) as usize;
        if self.occupied[0] & (1u64 << slot) == 0 {
            return 0;
        }

        let mut fired = 0usize;
        let mut index = self.take(0, slot);
        while index != NIL {
            let next = self.entries[index as usize].next;
            debug_assert!(self.entries[index as usize].deadline <= self.cursor);
            let token = self.entries[index as usize]
                .token
                .take()
                .expect("a filed entry always holds a token");
            self.release(index);
            fired += 1;
            emit(token);
            index = next;
        }
        fired
    }

    fn take(&mut self, level: usize, slot: usize) -> u32 {
        let head = self.heads[level][slot];
        self.heads[level][slot] = NIL;
        self.occupied[level] &= !(1u64 << slot);
        head
    }

    fn unlink(&mut self, index: u32) {
        let (level, slot, prev, next) = {
            let entry = &self.entries[index as usize];
            (entry.level as usize, entry.slot as usize, entry.prev, entry.next)
        };

        if prev == NIL {
            self.heads[level][slot] = next;
            if next == NIL {
                self.occupied[level] &= !(1u64 << slot);
            }
        } else {
            self.entries[prev as usize].next = next;
        }
        if next != NIL {
            self.entries[next as usize].prev = prev;
        }
    }

    // ---- Fixed slab ----------------------------------------------------

    fn allocate(&mut self, deadline: u64, token: T) -> u32 {
        let index = self.free;
        debug_assert_ne!(index, NIL);
        let entry = &mut self.entries[index as usize];
        debug_assert!(entry.token.is_none());
        debug_assert_eq!(entry.level, FREE_LEVEL);
        debug_assert_ne!(entry.generation, u64::MAX);

        self.free = entry.next;
        entry.deadline = deadline;
        entry.generation += 1;
        entry.token = Some(token);
        entry.next = NIL;
        entry.prev = NIL;
        self.live += 1;
        index
    }

    fn release(&mut self, index: u32) {
        let entry = &mut self.entries[index as usize];
        debug_assert!(entry.token.is_none());
        entry.prev = NIL;
        entry.slot = 0;
        self.live -= 1;

        if entry.generation == u64::MAX {
            entry.next = NIL;
            entry.level = RETIRED_LEVEL;
            self.retired += 1;
        } else {
            entry.next = self.free;
            entry.level = FREE_LEVEL;
            self.free = index;
        }
    }

    // ---- Diagnostics ---------------------------------------------------

    /// Verify linkage, occupancy, handle generations, free/retired state, and
    /// the exact next-event cache.
    ///
    /// This diagnostic is linear in configured capacity and allocates one byte
    /// per slab position.  It is intended for tests, fuzzing, and offline
    /// diagnosis, never the reactor hot path.
    pub fn check_invariants(&self) -> Result<(), &'static str> {
        let mut seen = vec![0u8; self.entries.len()];
        let mut live = 0usize;

        for level in 0..LEVELS {
            for slot in 0..SLOTS {
                let head = self.heads[level][slot];
                let occupied = self.occupied[level] & (1u64 << slot) != 0;
                if occupied != (head != NIL) {
                    return Err("occupancy disagrees with slot head");
                }

                let mut index = head;
                let mut prev = NIL;
                while index != NIL {
                    let Some(entry) = self.entries.get(index as usize) else {
                        return Err("slot list contains an out-of-range index");
                    };
                    if seen[index as usize] != 0 {
                        return Err("an entry appears more than once or a list cycles");
                    }
                    seen[index as usize] = 1;
                    live += 1;

                    if entry.token.is_none() {
                        return Err("a filed entry holds no token");
                    }
                    if entry.generation == 0 {
                        return Err("a live entry has never been issued");
                    }
                    if entry.level as usize != level || entry.slot as usize != slot {
                        return Err("an entry disagrees with its slot");
                    }
                    if entry.prev != prev {
                        return Err("slot back-link disagrees with forward linkage");
                    }
                    if self.slot_wakeup(level, slot) > entry.deadline.max(self.cursor) {
                        return Err("a slot would be visited after its deadline");
                    }

                    prev = index;
                    index = entry.next;
                }
            }
        }

        if live != self.live {
            return Err("live counter disagrees with filed entries");
        }

        let mut free = 0usize;
        let mut index = self.free;
        while index != NIL {
            let Some(entry) = self.entries.get(index as usize) else {
                return Err("free list contains an out-of-range index");
            };
            if seen[index as usize] != 0 {
                return Err(
                    "an entry appears in both a slot and the free list, or the free list cycles",
                );
            }
            seen[index as usize] = 2;
            free += 1;
            if entry.token.is_some() || entry.level != FREE_LEVEL {
                return Err("a free entry has live state");
            }
            index = entry.next;
        }

        let mut retired = 0usize;
        for (index, entry) in self.entries.iter().enumerate() {
            if entry.level == RETIRED_LEVEL {
                if seen[index] != 0 || entry.token.is_some() || entry.generation != u64::MAX {
                    return Err("a retired entry has inconsistent state");
                }
                seen[index] = 3;
                retired += 1;
            }
        }

        if retired != self.retired {
            return Err("retired counter disagrees with retired entries");
        }
        if free + live + retired != self.entries.len() || seen.contains(&0) {
            return Err("the slab contains an unreachable entry");
        }
        if self.available() != free {
            return Err("available count disagrees with the free list");
        }

        let actual = self.compute_next_event().unwrap_or(u64::MAX);
        if actual != self.next_event {
            return Err("cached next event is not exact");
        }
        if self.next_event < self.cursor {
            return Err("cached next event is behind the cursor");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, HashMap};

    type Ns = Wheel<u64, 0>;
    type Us = Wheel<u64, 10>;

    fn drain(wheel: &mut Ns, at: u64) -> Vec<u64> {
        let mut due = Vec::new();
        wheel.advance_to(at, &mut due);
        assert_eq!(wheel.check_invariants(), Ok(()));
        due.sort_unstable();
        due
    }

    #[test]
    fn size_and_time_constants_are_as_documented() {
        assert_eq!(Us::granularity(), Duration::from_nanos(1_024));
        assert!(Us::max_delay() > Duration::from_secs(9 * 365 * 86_400));
        assert!(Us::max_delay() < Duration::from_secs(10 * 365 * 86_400));
        assert_eq!(std::mem::size_of::<Option<TimerId>>(), std::mem::size_of::<TimerId>());
        assert_eq!(std::mem::size_of::<TimerId>(), 16);
    }

    #[test]
    fn duration_deadlines_round_up_and_current_time_rounds_down() {
        assert_eq!(Us::duration_to_tick_ceil(Duration::from_nanos(0)), Some(0));
        assert_eq!(Us::duration_to_tick_ceil(Duration::from_nanos(1)), Some(1));
        assert_eq!(Us::duration_to_tick_ceil(Duration::from_nanos(1_024)), Some(1));
        assert_eq!(Us::duration_to_tick_ceil(Duration::from_nanos(1_025)), Some(2));
        assert_eq!(Us::duration_to_tick_floor(Duration::from_nanos(1_023)), Some(0));
        assert_eq!(Us::duration_to_tick_floor(Duration::from_nanos(1_024)), Some(1));

        let mut wheel = Us::with_capacity(1);
        wheel
            .try_insert_duration(Duration::from_nanos(1_500), 7)
            .expect("within capacity and horizon");
        let mut due = Vec::with_capacity(1);
        wheel.advance_duration(Duration::from_nanos(2_047), &mut due).unwrap();
        assert!(due.is_empty(), "the 1,500ns deadline must not fire at tick one");
        wheel.advance_duration(Duration::from_nanos(2_048), &mut due).unwrap();
        assert_eq!(due, [7]);
    }

    #[test]
    fn fires_once_at_deadline_and_skips_empty_ranges() {
        let mut wheel = Ns::with_capacity(4);
        wheel.try_insert_at(100, 1).unwrap();
        assert!(!wheel.is_due_at(63));
        assert!(wheel.is_due_at(64), "the level-one bucket needs cascading before the deadline");
        assert_eq!(drain(&mut wheel, 50), []);
        assert_eq!(drain(&mut wheel, 99), []);
        assert!(wheel.is_due_at(100));
        assert_eq!(drain(&mut wheel, 100), [1]);
        assert_eq!(drain(&mut wheel, 1_000_000_000_000), []);
        assert!(wheel.is_empty());
    }

    #[test]
    fn stale_handle_cannot_cancel_or_reschedule_a_reused_position() {
        let mut wheel = Ns::with_capacity(1);
        let old = wheel.try_insert_at(10, 11).unwrap();
        assert_eq!(drain(&mut wheel, 10), [11]);

        let new = wheel.try_insert_at(20, 22).unwrap();
        assert_ne!(old, new);
        assert_eq!(wheel.cancel(old), None);
        assert_eq!(wheel.reschedule_at(old, 30), Err(RescheduleError::StaleId));
        assert!(wheel.contains(new));
        assert_eq!(drain(&mut wheel, 20), [22]);
    }

    #[test]
    fn a_handle_from_another_wheel_is_harmless() {
        let mut left = Ns::with_capacity(1);
        let foreign = left.try_insert_at(10, 1).unwrap();
        let mut right = Ns::with_capacity(1);
        let local = right.try_insert_at(10, 2).unwrap();

        assert_eq!(right.cancel(foreign), None);
        assert_eq!(right.reschedule_at(foreign, 20), Err(RescheduleError::StaleId));
        assert!(right.contains(local));
    }

    #[test]
    fn cancellation_is_idempotent_and_unlinks_every_list_position() {
        for victim in 0..3usize {
            let mut wheel = Ns::with_capacity(3);
            let ids: Vec<_> =
                (0..3u64).map(|token| wheel.try_insert_at(100, token).unwrap()).collect();
            assert_eq!(wheel.cancel(ids[victim]), Some(victim as u64));
            assert_eq!(wheel.cancel(ids[victim]), None);
            assert_eq!(wheel.check_invariants(), Ok(()));
            let expected: Vec<_> = (0..3u64).filter(|token| *token != victim as u64).collect();
            assert_eq!(drain(&mut wheel, 100), expected);
        }
    }

    #[test]
    fn reschedule_keeps_the_handle_and_updates_the_cached_wakeup() {
        let mut wheel = Ns::with_capacity(3);
        let id = wheel.try_insert_at(10_000, 1).unwrap();
        wheel.try_insert_at(20_000, 2).unwrap();
        let first = wheel.next_wakeup_tick().unwrap();
        assert!(first <= 10_000);

        wheel.reschedule_at(id, 30_000).unwrap();
        assert!(wheel.contains(id));
        assert_eq!(drain(&mut wheel, 20_000), [2]);
        assert_eq!(drain(&mut wheel, 29_999), []);
        assert_eq!(drain(&mut wheel, 30_000), [1]);
    }

    #[test]
    fn past_deadlines_fire_on_the_next_sweep() {
        let mut wheel = Ns::with_capacity_at_tick(4, 1_000);
        wheel.try_insert_at(10, 1).unwrap();
        wheel.try_insert_at(999, 2).unwrap();
        assert!(wheel.is_due_at(1_000));
        assert_eq!(drain(&mut wheel, 1_000), [1, 2]);
    }

    #[test]
    fn fixed_capacity_never_grows_and_failure_returns_the_token() {
        let mut wheel = Ns::with_capacity(2);
        wheel.try_insert_at(10, 1).unwrap();
        wheel.try_insert_at(20, 2).unwrap();
        let error = wheel.try_insert_at(30, 3).unwrap_err();
        assert_eq!(error.kind(), InsertErrorKind::Full);
        assert_eq!(error.into_token(), 3);
        assert_eq!(wheel.capacity(), 2);
        assert_eq!(wheel.available(), 0);
    }

    #[test]
    fn horizon_is_explicit_and_has_an_exact_boundary() {
        let cursor = 123_456;
        let mut wheel = Ns::with_capacity_at_tick(2, cursor);
        wheel.try_insert_at(cursor + MAX_DELAY_TICKS, 1).unwrap();
        let error = wheel.try_insert_at(cursor + SPAN_TICKS, 2).unwrap_err();
        assert_eq!(error.kind(), InsertErrorKind::DeadlineTooFar);
        assert_eq!(error.into_token(), 2);

        assert_eq!(drain(&mut wheel, cursor + MAX_DELAY_TICKS - 1), []);
        assert_eq!(drain(&mut wheel, cursor + MAX_DELAY_TICKS), [1]);
    }

    #[test]
    fn crosses_the_full_rotation_boundary_without_wrapping_slots() {
        let cursor = SPAN_TICKS - 17;
        let deadline = cursor + 10_000;
        let mut wheel = Ns::with_capacity_at_tick(4, cursor);
        wheel.try_insert_at(deadline, 1).unwrap();
        assert_eq!(wheel.check_invariants(), Ok(()));
        assert_eq!(drain(&mut wheel, deadline - 1), []);
        assert_eq!(drain(&mut wheel, deadline), [1]);
    }

    #[test]
    fn works_near_the_maximum_tick_without_arithmetic_wrap() {
        let cursor = MAX_TICK - 10_000;
        let mut wheel = Ns::with_capacity_at_tick(4, cursor);
        wheel.try_insert_at(MAX_TICK - 1, 1).unwrap();
        wheel.try_insert_at(MAX_TICK, 2).unwrap();
        assert_eq!(drain(&mut wheel, MAX_TICK - 2), []);
        assert_eq!(drain(&mut wheel, MAX_TICK - 1), [1]);
        assert_eq!(drain(&mut wheel, MAX_TICK), [2]);
        assert!(!wheel.is_due_at(MAX_TICK));
    }

    #[test]
    fn deadlines_across_all_levels_fire_in_tick_order() {
        let mut wheel = Ns::with_capacity(LEVELS);
        let mut deadlines = Vec::new();
        for level in 0..LEVELS {
            let deadline = if level == 0 { 1 } else { (1u64 << (level as u32 * SLOT_BITS)) + 3 };
            deadlines.push(deadline);
            wheel.try_insert_at(deadline, deadline).unwrap();
        }

        let mut emitted = Vec::new();
        wheel.advance_to(*deadlines.last().unwrap(), &mut emitted);
        assert_eq!(emitted, deadlines);
        assert_eq!(wheel.check_invariants(), Ok(()));
    }

    #[test]
    fn a_large_coarse_bucket_cascades_without_loss() {
        const COUNT: usize = 20_000;
        let base = 1u64 << 30;
        let mut wheel = Ns::with_capacity(COUNT);
        for token in 0..COUNT as u64 {
            let deadline = base + (token % 100_000);
            wheel.try_insert_at(deadline, token).unwrap();
        }

        let mut due = Vec::with_capacity(COUNT);
        wheel.advance_to(base + 100_000, &mut due);
        due.sort_unstable();
        assert_eq!(due, (0..COUNT as u64).collect::<Vec<_>>());
        assert_eq!(wheel.check_invariants(), Ok(()));
    }

    #[test]
    fn callback_api_avoids_an_intermediate_output_buffer() {
        let mut wheel = Ns::with_capacity(4);
        for token in 1..=4 {
            wheel.try_insert_at(token, token).unwrap();
        }
        let mut sum = 0;
        let fired = wheel.advance_to_with(4, |token| sum += token);
        assert_eq!(fired, 4);
        assert_eq!(sum, 10);
    }

    #[test]
    fn a_generation_is_retired_instead_of_wrapping() {
        let mut wheel = Ns::with_capacity(1);
        wheel.entries[0].generation = u64::MAX - 1;
        let id = wheel.try_insert_at(1, 1).unwrap();
        assert_eq!(drain(&mut wheel, 1), [1]);
        assert_eq!(wheel.retired(), 1);
        assert_eq!(wheel.available(), 0);
        assert_eq!(wheel.cancel(id), None);
        assert_eq!(wheel.try_insert_at(2, 2).unwrap_err().kind(), InsertErrorKind::Full);
        assert_eq!(wheel.check_invariants(), Ok(()));
    }

    #[test]
    fn one_position_survives_heavy_recycling_without_accepting_old_handles() {
        const ROUNDS: u64 = 100_000;
        let mut wheel = Ns::with_capacity(1);
        let mut stale = Vec::new();
        let mut due = Vec::with_capacity(1);

        for tick in 1..=ROUNDS {
            let id = wheel.try_insert_at(tick, tick).unwrap();
            if tick.is_multiple_of(10_000) {
                stale.push(id);
            }
            due.clear();
            wheel.advance_to(tick, &mut due);
            assert_eq!(due, [tick]);
        }

        let current = wheel.try_insert_at(ROUNDS + 1, ROUNDS + 1).unwrap();
        for id in stale {
            assert_eq!(wheel.cancel(id), None);
            assert!(wheel.contains(current));
        }
        assert_eq!(wheel.check_invariants(), Ok(()));
    }

    #[test]
    fn rescheduling_separates_an_unrepresentable_tick_from_an_unreachable_one() {
        let mut wheel = Ns::with_capacity(2);
        let id = wheel.try_insert_at(4, 7).unwrap();

        // `u64::MAX` is the sentinel for "no event", not a tick anyone can ask
        // for. It is out of range whatever the wheel's current reach.
        assert_eq!(wheel.reschedule_at(id, u64::MAX), Err(RescheduleError::TimeOutOfRange));

        // `MAX_TICK` is representable, so it gets the other answer: too far for
        // this wheel to file, not impossible to express. The two refusals mean
        // different things to a caller, and testing only that "something big
        // fails" cannot tell the boundary between them from either side of it.
        assert_eq!(wheel.reschedule_at(id, MAX_TICK), Err(RescheduleError::DeadlineTooFar));

        // Neither refusal disturbed the timer.
        assert_eq!(wheel.reschedule_at(id, 6), Ok(()));
        assert!(wheel.contains(id));
    }

    #[test]
    fn malformed_timer_ids_are_rejected() {
        assert!(TimerId::from_raw(0).is_none());
        assert!(TimerId::from_raw(1).is_none(), "wheel and generation fields are zero");

        // Each field is rejected on its own account. Testing them only in
        // combination, as the two cases above do, cannot tell the three
        // conditions apart: any one of them explains the rejection, so a
        // decoder that had stopped checking two of them would still pass.
        //
        // The layout is wheel in the top 32 bits, generation in the next 64,
        // and the index, stored one above its real value, in the low 32.
        const WHEEL: u128 = 1 << 96;
        const GENERATION: u128 = 1 << 32;
        assert!(TimerId::from_raw(GENERATION | 5).is_none(), "no wheel");
        assert!(TimerId::from_raw(WHEEL | 5).is_none(), "no generation");
        assert!(TimerId::from_raw(WHEEL | GENERATION).is_none(), "no index");

        let mut wheel = Ns::with_capacity(1);
        let id = wheel.try_insert_at(1, 1).unwrap();
        assert_eq!(TimerId::from_raw(id.get()), Some(id));
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    fn remove_model(model: &mut BTreeMap<u64, Vec<u64>>, deadline: u64, token: u64) {
        let bucket = model.get_mut(&deadline).expect("model contains live deadline");
        let position = bucket.iter().position(|held| *held == token).expect("model contains token");
        bucket.swap_remove(position);
        if bucket.is_empty() {
            model.remove(&deadline);
        }
    }

    #[test]
    fn randomized_battle_against_an_ordered_reference_model() {
        const CAPACITY: usize = 256;
        // Miri interprets every slot visit, and `check_invariants` walks all of
        // them each step; the full sweep would run for hours there.
        const STEPS: usize = if cfg!(miri) { 150 } else { 30_000 };

        let seeds: &[u64] =
            if cfg!(miri) { &[1] } else { &[1, 0x5eed, 0xdead_beef, 0x1234_5678_9abc_def0] };
        for &seed in seeds {
            let mut rng = Rng(seed);
            let mut wheel = Ns::with_capacity(CAPACITY);
            let mut model: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
            let mut live: HashMap<u64, (u64, TimerId)> = HashMap::new();
            let mut stale = Vec::<TimerId>::new();
            let mut next_token = 0u64;
            let mut now = 0u64;
            let mut due = Vec::with_capacity(CAPACITY);

            for step in 0..STEPS {
                match rng.next() % 12 {
                    0..=4 if live.len() < CAPACITY => {
                        next_token += 1;
                        let roll = rng.next();
                        let ahead = match roll % 8 {
                            0 => roll % (1 << 6),
                            1 => roll % (1 << 12),
                            2 => roll % (1 << 24),
                            3 => roll % (1 << 42),
                            4 => MAX_DELAY_TICKS - (roll % 10_000),
                            _ => roll % 10_000,
                        };
                        let deadline = now.checked_add(ahead).filter(|tick| *tick <= MAX_TICK);
                        if let Some(deadline) = deadline {
                            match wheel.try_insert_at(deadline, next_token) {
                                Ok(id) => {
                                    model.entry(deadline).or_default().push(next_token);
                                    live.insert(next_token, (deadline, id));
                                }
                                Err(error) => {
                                    assert_eq!(error.kind(), InsertErrorKind::DeadlineTooFar);
                                    assert_eq!(error.into_token(), next_token);
                                }
                            }
                        }
                    }
                    5 if !live.is_empty() => {
                        let victim_index = (rng.next() as usize) % live.len();
                        let victim = *live.keys().nth(victim_index).unwrap();
                        let (deadline, id) = live.remove(&victim).unwrap();
                        assert_eq!(wheel.cancel(id), Some(victim));
                        remove_model(&mut model, deadline, victim);
                        stale.push(id);
                    }
                    6 if !live.is_empty() => {
                        let victim_index = (rng.next() as usize) % live.len();
                        let victim = *live.keys().nth(victim_index).unwrap();
                        let (old_deadline, id) = live.get(&victim).copied().unwrap();
                        let ahead = rng.next() % 1_000_000;
                        let deadline = now + ahead;
                        wheel.reschedule_at(id, deadline).unwrap();
                        remove_model(&mut model, old_deadline, victim);
                        model.entry(deadline).or_default().push(victim);
                        live.insert(victim, (deadline, id));
                    }
                    7 if !stale.is_empty() => {
                        let id = stale[(rng.next() as usize) % stale.len()];
                        assert_eq!(
                            wheel.cancel(id),
                            None,
                            "step {step}: stale cancel affected a timer"
                        );
                        assert_eq!(
                            wheel.reschedule_at(id, now),
                            Err(RescheduleError::StaleId),
                            "step {step}: stale reschedule affected a timer"
                        );
                    }
                    _ => {
                        let jump = match rng.next() % 8 {
                            0 => rng.next() % (1 << 24),
                            1 => rng.next() % (1 << 36),
                            _ => rng.next() % 2_000,
                        };
                        now = now.saturating_add(jump).min(MAX_TICK);
                        due.clear();
                        wheel.advance_to(now, &mut due);
                        due.sort_unstable();

                        let expired: Vec<u64> =
                            model.range(..=now).map(|(deadline, _)| *deadline).collect();
                        let mut expected = Vec::new();
                        for deadline in expired {
                            expected
                                .extend(model.remove(&deadline).expect("deadline was just listed"));
                        }
                        expected.sort_unstable();
                        assert_eq!(due, expected, "step {step}, seed {seed:#x}, now {now}");
                        for token in &expected {
                            let (_, id) = live.remove(token).expect("fired timer is live in model");
                            stale.push(id);
                        }
                    }
                }

                assert_eq!(wheel.len(), live.len(), "step {step}, seed {seed:#x}");
                assert_eq!(wheel.check_invariants(), Ok(()), "step {step}, seed {seed:#x}");
                match model.keys().next().copied() {
                    Some(earliest) => {
                        assert!(wheel.next_wakeup_tick().unwrap() <= earliest);
                    }
                    None => assert!(wheel.is_empty()),
                }
            }
        }
    }
}
