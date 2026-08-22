//! Placement of work onto the shard owning its key.
//!
//! Routing goes through a slot table rather than hashing straight onto a shard
//! index. The indirection costs one extra cached load and buys the ability to
//! change which shard owns a slot without changing how keys hash — the
//! foundation for rebalancing a skewed workload later. Work stealing is not
//! and never will be an option here, since it would break the guarantee that
//! exactly one shard owns a key's state.

use crate::clock::{Clock, SystemClock};
use crate::key::{ShardKey, mix};
use crate::mailbox::{Mailbox, TrySendError};
use crate::work::{Envelope, Work};
use std::fmt;
use std::time::Duration;

/// Slots per shard. More slots mean finer future rebalancing and a smaller
/// worst-case imbalance when the shard count is not a power of two, at two
/// bytes each.
const SLOTS_PER_SHARD: usize = 64;

/// Why a submission did not reach a shard. Each variant hands the work back, so
/// the caller can shed it deliberately — answer the client, count it, or retry
/// somewhere else — rather than discovering it was dropped.
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SubmitError<W> {
    /// The target shard's mailbox is full. Only [`Router::try_submit`] returns
    /// this; [`Router::submit`] waits instead, which is the backpressure path.
    Full(W),
    /// The target shard is gone, most likely because the runtime is shutting
    /// down.
    ShardDown(W),
    /// `Work::class()` returned a class outside `0..CLASSES`.
    InvalidClass(W),
}

impl<W> SubmitError<W> {
    pub fn into_work(self) -> W {
        match self {
            Self::Full(work) | Self::ShardDown(work) | Self::InvalidClass(work) => work,
        }
    }

    /// Rewrap the returned work, so a layer that wrapped it can hand the
    /// caller back what the caller actually submitted.
    pub fn map<T>(self, transform: impl FnOnce(W) -> T) -> SubmitError<T> {
        match self {
            Self::Full(work) => SubmitError::Full(transform(work)),
            Self::ShardDown(work) => SubmitError::ShardDown(transform(work)),
            Self::InvalidClass(work) => SubmitError::InvalidClass(transform(work)),
        }
    }
}

/// What a batch submission left undone.
///
/// A batch is not all-or-nothing: the items that landed are already being worked
/// on and cannot be taken back, so the only honest report is which ones did not.
/// Every rejected item comes back inside its [`SubmitError`], which means the caller still
/// owns the work and can answer, retry or shed it — nothing is dropped on the
/// caller's behalf.
#[derive(Debug, PartialEq, Eq)]
pub struct BatchError<W> {
    submitted: usize,
    rejected: Vec<SubmitError<W>>,
}

// A manual impl: the error is empty whatever `W` is, and a derive would demand
// `W: Default` for no reason.
impl<W> Default for BatchError<W> {
    fn default() -> Self {
        Self { submitted: 0, rejected: Vec::new() }
    }
}

impl<W> BatchError<W> {
    /// How many items of the batch were accepted.
    pub fn submitted(&self) -> usize {
        self.submitted
    }

    /// The items that were not, each with the reason it was refused.
    pub fn rejected(&self) -> &[SubmitError<W>] {
        &self.rejected
    }

    /// Take the rejected items, to answer or resubmit them.
    pub fn into_rejected(self) -> Vec<SubmitError<W>> {
        self.rejected
    }

    /// Take just the work back, for a caller that treats every rejection the
    /// same way.
    pub fn into_work(self) -> impl Iterator<Item = W> {
        self.rejected.into_iter().map(SubmitError::into_work)
    }

    fn record(&mut self, outcome: Result<(), SubmitError<W>>) {
        match outcome {
            Ok(()) => self.submitted += 1,
            Err(error) => self.rejected.push(error),
        }
    }

    /// Nothing refused is not an error. The `Vec` never allocates on that
    /// path, so a batch that lands whole costs no allocation at all.
    fn into_result(self) -> Result<(), Self> {
        if self.rejected.is_empty() { Ok(()) } else { Err(self) }
    }
}

impl<W> fmt::Display for BatchError<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} of {} submitted items were refused",
            self.rejected.len(),
            self.submitted + self.rejected.len()
        )
    }
}

impl<W: fmt::Debug> std::error::Error for BatchError<W> {}

pub struct Router<W: Work, C: Clock = SystemClock, const CLASSES: usize = 2> {
    shards: Vec<Mailbox<Envelope<W>>>,
    slots: Box<[u16]>,
    mask: u64,
    clock: C,
    stamp_arrival: bool,
}

impl<W: Work, C: Clock, const CLASSES: usize> Router<W, C, CLASSES> {
    pub fn new(shards: Vec<Mailbox<Envelope<W>>>, clock: C) -> Self {
        Self::with_options(shards, clock, true)
    }

    /// `stamp_arrival` records a submission timestamp on every item, which
    /// powers queue-wait metrics and deadlines. Turning it off saves a clock
    /// read per submission — worth roughly a few percent of a core at millions
    /// of items per second — at the cost of both features.
    pub fn with_options(shards: Vec<Mailbox<Envelope<W>>>, clock: C, stamp_arrival: bool) -> Self {
        assert!(!shards.is_empty(), "a router needs at least one shard");
        assert!(
            shards.len() <= usize::from(u16::MAX) + 1,
            "a router supports at most {} shards",
            usize::from(u16::MAX) + 1
        );
        let count = (shards.len() * SLOTS_PER_SHARD).next_power_of_two();
        let slots = (0..count).map(|slot| (slot % shards.len()) as u16).collect::<Box<[u16]>>();
        Self { shards, slots, mask: count as u64 - 1, clock, stamp_arrival }
    }

    pub fn shards(&self) -> usize {
        self.shards.len()
    }

    /// Which shard owns `key`. Stable for the lifetime of the router.
    #[inline]
    pub fn shard_index(&self, key: W::Key) -> usize {
        usize::from(self.slots[(mix(key.shard_hash()) & self.mask) as usize])
    }

    /// Submit, waiting if the target shard's mailbox is full.
    ///
    /// That wait is the system's backpressure: when a shard is saturated it
    /// stops admitting, its mailbox fills, and this call suspends the caller
    /// rather than growing an unbounded queue.
    pub async fn submit(&self, work: W) -> Result<(), SubmitError<W>> {
        let envelope = self.stamp(work, self.arrival())?;
        self.send(envelope).await
    }

    /// Submit without ever waiting, reporting a full mailbox instead.
    ///
    /// Use this when shedding load is better than queueing it, which is often
    /// true under a latency objective.
    pub fn try_submit(&self, work: W) -> Result<(), SubmitError<W>> {
        let envelope = self.stamp(work, self.arrival())?;
        self.try_send(envelope)
    }

    /// Submit many items, waiting on backpressure, and report everything that
    /// did not land.
    ///
    /// Every item is attempted. One shard being unreachable does not stop
    /// items bound for the others, so a batch spanning shards is not held
    /// hostage by the worst of them.
    ///
    /// The whole batch shares one arrival stamp. That is not an approximation:
    /// the caller held these items at one instant and handed them over at one
    /// instant, so that instant is when they arrived. Queue-wait is measured
    /// from it, and a deadline runs from it — including across a wait for
    /// mailbox space, because that wait is queueing, which is exactly what a
    /// deadline is meant to account for.
    ///
    /// # Errors
    ///
    /// [`BatchError`] carries every rejected item back with its reason, so a
    /// caller can answer, retry or shed each one. Items not named there were
    /// accepted.
    pub async fn submit_batch<I>(&self, work: I) -> Result<(), BatchError<W>>
    where
        I: IntoIterator<Item = W>,
    {
        let arrival = self.arrival();
        let mut batch = BatchError::default();
        for item in work {
            match self.stamp(item, arrival) {
                Ok(envelope) => batch.record(self.send(envelope).await),
                Err(error) => batch.rejected.push(error),
            }
        }
        batch.into_result()
    }

    /// Submit many items without ever waiting, reporting everything that did
    /// not land.
    ///
    /// The shedding counterpart of [`submit_batch`], and the one to reach for
    /// under a latency objective: a full mailbox rejects that item and the
    /// batch carries on rather than blocking behind it.
    ///
    /// [`submit_batch`]: Router::submit_batch
    pub fn try_submit_batch<I>(&self, work: I) -> Result<(), BatchError<W>>
    where
        I: IntoIterator<Item = W>,
    {
        let arrival = self.arrival();
        let mut batch = BatchError::default();
        for item in work {
            match self.stamp(item, arrival) {
                Ok(envelope) => batch.record(self.try_send(envelope)),
                Err(error) => batch.rejected.push(error),
            }
        }
        batch.into_result()
    }

    async fn send(&self, envelope: Envelope<W>) -> Result<(), SubmitError<W>> {
        let index = self.shard_index(envelope.key);
        self.shards[index]
            .send(envelope)
            .await
            .map_err(|closed| SubmitError::ShardDown(closed.into_inner().work))
    }

    fn try_send(&self, envelope: Envelope<W>) -> Result<(), SubmitError<W>> {
        let index = self.shard_index(envelope.key);
        match self.shards[index].try_send(envelope) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(envelope)) => Err(SubmitError::Full(envelope.work)),
            Err(TrySendError::Closed(envelope)) => Err(SubmitError::ShardDown(envelope.work)),
        }
    }

    /// The arrival stamp for a submission, or `None` when stamping is off.
    ///
    /// Separated from [`Router::stamp`] so a batch can read the clock once and
    /// spend that reading across every item, the way the reactor spends one
    /// reading across a turn.
    #[inline]
    fn arrival(&self) -> Option<Duration> {
        self.stamp_arrival.then(|| self.clock.now())
    }

    fn stamp(&self, work: W, arrival: Option<Duration>) -> Result<Envelope<W>, SubmitError<W>> {
        let class = work.class();
        if usize::from(class) >= CLASSES {
            return Err(SubmitError::InvalidClass(work));
        }
        let key = work.key();
        let (enqueued, expires_at) = match arrival {
            Some(now) => (now, work.time_to_live().map(|ttl| now.saturating_add(ttl))),
            // Without an arrival stamp a deadline would be measured from the
            // clock's origin, which would expire everything immediately, so it
            // is ignored rather than misapplied.
            None => (Duration::ZERO, None),
        };
        let request_id = work.request_id();
        Ok(Envelope { key, class, request_id, expires_at, enqueued, work })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::ManualClock;
    use crate::mailbox::{Inbox, channel};
    use grommet_core::ClassId;

    #[derive(Debug)]
    struct Item {
        key: u64,
        class: ClassId,
        ttl: Option<Duration>,
    }

    impl Item {
        fn new(key: u64) -> Self {
            Self { key, class: 0, ttl: None }
        }
    }

    impl Work for Item {
        type Key = u64;
        type Id = ();
        fn key(&self) -> u64 {
            self.key
        }
        fn class(&self) -> ClassId {
            self.class
        }
        fn time_to_live(&self) -> Option<Duration> {
            self.ttl
        }
    }

    fn router(shards: usize) -> (Router<Item, ManualClock, 2>, Vec<Inbox<Envelope<Item>>>) {
        let clock = ManualClock::new();
        let (senders, receivers): (Vec<_>, Vec<_>) =
            (0..shards).map(|_| channel(4)).collect::<Vec<_>>().into_iter().unzip();
        (Router::new(senders, clock), receivers)
    }

    #[test]
    fn placement_is_stable_total_and_reaches_every_shard() {
        let (router, _receivers) = router(7);
        assert_eq!(router.shards(), 7, "the fan-out a caller sizes its own sharding against");
        let mut reached = vec![false; 7];
        for key in 0..10_000 {
            let shard = router.shard_index(key);
            assert!(shard < 7);
            assert_eq!(shard, router.shard_index(key), "placement must be stable");
            reached[shard] = true;
        }
        assert!(reached.into_iter().all(|hit| hit));
    }

    #[tokio::test]
    async fn a_batch_reaches_every_shard_that_owns_one_of_its_keys() {
        // Mailboxes deep enough for the whole batch: this is about where items
        // land, and nothing is draining, so any backpressure here would just
        // be the test deadlocking on itself.
        let clock = ManualClock::new();
        let (senders, mut receivers): (Vec<_>, Vec<_>) =
            (0..4).map(|_| channel(64)).collect::<Vec<_>>().into_iter().unzip();
        let router = Router::<Item, ManualClock, 2>::new(senders, clock);
        let keys: Vec<u64> = (0..64).collect();
        let expected: Vec<usize> = keys.iter().map(|key| router.shard_index(*key)).collect();

        router.submit_batch(keys.iter().map(|key| Item::new(*key))).await.unwrap();

        // Every item is where single submission would have put it: batching
        // amortizes the submission, it does not re-route anything.
        for (shard, receiver) in receivers.iter_mut().enumerate() {
            let mut landed = Vec::new();
            while let Ok(envelope) = receiver.try_recv() {
                landed.push(envelope.key());
            }
            let want: Vec<u64> = keys
                .iter()
                .zip(&expected)
                .filter(|(_, owner)| **owner == shard)
                .map(|(key, _)| *key)
                .collect();
            assert_eq!(landed, want, "shard {shard} received the wrong items, or the wrong order");
        }
    }

    #[tokio::test]
    async fn one_batch_shares_one_arrival_stamp() {
        let clock = ManualClock::new();
        let (sender, mut receiver) = channel(8);
        let router = Router::<Item, ManualClock, 2>::new(vec![sender], clock.clone());
        clock.set(Duration::from_secs(3));

        let ttl = Duration::from_millis(10);
        let batch = (0..4).map(|key| Item { key, class: 0, ttl: Some(ttl) });
        router.submit_batch(batch).await.unwrap();

        // The caller held these at one instant and handed them over at one
        // instant, so one instant is when they arrived. Reading the clock per
        // item would also stagger their deadlines, which nothing asked for.
        for _ in 0..4 {
            let envelope = receiver.try_recv().expect("the batch landed");
            assert_eq!(envelope.enqueued, Duration::from_secs(3));
            assert_eq!(envelope.expires_at, Some(Duration::from_secs(3) + ttl));
        }
    }

    #[tokio::test]
    async fn a_batch_hands_back_every_item_it_could_not_place() {
        let (sender, mut receiver) = channel(2);
        let router = Router::<Item, ManualClock, 2>::new(vec![sender], ManualClock::new());

        // Two fit; one is refused for a bad class whatever the room; the rest
        // overflow. Nothing may be silently dropped.
        let batch = vec![
            Item::new(1),
            Item { key: 2, class: 9, ttl: None },
            Item::new(3),
            Item::new(4),
            Item::new(5),
        ];
        let error = router.try_submit_batch(batch).expect_err("the mailbox holds two");

        assert_eq!(error.submitted(), 2);
        assert_eq!(error.rejected().len(), 3);
        assert!(matches!(error.rejected()[0], SubmitError::InvalidClass(Item { key: 2, .. })));
        assert!(matches!(error.rejected()[1], SubmitError::Full(Item { key: 4, .. })));
        assert!(matches!(error.rejected()[2], SubmitError::Full(Item { key: 5, .. })));
        assert!(error.to_string().contains("3 of 5"));

        let returned: Vec<u64> = error.into_work().map(|item| item.key).collect();
        assert_eq!(returned, vec![2, 4, 5], "the caller gets its own work back to answer");

        let landed: Vec<u64> =
            std::iter::from_fn(|| receiver.try_recv().ok()).map(|e| e.key()).collect();
        assert_eq!(landed, vec![1, 3], "and exactly what was accepted was delivered");
    }

    #[tokio::test]
    async fn one_unreachable_shard_does_not_strand_the_rest_of_a_batch() {
        let clock = ManualClock::new();
        let (first, mut receiver) = channel(8);
        let (second, closed) = channel(8);
        drop(closed);
        let router = Router::<Item, ManualClock, 2>::new(vec![first, second], clock);

        // Find a key each shard owns, so the batch genuinely spans both.
        let alive =
            (0..1000).find(|key| router.shard_index(*key) == 0).expect("shard 0 owns a key");
        let gone = (0..1000).find(|key| router.shard_index(*key) == 1).expect("shard 1 owns a key");

        let error = router
            .submit_batch(vec![Item::new(gone), Item::new(alive)])
            .await
            .expect_err("one shard is gone");
        assert_eq!(error.submitted(), 1);
        assert!(matches!(error.rejected()[0], SubmitError::ShardDown(_)));
        assert_eq!(
            receiver.try_recv().map(|envelope| envelope.key()),
            Ok(alive),
            "the living shard is not held hostage by the dead one"
        );
    }

    #[tokio::test]
    async fn an_empty_batch_is_accepted_and_does_nothing() {
        let (router, mut receivers) = router(2);
        router.submit_batch(Vec::<Item>::new()).await.unwrap();
        router.try_submit_batch(Vec::<Item>::new()).unwrap();
        assert!(receivers[0].try_recv().is_err());
    }

    #[tokio::test]
    async fn submission_stamps_arrival_and_deadline() {
        let clock = ManualClock::new();
        let (sender, mut receiver) = channel(4);
        let router = Router::<Item, ManualClock, 2>::new(vec![sender], clock.clone());
        clock.set(Duration::from_secs(5));

        let ttl = Duration::from_millis(250);
        // Class 1 rather than 0: an envelope that lost the class on the way
        // would still route to a real ring, just the wrong one, so the value
        // has to differ from the default to be worth asserting.
        router.submit(Item { key: 1, class: 1, ttl: Some(ttl) }).await.unwrap();
        let envelope = receiver.recv().await.unwrap();
        assert_eq!(envelope.enqueued, Duration::from_secs(5));
        assert_eq!(envelope.expires_at, Some(Duration::from_secs(5) + ttl));
        assert_eq!(envelope.key(), 1);
        assert_eq!(envelope.class(), 1);
    }

    #[tokio::test]
    async fn disabling_arrival_stamping_also_disables_deadlines() {
        let clock = ManualClock::new();
        clock.set(Duration::from_secs(5));
        let (sender, mut receiver) = channel(4);
        let router = Router::<Item, ManualClock, 2>::with_options(vec![sender], clock, false);

        router
            .submit(Item { key: 1, class: 0, ttl: Some(Duration::from_millis(1)) })
            .await
            .unwrap();
        let envelope = receiver.recv().await.unwrap();
        assert_eq!(envelope.enqueued, Duration::ZERO);
        assert_eq!(envelope.expires_at, None, "a deadline without an origin must not be applied");
    }

    #[test]
    fn an_out_of_range_class_is_rejected_and_the_work_is_returned() {
        let (router, _receivers) = router(1);
        let error = router.try_submit(Item { key: 1, class: 9, ttl: None }).unwrap_err();
        assert!(matches!(error, SubmitError::InvalidClass(_)));
        assert_eq!(error.into_work().key, 1);
    }

    #[test]
    fn a_full_mailbox_sheds_instead_of_blocking() {
        let (router, _receivers) = router(1);
        for key in 0..4 {
            router.try_submit(Item::new(key)).expect("mailbox has room");
        }
        let error = router.try_submit(Item::new(4)).unwrap_err();
        assert!(matches!(error, SubmitError::Full(_)));
        assert_eq!(error.into_work().key, 4, "shed work is handed back to its submitter");
    }

    #[tokio::test]
    async fn a_closed_shard_is_reported_by_both_submission_paths() {
        let (sender, receiver) = channel(1);
        drop(receiver);
        let router = Router::<Item, ManualClock, 2>::new(vec![sender], ManualClock::new());

        assert!(matches!(router.try_submit(Item::new(1)), Err(SubmitError::ShardDown(_))));
        assert!(matches!(router.submit(Item::new(2)).await, Err(SubmitError::ShardDown(_))));
    }

    #[test]
    fn the_slot_table_spreads_evenly_when_shards_are_not_a_power_of_two() {
        let (router, _receivers) = router(9);
        let mut counts = vec![0usize; 9];
        for key in 0..90_000u64 {
            counts[router.shard_index(key)] += 1;
        }
        let fair = 10_000;
        for count in counts {
            let skew = (count as f64 - fair as f64).abs() / fair as f64;
            assert!(skew < 0.05, "shard skew {skew} exceeds the slot table's bound");
        }
    }
}
