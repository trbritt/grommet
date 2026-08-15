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
use crate::work::{Envelope, Work};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;

/// Slots per shard. More slots mean finer future rebalancing and a smaller
/// worst-case imbalance when the shard count is not a power of two, at two
/// bytes each.
const SLOTS_PER_SHARD: usize = 64;

/// Why a submission did not reach a shard. Each variant hands the work back, so
/// the caller can shed it deliberately — answer the client, count it, or retry
/// somewhere else — rather than discovering it was dropped.
#[derive(Debug, PartialEq, Eq)]
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

pub struct Router<W: Work, C: Clock = SystemClock, const CLASSES: usize = 2> {
    shards: Vec<mpsc::Sender<Envelope<W>>>,
    slots: Box<[u16]>,
    mask: u64,
    clock: C,
    stamp_arrival: bool,
}

impl<W: Work, C: Clock, const CLASSES: usize> Router<W, C, CLASSES> {
    pub fn new(shards: Vec<mpsc::Sender<Envelope<W>>>, clock: C) -> Self {
        Self::with_options(shards, clock, true)
    }

    /// `stamp_arrival` records a submission timestamp on every item, which
    /// powers queue-wait metrics and deadlines. Turning it off saves a clock
    /// read per submission — worth roughly a few percent of a core at millions
    /// of items per second — at the cost of both features.
    pub fn with_options(
        shards: Vec<mpsc::Sender<Envelope<W>>>,
        clock: C,
        stamp_arrival: bool,
    ) -> Self {
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
        let envelope = self.stamp(work)?;
        let index = self.shard_index(envelope.key);
        self.shards[index]
            .send(envelope)
            .await
            .map_err(|error| SubmitError::ShardDown(error.0.work))
    }

    /// Submit without ever waiting, reporting a full mailbox instead.
    ///
    /// Use this when shedding load is better than queueing it, which is often
    /// true under a latency objective.
    pub fn try_submit(&self, work: W) -> Result<(), SubmitError<W>> {
        let envelope = self.stamp(work)?;
        let index = self.shard_index(envelope.key);
        match self.shards[index].try_send(envelope) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(envelope)) => Err(SubmitError::Full(envelope.work)),
            Err(TrySendError::Closed(envelope)) => Err(SubmitError::ShardDown(envelope.work)),
        }
    }

    fn stamp(&self, work: W) -> Result<Envelope<W>, SubmitError<W>> {
        let class = work.class();
        if usize::from(class) >= CLASSES {
            return Err(SubmitError::InvalidClass(work));
        }
        let key = work.key();
        let (enqueued, expires_at) = if self.stamp_arrival {
            let now = self.clock.now();
            (now, work.time_to_live().map(|ttl| now.saturating_add(ttl)))
        } else {
            // Without an arrival stamp a deadline would be measured from the
            // clock's origin, which would expire everything immediately, so it
            // is ignored rather than misapplied.
            (Duration::ZERO, None)
        };
        let request_id = work.request_id();
        Ok(Envelope { key, class, request_id, expires_at, enqueued, work })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::ManualClock;
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

    fn router(
        shards: usize,
    ) -> (Router<Item, ManualClock, 2>, Vec<mpsc::Receiver<Envelope<Item>>>) {
        let clock = ManualClock::new();
        let (senders, receivers): (Vec<_>, Vec<_>) =
            (0..shards).map(|_| mpsc::channel(4)).collect::<Vec<_>>().into_iter().unzip();
        (Router::new(senders, clock), receivers)
    }

    #[test]
    fn placement_is_stable_total_and_reaches_every_shard() {
        let (router, _receivers) = router(7);
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
    async fn submission_stamps_arrival_and_deadline() {
        let clock = ManualClock::new();
        let (sender, mut receiver) = mpsc::channel(4);
        let router = Router::<Item, ManualClock, 2>::new(vec![sender], clock.clone());
        clock.set(Duration::from_secs(5));

        let ttl = Duration::from_millis(250);
        router.submit(Item { key: 1, class: 0, ttl: Some(ttl) }).await.unwrap();
        let envelope = receiver.recv().await.unwrap();
        assert_eq!(envelope.enqueued, Duration::from_secs(5));
        assert_eq!(envelope.expires_at, Some(Duration::from_secs(5) + ttl));
        assert_eq!(envelope.key(), 1);
        assert_eq!(envelope.class(), 0);
    }

    #[tokio::test]
    async fn disabling_arrival_stamping_also_disables_deadlines() {
        let clock = ManualClock::new();
        clock.set(Duration::from_secs(5));
        let (sender, mut receiver) = mpsc::channel(4);
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
        let (sender, receiver) = mpsc::channel(1);
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
