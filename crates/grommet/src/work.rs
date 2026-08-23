//! Work items, and the envelope the runtime stamps around them.

use crate::key::{RequestId, ShardKey};
use grommet_core::ClassId;
use std::time::Duration;

/// The latency-bound half of the conventional two-class split.
///
/// A class is an index into the per-class in-flight budgets and ready rings, and
/// a runtime can carry as many as it wants. Two covers most deployments: work
/// that spends its time waiting on something else, and work that spends its time
/// burning a core. Separating them is what stops a batch of the second from
/// delaying the first.
///
/// These are named here so that every crate downstream stops redeclaring them.
/// A workload that needs a different split should define its own constants and
/// pass its own `CLASSES`; nothing in the runtime privileges these values beyond
/// [`CLASSES`] being what every `CLASSES` parameter defaults to.
pub const IO: ClassId = 0;

/// The CPU-bound half of the conventional two-class split. See [`IO`].
pub const COMPUTE: ClassId = 1;

/// The number of classes in the [`IO`] plus [`COMPUTE`] split.
///
/// Every `CLASSES` parameter in this crate defaults to this, so the common case
/// never has to name it. `Scheduler<P>`, `Router<W>`, `ShardConfig` and
/// `ShardStats` all mean the two-class versions when the argument is left off,
/// and the first two default their clock to
/// [`SystemClock`](crate::clock::SystemClock) as well.
pub const CLASSES: usize = 2;

/// A unit of work routed to the shard owning its affine key.
///
/// Work crosses a thread boundary exactly once: from whoever submitted it to
/// the shard that owns its key, which is why it must be `Send`. Nothing else
/// in the pipeline is: per-key state and the futures processing it stay on the
/// shard's own core and are free to be `!Send`.
///
/// Each of these methods is called exactly once, at submission, and the answer
/// is stamped into the envelope. The scheduler never asks again, so an
/// implementation that answered differently on a second call cannot corrupt
/// anything.
pub trait Work: Send + 'static {
    type Key: ShardKey;

    /// The idempotency key type. Use `()` when retries are not deduplicated,
    /// which leaves every duplicate-related feature inert.
    type Id: RequestId;

    /// The affine key. Everything sharing a key is processed in submission
    /// order, one item at a time, by one shard.
    fn key(&self) -> Self::Key;

    /// The caller-stable identity of this operation, which a retry reuses.
    ///
    /// When duplicate coalescing is enabled, an item whose id matches one
    /// already queued or in flight *for the same key* is not dispatched a
    /// second time; it goes to
    /// [`Processor::on_coalesced`](crate::processor::Processor::on_coalesced)
    /// instead. Returning `None` opts an individual item out.
    ///
    /// This covers concurrent retries only. A retry arriving after the original
    /// has completed is a durable-deduplication problem, which needs your store.
    fn request_id(&self) -> Option<Self::Id> {
        None
    }

    /// Which class budget and ready ring this item belongs to, in
    /// `0..CLASSES`. Submitting a class outside that range is rejected rather
    /// than silently misrouted.
    fn class(&self) -> ClassId;

    /// How long this item is worth doing. Once the deadline passes the item is
    /// discarded at dispatch rather than spending a turn, and the processor is
    /// told through [`Processor::on_expired`]. Deadlines require arrival
    /// stamping to be enabled on the router, which it is by default.
    ///
    /// [`Processor::on_expired`]: crate::processor::Processor::on_expired
    fn time_to_live(&self) -> Option<Duration> {
        None
    }
}

/// A work item plus the scheduling metadata stamped at submission.
pub struct Envelope<W: Work> {
    pub(crate) key: W::Key,
    pub(crate) class: ClassId,
    pub(crate) request_id: Option<W::Id>,
    pub(crate) expires_at: Option<Duration>,
    pub(crate) enqueued: Duration,
    pub(crate) work: W,
}

impl<W: Work> Envelope<W> {
    pub fn key(&self) -> W::Key {
        self.key
    }

    pub fn class(&self) -> ClassId {
        self.class
    }

    pub fn work(&self) -> &W {
        &self.work
    }

    pub fn into_work(self) -> W {
        self.work
    }
}

/// What a shard's scheduler stores: the work, when it was submitted so
/// queue-wait latency can be measured at dispatch, and its idempotency key so
/// the coalescing index can be cleared when the item leaves.
pub(crate) struct Stamped<W: Work> {
    pub(crate) enqueued: Duration,
    pub(crate) request_id: Option<W::Id>,
    pub(crate) work: W,
}
