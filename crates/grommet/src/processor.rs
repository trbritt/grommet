//! The behaviour a shard drives: the one trait every user of this crate writes.

use crate::error::ProcessError;
use crate::work::Work;
use grommet_core::Disposition;

/// The affine key type reached through a processor's work type.
pub type KeyOf<P> = <<P as Processor>::Work as Work>::Key;

/// Processes work for keys owned by one shard.
///
/// # Threading
///
/// A processor instance belongs to exactly one shard, running on one pinned
/// core, and is never shared across threads. Its futures are therefore allowed
/// to be `!Send`, which is the point of the whole design: connection pools,
/// caches and per-key state can be held behind `Rc` and mutated through `Cell`
/// or `RefCell` with no synchronization at all. If you need `Send` futures and
/// work stealing, use an ordinary multi-threaded executor — this crate is
/// deliberately the other thing.
///
/// `Clone` is required because each dispatched item owns a handle for the
/// duration of its future, so the future can be `'static` and live in a
/// non-boxed `FuturesUnordered`. Clone should be cheap: hold shared
/// dependencies behind `Rc` and clone that.
///
/// # State ownership
///
/// While `process` runs, it holds the *only* copy of that key's resident
/// state. No other future for the same key can be in flight, so state needs no
/// locking. Returning [`Disposition::Keep`] hands it back for the next
/// dispatch, and [`Disposition::Drop`] declares it untrustworthy so the next
/// dispatch reloads from the authoritative source. Any operation whose outcome
/// is unknown — a timeout, a lost acknowledgement — must return `Drop`.
#[allow(async_fn_in_trait)]
pub trait Processor: Clone + 'static {
    type Work: Work;
    /// Per-key state kept resident between dispatches. Use `()` if the work is
    /// stateless.
    type State: 'static;
    /// Failures this processor can report. Use
    /// [`Infallible`](std::convert::Infallible) if it cannot fail.
    type Error: ProcessError;

    /// Process one item. `state` is the key's resident state, or `None` when
    /// nothing is resident and it must be loaded.
    ///
    /// Returning `Err` always discards the key's resident state — it was moved
    /// into this future and cannot be recovered — so the next dispatch reloads.
    /// A failure that leaves your state intact is not an error here: return
    /// `Ok(Disposition::Keep(state))` and answer your caller yourself.
    async fn process(
        &self,
        key: KeyOf<Self>,
        state: Option<Self::State>,
        work: Self::Work,
    ) -> Result<Disposition<Self::State>, Self::Error>;

    /// Observe a failure. Runs on the shard's turn, so it must not block.
    ///
    /// The runtime already counts these and separates out the in-doubt ones;
    /// this is for logging, alerting, or answering a waiting caller.
    fn on_error(&self, key: KeyOf<Self>, error: &Self::Error) {
        let _ = (key, error);
    }

    /// Called for an item suppressed because another item with the same
    /// request id was already queued or in flight for this key.
    ///
    /// Only reached when duplicate coalescing is enabled. The suppressed work
    /// never runs, so if a caller is waiting on it, this is where it is
    /// answered. It runs on the shard's turn and must not block.
    fn on_coalesced(&self, key: KeyOf<Self>, work: Self::Work) {
        let _ = (key, work);
    }

    /// Flush state that is about to be released, because the key went idle
    /// past its window or the resident cap is under pressure.
    ///
    /// The key is quiesced for the duration: newly arriving work queues behind
    /// this call rather than racing it, and will see no resident state
    /// afterwards. That makes a write-back flush safe here.
    async fn on_evict(&self, key: KeyOf<Self>, state: Self::State) {
        let _ = (key, state);
    }

    /// Called for work discarded at dispatch because its deadline had passed.
    /// This is where a caller waiting on a reply is told it was shed. It runs
    /// on the shard's turn, so it must not block.
    fn on_expired(&self, key: KeyOf<Self>, work: Self::Work) {
        let _ = (key, work);
    }
}

/// What a shard does when a [`Processor::process`] future panics.
///
/// The panic is always caught — otherwise it would unwind the shard's reactor
/// loop, killing every key that shard owns and leaving the scheduler's
/// in-flight accounting permanently wrong. The state that was moved into the
/// panicking future is gone either way, so the key is always treated as
/// [`Disposition::Drop`] and reloads on its next dispatch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PanicPolicy {
    /// Count the panic, drop the key's state, and keep serving.
    ///
    /// The work itself was moved into the panicking future and dies with it, so
    /// any reply channel it carried is dropped rather than answered. A caller
    /// awaiting that channel observes a cancellation, not a hang.
    #[default]
    Continue,
    /// Publish metrics and abort the process. Choose this when a panic means an
    /// invariant you rely on is already broken and continuing would do damage.
    Abort,
}
