//! Classifying failures by what they imply about durable state.
//!
//! The hardest bug in a stateful processor is not a failure: it is a failure
//! whose outcome you do not know. A commit that timed out may or may not have
//! landed, so every in-memory copy derived from the old value is now a guess.
//! Treating that case like an ordinary error is how systems silently serve
//! stale data.
//!
//! Making [`Processor::process`] return a classified error means the case
//! cannot be forgotten: you have to say which kind of failure this was.
//!
//! [`Processor::process`]: crate::processor::Processor::process

use std::convert::Infallible;

/// What a failure implies about the *authoritative* store.
///
/// This describes durable state, not the resident copy. An `Err` from
/// `process` always discards resident state: it was moved into the future and
/// is gone either way, so if your state is intact and you simply want to
/// report a business-level rejection, return `Ok(Disposition::Keep(state))`
/// instead and answer your caller yourself. `Err` means "I no longer hold
/// something I can trust."
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallout {
    /// The operation definitely did not take effect: a request rejected before
    /// it was sent, a connection refused, a precondition that failed. A retry
    /// is safe and the store is exactly where it was.
    Untouched,
    /// The outcome is unknown. It may or may not have taken effect, so nothing
    /// may assume either way until it is reconciled against the store. A retry
    /// is only safe if it is idempotent.
    ///
    /// This is the case worth counting and alerting on.
    InDoubt,
}

impl Fallout {
    pub fn is_in_doubt(self) -> bool {
        matches!(self, Self::InDoubt)
    }
}

/// An error a processor can return, classified by [`Fallout`].
///
/// Implement it on your own error type, so the runtime can count in-doubt
/// operations and the testkit can insist they reconcile, while you keep
/// whatever error detail your domain needs.
pub trait ProcessError: 'static {
    fn fallout(&self) -> Fallout;
}

/// A processor that cannot fail uses `type Error = Infallible`.
impl ProcessError for Infallible {
    fn fallout(&self) -> Fallout {
        match *self {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    enum StoreError {
        Refused,
        CommitTimedOut,
    }

    impl ProcessError for StoreError {
        fn fallout(&self) -> Fallout {
            match self {
                Self::Refused => Fallout::Untouched,
                Self::CommitTimedOut => Fallout::InDoubt,
            }
        }
    }

    #[test]
    fn a_domain_error_classifies_itself() {
        assert_eq!(StoreError::Refused.fallout(), Fallout::Untouched);
        assert!(!StoreError::Refused.fallout().is_in_doubt());
        assert_eq!(StoreError::CommitTimedOut.fallout(), Fallout::InDoubt);
        assert!(StoreError::CommitTimedOut.fallout().is_in_doubt());
    }
}
