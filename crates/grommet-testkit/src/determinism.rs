//! Declaring, and then actually checking, that a processor is reproducible.

/// Exposes everything about a run that a repeat of the same run must reproduce.
///
/// Implementing this is a *promise*, not a proof — the type system cannot see
/// inside your processor and tell whether it read the wall clock, hashed a
/// pointer, or iterated a `HashMap`. What it does is force you to say what
/// "the same result" means for your workload, so
/// [`assert_deterministic`](crate::conformance::assert_deterministic) can check
/// the promise by running the workload repeatedly and comparing.
///
/// A good digest covers the durable state a workload produced and the
/// responses it returned, in order. A digest that covers nothing will pass and
/// prove nothing.
pub trait Deterministic {
    type Digest: PartialEq + std::fmt::Debug;

    fn digest(&self) -> Self::Digest;
}
