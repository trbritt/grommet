//! Testing tools for schedulers and processors built on `grommet`.
//!
//! The runtime can be proven correct on its own, but your processor is where
//! the interesting failures live: a lost acknowledgement, a retry that applies
//! twice, a cache that outlives the value it caches. This crate provides the
//! machinery to find those, generic over your fault labels and your workload.
//!
//! - [`FaultPlan`] injects failures at points you label, either in a fixed
//!   order, at one chosen position, or steered by a fuzzer.
//! - [`conformance::single_fault_campaign`] runs your workload once per
//!   reachable failure position and insists every one of them recovers.
//! - [`conformance::assert_idempotent`] replays one request and insists only
//!   the first application changed anything.
//! - [`conformance::scheduler_sweep`] drives your scheduler configuration
//!   through a mixed workload, checking every structural invariant and
//!   measuring the starvation bound.
//! - [`Deterministic`] plus [`conformance::assert_deterministic`] turn "this is
//!   reproducible" from an unchecked claim into a test.
//!
//! Nothing here is a marker trait that grants correctness by being implemented.
//! Every check runs your code and reports what it observed.

#![deny(unsafe_code)]

pub mod conformance;
pub mod determinism;
pub mod fault;

pub use conformance::{CampaignReport, CaseOutcome, SweepReport, SweepSpec};
pub use determinism::Deterministic;
pub use fault::{FaultPlan, FaultPoint};

pub use grommet::clock::ManualClock;
pub use grommet::offload::InlineOffload;
