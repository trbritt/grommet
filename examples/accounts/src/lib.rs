//! An account service built on `grommet`, kept as the worked example.
//!
//! It exists to prove the runtime's abstractions survive contact with a real
//! workload: durable state behind an idempotency key, a non-authoritative
//! cache, CPU-bound work that must not stall the reactor, and a commit whose
//! acknowledgement can be lost. An abstract scheduler with no concrete user is
//! how these designs acquire traits nobody can implement.
//!
//! The division of labour is the point:
//!
//! - `grommet` owns affinity, fairness, budgets, backpressure, deadlines,
//!   eviction and panic containment.
//! - This crate owns what an account *is*, when a retry is a duplicate, and
//!   what to do when a commit's outcome is unknown.
//!
//! Neither knows the other's business, and the seam between them is
//! [`Processor`](grommet::Processor).

#![cfg_attr(coverage, feature(coverage_attribute))]

pub mod domain;
pub mod frontdoor;
pub mod net;
pub mod ports;
pub mod processor;
pub mod prod;
pub mod sim;

pub mod grpc {
    tonic::include_proto!("xt.prime.v1");
}

#[cfg(all(sim, not(feature = "sim")))]
compile_error!("cfg(sim) requires Cargo feature `sim`");
