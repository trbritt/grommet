//! Hardware-aware runtime layout for `grommet`.
//!
//! Thread-per-core placement is only as good as its picture of the machine. The
//! previous picture was a flat list of logical CPU identifiers split by taking
//! the first `n`, which assumes identifier adjacency means hardware adjacency.
//! It does not, on any machine worth tuning for.
//!
//! Discovery is libhwloc, unmodified and unwrapped. It already knows the cache
//! hierarchy, NUMA distances, Windows processor groups and the cpukinds ranking
//! for hybrid parts, so there is no local model to drift from it — `plan()`
//! reads an `hwlocality::Topology` directly.
//!
//! Two things hwloc does not answer are supplied here:
//!
//! - **CFS bandwidth limits.** hwloc reports which CPUs you may run on, not how
//!   much CPU time you may use. A container capped at two cores on a
//!   ninety-six core host still sees ninety-six. See [`cgroup`].
//! - **Whether this build is fit to be measured.** A debug build runs the
//!   scheduler's invariant check every reactor iteration.
//!
//! ```
//! use grommet_topology::{Workload, detect};
//!
//! let layout = detect(&Workload::default()).expect("read this machine");
//! for note in &layout.notes {
//!     eprintln!("topology: {note}");
//! }
//! ```
//!
//! The plan is then applied by the threads it describes, each binding itself as
//! it starts (see [`bind`]), and the split it chose can be checked against what
//! the workload turned out to need (see [`calibrate`]) — offline, as a
//! configuration change, never as a live adjustment.
//!
//! # Testing without the hardware
//!
//! hwloc builds a topology from a synthetic description or an XML capture, so
//! a two-socket SMT server, a throttled container or a hybrid laptop are all
//! unit tests on whatever machine is to hand. See `plan()` for how.
//!
//! # Without hwloc
//!
//! The `hwloc` feature is on by default and is what makes any of the above
//! true. Turning it off replaces discovery with `fallback::detect`, which
//! plans from [`std::thread::available_parallelism`] alone: one memory node,
//! no SMT knowledge, and no binding. Every type here stays, so code that plans
//! and starts a runtime compiles either way — it simply gets a layout that
//! admits it is guessing.

#![deny(unsafe_code)]

pub mod bind;
pub mod calibrate;
pub mod cgroup;
#[cfg(not(feature = "hwloc"))]
pub mod fallback;
pub mod plan;

pub use bind::Bound;
pub use calibrate::{Advice, Observation, Verdict};
pub use cgroup::{Quota, QuotaSource};
pub use plan::{OffloadPool, Plan, ShardPlacement, Workload};
#[cfg(feature = "hwloc")]
pub use plan::{plan, plan_shared};

/// Plan a layout for the machine this process is running on.
#[cfg(feature = "hwloc")]
pub fn detect(workload: &Workload) -> Result<Plan, String> {
    let topology = hwlocality::Topology::new()
        .map_err(|error| format!("hwloc could not read this machine: {error}"))?;
    Ok(plan_shared(std::sync::Arc::new(topology), workload, cgroup::detect()))
}

#[cfg(not(feature = "hwloc"))]
pub use fallback::detect;
