//! Planning a layout without reading the machine.
//!
//! This is what the crate does when the `hwloc` feature is off: a deliberate
//! opt-out, taken by builds that cannot afford a C toolchain, a network fetch
//! at build time, or a dependency on libhwloc at all: musl images,
//! air-gapped builds, distribution packaging.
//!
//! It plans from [`std::thread::available_parallelism`], which is the honest
//! extent of what the standard library knows. That number reflects CPU
//! affinity and, on Linux, cgroup quota, so the *count* is usually right. What
//! it cannot tell us is the shape: which CPUs share a physical core, which
//! memory node a core belongs to, or whether some cores are faster than
//! others. So this assigns CPU indices `0..n` in order, puts everything on one
//! memory node, and says so in the plan's notes.
//!
//! The resulting plan reports `can_bind_cpu: false` and `can_bind_memory:
//! false`, and its `bind_*` calls do nothing. Nothing here pretends otherwise:
//! a layout that claimed placement it never performed would turn every latency
//! measurement into a measurement of the OS scheduler instead.

use crate::plan::{OffloadPool, Plan, ShardPlacement, Workload};

/// Plan a layout from the CPU count alone.
///
/// The signature matches the hwloc-backed `detect`, so a caller compiles
/// unchanged either way. The `Result` is likewise kept: it never fails here,
/// but a caller that handled the hwloc error should not have to change shape
/// to build without it.
pub fn detect(workload: &Workload) -> Result<Plan, String> {
    let visible =
        std::thread::available_parallelism().map(std::num::NonZeroUsize::get).unwrap_or(1);
    Ok(from_cpu_count(visible, workload))
}

/// The planning rule, separated from reading the CPU count so it can be tested
/// against machine sizes this one does not have.
pub(crate) fn from_cpu_count(visible: usize, workload: &Workload) -> Plan {
    let visible = visible.max(1);
    let mut notes = vec![format!(
        "built without the `topology` feature: planning for {visible} logical CPUs from \
         available_parallelism, with no knowledge of SMT siblings, memory nodes or core kinds, \
         and no thread binding"
    )];

    // Always leave at least one reactor. A machine too small to honour the
    // reservation still has to be able to run.
    let budget = visible.saturating_sub(workload.reserve_cores).max(1);
    if budget < visible.saturating_sub(workload.reserve_cores).max(budget) {
        notes.push(format!("{visible} CPUs is too few to reserve {}", workload.reserve_cores));
    }

    // Round the compute split rather than truncating it, and never let it take
    // the last reactor or claim a worker there is no core for.
    let compute = ((budget as f64) * workload.compute_fraction).round() as usize;
    let compute = compute.min(budget.saturating_sub(1));
    let reactors = budget - compute;

    let shards = (0..reactors).map(|cpu| ShardPlacement { cpu, node: 0 }).collect();
    let offload = if compute == 0 {
        Vec::new()
    } else {
        vec![OffloadPool { node: 0, cpus: (reactors..reactors + compute).collect() }]
    };

    if workload.isolate_smt {
        notes.push(
            "SMT isolation was requested but cannot be honoured without hwloc: sibling threads \
             are indistinguishable from separate cores here, so a compute worker may share a \
             physical core with a reactor"
                .to_owned(),
        );
    }
    if cfg!(debug_assertions) {
        notes.push(
            "built with debug assertions: the scheduler checks its invariants every reactor \
             iteration, which is linear in resident keys, so timing from this build describes a \
             different program"
                .to_owned(),
        );
    }

    Plan { shards, offload, can_bind_cpu: false, can_bind_memory: false, notes }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workload() -> Workload {
        Workload { compute_fraction: 0.25, reserve_cores: 1, ..Workload::default() }
    }

    #[test]
    fn a_plan_splits_reactors_from_compute_and_never_overlaps_them() {
        let layout = from_cpu_count(16, &workload());
        assert_eq!(layout.shards.len(), 11, "15 usable, a quarter rounded to compute");
        assert_eq!(layout.offload_workers(), 4);

        let reactors: Vec<usize> = layout.shards.iter().map(|shard| shard.cpu).collect();
        let workers = &layout.offload.first().expect("a compute pool").cpus;
        assert_eq!(reactors, (0..11).collect::<Vec<_>>());
        assert_eq!(workers, &(11..15).collect::<Vec<_>>());
        assert!(
            workers.iter().all(|cpu| !reactors.contains(cpu)),
            "a CPU handed to both would be the one placement mistake this can still make"
        );
    }

    #[test]
    fn every_machine_size_still_produces_a_runnable_plan() {
        for cpus in [0, 1, 2, 3] {
            let layout = from_cpu_count(cpus, &workload());
            assert!(
                !layout.shards.is_empty(),
                "{cpus} CPUs left nothing to dispatch from; a runtime cannot start"
            );
        }
    }

    #[test]
    fn a_pure_io_workload_reserves_no_compute_cores() {
        let layout = from_cpu_count(8, &Workload { compute_fraction: 0.0, ..workload() });
        assert!(layout.offload.is_empty());
        assert_eq!(layout.shards.len(), 7);
    }

    #[test]
    fn a_compute_heavy_workload_still_leaves_a_reactor() {
        let layout = from_cpu_count(4, &Workload { compute_fraction: 1.0, ..workload() });
        assert_eq!(layout.shards.len(), 1, "a shard with no reactor cannot dispatch anything");
        assert_eq!(layout.offload_workers(), 2);
    }

    #[test]
    fn the_plan_admits_what_it_does_not_know() {
        let layout = from_cpu_count(8, &workload());
        assert!(!layout.can_bind_cpu, "nothing here can bind a thread");
        assert!(!layout.can_bind_memory);
        assert!(
            layout.notes.iter().any(|note| note.contains("without the `topology` feature")),
            "an operator reading the notes has to be able to tell which planner ran"
        );
        assert!(
            layout.notes.iter().any(|note| note.contains("SMT")),
            "a requested isolation that cannot be honoured must be reported, not ignored"
        );
    }

    #[test]
    fn binding_reports_that_it_did_nothing() {
        let layout = from_cpu_count(8, &workload());
        let placement = layout.shards.first().expect("a reactor");
        assert!(layout.bind_shard(placement).is_floating());
        assert!(layout.bind_current(&[0], 0).is_floating());
        let pool = layout.offload.first().expect("a compute pool");
        assert!(layout.bind_offload_worker(pool, 0).is_floating());
    }

    #[test]
    fn a_shard_finds_the_only_pool_whatever_node_it_asks_for() {
        let layout = from_cpu_count(8, &workload());
        assert!(layout.pool_for(0).is_some());
        assert!(layout.pool_for(7).is_some(), "one node means every shard shares one pool");
    }
}
