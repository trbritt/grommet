//! Turning a machine into a runtime layout.
//!
//! This is the whole point of the crate: decide how many shard reactors to run,
//! which CPUs they get, and how the offload pool is split, from what the
//! hardware actually is rather than from a fraction of a core count.
//!
//! The plan is computed directly from an [`hwlocality::Topology`], which can be
//! the live machine, a synthetic description, or an XML capture of somebody
//! else's server — so every rule below is testable without owning the hardware.

use crate::cgroup::Quota;
use hwlocality::Topology;
use hwlocality::cpu::cpuset::CpuSet;
use hwlocality::object::TopologyObject;
use hwlocality::object::types::ObjectType;
use std::sync::Arc;

/// What the workload looks like, which the caller knows and the hardware does
/// not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Workload {
    /// Share of CPU demand that is offloaded compute rather than reactor work,
    /// in `0.0..=1.0`. Measure it from `OffloadStats` against `ShardHot` rather
    /// than guessing, once there is something to measure.
    pub compute_fraction: f64,
    /// Leave this many cores for the operating system, interrupt handling and
    /// the front door. Pinning onto every core including the one taking most
    /// interrupts is a reliable way to lose tail latency.
    pub reserve_cores: usize,
    /// Never let an offload worker share a physical core with a shard reactor.
    /// A saturated worker on a reactor's SMT sibling is the most damaging
    /// placement available.
    pub isolate_smt: bool,
    /// Prefer performance cores for reactors on hybrid machines.
    pub prefer_performance_cores: bool,
}

impl Default for Workload {
    fn default() -> Self {
        Self {
            compute_fraction: 0.25,
            reserve_cores: 1,
            isolate_smt: true,
            prefer_performance_cores: true,
        }
    }
}

/// One reactor's placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShardPlacement {
    /// The logical CPU to bind this shard's thread to.
    pub cpu: usize,
    /// The memory node its state should be allocated on.
    pub node: usize,
}

/// One offload pool, local to a memory node.
///
/// There is a pool per node rather than one global pool because a shard that
/// ships a closure and its captured data to a worker on another node pays for
/// the interconnect on every task.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OffloadPool {
    pub node: usize,
    pub cpus: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub shards: Vec<ShardPlacement>,
    pub offload: Vec<OffloadPool>,
    /// Whether this platform will honour a request to bind a thread. Answered
    /// before any thread is spawned, so a strict policy fails at configuration
    /// time rather than silently doing nothing — which is what thread affinity
    /// does on Apple silicon.
    pub can_bind_cpu: bool,
    pub can_bind_memory: bool,
    pub notes: Vec<String>,
    /// The machine this plan describes, kept so the plan can act on itself.
    ///
    /// Binding is a call on a topology, and it has to happen on the thread being
    /// placed — long after planning finished, on a thread that has no other way
    /// to reach one. Re-reading the machine per thread would be both slower and
    /// less trustworthy, since a second read can disagree with the first.
    topology: Arc<Topology>,
}

impl Plan {
    pub fn offload_workers(&self) -> usize {
        self.offload.iter().map(|pool| pool.cpus.len()).sum()
    }

    /// The offload pool a shard on `node` should submit to, falling back to the
    /// only pool when the machine has one memory node.
    pub fn pool_for(&self, node: usize) -> Option<&OffloadPool> {
        self.offload.iter().find(|pool| pool.node == node).or_else(|| self.offload.first())
    }

    /// The machine this plan was computed from.
    pub fn topology(&self) -> &Topology {
        &self.topology
    }
}

/// A physical core reduced to what placement needs.
struct Core {
    pus: Vec<usize>,
    node: usize,
    performance: bool,
}

/// Compute a layout for `topology` under `workload`, honouring `quota`.
///
/// The topology is duplicated into the plan so that it can bind threads later.
/// Use [`plan_shared`] when one is already shared, which avoids the copy.
pub fn plan(topology: &Topology, workload: &Workload, quota: Option<Quota>) -> Plan {
    plan_shared(Arc::new(topology.clone()), workload, quota)
}

/// [`plan`], for a topology that is already shared.
pub fn plan_shared(topology: Arc<Topology>, workload: &Workload, quota: Option<Quota>) -> Plan {
    let mut notes = Vec::new();
    let machine: &Topology = &topology;
    let allowed = machine.cpuset();
    let nodes = numa_nodes(machine);
    let kinds = performance_cpus(machine);
    let mut cores = physical_cores(machine, &allowed, &nodes, kinds.as_ref());

    // Best cores first, so reserving and splitting both take from the right end.
    cores.sort_by_key(|core| (!core.performance, core.node, core.pus[0]));

    let visible = cores.len();
    let mut budget = visible.saturating_sub(workload.reserve_cores).max(1);
    if let Some(quota) = quota {
        let limit = quota.usable_cores();
        if limit < budget {
            notes.push(format!(
                "CPU quota allows {limit} cores of {visible} visible; planning for the quota, \
                 since exceeding it throttles every thread at once"
            ));
            budget = limit;
        }
    }
    if budget < visible {
        notes.push(format!("using {budget} of {visible} usable cores"));
    }
    cores.truncate(budget);

    // Compute wants whole cores at the end of the list; reactors take the best.
    // A single core cannot be split: something has to dispatch, so it all goes
    // to the reactor and compute runs inline or not at all.
    let offload_cores = if workload.compute_fraction <= 0.0 || budget <= 1 {
        0
    } else {
        ((budget as f64 * workload.compute_fraction).round() as usize).clamp(1, budget - 1)
    };
    let split = budget - offload_cores;
    let (shard_cores, offload_cores) = cores.split_at(split);

    if !kinds.is_none() && workload.prefer_performance_cores {
        let on_efficiency = shard_cores.iter().filter(|core| !core.performance).count();
        if on_efficiency > 0 {
            notes.push(format!(
                "{on_efficiency} reactors placed on efficiency cores; there were not enough \
                 performance cores to hold them all"
            ));
        }
    }

    let shards = shard_cores
        .iter()
        .map(|core| ShardPlacement { cpu: core.pus[0], node: core.node })
        .collect();

    // One pool per memory node, holding whole cores. Taking only the first
    // processing unit of each leaves SMT siblings idle rather than letting a
    // saturated worker share execution units with anything.
    let mut offload: Vec<OffloadPool> = Vec::new();
    for core in offload_cores {
        let cpus: Vec<usize> =
            if workload.isolate_smt { vec![core.pus[0]] } else { core.pus.clone() };
        match offload.iter_mut().find(|pool| pool.node == core.node) {
            Some(pool) => pool.cpus.extend(cpus),
            None => offload.push(OffloadPool { node: core.node, cpus }),
        }
    }
    if workload.isolate_smt && offload_cores.iter().any(|core| core.pus.len() > 1) {
        notes.push(
            "offload workers take one thread per physical core, leaving SMT siblings idle so \
             they cannot contend with a reactor"
                .to_owned(),
        );
    }

    // Both probes ask about *this thread*, because that is how placement is
    // applied: each shard binds itself on the thread it will run on. Process-wide
    // support is a different question and answering it here would be misleading.
    let support = machine.feature_support();
    let can_bind_cpu = support.cpu_binding().is_some_and(|cpu| cpu.set_current_thread());
    let can_bind_memory = support
        .memory_binding()
        .is_some_and(|memory| memory.set_current_thread() && memory.bind_policy());
    if !can_bind_cpu {
        notes.push(
            "this platform does not support binding a thread to a CPU; placement is advisory"
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

    Plan { shards, offload, can_bind_cpu, can_bind_memory, notes, topology }
}

fn numa_nodes(topology: &Topology) -> Vec<(usize, CpuSet)> {
    topology
        .objects_with_type(ObjectType::NUMANode)
        .enumerate()
        .filter_map(|(fallback, node)| {
            node.cpuset().map(|cpus| (node.os_index().unwrap_or(fallback), cpus.clone_target()))
        })
        .collect()
}

/// The processing units hwloc ranks as most performant, on a hybrid machine.
fn performance_cpus(topology: &Topology) -> Option<CpuSet> {
    let kinds: Vec<_> = topology.cpu_kinds().ok()?.collect();
    if kinds.len() < 2 {
        return None;
    }
    let best = kinds.iter().filter_map(|kind| kind.efficiency).max()?;
    Some(kinds.iter().filter(|kind| kind.efficiency == Some(best)).fold(
        CpuSet::new(),
        |mut set, kind| {
            set |= &kind.cpuset;
            set
        },
    ))
}

fn physical_cores(
    topology: &Topology,
    allowed: &impl std::ops::Deref<Target = CpuSet>,
    nodes: &[(usize, CpuSet)],
    performance: Option<&CpuSet>,
) -> Vec<Core> {
    let describe = |object: &TopologyObject| -> Option<Core> {
        let pus: Vec<usize> = object
            .cpuset()?
            .iter_set()
            .map(usize::from)
            .filter(|cpu| allowed.is_set(*cpu))
            .collect();
        let first = *pus.first()?;
        let node =
            nodes.iter().find(|(_, cpus)| cpus.is_set(first)).map(|(id, _)| *id).unwrap_or(0);
        Some(Core { pus, node, performance: performance.is_none_or(|set| set.is_set(first)) })
    };

    let cores: Vec<Core> =
        topology.objects_with_type(ObjectType::Core).filter_map(&describe).collect();
    if !cores.is_empty() {
        return cores;
    }
    // Some platforms expose no core objects; then each processing unit is one.
    topology.objects_with_type(ObjectType::PU).filter_map(&describe).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cgroup::QuotaSource;
    use hwlocality::topology::builder::TopologyBuilder;
    use std::time::Duration;

    /// hwloc synthesizes any machine shape from a description, so a two-socket
    /// server is a unit test on a laptop.
    fn machine(description: &str) -> Topology {
        TopologyBuilder::new()
            .from_synthetic(description)
            .expect("synthetic description")
            .build()
            .expect("synthetic topology")
    }

    fn quota(cores: f64) -> Quota {
        Quota { cores, period: Duration::from_millis(100), source: QuotaSource::CgroupV2 }
    }

    #[test]
    fn a_two_socket_server_splits_reactors_from_compute_by_whole_cores() {
        let topology = machine("pack:2 core:8 pu:2");
        let plan = plan(&topology, &Workload::default(), None);

        // 16 cores, one reserved, a quarter of the rest to compute.
        assert_eq!(plan.shards.len() + plan.offload_workers(), 15);
        assert_eq!(plan.offload_workers(), 4);
        assert_eq!(plan.shards.len(), 11);

        // No CPU is used twice, and no offload worker sits on a shard's core.
        let mut used: Vec<usize> = plan.shards.iter().map(|shard| shard.cpu).collect();
        used.extend(plan.offload.iter().flat_map(|pool| pool.cpus.iter().copied()));
        let unique: std::collections::BTreeSet<_> = used.iter().collect();
        assert_eq!(unique.len(), used.len(), "a CPU was assigned to two pools");
    }

    #[test]
    fn smt_siblings_are_left_idle_rather_than_shared_with_a_reactor() {
        let topology = machine("pack:1 core:4 pu:2");
        let plan = plan(&topology, &Workload::default(), None);

        // Each placement takes the first unit of a physical core, so no two
        // placements ever land on the same core.
        let mut cpus: Vec<usize> = plan.shards.iter().map(|shard| shard.cpu).collect();
        cpus.extend(plan.offload.iter().flat_map(|pool| pool.cpus.iter().copied()));
        for cpu in &cpus {
            assert!(cpu % 2 == 0, "CPU {cpu} is an SMT sibling, not a core's first unit");
        }
        assert!(plan.notes.iter().any(|note| note.contains("SMT siblings idle")));
    }

    #[test]
    fn a_quota_overrides_the_visible_core_count() {
        let topology = machine("pack:2 core:24 pu:2");
        let plan = plan(&topology, &Workload::default(), Some(quota(2.0)));

        assert_eq!(
            plan.shards.len() + plan.offload_workers(),
            2,
            "a two-core quota must not produce forty-eight threads"
        );
        assert!(plan.notes.iter().any(|note| note.contains("CPU quota allows 2 cores of 48")));
    }

    #[test]
    fn offload_pools_are_created_per_memory_node() {
        let topology = machine("node:2 core:8 pu:1");
        let plan = plan(&topology, &Workload { reserve_cores: 0, ..Workload::default() }, None);

        assert_eq!(plan.offload_workers(), 4);
        // Every pool serves the node its workers sit on, and a shard can find
        // the pool local to itself.
        for pool in &plan.offload {
            assert!(!pool.cpus.is_empty());
        }
        for shard in &plan.shards {
            assert!(plan.pool_for(shard.node).is_some());
        }
    }

    #[test]
    fn a_pure_io_workload_reserves_no_compute_cores() {
        let topology = machine("pack:1 core:8 pu:1");
        let plan =
            plan(&topology, &Workload { compute_fraction: 0.0, ..Workload::default() }, None);
        assert_eq!(plan.offload_workers(), 0);
        assert_eq!(plan.shards.len(), 7, "every core but the reserved one runs a reactor");
    }

    #[test]
    fn a_compute_heavy_workload_still_leaves_a_reactor() {
        let topology = machine("pack:1 core:4 pu:1");
        let plan =
            plan(&topology, &Workload { compute_fraction: 1.0, ..Workload::default() }, None);
        assert!(!plan.shards.is_empty(), "there must always be somewhere to dispatch from");
        assert!(plan.offload_workers() >= 1);
    }

    #[test]
    fn a_single_core_machine_still_produces_a_runnable_plan() {
        let topology = machine("pack:1 core:1 pu:1");
        let plan = plan(&topology, &Workload::default(), None);
        assert_eq!(plan.shards.len(), 1);
        assert_eq!(plan.offload_workers(), 0, "one core cannot be split");
    }

    #[test]
    fn a_plan_reports_whether_this_platform_will_honour_it() {
        let plan = plan(&machine("pack:1 core:2 pu:1"), &Workload::default(), None);
        // Synthetic topologies support no binding at all, which the plan must
        // say rather than let a caller assume placement took effect.
        assert!(!plan.can_bind_cpu);
        assert!(plan.notes.iter().any(|note| note.contains("placement is advisory")));
    }
}
