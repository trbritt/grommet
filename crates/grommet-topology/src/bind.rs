//! Making a plan take effect.
//!
//! A placement nothing acts on is a comment. These are the calls that turn one
//! into a bound thread, and they report what the operating system actually did
//! rather than what was asked for: binding is advisory on more platforms than
//! people expect, and a benchmark run on unbound threads measures the OS
//! scheduler rather than this one.
//!
//! CPU and memory binding are done together, because doing only the first is
//! the more expensive half of getting NUMA wrong. A shard thread pinned to a
//! core on node 1 whose state was first touched while it ran on node 0 pays the
//! interconnect on every subsequent access, for the life of the process, and no
//! amount of correct CPU placement recovers it.

use crate::plan::{OffloadPool, Plan, ShardPlacement};
#[cfg(feature = "hwloc")]
use hwlocality::cpu::binding::CpuBindingFlags;
#[cfg(feature = "hwloc")]
use hwlocality::cpu::cpuset::CpuSet;
#[cfg(feature = "hwloc")]
use hwlocality::memory::binding::{MemoryBindingFlags, MemoryBindingPolicy};
#[cfg(feature = "hwloc")]
use hwlocality::memory::nodeset::NodeSet;
#[cfg(feature = "hwloc")]
use hwlocality::object::types::ObjectType;

/// What binding achieved for one thread, as distinct from what was requested.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Bound {
    /// The thread will run only on the CPUs it was given.
    pub cpu: bool,
    /// Its allocations will come from its own memory node.
    pub memory: bool,
}

impl Bound {
    /// Nothing took effect: this thread is wherever the OS decides to put it.
    pub fn is_floating(&self) -> bool {
        !self.cpu && !self.memory
    }
}

#[cfg(feature = "hwloc")]
impl Plan {
    /// Bind the calling thread to `placement`. Call this from the thread being
    /// placed, as the first thing it does: memory binding only governs pages
    /// touched after it is set.
    pub fn bind_shard(&self, placement: &ShardPlacement) -> Bound {
        self.bind_current(std::slice::from_ref(&placement.cpu), placement.node)
    }

    /// Bind one offload worker. Workers take a CPU each rather than sharing the
    /// pool's set: Rayon steals work between them anyway, so letting each float
    /// across the pool buys nothing and costs the cache locality that made a
    /// per-node pool worth having.
    pub fn bind_offload_worker(&self, pool: &OffloadPool, worker: usize) -> Bound {
        match pool.cpus.get(worker % pool.cpus.len().max(1)) {
            Some(cpu) => self.bind_current(std::slice::from_ref(cpu), pool.node),
            None => Bound::default(),
        }
    }

    /// Bind the calling thread to `cpus` and its allocations to `node`.
    pub fn bind_current(&self, cpus: &[usize], node: usize) -> Bound {
        let mut set = CpuSet::new();
        for cpu in cpus {
            set.set(*cpu);
        }
        if set.is_empty() {
            return Bound::default();
        }

        // Both calls are gated on hwloc's own support probe rather than on their
        // return value, because the return value is not trustworthy on its own:
        // asked to bind against a topology that is not this machine: a
        // synthetic description, an XML capture: hwloc returns success without
        // binding anything. Reporting that as `cpu: true` would make every test
        // on a synthetic machine claim placement it did not get.
        //
        // `THREAD` rather than `PROCESS`: every other shard is binding itself at
        // the same moment, and a process-wide binding would be whichever one
        // happened to run last.
        let cpu =
            self.can_bind_cpu && self.topology().bind_cpu(&set, CpuBindingFlags::THREAD).is_ok();
        let memory = self.can_bind_memory
            && self.nodeset(node).is_some_and(|nodes| {
                self.topology()
                    .bind_memory(&nodes, MemoryBindingPolicy::Bind, MemoryBindingFlags::THREAD)
                    .is_ok()
            });
        Bound { cpu, memory }
    }

    /// The memory nodes behind a NUMA node index from a placement.
    fn nodeset(&self, node: usize) -> Option<NodeSet> {
        self.topology()
            .objects_with_type(ObjectType::NUMANode)
            .enumerate()
            .find(|(fallback, object)| object.os_index().unwrap_or(*fallback) == node)
            .and_then(|(_, object)| object.nodeset().map(|nodes| nodes.clone_target()))
    }
}

/// Without hwloc there is no way to ask the operating system for a placement,
/// so these report exactly that.
///
/// They are deliberately not errors. A plan built by the fallback already says
/// `can_bind_cpu: false` and carries a note explaining why, the runtime's
/// `TopologyReport` will show nothing pinned, and `PinPolicy::Require` refuses
/// to start at all, so the shortfall is visible at three levels before any
/// timing is measured. Making the calls themselves fail would only force every
/// caller to handle a case that is already reported.
#[cfg(not(feature = "hwloc"))]
impl Plan {
    /// Bind the calling thread to `placement`. Always floating in this build.
    pub fn bind_shard(&self, placement: &ShardPlacement) -> Bound {
        let _ = placement;
        Bound::default()
    }

    /// Bind one offload worker. Always floating in this build.
    pub fn bind_offload_worker(&self, pool: &OffloadPool, worker: usize) -> Bound {
        let _ = (pool, worker);
        Bound::default()
    }

    /// Bind the calling thread to `cpus` and its allocations to `node`. Always
    /// floating in this build.
    pub fn bind_current(&self, cpus: &[usize], node: usize) -> Bound {
        let _ = (cpus, node);
        Bound::default()
    }
}

#[cfg(test)]
#[cfg(feature = "hwloc")]
mod tests {
    use super::*;
    use crate::Workload;
    use crate::plan::plan;
    use hwlocality::Topology;
    use hwlocality::topology::builder::TopologyBuilder;

    fn synthetic() -> Topology {
        TopologyBuilder::new()
            .from_synthetic("node:2 core:4 pu:2")
            .expect("synthetic description")
            .build()
            .expect("synthetic topology")
    }

    #[test]
    fn binding_against_a_synthetic_machine_reports_failure_rather_than_pretending() {
        // hwloc answers `Ok(())` here. It has not bound anything and cannot,
        // the topology describes a machine that is not this one, but the call
        // succeeds, so a `Bound` derived from the return value alone would claim
        // placement that never happened, on every test in this crate.
        let layout = plan(&synthetic(), &Workload::default(), None);
        let placement = layout.shards.first().expect("a synthetic machine has cores");
        assert!(layout.bind_shard(placement).is_floating());
        assert!(!layout.can_bind_cpu, "the plan warned about this before a thread was spawned");
    }

    #[test]
    fn an_empty_cpu_set_binds_nothing_rather_than_binding_everything() {
        // An empty hwloc cpuset is not "no constraint", it is an error. Guarding
        // here keeps a pool with no CPUs from widening a thread's affinity.
        let layout = plan(&synthetic(), &Workload::default(), None);
        assert_eq!(layout.bind_current(&[], 0), Bound::default());
        let empty = OffloadPool { node: 0, cpus: Vec::new() };
        assert_eq!(layout.bind_offload_worker(&empty, 0), Bound::default());
    }

    #[test]
    fn a_worker_index_past_the_end_wraps_onto_the_pool() {
        let layout = plan(&synthetic(), &Workload::default(), None);
        let pool = OffloadPool { node: 0, cpus: vec![0, 2] };
        // Wrapping, not panicking: a pool may run more workers than CPUs when a
        // quota forced the split narrower than the thread count.
        assert_eq!(layout.bind_offload_worker(&pool, 5), layout.bind_offload_worker(&pool, 1));
    }

    #[test]
    fn every_numa_node_in_the_plan_resolves_to_a_nodeset() {
        // If a placement's node did not map back to real memory, memory binding
        // would fail for a reason that has nothing to do with OS support.
        let layout = plan(&synthetic(), &Workload::default(), None);
        for shard in &layout.shards {
            assert!(layout.nodeset(shard.node).is_some(), "node {} is unknown", shard.node);
        }
        assert!(layout.nodeset(usize::MAX).is_none());
    }

    #[test]
    fn binding_to_this_machine_is_attempted_and_answers_consistently() {
        // The live machine may or may not support binding: macOS does not, so
        // the assertion is consistency, not success: the plan's advertised
        // support must agree with what binding actually returns.
        let topology = Topology::new().expect("read this machine");
        let layout = plan(&topology, &Workload::default(), None);
        let placement = layout.shards.first().expect("this machine has a core");
        let bound = layout.bind_shard(placement);

        // Undo it before asserting. On a platform where this worked, leaving it
        // in place would pin the whole test binary: every later test in this
        // process: to one CPU.
        let all = topology.cpuset().clone_target();
        let _ = topology.bind_cpu(&all, CpuBindingFlags::THREAD);

        assert_eq!(
            bound.cpu, layout.can_bind_cpu,
            "the plan promised cpu binding = {} and got {}",
            layout.can_bind_cpu, bound.cpu
        );
    }
}
