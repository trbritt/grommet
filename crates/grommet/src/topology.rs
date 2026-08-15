//! Where shards and offload workers actually run.
//!
//! Thread-per-core is only worth anything if the threads stay on their cores,
//! and on the cores they were meant to have. Deciding which those are is a
//! hardware question, answered by [`grommet_topology`] against libhwloc: NUMA
//! nodes, SMT siblings, hybrid core kinds, Windows processor groups and cgroup
//! bandwidth limits all change the answer, and none of them are visible in a
//! CPU count.
//!
//! What is left here is the scheduler's half of the arrangement — what to do
//! when placement is unavailable, and what actually happened once the threads
//! started. Both matter because binding fails quietly: on macOS, thread affinity
//! is a hint the kernel is free to ignore, and a latency number measured on
//! threads that were never bound is measuring the OS scheduler.

pub use grommet_topology::{
    Advice, Bound, Observation, OffloadPool, Plan, Quota, ShardPlacement, Verdict, Workload, bind,
    calibrate, cgroup, detect, plan, plan_shared,
};

/// What to do when a shard thread cannot be bound to its placement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PinPolicy {
    /// Refuse to start. Choose this when a benchmark or latency budget is only
    /// meaningful with real pinning — it turns a silently unpinned run into a
    /// startup error instead of a misleading measurement.
    Require,
    /// Start anyway and report it. The scheduler is still correct; its timing is
    /// just at the OS scheduler's mercy.
    #[default]
    BestEffort,
    /// Do not attempt to bind at all, and do not plan a layout to bind to.
    Disabled,
}

/// What placement achieved, as opposed to what was requested.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopologyReport {
    pub shards: usize,
    /// Distinct CPUs the shards were spread across. Fewer than `shards` means
    /// threads share a CPU.
    pub distinct_cores: usize,
    /// Shard threads whose CPU binding took effect.
    pub pinned: usize,
    /// Shard threads whose allocations were bound to their own memory node.
    pub memory_bound: usize,
    pub policy: PinPolicy,
}

impl TopologyReport {
    /// More shard threads than distinct CPUs: threads contend for a CPU, and any
    /// throughput measured is measuring that contention.
    pub fn oversubscribed(&self) -> bool {
        self.shards > self.distinct_cores
    }

    /// Bound on the CPU but not to a memory node. Harmless on a single-node
    /// machine; on a multi-socket one it means a shard's state may live an
    /// interconnect hop away from the core that touches it on every item.
    pub fn memory_unbound(&self) -> usize {
        self.pinned.saturating_sub(self.memory_bound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_report_names_oversubscription() {
        let report = TopologyReport {
            shards: 64,
            distinct_cores: 9,
            pinned: 64,
            memory_bound: 64,
            policy: PinPolicy::BestEffort,
        };
        assert!(report.oversubscribed());
        assert!(!TopologyReport { shards: 4, distinct_cores: 4, ..report }.oversubscribed());
    }

    #[test]
    fn a_report_names_shards_that_got_a_core_but_not_a_memory_node() {
        let report = TopologyReport {
            shards: 8,
            distinct_cores: 8,
            pinned: 8,
            memory_bound: 3,
            policy: PinPolicy::BestEffort,
        };
        assert_eq!(report.memory_unbound(), 5);

        // Memory binding without CPU binding is not a deficit to report; it is
        // the platform answering a different question, so this must not wrap.
        assert_eq!(TopologyReport { pinned: 0, memory_bound: 8, ..report }.memory_unbound(), 0);
    }

    #[test]
    fn planning_this_machine_produces_something_a_runtime_can_use() {
        let layout = detect(&Workload::default()).expect("read this machine");
        assert!(!layout.shards.is_empty(), "every machine has at least one core to dispatch from");
        for shard in &layout.shards {
            assert!(layout.pool_for(shard.node).is_some() || layout.offload.is_empty());
        }
    }
}
