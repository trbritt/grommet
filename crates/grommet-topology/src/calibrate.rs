//! Choosing [`compute_fraction`] from measurement instead of from a guess.
//!
//! [`compute_fraction`]: crate::Workload::compute_fraction
//!
//! # This does not run the service
//!
//! Nothing here adjusts anything. Threads are placed once, at startup, and stay
//! where they were put; there is no controller moving cores between the reactors
//! and the offload pool while requests are in flight. That is deliberate. A
//! feedback loop over placement would be a second scheduler sitting above the
//! first, with its own oscillation, its own warm-up and its own failure mode
//! under exactly the load spike where you least want to find out about it — and
//! re-binding a thread throws away the cache and NUMA locality that pinning it
//! was for.
//!
//! What this module produces is a **number to put in configuration**. Run the
//! workload, read the counters, get a recommendation, decide whether you agree,
//! and restart with it. The split stays static and stays yours.
//!
//! # What is actually measured
//!
//! Two of the three inputs are solid and one is a floor:
//!
//! - `offload_busy` is exact. Offload workers are saturated for the whole of a
//!   task, so summed task duration over wall time is the compute demand that was
//!   *served*, in cores.
//! - `permit_wait` is demand that was *not* served: time shards spent blocked
//!   because every permit was taken. It is an approximation of unmet demand
//!   rather than a measurement of it, since a blocked reactor is not itself
//!   computing.
//! - `reactor_busy` is a **lower bound** on reactor CPU demand, not the demand.
//!   It counts scheduling bookkeeping only; a processor doing its own work
//!   inline is invisible to it. Recommendations therefore lean towards giving
//!   compute slightly more than it needs, which is the safer direction — the
//!   reactors degrade gracefully under a shortfall, and a starved offload pool
//!   blocks them outright.

use crate::plan::Workload;
use std::fmt;
use std::time::Duration;

/// Reactors blocked on a compute permit for more than this share of their time
/// are being held up by the offload pool rather than by their own work.
const CONTENTION_HIGH: f64 = 0.05;
/// Below this, blocking is incidental and not worth moving a core for.
const CONTENTION_LOW: f64 = 0.01;
/// An offload pool busy less than half the time is holding cores that the
/// reactors would put to better use.
const UTILIZATION_LOW: f64 = 0.5;

/// Counters read from a running system over one window.
///
/// Each duration is a **sum across threads** over `wall`, which is what makes
/// dividing by `wall` yield cores rather than a ratio. Take two readings of the
/// cumulative counters and pass the difference; they never reset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Observation {
    /// How long the window was.
    pub wall: Duration,
    pub shards: usize,
    pub offload_workers: usize,
    /// Summed duration of offload tasks that completed in the window.
    pub offload_busy: Duration,
    /// Summed time shards spent waiting for an offload permit.
    pub permit_wait: Duration,
    /// Summed scheduler bookkeeping time across shards.
    pub reactor_busy: Duration,
}

/// What the [`Observation`] implies about the split.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Advice {
    /// Cores of compute that were served.
    pub compute_cores: f64,
    /// Cores of compute that waited for a permit.
    pub blocked_cores: f64,
    /// Cores of scheduler overhead. A floor; see the module docs.
    pub reactor_cores: f64,
    /// Share of the offload pool that was busy, in `0.0..=1.0`.
    pub utilization: f64,
    /// Permit wait per shard-second. Above `1.0` means several items on one
    /// shard were waiting at the same time.
    pub contention: f64,
    /// The [`Workload::compute_fraction`] this observation argues for.
    pub compute_fraction: f64,
    pub verdict: Verdict,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Nothing ran. A recommendation from an idle window describes nothing.
    Idle,
    /// The split matches the load. Leave it alone.
    Balanced,
    /// Reactors are blocking on compute. Move cores to the offload pool.
    ComputeStarved,
    /// The offload pool is mostly idle while holding cores the reactors want.
    ComputeOversized,
}

impl Observation {
    /// Read the split off the counters.
    pub fn advise(&self) -> Advice {
        let seconds = self.wall.as_secs_f64();
        let budget = self.shards + self.offload_workers;
        if seconds <= 0.0 || budget == 0 {
            return Advice {
                compute_cores: 0.0,
                blocked_cores: 0.0,
                reactor_cores: 0.0,
                utilization: 0.0,
                contention: 0.0,
                compute_fraction: 0.0,
                verdict: Verdict::Idle,
            };
        }

        let compute_cores = self.offload_busy.as_secs_f64() / seconds;
        let blocked_cores = self.permit_wait.as_secs_f64() / seconds;
        let reactor_cores = self.reactor_busy.as_secs_f64() / seconds;
        let utilization = if self.offload_workers == 0 {
            0.0
        } else {
            compute_cores / self.offload_workers as f64
        };
        let contention = if self.shards == 0 { 0.0 } else { blocked_cores / self.shards as f64 };

        // Give compute what it consumed plus what it could not get, and leave
        // the rest to the reactors. The ceiling keeps at least one reactor: a
        // machine with nowhere to dispatch from computes nothing.
        let ceiling = (budget - 1) as f64 / budget as f64;
        let compute_fraction =
            ((compute_cores + blocked_cores) / budget as f64).clamp(0.0, ceiling.max(0.0));

        let idle = compute_cores == 0.0 && blocked_cores == 0.0 && reactor_cores == 0.0;
        let verdict = if idle {
            Verdict::Idle
        } else if contention > CONTENTION_HIGH || (self.offload_workers == 0 && blocked_cores > 0.0)
        {
            Verdict::ComputeStarved
        } else if self.offload_workers > 0
            && utilization < UTILIZATION_LOW
            && contention < CONTENTION_LOW
        {
            Verdict::ComputeOversized
        } else {
            Verdict::Balanced
        };

        Advice {
            compute_cores,
            blocked_cores,
            reactor_cores,
            utilization,
            contention,
            compute_fraction,
            verdict,
        }
    }
}

impl Advice {
    /// `workload` with the recommended fraction, for writing back to config.
    ///
    /// Nothing calls this on your behalf. An [`Idle`] window is refused rather
    /// than obeyed, because a system under no load argues for no compute cores
    /// and would talk you out of your pool.
    ///
    /// [`Idle`]: Verdict::Idle
    pub fn apply(&self, workload: &Workload) -> Option<Workload> {
        (self.verdict != Verdict::Idle)
            .then_some(Workload { compute_fraction: self.compute_fraction, ..*workload })
    }
}

impl fmt::Display for Advice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let action = match self.verdict {
            Verdict::Idle => return f.write_str("calibration: no load in this window"),
            Verdict::Balanced => "split looks right",
            Verdict::ComputeStarved => "reactors are blocking on compute",
            Verdict::ComputeOversized => "compute pool is oversized",
        };
        write!(
            f,
            "calibration: {action} — compute {:.2} cores served, {:.2} blocked, \
             pool {:.0}% busy, reactor overhead {:.2} cores; \
             compute_fraction = {:.2}",
            self.compute_cores,
            self.blocked_cores,
            self.utilization * 100.0,
            self.reactor_cores,
            self.compute_fraction,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(offload_busy: f64, permit_wait: f64, reactor_busy: f64) -> Observation {
        Observation {
            wall: Duration::from_secs(10),
            shards: 12,
            offload_workers: 4,
            offload_busy: Duration::from_secs_f64(offload_busy),
            permit_wait: Duration::from_secs_f64(permit_wait),
            reactor_busy: Duration::from_secs_f64(reactor_busy),
        }
    }

    #[test]
    fn a_saturated_pool_with_waiting_reactors_asks_for_more_compute_cores() {
        // Four workers busy the whole window, and reactors blocked for 20% of
        // theirs: the pool is the bottleneck.
        let advice = window(40.0, 24.0, 6.0).advise();
        assert_eq!(advice.verdict, Verdict::ComputeStarved);
        assert_eq!(advice.utilization, 1.0);
        assert!(advice.contention > CONTENTION_HIGH);
        // 4 cores served plus 2.4 blocked, over a 16 core budget.
        assert!((advice.compute_fraction - 0.4).abs() < 1e-9, "{advice:?}");
    }

    #[test]
    fn an_idle_pool_that_nobody_waits_for_should_give_cores_back() {
        let advice = window(4.0, 0.0, 6.0).advise();
        assert_eq!(advice.verdict, Verdict::ComputeOversized);
        assert_eq!(advice.utilization, 0.1);
        assert!(advice.compute_fraction < 0.05, "{advice:?}");
    }

    #[test]
    fn a_pool_that_is_busy_but_not_blocking_anyone_is_left_alone() {
        let advice = window(32.0, 0.5, 6.0).advise();
        assert_eq!(advice.verdict, Verdict::Balanced);
        assert!(advice.utilization > UTILIZATION_LOW);
    }

    #[test]
    fn a_workload_with_no_pool_at_all_is_starved_rather_than_balanced() {
        // `compute_fraction: 0.0` was configured, but work is queueing for a
        // pool that does not exist. That must be visible.
        let starved = Observation {
            offload_workers: 0,
            shards: 16,
            offload_busy: Duration::ZERO,
            permit_wait: Duration::from_secs(8),
            ..window(0.0, 0.0, 6.0)
        };
        let advice = starved.advise();
        assert_eq!(advice.verdict, Verdict::ComputeStarved);
        assert_eq!(advice.utilization, 0.0, "an absent pool cannot be busy");
        assert!(advice.compute_fraction > 0.0, "it should ask for a pool");
    }

    #[test]
    fn an_idle_window_recommends_nothing_and_refuses_to_be_applied() {
        let advice = Observation { wall: Duration::from_secs(10), ..Observation::default() };
        assert_eq!(advice.advise().verdict, Verdict::Idle);
        assert_eq!(advice.advise().apply(&Workload::default()), None);
        assert_eq!(advice.advise().to_string(), "calibration: no load in this window");

        // A zero-length window is the other way to have measured nothing.
        assert_eq!(window(40.0, 24.0, 6.0).advise().verdict, Verdict::ComputeStarved);
        assert_eq!(
            Observation { wall: Duration::ZERO, ..window(40.0, 24.0, 6.0) }.advise().verdict,
            Verdict::Idle,
        );
    }

    #[test]
    fn a_recommendation_always_leaves_somewhere_to_dispatch_from() {
        // Compute demand far beyond the machine must not consume every core.
        let greedy = Observation { shards: 3, offload_workers: 1, ..window(400.0, 400.0, 1.0) };
        let advice = greedy.advise();
        assert!(advice.compute_fraction <= 0.75, "{advice:?}");

        // Even on a single core, where there is no split to make.
        let tiny = Observation { shards: 1, offload_workers: 0, ..window(40.0, 40.0, 1.0) };
        assert_eq!(tiny.advise().compute_fraction, 0.0);
    }

    #[test]
    fn applying_advice_changes_only_the_fraction() {
        let workload = Workload { reserve_cores: 3, isolate_smt: false, ..Workload::default() };
        let tuned = window(40.0, 24.0, 6.0).advise().apply(&workload).expect("not idle");
        assert_eq!(tuned.reserve_cores, 3);
        assert!(!tuned.isolate_smt);
        assert_ne!(tuned.compute_fraction, workload.compute_fraction);
    }

    #[test]
    fn advice_describes_itself_for_an_operator() {
        let message = window(40.0, 24.0, 6.0).advise().to_string();
        assert!(message.contains("reactors are blocking on compute"), "{message}");
        assert!(message.contains("compute_fraction = 0.40"), "{message}");
    }
}
