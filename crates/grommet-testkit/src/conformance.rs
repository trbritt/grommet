//! Checks you can run against your own processor and configuration.
//!
//! None of this is enforced by the type system, and it deliberately does not
//! pretend to be. A marker trait asserting "my processor is deterministic"
//! would be an unchecked promise; what is here instead runs your code and
//! reports what it actually did.

use crate::determinism::Deterministic;
use grommet_core::{Admit, ClassId, Completion, Config, Disposition, Scheduler};
use std::future::Future;

/// A randomized but reproducible workload for [`scheduler_sweep`].
#[derive(Clone, Copy, Debug)]
pub struct SweepSpec {
    /// Distinct affine keys in the workload.
    pub keys: u64,
    /// Operations to perform.
    pub steps: usize,
    pub seed: u64,
}

impl Default for SweepSpec {
    fn default() -> Self {
        Self { keys: 64, steps: 20_000, seed: 0x5eed }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SweepReport {
    pub admitted: usize,
    pub dispatched: usize,
    pub evicted: usize,
    /// The most dispatches *of its own class* that any key waited between
    /// becoming ready and running. This is the starvation bound, measured
    /// rather than assumed: it can never exceed one rotation of that ring.
    pub worst_dispatch_gap: usize,
    /// Peak simultaneously queued items, for sizing `Config::queue_reserve`.
    pub queue_capacity: usize,
    /// The largest eviction worklist seen at any point in the sweep. Idle
    /// tracking is per resident key rather than per completion, so this is
    /// bounded by the key count no matter how much throughput passes
    /// through — a value that scales with `steps` instead means the
    /// scheduler is accumulating one entry per operation.
    pub peak_eviction_backlog: usize,
}

/// An independent model of what the scheduler owes each key, so the sweep can
/// tell exactly when a key became ready and in which ring it was waiting.
#[derive(Default)]
struct KeyModel {
    queued: std::collections::VecDeque<ClassId>,
    in_flight: bool,
    /// The ring this key is waiting in, and that ring's dispatch count at the
    /// moment it joined.
    ready_at: Option<(ClassId, usize)>,
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Drive a scheduler configured exactly as yours through a mixed workload,
/// checking every structural invariant after every operation and measuring how
/// long a key can wait.
///
/// This exercises your *configuration* — class budgets, caps, eviction windows
/// — without needing your processor, so it is cheap to run in CI.
///
/// # Panics
///
/// If an invariant breaks, or if a key waits longer than one full rotation of
/// every ring, which would mean starvation is not bounded.
pub fn scheduler_sweep<const CLASSES: usize>(cfg: Config<CLASSES>, spec: SweepSpec) -> SweepReport {
    assert!(spec.keys > 0, "a sweep needs at least one key");
    let mut book: Scheduler<u64, u64, u64, CLASSES> = Scheduler::new(cfg);
    let mut rng = Rng(spec.seed | 1);
    let mut report = SweepReport::default();
    let mut inflight: Vec<(u64, ClassId)> = Vec::new();
    let mut evicted = Vec::new();
    let mut model: Vec<KeyModel> = (0..spec.keys).map(|_| KeyModel::default()).collect();
    let mut dispatched_per_class = [0usize; CLASSES];
    let mut now = std::time::Duration::ZERO;

    for step in 0..spec.steps {
        let roll = rng.next();
        let key = roll % spec.keys;
        let class = ((roll >> 16) % CLASSES as u64) as ClassId;
        now += std::time::Duration::from_micros(1);

        match roll >> 8 & 0b11 {
            // Admit
            0 | 1 if !book.is_saturated() => {
                let entry = &mut model[key as usize];
                entry.queued.push_back(class);
                // A key already waiting keeps its place; only one that was idle
                // joins a ring now.
                if !entry.in_flight && entry.ready_at.is_none() {
                    let head = *entry.queued.front().expect("just pushed");
                    entry.ready_at = Some((head, dispatched_per_class[head as usize]));
                }
                book.admit(Admit { key, class, expires_at: None, payload: roll });
                report.admitted += 1;
            }
            // Dispatch
            2 => {
                if let Some(dispatch) = book.next(class, now) {
                    let entry = &mut model[dispatch.key as usize];
                    let (ring, since) =
                        entry.ready_at.take().expect("a dispatched key was waiting to run");
                    assert_eq!(ring, class, "a key was dispatched from the wrong ring");
                    report.worst_dispatch_gap =
                        report.worst_dispatch_gap.max(dispatched_per_class[class as usize] - since);
                    entry.queued.pop_front();
                    entry.in_flight = true;
                    dispatched_per_class[class as usize] += 1;
                    report.dispatched += 1;
                    inflight.push((dispatch.key, dispatch.class));
                }
            }
            // Complete
            _ if !inflight.is_empty() => {
                let index = (roll >> 32) as usize % inflight.len();
                let (key, class) = inflight.swap_remove(index);
                book.complete(Completion { key, class, state: Disposition::Keep(roll) }, now);
                let entry = &mut model[key as usize];
                entry.in_flight = false;
                if let Some(&head) = entry.queued.front() {
                    entry.ready_at = Some((head, dispatched_per_class[head as usize]));
                }
            }
            _ => {}
        }

        assert!(book.pop_expired().is_none(), "the sweep issues no deadlines");
        book.evict(now, &mut evicted);
        report.evicted += evicted.len();
        for (key, _) in evicted.drain(..) {
            book.finish_evict(key, now);
        }

        assert_eq!(book.check_invariants(), Ok(()), "invariant broke at step {step}");

        // Idle tracking is the one structure that grew per operation rather
        // than per key, so it is measured on every step rather than sampled:
        // a leak here is silent, costs nothing until the shard has been up
        // for hours, and is exactly what a sweep is placed to catch.
        let live = book.snapshot();
        assert!(
            live.eviction_backlog <= live.resident,
            "step {step}: {} keys awaiting eviction against {} resident — \
             idle tracking is not bounded by the key count",
            live.eviction_backlog,
            live.resident,
        );
        report.peak_eviction_backlog = report.peak_eviction_backlog.max(live.eviction_backlog);
    }

    // A ring holds each key at most once, so a key can wait behind at most one
    // full rotation of it.
    let bound = spec.keys as usize;
    assert!(
        report.worst_dispatch_gap <= bound,
        "a key waited {} dispatches of its own class, beyond the rotation bound of {bound}",
        report.worst_dispatch_gap
    );
    report.queue_capacity = book.snapshot().queue_capacity;

    // Shut down from wherever the sweep happened to stop, in the order a shard
    // does it: drain every queued item, take back every dispatch, then flush
    // what is still resident. These dispatches are verification rather than
    // workload, so they stay out of the report and out of the starvation model.
    loop {
        for class in 0..CLASSES {
            while let Some(dispatch) = book.next(class as ClassId, now) {
                inflight.push((dispatch.key, dispatch.class));
            }
        }
        if inflight.is_empty() {
            break;
        }
        for (key, class) in inflight.drain(..) {
            book.complete(Completion { key, class, state: Disposition::Keep(0) }, now);
        }
    }
    book.evict_all(&mut evicted);
    report.evicted += evicted.len();
    for (key, _) in evicted.drain(..) {
        book.finish_evict(key, now);
    }
    assert_eq!(book.check_invariants(), Ok(()), "an invariant broke during shutdown");
    let left = book.snapshot();
    assert_eq!(left.resident, 0, "shutdown left {} keys holding unflushed state", left.resident);
    assert_eq!(left.pending, 0, "shutdown left {} items undrained", left.pending);
    report
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaseOutcome {
    /// Whether the plan actually injected a fault in this case.
    pub fired: bool,
    /// Whether the workload reached its expected final state afterwards.
    pub converged: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CampaignReport {
    /// Failure positions visited, which is every one the workload can reach.
    pub cases: usize,
}

/// Run a workload once per single-failure position, until a position exists
/// that the workload never reaches.
///
/// Give it a closure that builds a fresh world with `FaultPlan::countdown(n)`,
/// runs the workload, retries, and reports whether the fault fired and whether
/// the world converged. Every reachable failure position gets its own run, so
/// "we recover from a failure here" is checked everywhere rather than at the
/// one place a hand-written test remembered.
///
/// # Panics
///
/// If any case fails to converge, or if `max_cases` is reached without finding
/// an unreachable position — which means the workload has more failure points
/// than the bound and the campaign proved less than it claims.
pub async fn single_fault_campaign<F, Fut>(max_cases: usize, mut case: F) -> CampaignReport
where
    F: FnMut(i64) -> Fut,
    Fut: Future<Output = CaseOutcome>,
{
    for n in 0..max_cases {
        let outcome = case(n as i64).await;
        assert!(outcome.converged, "the workload did not converge with a fault at position {n}");
        if !outcome.fired {
            return CampaignReport { cases: n };
        }
    }
    panic!("the workload has more than {max_cases} failure positions; raise the bound");
}

/// Run the same workload repeatedly and assert it lands in the same observable
/// state every time.
///
/// This is what a "my processor is deterministic" claim actually rests on. Run
/// it in your own test suite; a marker trait cannot check this for you.
///
/// # Panics
///
/// If any run's digest differs from the first.
pub async fn assert_deterministic<T, F, Fut>(runs: usize, mut workload: F)
where
    F: FnMut() -> Fut,
    Fut: Future<Output = T>,
    T: Deterministic,
{
    assert!(runs >= 2, "determinism needs at least two runs to compare");
    let first = workload().await.digest();
    for run in 1..runs {
        let next = workload().await.digest();
        assert!(first == next, "run {run} diverged from the first: {first:?} then {next:?}");
    }
}

/// Replay one operation against a shared world and assert that only the first
/// application changes anything.
///
/// The closure should apply the *same* request id every time to the *same*
/// world, and return something whose [`Deterministic::digest`] covers the
/// durable state and the response. Anything that double-applies — a retry that
/// credits twice, a counter that increments per attempt — shows up as a digest
/// that keeps moving.
///
/// This is the check that gives an idempotency key its meaning. The runtime
/// suppresses retries that are still concurrent with their original, but a
/// retry arriving after the original completed reaches your processor, and only
/// your store can answer it correctly.
///
/// # Panics
///
/// If any replay after the first changes the digest.
pub async fn assert_idempotent<T, F, Fut>(attempts: usize, mut attempt: F)
where
    F: FnMut(usize) -> Fut,
    Fut: Future<Output = T>,
    T: Deterministic,
{
    assert!(attempts >= 2, "idempotency needs at least one replay to observe");
    let settled = attempt(0).await.digest();
    for replay in 1..attempts {
        let repeated = attempt(replay).await.digest();
        assert!(
            settled == repeated,
            "replay {replay} was applied again: {settled:?} became {repeated:?}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    #[test]
    fn a_sweep_exercises_the_scheduler_and_bounds_starvation() {
        let mut cfg = Config::<2>::new([4, 2]);
        cfg.max_pending = 256;
        // Under constant traffic almost nothing stays idle for long, so the
        // eviction path is only reached with a window this aggressive.
        cfg.evict_after = std::time::Duration::ZERO;
        let report = scheduler_sweep(cfg, SweepSpec { keys: 32, steps: 20_000, seed: 7 });

        assert!(report.admitted > 0 && report.dispatched > 0);
        assert!(report.evicted > 0, "a zero idle window must reach the eviction path");
        assert!(report.queue_capacity > 0);
        assert!(
            report.worst_dispatch_gap > 0,
            "a sweep this long must make some key wait, or it is proving nothing"
        );
    }

    /// Throughput must not accumulate. Sixteen keys are cycled tens of
    /// thousands of times with an eviction window nothing ever reaches, so
    /// every completion that goes idle has to reuse the same entry. A
    /// structure that recorded one candidate per completion would end this
    /// run holding `dispatched` of them; this one is capped at the key count.
    #[test]
    fn idle_tracking_is_bounded_by_keys_rather_than_by_throughput() {
        const KEYS: u64 = 16;
        let mut cfg = Config::<2>::new([4, 2]);
        cfg.max_pending = 256;
        // Long enough that no key is ever old enough to age out mid-run, so
        // the worklist is only ever added to — the leaking case.
        cfg.evict_after = std::time::Duration::from_secs(3600);
        let report = scheduler_sweep(cfg, SweepSpec { keys: KEYS, steps: 50_000, seed: 11 });

        // Nothing ages out, so the only evictions are the shutdown flush: one
        // per key, however long the run was.
        assert_eq!(report.evicted, KEYS as usize, "only the shutdown flush should evict");
        assert!(report.dispatched > 10_000, "the run has to be long enough to expose growth");
        assert!(
            report.peak_eviction_backlog <= KEYS as usize,
            "idle tracking reached {} entries against {KEYS} keys and {} dispatches",
            report.peak_eviction_backlog,
            report.dispatched,
        );
    }

    #[test]
    fn a_sweep_is_reproducible_from_its_seed() {
        let cfg = Config::<2>::new([2, 2]);
        let spec = SweepSpec { keys: 8, steps: 2_000, seed: 99 };
        assert_eq!(scheduler_sweep(cfg, spec), scheduler_sweep(cfg, spec));
    }

    #[test]
    fn a_three_class_configuration_sweeps_too() {
        let report = scheduler_sweep(
            Config::<3>::new([2, 1, 1]),
            SweepSpec { keys: 16, steps: 5_000, seed: 3 },
        );
        assert!(report.dispatched > 0);
    }

    #[tokio::test]
    async fn a_campaign_visits_every_reachable_failure_position() {
        // A workload with exactly four injectable operations.
        let report =
            single_fault_campaign(
                64,
                |n| async move { CaseOutcome { fired: n < 4, converged: true } },
            )
            .await;
        assert_eq!(report.cases, 4);
    }

    #[tokio::test]
    #[should_panic(expected = "did not converge")]
    async fn a_campaign_fails_loudly_when_recovery_does_not_converge() {
        single_fault_campaign(8, |n| async move { CaseOutcome { fired: true, converged: n != 2 } })
            .await;
    }

    #[tokio::test]
    #[should_panic(expected = "raise the bound")]
    async fn a_campaign_refuses_to_claim_more_than_it_checked() {
        single_fault_campaign(4, |_| async { CaseOutcome { fired: true, converged: true } }).await;
    }

    struct Run(u64);

    impl Deterministic for Run {
        type Digest = u64;
        fn digest(&self) -> u64 {
            self.0
        }
    }

    #[tokio::test]
    async fn repeated_runs_of_a_deterministic_workload_agree() {
        assert_deterministic(4, || async { Run(42) }).await;
    }

    #[tokio::test]
    #[should_panic(expected = "diverged from the first")]
    async fn a_workload_that_drifts_is_reported() {
        let counter = Rc::new(Cell::new(0));
        assert_deterministic(3, || {
            let counter = counter.clone();
            async move {
                counter.set(counter.get() + 1);
                Run(counter.get())
            }
        })
        .await;
    }

    /// A store that records each request id once, the way durable dedup works.
    #[derive(Default)]
    struct Ledger {
        balance: Cell<i64>,
        applied: std::cell::RefCell<std::collections::HashSet<u64>>,
    }

    struct Snapshot(i64);

    impl Deterministic for Snapshot {
        type Digest = i64;
        fn digest(&self) -> i64 {
            self.0
        }
    }

    impl Ledger {
        fn credit(&self, request: u64, amount: i64) -> Snapshot {
            if self.applied.borrow_mut().insert(request) {
                self.balance.set(self.balance.get() + amount);
            }
            Snapshot(self.balance.get())
        }
    }

    #[tokio::test]
    async fn replaying_a_recorded_request_does_not_apply_it_again() {
        let ledger = Ledger::default();
        assert_idempotent(4, |_| async { ledger.credit(1, 100) }).await;
        assert_eq!(ledger.balance.get(), 100);
    }

    #[tokio::test]
    #[should_panic(expected = "was applied again")]
    async fn a_retry_that_double_applies_is_caught() {
        let balance = Cell::new(0i64);
        assert_idempotent(3, |_| async {
            balance.set(balance.get() + 100);
            Snapshot(balance.get())
        })
        .await;
    }
}
