//! Recovery from every single failure position the workload can reach.
//!
//! Individual tests can only check the failures someone thought to write down.
//! This drives the whole shard: router, scheduler, processor, adapters: once
//! per injectable operation, and insists each one reconciles.

use accounts::domain::{Account, Op, Reply, RequestId};
use accounts::processor::{AccountCall, Request};
use accounts::sim::{Plan, SimWorld};
use grommet::metrics::ShardStats;
use grommet::{ManualClock, Router, ShardConfig, shard};
use grommet_testkit::Deterministic;
use grommet_testkit::conformance::{CaseOutcome, assert_idempotent, single_fault_campaign};
use std::sync::Arc;

const ACCOUNT: u64 = 55;
const REQUESTS: u128 = 4;
const CREDIT: i64 = 10;

type AccountRouter = Router<AccountCall, ManualClock, { accounts::domain::CLASSES }>;

/// What the workload settled on, as an outside observer would see it.
struct Settled {
    durable: Account,
    applied: usize,
}

impl Deterministic for Settled {
    type Digest = (i64, u64, usize);

    fn digest(&self) -> Self::Digest {
        (self.durable.balance, self.durable.version, self.applied)
    }
}

fn expected() -> Account {
    Account { balance: CREDIT * REQUESTS as i64, version: REQUESTS as u64 }
}

/// Run a closure against a live single-shard runtime, then shut it down.
async fn with_shard<F, Fut, T>(world: &SimWorld, driver: F) -> T
where
    F: FnOnce(Arc<AccountRouter>) -> Fut,
    Fut: Future<Output = T>,
{
    let clock = ManualClock::new();
    let (tx, rx) = grommet::channel(64);
    let router = Arc::new(AccountRouter::new(vec![tx], clock.clone()));
    let mut cfg = ShardConfig::new([64, 8]);
    cfg.tick = std::time::Duration::from_millis(1);
    let engine = shard::run(rx, world.processor(), clock, Arc::new(ShardStats::default()), cfg);

    let handle = router.clone();
    let (_, result) = tokio::join!(engine, async move {
        let result = driver(handle).await;
        drop(router);
        result
    });
    result
}

/// Submit the same four credits under the same four ids, retrying until each
/// is either applied or recognised as already applied.
async fn settle(world: &SimWorld, rounds: usize) -> bool {
    with_shard(world, |router| async move {
        for _ in 0..rounds {
            let mut clean = true;
            for index in 0..REQUESTS {
                let request = Request {
                    req_id: RequestId::from(index + 1),
                    account: ACCOUNT,
                    op: Op::Credit(CREDIT),
                };
                let reply = router.call(request).await;
                clean &= matches!(reply, Ok(Reply::Ok(_) | Reply::Duplicate(_)));
            }
            if clean {
                return true;
            }
        }
        false
    })
    .await
}

#[tokio::test(flavor = "current_thread")]
async fn every_single_failure_position_reconciles_under_replay() {
    let report = single_fault_campaign(256, |position| async move {
        let world = SimWorld::new(Plan::countdown(position));
        // Retries are what an idempotency key is for; the campaign checks that
        // replaying under the same ids always converges on the same state.
        let converged = settle(&world, 8).await && world.store.account(ACCOUNT) == expected();
        CaseOutcome { fired: world.plan.did_fire(), converged }
    })
    .await;

    // The exact count follows from the workload's shape, so this is a floor
    // against a campaign that silently proves nothing: if the workload stopped
    // reaching the adapters, it would find no positions and pass trivially.
    assert!(
        report.cases >= 20,
        "the campaign only reached {} failure positions; the workload is no longer \
         exercising the adapters it claims to",
        report.cases
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_healthy_workload_applies_each_request_exactly_once() {
    let world = SimWorld::healthy();
    assert!(settle(&world, 1).await);
    assert_eq!(world.store.account(ACCOUNT), expected());
    assert_eq!(world.store.applied_requests(), REQUESTS as usize);
}

#[tokio::test(flavor = "current_thread")]
async fn replaying_the_whole_workload_never_applies_it_twice() {
    let world = SimWorld::healthy();
    assert_idempotent(4, |_| async {
        assert!(settle(&world, 1).await);
        Settled { durable: world.store.account(ACCOUNT), applied: world.store.applied_requests() }
    })
    .await;
    assert_eq!(world.store.account(ACCOUNT), expected());
}

#[tokio::test(flavor = "current_thread")]
async fn a_retry_arriving_while_its_original_is_outstanding_is_coalesced() {
    let world = SimWorld::healthy();
    let clock = ManualClock::new();
    let (tx, rx) = grommet::channel(16);
    let router = Arc::new(AccountRouter::new(vec![tx], clock.clone()));
    let mut cfg = ShardConfig::new([64, 8]);
    cfg.tick = std::time::Duration::from_millis(1);
    cfg.coalesce_duplicates = true;
    let stats = Arc::new(ShardStats::default());
    let engine = shard::run(rx, world.processor(), clock, stats.clone(), cfg);

    let handle = router.clone();
    tokio::join!(engine, async move {
        let request =
            || Request { req_id: RequestId::from(1u128), account: ACCOUNT, op: Op::Credit(CREDIT) };
        // Both reach the mailbox before the shard runs.
        let first = handle.try_call(request()).expect("first is accepted");
        let second = handle.try_call(request()).expect("the retry is accepted, then suppressed");
        let (first, second) = tokio::join!(first, second);

        assert_eq!(first, Ok(Reply::Ok(CREDIT)));
        assert!(
            matches!(&second, Ok(Reply::Err(message)) if message.contains("already in flight")),
            "the suppressed retry is still answered, not dropped: {second:?}"
        );
        drop(router);
    });

    assert_eq!(stats.coalesced.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(
        world.store.account(ACCOUNT),
        Account { balance: CREDIT, version: 1 },
        "the credit applied once, without a second round trip to the store"
    );
}
