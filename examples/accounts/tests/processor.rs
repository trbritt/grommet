//! What the processor does with one request, under every failure it can meet.

use accounts::domain::{Account, Op, Reply, RequestId, RevalueParams};
use accounts::processor::{AccountCall, Failure, Request};
use accounts::sim::{FaultPoint, Plan, SimWorld};
use grommet::{Disposition, Fallout, ProcessError, Processor};

const ACCOUNT: u64 = 7;

fn request(req_id: u128, op: Op) -> (AccountCall, tokio::sync::oneshot::Receiver<Reply>) {
    AccountCall::new(Request { req_id: RequestId::from(req_id), account: ACCOUNT, op })
}

/// Run one request against a world, returning the reply and what the runtime
/// was told to do with the account's resident state.
async fn run(
    world: &SimWorld,
    state: Option<Account>,
    req_id: u128,
    op: Op,
) -> (Reply, Result<Disposition<Account>, Failure>) {
    let (call, reply) = request(req_id, op);
    let outcome = world.processor().process(ACCOUNT, state, call).await;
    (reply.await.expect("the processor must always answer"), outcome)
}

fn resident(outcome: &Result<Disposition<Account>, Failure>) -> Option<Account> {
    match outcome {
        Ok(Disposition::Keep(account)) => Some(account.clone()),
        Ok(Disposition::Drop) | Err(_) => None,
    }
}

#[tokio::test]
async fn a_credit_applies_once_and_stays_resident() {
    let world = SimWorld::healthy();
    let (reply, outcome) = run(&world, None, 1, Op::Credit(100)).await;

    assert_eq!(reply, Reply::Ok(100));
    assert_eq!(resident(&outcome), Some(Account { balance: 100, version: 1 }));
    assert_eq!(world.store.account(ACCOUNT), Account { balance: 100, version: 1 });
    assert_eq!(world.cache.account(ACCOUNT), Some(Account { balance: 100, version: 1 }));
}

#[tokio::test]
async fn a_lost_acknowledgement_is_reported_in_doubt_and_evicts_resident_state() {
    let world = SimWorld::new(Plan::ordered([FaultPoint::CommitAfterApply]));
    let (reply, outcome) = run(&world, None, 42, Op::Credit(100)).await;

    assert!(matches!(reply, Reply::Err(_)));
    assert_eq!(outcome, Err(Failure::InDoubt));
    assert_eq!(
        Failure::InDoubt.fallout(),
        Fallout::InDoubt,
        "the runtime must be told the durable outcome is unknown"
    );
    assert_eq!(resident(&outcome), None, "an unknown outcome must not leave state resident");
    assert_eq!(world.cache.account(ACCOUNT), None, "nor anything derived from it cached");

    // It did in fact apply. Only a reload can discover that.
    assert_eq!(world.store.account(ACCOUNT), Account { balance: 100, version: 1 });
    assert!(world.plan.is_exhausted());
}

#[tokio::test]
async fn replaying_an_in_doubt_request_under_the_same_id_reconciles_without_double_applying() {
    let world = SimWorld::new(Plan::ordered([FaultPoint::CommitAfterApply]));
    let (_, first) = run(&world, None, 42, Op::Credit(100)).await;
    assert_eq!(first, Err(Failure::InDoubt));

    // The retry reuses the id and starts from no resident state, exactly as the
    // runtime leaves it.
    let (reply, outcome) = run(&world, resident(&first), 42, Op::Credit(100)).await;
    assert_eq!(reply, Reply::Duplicate(100), "the store recognises the replay");
    assert_eq!(resident(&outcome), Some(Account { balance: 100, version: 1 }));
    assert_eq!(
        world.store.account(ACCOUNT),
        Account { balance: 100, version: 1 },
        "the credit applied exactly once across both attempts"
    );
    assert_eq!(world.store.applied_requests(), 1);
}

#[tokio::test]
async fn a_version_conflict_is_definite_but_still_forces_a_reload() {
    let world = SimWorld::healthy();
    // Advance the durable account behind our back.
    let _ = run(&world, None, 1, Op::Credit(50)).await;

    // Now submit against a stale version-0 snapshot.
    let (reply, outcome) = run(&world, Some(Account::default()), 2, Op::Credit(10)).await;
    assert!(matches!(reply, Reply::Err(_)));
    assert_eq!(outcome, Err(Failure::Stale));
    assert_eq!(
        Failure::Stale.fallout(),
        Fallout::Untouched,
        "a conflict means the mutation definitely did not apply"
    );
    assert_eq!(world.store.account(ACCOUNT), Account { balance: 50, version: 1 });
}

#[tokio::test]
async fn a_definite_precommit_failure_keeps_the_account_resident() {
    let world = SimWorld::new(Plan::ordered([FaultPoint::CommitBeforeApply]));
    let held = Account { balance: 20, version: 0 };
    let (reply, outcome) = run(&world, Some(held.clone()), 3, Op::Debit(4)).await;

    assert!(matches!(reply, Reply::Err(_)));
    assert_eq!(
        resident(&outcome),
        Some(held),
        "a failure that cannot have applied is no reason to reload"
    );
    assert_eq!(world.store.account(ACCOUNT), Account::default());
    assert!(world.plan.is_exhausted());
}

#[tokio::test]
async fn a_failed_cache_write_never_fails_a_committed_request() {
    let world = SimWorld::new(Plan::ordered([FaultPoint::CachePut]));
    let (reply, outcome) = run(&world, Some(Account::default()), 4, Op::Credit(9)).await;

    assert_eq!(reply, Reply::Ok(9));
    assert_eq!(resident(&outcome), Some(Account { balance: 9, version: 1 }));
    assert_eq!(world.store.account(ACCOUNT), Account { balance: 9, version: 1 });
    assert_eq!(world.cache.account(ACCOUNT), None, "a failed write leaves no known-stale value");
}

#[tokio::test]
async fn reads_and_rejected_amounts_never_touch_the_store() {
    let world = SimWorld::healthy();
    let held = Account { balance: 20, version: 3 };

    for (id, op, expected) in [
        (10u128, Op::Balance, Reply::Ok(20)),
        (11, Op::Debit(-1), Reply::Err("debit amount must be non-negative".to_owned())),
        (12, Op::Credit(-1), Reply::Err("credit amount must be non-negative".to_owned())),
    ] {
        let (reply, outcome) = run(&world, Some(held.clone()), id, op).await;
        assert_eq!(reply, expected);
        assert_eq!(resident(&outcome), Some(held.clone()));
    }
    assert_eq!(world.store.account(ACCOUNT), Account::default());
    assert_eq!(world.store.applied_requests(), 0);
}

#[tokio::test]
async fn a_compute_failure_cannot_mutate_durable_or_resident_state() {
    let world = SimWorld::new(Plan::ordered([FaultPoint::Compute]));
    let held = Account { balance: 11, version: 4 };
    let (reply, outcome) =
        run(&world, Some(held.clone()), 5, Op::Revalue(RevalueParams { scenarios: 8 })).await;

    assert!(matches!(reply, Reply::Err(_)));
    assert_eq!(resident(&outcome), Some(held));
    assert_eq!(world.store.account(ACCOUNT), Account::default());
    assert!(world.plan.is_exhausted());
}

#[tokio::test]
async fn a_successful_revalue_writes_the_computed_balance() {
    let world = SimWorld::healthy();
    let (reply, outcome) =
        run(&world, Some(Account::default()), 6, Op::Revalue(RevalueParams { scenarios: 0 })).await;

    assert_eq!(reply, Reply::Ok(0));
    assert_eq!(resident(&outcome), Some(Account { balance: 0, version: 1 }));
}

#[tokio::test]
async fn a_restart_reloads_durable_state_and_still_recognises_the_request() {
    let world = SimWorld::healthy();
    let (reply, _) = run(&world, None, 0xfeed, Op::Credit(25)).await;
    assert_eq!(reply, Reply::Ok(25));

    // A fresh processor has neither resident nor cached state; only the durable
    // account and request records survive.
    let restarted = SimWorld {
        store: world.store.clone(),
        cache: accounts::sim::SimCache::new(Plan::off()),
        offload: world.offload.clone(),
        plan: world.plan.clone(),
    };
    let (reply, outcome) = run(&restarted, None, 0xfeed, Op::Credit(25)).await;
    assert_eq!(reply, Reply::Duplicate(25));
    assert_eq!(resident(&outcome), Some(Account { balance: 25, version: 1 }));
    assert_eq!(restarted.store.account(ACCOUNT), Account { balance: 25, version: 1 });
}

#[tokio::test]
async fn an_old_duplicate_cannot_regress_newer_state() {
    let world = SimWorld::healthy();
    let mut state = None;
    for id in [1u128, 3, 4] {
        let (reply, outcome) = run(&world, state, id, Op::Credit(10)).await;
        assert!(matches!(reply, Reply::Ok(_)));
        state = resident(&outcome);
    }
    assert_eq!(state, Some(Account { balance: 30, version: 3 }));

    // Replaying the oldest id returns its historical balance, but must not roll
    // the resident snapshot back to it — that would livelock the missing id 2
    // behind version conflicts forever.
    let (reply, outcome) = run(&world, state, 1, Op::Credit(10)).await;
    assert_eq!(reply, Reply::Duplicate(10));
    assert_eq!(resident(&outcome), Some(Account { balance: 30, version: 3 }));

    let (reply, outcome) = run(&world, resident(&outcome), 2, Op::Credit(10)).await;
    assert_eq!(reply, Reply::Ok(40), "the genuinely missing request still applies");
    assert_eq!(resident(&outcome), Some(Account { balance: 40, version: 4 }));
}
