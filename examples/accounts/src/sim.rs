//! Deterministic models of PostgreSQL and Redis, with injectable faults.
//!
//! These are decorators around the same contracts production implements, not
//! mocks of the processor. In particular [`FaultPoint::CommitAfterApply`]
//! writes durable state and only then reports [`CommitOutcome::InDoubt`],
//! reproducing the lost-acknowledgement window that is otherwise very hard to
//! force against a real server.

use crate::domain::{Account, AccountId, Mutation, RequestId, RevalueParams, heavy_revalue};
use crate::ports::{AccountCache, AccountStore, CacheError, CommitOutcome, StoreError};
use ahash::AHashMap;
use grommet::{Offload, OffloadError};
use grommet_testkit::FaultPlan;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

/// Every place this service can fail against an external dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultPoint {
    StoreAcquire,
    StoreLoad,
    /// Fails before anything durable changes.
    CommitBeforeApply,
    /// Applies durably, then loses the acknowledgement.
    CommitAfterApply,
    CacheGet,
    CachePut,
    CacheInvalidate,
    Compute,
}

pub type Plan = FaultPlan<FaultPoint>;

#[derive(Clone, Debug)]
struct AppliedRequest {
    account_id: AccountId,
    mutation: Mutation,
    account: Account,
}

#[derive(Default)]
struct StoreState {
    accounts: AHashMap<AccountId, Account>,
    requests: AHashMap<RequestId, AppliedRequest>,
}

#[derive(Clone, Default)]
pub struct SimStore {
    state: Rc<RefCell<StoreState>>,
    faults: Plan,
}

impl SimStore {
    pub fn new(faults: Plan) -> Self {
        Self { state: Rc::new(RefCell::new(StoreState::default())), faults }
    }

    pub fn account(&self, id: AccountId) -> Account {
        self.state.borrow().accounts.get(&id).cloned().unwrap_or_default()
    }

    /// Requests durably recorded, which is what makes a replay a duplicate
    /// rather than a second application.
    pub fn applied_requests(&self) -> usize {
        self.state.borrow().requests.len()
    }
}

impl AccountStore for SimStore {
    type Session = ();

    async fn acquire(&self) -> Result<Self::Session, StoreError> {
        if self.faults.fires(FaultPoint::StoreAcquire) {
            Err(StoreError::PoolExhausted)
        } else {
            Ok(())
        }
    }

    async fn load(
        &self,
        _session: &mut Self::Session,
        id: AccountId,
    ) -> Result<Account, StoreError> {
        if self.faults.fires(FaultPoint::StoreLoad) {
            Err(StoreError::ConnectionReset)
        } else {
            Ok(self.account(id))
        }
    }

    async fn commit(
        &self,
        _session: &mut Self::Session,
        req_id: RequestId,
        id: AccountId,
        current: &Account,
        mutation: Mutation,
    ) -> Result<CommitOutcome, StoreError> {
        if self.faults.fires(FaultPoint::CommitBeforeApply) {
            return Err(StoreError::ConnectionReset);
        }

        let mut state = self.state.borrow_mut();
        if let Some(prior) = state.requests.get(&req_id) {
            if prior.account_id != id || prior.mutation != mutation {
                return Err(StoreError::RequestConflict);
            }
            return Ok(CommitOutcome::Committed {
                account: prior.account.clone(),
                duplicate: true,
            });
        }

        let durable = state.accounts.get(&id).cloned().unwrap_or_default();
        if durable.version != current.version {
            return Err(StoreError::VersionConflict);
        }
        let balance = match mutation {
            Mutation::Delta(delta) => durable
                .balance
                .checked_add(delta)
                .ok_or_else(|| StoreError::Constraint("balance overflow".to_owned()))?,
            Mutation::SetBalance(balance) => balance,
        };
        let version = durable
            .version
            .checked_add(1)
            .ok_or_else(|| StoreError::Constraint("version overflow".to_owned()))?;
        let account = Account { balance, version };
        state.accounts.insert(id, account.clone());
        state
            .requests
            .insert(req_id, AppliedRequest { account_id: id, mutation, account: account.clone() });
        drop(state);

        // Durably applied, acknowledgement lost.
        if self.faults.fires(FaultPoint::CommitAfterApply) {
            Ok(CommitOutcome::InDoubt)
        } else {
            Ok(CommitOutcome::Committed { account, duplicate: false })
        }
    }
}

#[derive(Clone, Default)]
pub struct SimCache {
    values: Rc<RefCell<AHashMap<AccountId, Account>>>,
    faults: Plan,
}

impl SimCache {
    pub fn new(faults: Plan) -> Self {
        Self { values: Rc::new(RefCell::new(AHashMap::new())), faults }
    }

    pub fn account(&self, id: AccountId) -> Option<Account> {
        self.values.borrow().get(&id).cloned()
    }
}

impl AccountCache for SimCache {
    async fn get(&self, id: AccountId) -> Result<Option<Account>, CacheError> {
        if self.faults.fires(FaultPoint::CacheGet) {
            Err(CacheError::ConnectionReset)
        } else {
            Ok(self.account(id))
        }
    }

    async fn put(
        &self,
        id: AccountId,
        account: &Account,
        _ttl: Duration,
    ) -> Result<(), CacheError> {
        if self.faults.fires(FaultPoint::CachePut) {
            Err(CacheError::ConnectionReset)
        } else {
            self.values.borrow_mut().insert(id, account.clone());
            Ok(())
        }
    }

    async fn invalidate(&self, id: AccountId) -> Result<(), CacheError> {
        if self.faults.fires(FaultPoint::CacheInvalidate) {
            Err(CacheError::ConnectionReset)
        } else {
            self.values.borrow_mut().remove(&id);
            Ok(())
        }
    }
}

/// Runs the same calculation as production, inline and without worker threads,
/// so a simulation has no scheduling nondeterminism at all.
#[derive(Clone, Default)]
pub struct SimOffload {
    faults: Plan,
}

impl SimOffload {
    pub fn new(faults: Plan) -> Self {
        Self { faults }
    }
}

impl Offload for SimOffload {
    async fn run<F, T>(&self, task: F) -> Result<T, OffloadError>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        if self.faults.fires(FaultPoint::Compute) {
            Err(OffloadError::Injected)
        } else {
            Ok(task())
        }
    }
}

/// A whole deterministic world: store, cache and compute sharing one plan.
pub struct SimWorld {
    pub store: SimStore,
    pub cache: SimCache,
    pub offload: SimOffload,
    pub plan: Plan,
}

impl SimWorld {
    pub fn new(plan: Plan) -> Self {
        Self {
            store: SimStore::new(plan.clone()),
            cache: SimCache::new(plan.clone()),
            offload: SimOffload::new(plan.clone()),
            plan,
        }
    }

    pub fn healthy() -> Self {
        Self::new(Plan::off())
    }

    pub fn processor(&self) -> crate::processor::AccountProcessor<SimStore, SimCache, SimOffload> {
        crate::processor::AccountProcessor::new(
            Rc::new(self.store.clone()),
            Rc::new(self.cache.clone()),
            self.offload.clone(),
        )
    }
}

/// The pure calculation, exposed so benchmarks can measure it without a shard.
pub fn revalue(account: &Account, params: &RevalueParams) -> i64 {
    heavy_revalue(account, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Op;
    use grommet::Processor as _;

    const ACCOUNT: AccountId = 3;

    /// These doubles are the specification every fault test is measured
    /// against, so what they promise is worth checking directly rather than
    /// only through the processor that consumes them.
    // `SimStore::Session` is a unit — there is no pool to hold — so binding
    // one reads as a unit binding. It is written this way so the tests still
    // say how the port is meant to be used.
    #[allow(clippy::let_unit_value)]
    #[tokio::test]
    async fn the_store_reads_back_what_it_committed() {
        let store = SimStore::new(FaultPlan::off());
        let mut session = store.acquire().await.expect("a healthy store hands out sessions");
        assert_eq!(
            store.load(&mut session, ACCOUNT).await.unwrap(),
            Account::default(),
            "an account nobody has touched reads as empty rather than missing"
        );

        let current = Account::default();
        store
            .commit(&mut session, RequestId::from(1u128), ACCOUNT, &current, Mutation::Delta(40))
            .await
            .expect("the commit applies");
        assert_eq!(
            store.load(&mut session, ACCOUNT).await.unwrap(),
            Account { balance: 40, version: 1 },
            "a load that ignored what was written would make every fault test vacuous"
        );
        assert_eq!(store.account(ACCOUNT), Account { balance: 40, version: 1 });
    }

    #[allow(clippy::let_unit_value)]
    #[tokio::test]
    async fn reusing_a_request_id_for_different_work_is_a_conflict() {
        let store = SimStore::new(FaultPlan::off());
        let mut session = store.acquire().await.unwrap();
        let id = RequestId::from(7u128);
        let empty = Account::default();
        store
            .commit(&mut session, id, ACCOUNT, &empty, Mutation::Delta(5))
            .await
            .expect("the first use of the id applies");
        let applied = Account { balance: 5, version: 1 };

        // Same id, same account, different mutation.
        assert_eq!(
            store.commit(&mut session, id, ACCOUNT, &applied, Mutation::Delta(6)).await,
            Err(StoreError::RequestConflict),
        );
        // Same id, same mutation, different account.
        assert_eq!(
            store.commit(&mut session, id, ACCOUNT + 1, &empty, Mutation::Delta(5)).await,
            Err(StoreError::RequestConflict),
        );
        // Either one alone is enough to conflict, so an exact replay — and only
        // an exact replay — is recognised as the duplicate it is.
        assert_eq!(
            store.commit(&mut session, id, ACCOUNT, &applied, Mutation::Delta(5)).await,
            Ok(CommitOutcome::Committed { account: applied, duplicate: true }),
        );
    }

    #[tokio::test]
    async fn the_cache_reads_back_what_it_stored_until_it_is_invalidated() {
        let cache = SimCache::new(FaultPlan::off());
        assert_eq!(cache.get(ACCOUNT).await.unwrap(), None, "a cold cache is a miss");

        let account = Account { balance: 12, version: 4 };
        cache.put(ACCOUNT, &account, Duration::from_secs(30)).await.unwrap();
        assert_eq!(
            cache.get(ACCOUNT).await.unwrap(),
            Some(account),
            "a cache that always missed would never exercise the resident path"
        );

        cache.invalidate(ACCOUNT).await.unwrap();
        assert_eq!(cache.get(ACCOUNT).await.unwrap(), None);
    }

    #[test]
    fn the_benchmark_entry_point_measures_the_same_calculation_the_service_runs() {
        let account = Account { balance: 17, version: 2 };
        let params = RevalueParams { scenarios: 1 };
        assert_eq!(
            revalue(&account, &params),
            heavy_revalue(&account, &params),
            "a benchmark measuring something else would report a number about nothing"
        );
    }

    #[tokio::test]
    async fn a_world_wires_its_processor_to_the_same_doubles_it_exposes() {
        let world = SimWorld::healthy();
        let (call, reply) = crate::processor::AccountCall::new(crate::processor::Request {
            req_id: RequestId::from(1u128),
            account: ACCOUNT,
            op: Op::Credit(25),
        });
        world.processor().process(ACCOUNT, None, call).await.expect("a healthy world commits");

        assert_eq!(reply.await.unwrap(), crate::domain::Reply::Ok(25));
        assert_eq!(
            world.store.account(ACCOUNT),
            Account { balance: 25, version: 1 },
            "the world's store handle must be the one its processor writes through"
        );
    }
}
