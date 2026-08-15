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
