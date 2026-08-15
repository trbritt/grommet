//! The account processor: everything this service does with one request.
//!
//! Ordering, retries, reconciliation and cache-consistency policy live here,
//! where deterministic fault tests can reach them, rather than in the adapters.

use crate::domain::{Account, AccountId, Mutation, Op, Reply, RequestId, heavy_revalue};
use crate::ports::{AccountCache, AccountStore, CommitOutcome, StoreError};
use grommet::{Call, ClassId, Disposition, Fallout, Offload, ProcessError, Processor, Work};
use std::rc::Rc;
use std::time::Duration;

const CACHE_TTL: Duration = Duration::from_secs(30);

/// One submitted operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    /// The caller-stable idempotency key. A retry must reuse it.
    pub req_id: RequestId,
    pub account: AccountId,
    pub op: Op,
}

impl Work for Request {
    type Key = AccountId;
    /// ULIDs are 128-bit, so they are their own exact idempotency key.
    type Id = u128;

    fn key(&self) -> AccountId {
        self.account
    }

    fn class(&self) -> ClassId {
        self.op.class()
    }

    fn request_id(&self) -> Option<u128> {
        Some(self.req_id.into())
    }
}

/// A request paired with the channel its reply travels back on.
pub type AccountCall = Call<Request, Reply>;

/// A failure that leaves this processor unable to trust its resident state.
///
/// Failures that do *not* invalidate state are not modelled here: the processor
/// answers the caller and returns `Ok(Disposition::Keep(..))`, because a
/// rejected debit is not a reason to reload an account.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Failure {
    /// The commit was submitted and its outcome was never learned. PostgreSQL
    /// may or may not hold the mutation, so the account must be reloaded and
    /// the request replayed under the same id to find out.
    #[error("commit outcome unknown; retry with the same request id")]
    InDoubt,
    /// Another writer advanced the account past the version this operation was
    /// built on. The mutation definitely did not apply; the resident copy is
    /// simply stale.
    #[error("account version conflict")]
    Stale,
}

impl ProcessError for Failure {
    fn fallout(&self) -> Fallout {
        match self {
            Self::InDoubt => Fallout::InDoubt,
            Self::Stale => Fallout::Untouched,
        }
    }
}

pub struct AccountProcessor<S, K, O> {
    store: Rc<S>,
    cache: Rc<K>,
    offload: O,
}

impl<S, K, O> AccountProcessor<S, K, O> {
    pub fn new(store: Rc<S>, cache: Rc<K>, offload: O) -> Self {
        Self { store, cache, offload }
    }
}

// Derived `Clone` would demand `S: Clone`, but the whole point of the `Rc`s is
// that the shard's store and cache are shared by handle, not duplicated.
impl<S, K, O: Clone> Clone for AccountProcessor<S, K, O> {
    fn clone(&self) -> Self {
        Self { store: self.store.clone(), cache: self.cache.clone(), offload: self.offload.clone() }
    }
}

impl<S, K, O> Processor for AccountProcessor<S, K, O>
where
    S: AccountStore,
    K: AccountCache,
    O: Offload,
{
    type Work = AccountCall;
    type State = Account;
    type Error = Failure;

    async fn process(
        &self,
        id: AccountId,
        state: Option<Account>,
        call: AccountCall,
    ) -> Result<Disposition<Account>, Failure> {
        let (request, responder) = call.into_parts();

        // Resident state, else the cache, else the authoritative store. A cache
        // failure is never fatal: it just costs a read.
        let mut session = None;
        let account = match state {
            Some(account) => account,
            None => match self.cache.get(id).await {
                Ok(Some(account)) => account,
                Ok(None) | Err(_) => {
                    let mut acquired = match self.store.acquire().await {
                        Ok(session) => session,
                        Err(error) => {
                            responder.send(Reply::Err(error.to_string()));
                            return Ok(Disposition::Drop);
                        }
                    };
                    let loaded = match self.store.load(&mut acquired, id).await {
                        Ok(account) => account,
                        Err(error) => {
                            responder.send(Reply::Err(error.to_string()));
                            return Ok(Disposition::Drop);
                        }
                    };
                    let _ = self.cache.put(id, &loaded, CACHE_TTL).await;
                    session = Some(acquired);
                    loaded
                }
            },
        };

        let mutation = match self.plan(&account, &request.op).await {
            Ok(mutation) => mutation,
            // Answered without touching anything durable: a read, a rejected
            // amount, or a compute failure. The account we hold is still good.
            Err(reply) => {
                responder.send(reply);
                return Ok(Disposition::Keep(account));
            }
        };

        // Invalidate before the authoritative write, so a crash between the two
        // cannot leave a pre-commit value looking current. A Redis failure never
        // blocks PostgreSQL.
        let _ = self.cache.invalidate(id).await;
        let mut session = match session {
            Some(session) => session,
            None => match self.store.acquire().await {
                Ok(session) => session,
                Err(error) => {
                    responder.send(Reply::Err(error.to_string()));
                    return Ok(Disposition::Keep(account));
                }
            },
        };

        match self.store.commit(&mut session, request.req_id, id, &account, mutation).await {
            Ok(CommitOutcome::Committed { account: committed, duplicate }) => {
                let recorded = committed.balance;
                // A duplicate carries the result recorded for that request id,
                // which may be older than what we already hold. Letting it
                // overwrite a newer snapshot would livelock a genuinely missing
                // request behind version conflicts forever.
                let resident = if duplicate && account.version > committed.version {
                    account
                } else {
                    committed
                };
                if self.cache.put(id, &resident, CACHE_TTL).await.is_err() {
                    let _ = self.cache.invalidate(id).await;
                }
                responder.send(if duplicate {
                    Reply::Duplicate(recorded)
                } else {
                    Reply::Ok(recorded)
                });
                Ok(Disposition::Keep(resident))
            }
            Ok(CommitOutcome::InDoubt) => {
                let _ = self.cache.invalidate(id).await;
                responder.send(Reply::Err(Failure::InDoubt.to_string()));
                Err(Failure::InDoubt)
            }
            Err(StoreError::VersionConflict) => {
                let _ = self.cache.invalidate(id).await;
                responder.send(Reply::Err(StoreError::VersionConflict.to_string()));
                Err(Failure::Stale)
            }
            Err(error) => {
                // Every other store error is reported by the contract as having
                // definitely not applied, so the account we hold is still good.
                let _ = self.cache.invalidate(id).await;
                responder.send(Reply::Err(error.to_string()));
                Ok(Disposition::Keep(account))
            }
        }
    }

    fn on_expired(&self, _id: AccountId, call: AccountCall) {
        call.reply(Reply::Err("request deadline passed before it was dispatched".to_owned()));
    }

    fn on_coalesced(&self, _id: AccountId, call: AccountCall) {
        // The original attempt is still outstanding and will answer its own
        // caller. Retrying while that is true adds nothing but load.
        call.reply(Reply::Err("an attempt with this request id is already in flight".to_owned()));
    }
}

impl<S, K, O> AccountProcessor<S, K, O>
where
    S: AccountStore,
    K: AccountCache,
    O: Offload,
{
    /// Decide what this operation needs to write, or produce the reply that
    /// finishes it without writing anything.
    ///
    /// Compute-class work runs here, on the offload pool, so the shard core
    /// stays free to dispatch other keys while it runs.
    async fn plan(&self, account: &Account, op: &Op) -> Result<Mutation, Reply> {
        match op {
            Op::Balance => Err(Reply::Ok(account.balance)),
            Op::Debit(amount) if *amount < 0 => {
                Err(Reply::Err("debit amount must be non-negative".to_owned()))
            }
            Op::Credit(amount) if *amount < 0 => {
                Err(Reply::Err("credit amount must be non-negative".to_owned()))
            }
            Op::Debit(amount) => Ok(Mutation::Delta(-*amount)),
            Op::Credit(amount) => Ok(Mutation::Delta(*amount)),
            Op::Revalue(params) => {
                let snapshot = account.clone();
                let params = params.clone();
                match self.offload.run(move || heavy_revalue(&snapshot, &params)).await {
                    Ok(balance) => Ok(Mutation::SetBalance(balance)),
                    Err(error) => Err(Reply::Err(error.to_string())),
                }
            }
        }
    }
}
