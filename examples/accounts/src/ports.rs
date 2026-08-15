//! Semantic boundaries to PostgreSQL and Redis.
//!
//! Futures are intentionally allowed to be `!Send`: each shard and its pools
//! stay on one core, which is the whole point of the runtime underneath. Native
//! async trait methods keep static dispatch and avoid a boxed future per call.

#![allow(async_fn_in_trait)]

use crate::domain::{Account, AccountId, Mutation, RequestId};
use std::time::Duration;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommitOutcome {
    Committed {
        account: Account,
        duplicate: bool,
    },
    /// The commit was submitted but its acknowledgement was lost. It may or may
    /// not have taken effect, and nothing may assume either way until the
    /// account is reloaded from the authoritative store.
    InDoubt,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum StoreError {
    #[error("connection pool exhausted")]
    PoolExhausted,
    #[error("database operation timed out")]
    Timeout,
    #[error("database connection reset")]
    ConnectionReset,
    #[error("serialization failure")]
    SerializationFailure,
    #[error("deadlock detected")]
    Deadlock,
    #[error("account version conflict")]
    VersionConflict,
    #[error("request id was reused for different work")]
    RequestConflict,
    #[error("database constraint violation: {0}")]
    Constraint(String),
    #[error("database server error: {0}")]
    Server(String),
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum CacheError {
    #[error("cache connection pool exhausted")]
    PoolExhausted,
    #[error("cache operation timed out")]
    Timeout,
    #[error("cache connection reset")]
    ConnectionReset,
    #[error("cache key moved to slot {slot}")]
    Moved { slot: u16 },
    #[error("cache server error: {0}")]
    Server(String),
}

/// The authoritative store. Every mutation is recorded against a caller-stable
/// request id, which is what makes a retry safe.
pub trait AccountStore: 'static {
    type Session: 'static;

    async fn acquire(&self) -> Result<Self::Session, StoreError>;

    async fn load(&self, session: &mut Self::Session, id: AccountId)
    -> Result<Account, StoreError>;

    /// Apply `mutation` if `req_id` has not been applied before, returning the
    /// recorded result if it has. `current.version` is the optimistic
    /// precondition.
    async fn commit(
        &self,
        session: &mut Self::Session,
        req_id: RequestId,
        id: AccountId,
        current: &Account,
        mutation: Mutation,
    ) -> Result<CommitOutcome, StoreError>;
}

/// A non-authoritative cache. Every method may fail without failing the
/// operation that called it.
pub trait AccountCache: 'static {
    async fn get(&self, id: AccountId) -> Result<Option<Account>, CacheError>;
    async fn put(&self, id: AccountId, account: &Account, ttl: Duration) -> Result<(), CacheError>;
    async fn invalidate(&self, id: AccountId) -> Result<(), CacheError>;
}
