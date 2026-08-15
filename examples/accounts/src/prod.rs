//! Thin production PostgreSQL and Redis adapters.
//!
//! Ordering, retries, reconciliation, and cache-consistency policy belong in
//! the account processor, where deterministic fault tests can reach them. This
//! module only translates semantic port operations to client-library calls and
//! maps real error categories into the processor's error model.

use crate::domain::{Account, AccountId, Mutation, RequestId};
use crate::ports::{AccountCache, AccountStore, CacheError, CommitOutcome, StoreError};
use std::time::Duration;
use tokio_postgres::error::SqlState;

const LOAD_ACCOUNT: &str = "SELECT balance, version FROM xt_accounts WHERE id = $1";
const LOAD_REQUEST: &str = "SELECT account_id, mutation_kind, mutation_value, balance, version FROM xt_requests WHERE req_id = $1";
const INSERT_ACCOUNT: &str = "INSERT INTO xt_accounts (id, balance, version) VALUES ($1, $2, 1) ON CONFLICT DO NOTHING RETURNING balance, version";
const UPDATE_DELTA: &str = "UPDATE xt_accounts SET balance = balance + $3, version = version + 1 WHERE id = $1 AND version = $2 RETURNING balance, version";
const UPDATE_BALANCE: &str = "UPDATE xt_accounts SET balance = $3, version = version + 1 WHERE id = $1 AND version = $2 RETURNING balance, version";
const INSERT_REQUEST: &str = "INSERT INTO xt_requests (req_id, account_id, mutation_kind, mutation_value, balance, version) VALUES ($1, $2, $3, $4, $5, $6)";

pub const ACCOUNT_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS xt_accounts (
    id      bytea PRIMARY KEY CHECK (octet_length(id) = 8),
    balance bigint NOT NULL,
    version bigint NOT NULL CHECK (version >= 0)
);
CREATE TABLE IF NOT EXISTS xt_requests (
    req_id         bytea PRIMARY KEY CHECK (octet_length(req_id) = 16),
    account_id     bytea NOT NULL CHECK (octet_length(account_id) = 8),
    mutation_kind  smallint NOT NULL,
    mutation_value bigint NOT NULL,
    balance        bigint NOT NULL,
    version        bigint NOT NULL CHECK (version >= 0)
);
CREATE INDEX IF NOT EXISTS xt_requests_account ON xt_requests (account_id)
"#;

pub struct PgStore {
    pool: deadpool_postgres::Pool,
}

impl PgStore {
    pub fn new(pool: deadpool_postgres::Pool) -> Self {
        Self { pool }
    }

    /// Build an independent pool suitable for one shard reactor. Call this once
    /// per shard rather than cloning one pool across every core.
    pub fn from_url(url: &str, max_size: usize) -> Result<Self, String> {
        let config = url.parse::<tokio_postgres::Config>().map_err(|error| error.to_string())?;
        let manager = deadpool_postgres::Manager::new(config, tokio_postgres::NoTls);
        let pool = deadpool_postgres::Pool::builder(manager)
            .max_size(max_size)
            .runtime(deadpool_postgres::Runtime::Tokio1)
            .build()
            .map_err(|error| error.to_string())?;
        Ok(Self::new(pool))
    }

    pub async fn ensure_schema(&self) -> Result<(), StoreError> {
        let connection = self.pool.get().await.map_err(map_pg_pool)?;
        connection.batch_execute(ACCOUNT_SCHEMA).await.map_err(|error| map_pg(&error))
    }
}

fn map_pg(error: &tokio_postgres::Error) -> StoreError {
    match error.code() {
        Some(code) if *code == SqlState::T_R_SERIALIZATION_FAILURE => {
            StoreError::SerializationFailure
        }
        Some(code) if *code == SqlState::T_R_DEADLOCK_DETECTED => StoreError::Deadlock,
        Some(code) => StoreError::Constraint(code.code().to_owned()),
        None => StoreError::ConnectionReset,
    }
}

fn map_pg_pool(error: deadpool_postgres::PoolError) -> StoreError {
    match error {
        deadpool_postgres::PoolError::Timeout(_) => StoreError::Timeout,
        deadpool_postgres::PoolError::Backend(error) => map_pg(&error),
        _ => StoreError::PoolExhausted,
    }
}

fn account_from_row(row: &tokio_postgres::Row) -> Result<Account, StoreError> {
    account_from_row_offset(row, 0)
}

fn account_from_row_offset(
    row: &tokio_postgres::Row,
    offset: usize,
) -> Result<Account, StoreError> {
    let balance = row.get::<_, i64>(offset);
    let version = row.get::<_, i64>(offset + 1);
    let version = u64::try_from(version)
        .map_err(|_| StoreError::Constraint("negative account version".to_owned()))?;
    Ok(Account { balance, version })
}

impl AccountStore for PgStore {
    type Session = deadpool_postgres::Object;

    async fn acquire(&self) -> Result<Self::Session, StoreError> {
        self.pool.get().await.map_err(map_pg_pool)
    }

    async fn load(
        &self,
        session: &mut Self::Session,
        id: AccountId,
    ) -> Result<Account, StoreError> {
        let id = id.to_be_bytes();
        let statement = session.prepare_cached(LOAD_ACCOUNT).await.map_err(|e| map_pg(&e))?;
        let row = session.query_opt(&statement, &[&id.as_slice()]).await.map_err(|e| map_pg(&e))?;
        row.as_ref().map(account_from_row).transpose().map(|account| account.unwrap_or_default())
    }

    async fn commit(
        &self,
        session: &mut Self::Session,
        req_id: RequestId,
        id: AccountId,
        current: &Account,
        mutation: Mutation,
    ) -> Result<CommitOutcome, StoreError> {
        let id = id.to_be_bytes();
        let req_id = req_id.to_bytes();
        let expected = i64::try_from(current.version).map_err(|_| {
            StoreError::Constraint("account version exceeds PostgreSQL bigint".to_owned())
        })?;
        let transaction = session.transaction().await.map_err(|e| map_pg(&e))?;

        let (kind, value) = match mutation {
            Mutation::Delta(delta) => (0i16, delta),
            Mutation::SetBalance(balance) => (1i16, balance),
        };
        let request = transaction.prepare_cached(LOAD_REQUEST).await.map_err(|e| map_pg(&e))?;
        if let Some(row) =
            transaction.query_opt(&request, &[&req_id.as_slice()]).await.map_err(|e| map_pg(&e))?
        {
            let prior_id = row.get::<_, Vec<u8>>(0);
            let prior_kind = row.get::<_, i16>(1);
            let prior_value = row.get::<_, i64>(2);
            if prior_id.as_slice() != id || prior_kind != kind || prior_value != value {
                transaction.rollback().await.map_err(|e| map_pg(&e))?;
                return Err(StoreError::RequestConflict);
            }
            let account = account_from_row_offset(&row, 3)?;
            transaction.rollback().await.map_err(|e| map_pg(&e))?;
            return Ok(CommitOutcome::Committed { account, duplicate: true });
        }

        let row = if current.version == 0 {
            let statement =
                transaction.prepare_cached(INSERT_ACCOUNT).await.map_err(|e| map_pg(&e))?;
            transaction
                .query_opt(&statement, &[&id.as_slice(), &value])
                .await
                .map_err(|e| map_pg(&e))?
        } else {
            let sql = match mutation {
                Mutation::Delta(_) => UPDATE_DELTA,
                Mutation::SetBalance(_) => UPDATE_BALANCE,
            };
            let statement = transaction.prepare_cached(sql).await.map_err(|e| map_pg(&e))?;
            transaction
                .query_opt(&statement, &[&id.as_slice(), &expected, &value])
                .await
                .map_err(|e| map_pg(&e))?
        };

        let Some(row) = row else {
            transaction.rollback().await.map_err(|e| map_pg(&e))?;
            return Err(StoreError::VersionConflict);
        };
        let account = account_from_row(&row)?;
        let version = i64::try_from(account.version).map_err(|_| {
            StoreError::Constraint("account version exceeds PostgreSQL bigint".to_owned())
        })?;
        let insert_request =
            transaction.prepare_cached(INSERT_REQUEST).await.map_err(|e| map_pg(&e))?;
        transaction
            .execute(
                &insert_request,
                &[&req_id.as_slice(), &id.as_slice(), &kind, &value, &account.balance, &version],
            )
            .await
            .map_err(|e| map_pg(&e))?;
        match transaction.commit().await {
            Ok(()) => Ok(CommitOutcome::Committed { account, duplicate: false }),
            Err(error) if error.code().is_none() => Ok(CommitOutcome::InDoubt),
            Err(error) => Err(map_pg(&error)),
        }
    }
}

pub struct RedisCache {
    pool: deadpool_redis::Pool,
    key_prefix: String,
}

impl RedisCache {
    pub fn new(pool: deadpool_redis::Pool) -> Self {
        Self { pool, key_prefix: "xt:account:".to_owned() }
    }

    pub fn with_prefix(pool: deadpool_redis::Pool, key_prefix: impl Into<String>) -> Self {
        Self { pool, key_prefix: key_prefix.into() }
    }

    /// Build an independent Redis pool for a shard reactor.
    pub fn from_url(url: &str, max_size: usize) -> Result<Self, String> {
        let config = deadpool_redis::Config::from_url(url);
        let pool = config
            .builder()
            .map_err(|error| error.to_string())?
            .max_size(max_size)
            .runtime(deadpool_redis::Runtime::Tokio1)
            .build()
            .map_err(|error| error.to_string())?;
        Ok(Self::new(pool))
    }

    fn key(&self, id: AccountId) -> String {
        format!("{}{id}", self.key_prefix)
    }
}

fn map_redis(error: redis::RedisError) -> CacheError {
    use redis::ErrorKind;
    match error.kind() {
        ErrorKind::IoError => CacheError::ConnectionReset,
        ErrorKind::Moved | ErrorKind::Ask | ErrorKind::TryAgain => CacheError::Moved { slot: 0 },
        _ if error.is_timeout() => CacheError::Timeout,
        _ => CacheError::Server(error.to_string()),
    }
}

fn encode_account(account: &Account) -> [u8; 16] {
    let mut bytes = [0; 16];
    bytes[..8].copy_from_slice(&account.balance.to_le_bytes());
    bytes[8..].copy_from_slice(&account.version.to_le_bytes());
    bytes
}

fn decode_account(bytes: &[u8]) -> Option<Account> {
    let balance = i64::from_le_bytes(bytes.get(..8)?.try_into().ok()?);
    let version = u64::from_le_bytes(bytes.get(8..16)?.try_into().ok()?);
    (bytes.len() == 16).then_some(Account { balance, version })
}

impl AccountCache for RedisCache {
    async fn get(&self, id: AccountId) -> Result<Option<Account>, CacheError> {
        let mut connection = self.pool.get().await.map_err(|_| CacheError::PoolExhausted)?;
        let value: Option<Vec<u8>> = redis::cmd("GET")
            .arg(self.key(id))
            .query_async(&mut *connection)
            .await
            .map_err(map_redis)?;
        Ok(value.as_deref().and_then(decode_account))
    }

    async fn put(&self, id: AccountId, account: &Account, ttl: Duration) -> Result<(), CacheError> {
        let mut connection = self.pool.get().await.map_err(|_| CacheError::PoolExhausted)?;
        redis::cmd("SET")
            .arg(self.key(id))
            .arg(&encode_account(account))
            .arg("PX")
            .arg(ttl.as_millis() as u64)
            .query_async::<()>(&mut *connection)
            .await
            .map_err(map_redis)
    }

    async fn invalidate(&self, id: AccountId) -> Result<(), CacheError> {
        let mut connection = self.pool.get().await.map_err(|_| CacheError::PoolExhausted)?;
        redis::cmd("DEL")
            .arg(self.key(id))
            .query_async::<()>(&mut *connection)
            .await
            .map_err(map_redis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_codec_round_trips_signed_balances_and_large_versions() {
        for account in [
            Account { balance: 0, version: 0 },
            Account { balance: i64::MIN, version: u64::MAX },
            Account { balance: i64::MAX, version: 42 },
        ] {
            assert_eq!(decode_account(&encode_account(&account)), Some(account));
        }
    }

    #[test]
    fn cache_codec_rejects_truncated_and_trailing_data() {
        assert_eq!(decode_account(&[0; 15]), None);
        assert_eq!(decode_account(&[0; 17]), None);
    }
}
