//! The account domain: deterministic, dependency-free, and separately testable.

use grommet::ClassId;

pub type AccountId = u64;
pub type RequestId = ulid::Ulid;

/// Work classes this service uses. IO-bound operations and CPU-bound
/// revaluation get independent in-flight budgets, so a flood of one cannot
/// starve the other.
///
/// This is the conventional split, so the constants come from the runtime
/// rather than being declared again here.
pub use grommet::{CLASSES, COMPUTE, IO};

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "gen", derive(bolero_generator::TypeGenerator))]
pub enum Op {
    Debit(i64),
    Credit(i64),
    Balance,
    Revalue(RevalueParams),
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "gen", derive(bolero_generator::TypeGenerator))]
pub struct RevalueParams {
    pub scenarios: u32,
}

impl Op {
    #[inline]
    pub fn class(&self) -> ClassId {
        match self {
            Self::Debit(_) | Self::Credit(_) | Self::Balance => IO,
            Self::Revalue(_) => COMPUTE,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reply {
    Ok(i64),
    /// This request id was already applied; the balance is the historical
    /// result recorded for it, not a fresh application.
    Duplicate(i64),
    Err(String),
}

#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct Account {
    pub balance: i64,
    pub version: u64,
}

impl Account {
    pub fn empty() -> Self {
        Self::default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "gen", derive(bolero_generator::TypeGenerator))]
pub enum Mutation {
    Delta(i64),
    SetBalance(i64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApplyError {
    BalanceOverflow,
    VersionOverflow,
}

pub fn apply_mutation(current: &Account, mutation: Mutation) -> Result<Account, ApplyError> {
    let balance = match mutation {
        Mutation::Delta(delta) => {
            current.balance.checked_add(delta).ok_or(ApplyError::BalanceOverflow)?
        }
        Mutation::SetBalance(balance) => balance,
    };
    let version = current.version.checked_add(1).ok_or(ApplyError::VersionOverflow)?;
    Ok(Account { balance, version })
}

/// A deliberately expensive, deterministic calculation, standing in for the
/// CPU-bound work a real service offloads. Keeping it here lets the production
/// Rayon pool and the deterministic in-process executor run the same function.
pub fn heavy_revalue(account: &Account, params: &RevalueParams) -> i64 {
    let mut value = account.balance;
    let iterations = i64::from(params.scenarios) * 200_000;
    for i in 0..iterations {
        value = value.wrapping_add(i).wrapping_mul(2_654_435_761u32 as i64) ^ (value >> 13);
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutation_is_checked_and_advances_exactly_one_version() {
        let current = Account { balance: 10, version: 4 };
        assert_eq!(
            apply_mutation(&current, Mutation::Delta(-3)),
            Ok(Account { balance: 7, version: 5 })
        );
        assert_eq!(
            apply_mutation(&Account { balance: i64::MAX, version: 0 }, Mutation::Delta(1)),
            Err(ApplyError::BalanceOverflow)
        );
        assert_eq!(
            apply_mutation(&Account { balance: 0, version: u64::MAX }, Mutation::SetBalance(3)),
            Err(ApplyError::VersionOverflow)
        );
    }

    #[test]
    fn empty_and_revalue_are_deterministic() {
        assert_eq!(Account::empty(), Account::default());
        let account = Account { balance: 7, version: 2 };
        assert_eq!(heavy_revalue(&account, &RevalueParams { scenarios: 0 }), 7);
        assert_eq!(
            heavy_revalue(&account, &RevalueParams { scenarios: 1 }),
            heavy_revalue(&account, &RevalueParams { scenarios: 1 })
        );
    }

    #[test]
    fn operations_route_to_the_budget_that_matches_their_cost() {
        assert_eq!(Op::Balance.class(), IO);
        assert_eq!(Op::Debit(1).class(), IO);
        assert_eq!(Op::Credit(1).class(), IO);
        assert_eq!(Op::Revalue(RevalueParams { scenarios: 1 }).class(), COMPUTE);
    }
}
