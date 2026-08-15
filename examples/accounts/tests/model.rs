//! The account transition checked against an independent wide-integer model.

use accounts::domain::{Account, ApplyError, Mutation, apply_mutation};

#[test]
fn account_transitions_match_an_independent_wide_integer_model() {
    bolero::check!().with_type::<Vec<Mutation>>().for_each(|mutations| {
        let mut actual = Account::default();
        let mut balance = 0i128;
        let mut version = 0u128;

        for mutation in mutations.iter().take(1_024) {
            let next_balance = match mutation {
                Mutation::Delta(delta) => balance + i128::from(*delta),
                Mutation::SetBalance(set) => i128::from(*set),
            };
            let next_version = version + 1;
            let expected =
                if next_balance < i128::from(i64::MIN) || next_balance > i128::from(i64::MAX) {
                    Err(ApplyError::BalanceOverflow)
                } else if next_version > u128::from(u64::MAX) {
                    Err(ApplyError::VersionOverflow)
                } else {
                    Ok(Account { balance: next_balance as i64, version: next_version as u64 })
                };

            let observed = apply_mutation(&actual, *mutation);
            assert_eq!(observed, expected);
            if let Ok(next) = observed {
                actual = next;
                balance = next_balance;
                version = next_version;
            }
        }
    });
}
