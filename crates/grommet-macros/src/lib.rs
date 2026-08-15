//! SQLite-style coverage accounting with zero production abstraction cost.
//!
//! `always!` and `never!` are three separately compiled definitions: asserted
//! in debug, folded out of coverage, and literal pass-throughs in release.
//! Arguments must be side-effect free because coverage builds discard them.

#![cfg_attr(not(coverage), no_std)]

#[cfg(coverage)]
#[macro_export]
macro_rules! always {
    ($condition:expr) => {{
        let _ = &$condition;
        true
    }};
}

#[cfg(coverage)]
#[macro_export]
macro_rules! never {
    ($condition:expr) => {{
        let _ = &$condition;
        false
    }};
}

#[cfg(all(not(coverage), debug_assertions))]
#[macro_export]
macro_rules! always {
    ($condition:expr) => {{
        let value = $condition;
        debug_assert!(value, "ALWAYS violated: {}", stringify!($condition));
        value
    }};
}

#[cfg(all(not(coverage), debug_assertions))]
#[macro_export]
macro_rules! never {
    ($condition:expr) => {{
        let value = $condition;
        debug_assert!(!value, "NEVER violated: {}", stringify!($condition));
        value
    }};
}

#[cfg(all(not(coverage), not(debug_assertions)))]
#[macro_export]
macro_rules! always {
    ($condition:expr) => {
        $condition
    };
}

#[cfg(all(not(coverage), not(debug_assertions)))]
#[macro_export]
macro_rules! never {
    ($condition:expr) => {
        $condition
    };
}

#[cfg(coverage)]
#[macro_export]
macro_rules! testcase {
    ($condition:expr) => {{
        let value = $condition;
        $crate::coverage::record(file!(), line!(), value);
        value
    }};
}

#[cfg(not(coverage))]
#[macro_export]
macro_rules! testcase {
    ($condition:expr) => {
        $condition
    };
}

#[cfg(coverage)]
pub mod coverage {
    use std::collections::BTreeMap;
    use std::sync::{Mutex, OnceLock};

    const SAW_TRUE: u8 = 0b01;
    const SAW_FALSE: u8 = 0b10;
    type Table = BTreeMap<(&'static str, u32), u8>;

    fn table() -> &'static Mutex<Table> {
        static TABLE: OnceLock<Mutex<Table>> = OnceLock::new();
        TABLE.get_or_init(|| Mutex::new(Table::new()))
    }

    pub fn record(file: &'static str, line: u32, value: bool) {
        let bit = if value { SAW_TRUE } else { SAW_FALSE };
        *table().lock().expect("coverage table poisoned").entry((file, line)).or_insert(0) |= bit;
    }

    pub fn assert_all_covered() {
        let table = table().lock().expect("coverage table poisoned");
        let incomplete: Vec<_> = table
            .iter()
            .filter(|(_, mask)| **mask != SAW_TRUE | SAW_FALSE)
            .map(|(&(file, line), &mask)| {
                let missing = if mask & SAW_TRUE == 0 { "true" } else { "false" };
                format!("{file}:{line} never observed {missing}")
            })
            .collect();
        assert!(
            incomplete.is_empty(),
            "incomplete testcase! coverage:\n  {}",
            incomplete.join("\n  ")
        );
    }
}

#[cfg(all(fault_injection, not(debug_assertions), not(sim)))]
compile_error!(
    "fault injection is enabled in an optimized non-sim build; also pass --cfg sim for an intentional simulation"
);
