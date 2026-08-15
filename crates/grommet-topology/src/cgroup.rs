//! CFS bandwidth limits, the one thing hwloc cannot tell us.
//!
//! hwloc answers *which* CPUs this process may run on, and answers it far
//! better than anything hand-rolled. It does not answer *how much* CPU time the
//! process may consume, because that is not a topology property.
//!
//! That gap is what hurts in production. A container limited to two cores on a
//! ninety-six core host still sees ninety-six cores through every topology API,
//! and one pinned thread per core produces ninety-six threads that stall
//! together every time the period's budget runs out.
//!
//! The parsers here are pure functions over file contents, so every shape —
//! v1 against v2, fractional limits, unlimited groups, malformed files — is a
//! test without needing a container.

use std::time::Duration;

const V2_MAX: &str = "/sys/fs/cgroup/cpu.max";
const V1_QUOTA: &str = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us";
const V1_PERIOD: &str = "/sys/fs/cgroup/cpu/cpu.cfs_period_us";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuotaSource {
    CgroupV2,
    CgroupV1,
}

/// A CFS bandwidth limit, expressed in whole cores.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quota {
    /// The limit in cores, so `2.5` means two and a half cores of CPU time.
    pub cores: f64,
    pub period: Duration,
    pub source: QuotaSource,
}

impl Quota {
    /// The most threads that can stay runnable without the group being
    /// throttled. Half a core cannot be spent by half a thread.
    pub fn usable_cores(&self) -> usize {
        self.cores.floor().max(1.0) as usize
    }
}

/// Read this process's bandwidth limit, if it has one.
pub fn detect() -> Option<Quota> {
    let read = |path: &str| std::fs::read_to_string(path).ok();
    match read(V2_MAX) {
        Some(text) => parse_v2(&text),
        None => parse_v1(read(V1_QUOTA)?.as_str(), read(V1_PERIOD).as_deref()),
    }
}

/// cgroup v2 states the limit as `"<quota|max> <period>"` in microseconds.
pub fn parse_v2(text: &str) -> Option<Quota> {
    let mut fields = text.split_whitespace();
    let quota = fields.next()?;
    if quota == "max" {
        return None;
    }
    build(quota.parse().ok()?, fields.next(), QuotaSource::CgroupV2)
}

/// cgroup v1 splits the limit across two files and uses `-1` for unlimited.
pub fn parse_v1(quota: &str, period: Option<&str>) -> Option<Quota> {
    let quota: i64 = quota.trim().parse().ok()?;
    if quota <= 0 {
        return None;
    }
    build(quota as u64, period, QuotaSource::CgroupV1)
}

fn build(quota: u64, period: Option<&str>, source: QuotaSource) -> Option<Quota> {
    let period: u64 = period.and_then(|text| text.trim().parse().ok()).unwrap_or(100_000);
    (period > 0).then(|| Quota {
        cores: quota as f64 / period as f64,
        period: Duration::from_micros(period),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_v2_limit_is_read_as_a_core_count() {
        let quota = parse_v2("200000 100000\n").unwrap();
        assert_eq!(quota.cores, 2.0);
        assert_eq!(quota.period, Duration::from_millis(100));
        assert_eq!(quota.source, QuotaSource::CgroupV2);
        assert_eq!(quota.usable_cores(), 2);
    }

    #[test]
    fn a_fractional_limit_survives_but_cannot_buy_a_whole_thread() {
        // `resources.limits.cpu: 1500m` in Kubernetes.
        let quota = parse_v2("150000 100000").unwrap();
        assert_eq!(quota.cores, 1.5);
        assert_eq!(quota.usable_cores(), 1);
        // Even a tiny slice must leave one usable thread rather than zero.
        assert_eq!(parse_v2("1000 100000").unwrap().usable_cores(), 1);
    }

    #[test]
    fn an_unlimited_group_reports_no_quota() {
        assert_eq!(parse_v2("max 100000\n"), None);
        assert_eq!(parse_v1("-1", Some("100000")), None);
        assert_eq!(parse_v1("0", Some("100000")), None);
    }

    #[test]
    fn v1_is_read_from_its_two_files_and_defaults_its_period() {
        let quota = parse_v1("400000\n", Some("100000\n")).unwrap();
        assert_eq!(quota.cores, 4.0);
        assert_eq!(quota.source, QuotaSource::CgroupV1);
        assert_eq!(parse_v1("400000", None).unwrap().cores, 4.0, "the default period is 100ms");
    }

    #[test]
    fn a_nonsensical_file_is_ignored_rather_than_dividing_by_zero() {
        assert_eq!(parse_v2("200000 0"), None);
        assert_eq!(parse_v2("garbage 100000"), None);
        assert_eq!(parse_v2(""), None);
        assert_eq!(parse_v1("nonsense", Some("100000")), None);
    }
}
