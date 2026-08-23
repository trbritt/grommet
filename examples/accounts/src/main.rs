//! Production bootstrap for the account service.

use accounts::domain::{COMPUTE, IO};
use accounts::frontdoor::Frontdoor;
use accounts::processor::AccountProcessor;
use accounts::prod::{PgStore, RedisCache};
use grommet::metrics::ShardStats;
use grommet::topology::{Observation, Workload, detect};
use grommet::{PinPolicy, Scheduler, ShardConfig};
use grommet_offload::{OffloadPools, OffloadStats, RayonOffload};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Duration;

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set"))
}

fn setting(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
}

fn fraction(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
}

fn flag(name: &str, default: bool) -> bool {
    match std::env::var(name).ok().as_deref() {
        Some("1" | "true" | "yes") => true,
        Some("0" | "false" | "no") => false,
        _ => default,
    }
}

/// Print per-shard rates every `every`, and what the split looks like from the
/// counters.
///
/// The runtime publishes gauges once per tick from each shard's own thread; this
/// only reads them. The calibration line is advice for the next restart: the
/// placement it describes was fixed before the first request arrived and does
/// not move while the service runs.
fn start_exporter<const CLASSES: usize>(
    shards: Vec<Arc<ShardStats<CLASSES>>>,
    offload: Vec<Arc<OffloadStats>>,
    offload_workers: usize,
    every: Duration,
) {
    std::thread::spawn(move || {
        let mut previous = vec![(0u64, 0u64); shards.len()];
        let mut compute = (Duration::ZERO, Duration::ZERO);
        loop {
            std::thread::sleep(every);
            let seconds = every.as_secs_f64();
            let mut total = 0.0;
            let mut reactor_busy = Duration::ZERO;
            for (index, shard) in shards.iter().enumerate() {
                let completed = shard.completed.load(Relaxed);
                let busy = shard.busy_nanos.load(Relaxed);
                let rate = completed.saturating_sub(previous[index].0) as f64 / seconds;
                let elapsed = busy.saturating_sub(previous[index].1);
                reactor_busy += Duration::from_nanos(elapsed);
                previous[index] = (completed, busy);
                total += rate;
                println!(
                    "shard {index}: {rate:.0} req/s, scheduler {:.1}%, \
                     pending {}, resident {}, in-doubt {}, coalesced {}, expired {}",
                    elapsed as f64 / (seconds * 1e9) * 100.0,
                    shard.pending.load(Relaxed),
                    shard.resident.load(Relaxed),
                    shard.in_doubt.load(Relaxed),
                    shard.coalesced.load(Relaxed),
                    shard.expired.load(Relaxed),
                );
            }

            let busy: Duration = offload.iter().map(|pool| pool.busy()).sum();
            let waited: Duration = offload.iter().map(|pool| pool.permit_wait()).sum();
            let runs: u64 = offload.iter().map(|pool| pool.runs.load(Relaxed)).sum();
            let window = Observation {
                wall: every,
                shards: shards.len(),
                offload_workers,
                offload_busy: busy.saturating_sub(compute.0),
                permit_wait: waited.saturating_sub(compute.1),
                reactor_busy,
            };
            compute = (busy, waited);

            println!("total {total:.0} req/s, compute {runs} runs");
            println!("{}", window.advise());
        }
    });
}

fn main() {
    let postgres_url = required_env("GROMMET_POSTGRES_URL");
    let redis_url = required_env("GROMMET_REDIS_URL");
    let pool_size = setting("GROMMET_POOL_SIZE", 64);

    // What the hardware cannot tell us: how much of this service's CPU demand is
    // computation rather than reactor work. It is static, and deliberately so,
    // the exporter below prints what the counters say it should have been, for
    // the next restart to use.
    let workload = Workload {
        compute_fraction: fraction("GROMMET_COMPUTE_FRACTION", 0.25),
        reserve_cores: setting("GROMMET_RESERVE_CORES", 1),
        isolate_smt: flag("GROMMET_ISOLATE_SMT", true),
        prefer_performance_cores: flag("GROMMET_PREFER_P_CORES", true),
    };
    let plan = Arc::new(detect(&workload).expect("read this machine"));
    for note in &plan.notes {
        println!("topology: {note}");
    }
    assert!(!plan.shards.is_empty(), "no cores available to run a reactor on");

    let pools = OffloadPools::build(plan.clone(), setting("GROMMET_OFFLOAD_DEPTH", 4));
    assert!(
        !pools.is_empty(),
        "this service offloads its signing work, so it needs compute cores: \
         raise GROMMET_COMPUTE_FRACTION above 0"
    );
    let offload_stats = pools.stats();
    let offload_workers = pools.workers();

    // Migrate before any shard accepts work. This pool is discarded; every shard
    // below builds its own on its own core.
    let bootstrap = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("bootstrap runtime");
    let schema = PgStore::from_url(&postgres_url, 1).expect("PostgreSQL pool config");
    bootstrap.block_on(schema.ensure_schema()).expect("account schema");
    drop(bootstrap);

    let mut shard_config =
        ShardConfig::new([setting("GROMMET_MAX_IO", 2048), setting("GROMMET_MAX_COMPUTE", 64)]);
    // These two compose: a shard queues up to GROMMET_MAILBOX items before the
    // scheduler sees them, plus GROMMET_MAX_PENDING after, so the worst-case
    // wait is their sum. The builder refuses a mailbox deeper than
    // max_pending, since past that point most of the queue would sit where
    // neither the pending gauge nor the admission gate can account for it.
    shard_config.scheduler.max_pending = setting("GROMMET_MAX_PENDING", 8192);
    shard_config.scheduler.queue_reserve = shard_config.scheduler.max_pending;
    shard_config.scheduler.evict_after = Duration::from_secs(60);
    // A client retrying while its first attempt is still outstanding should not
    // cost a second round trip to PostgreSQL.
    shard_config.coalesce_duplicates = true;

    let shards = setting("GROMMET_SHARDS", plan.shards.len());
    let runtime = Scheduler::<AccountProcessor<PgStore, RedisCache, RayonOffload>>::builder(
        shards,
        [
            shard_config.scheduler.max_inflight[IO as usize],
            shard_config.scheduler.max_inflight[COMPUTE as usize],
        ],
    )
    .shard_config(shard_config)
    .plan(plan.clone())
    .pin(PinPolicy::BestEffort)
    .mailbox(setting("GROMMET_MAILBOX", 1024))
    .spawn(move |shard| {
        // Each shard builds its own pools, on its own core, so no connection
        // is ever shared across cores, and takes the compute pool local to
        // its own memory node, so a closure it ships does not cross the
        // interconnect on the way to a worker.
        let store = PgStore::from_url(&postgres_url, pool_size).expect("PostgreSQL pool");
        let cache = RedisCache::from_url(&redis_url, pool_size).expect("Redis pool");
        let offload = pools.for_node(shard.node()).expect("a compute pool for this shard");
        AccountProcessor::new(Rc::new(store), Rc::new(cache), offload)
    })
    .expect("start shards");

    let report = runtime.topology();
    println!(
        "topology: {} shards on {} distinct cores, {} pinned, {} on their own memory node{}",
        report.shards,
        report.distinct_cores,
        report.pinned,
        report.memory_bound,
        if report.oversubscribed() { " (oversubscribed)" } else { "" },
    );
    println!("topology: {offload_workers} compute workers across {} pools", plan.offload.len(),);

    start_exporter(
        runtime.stats().to_vec(),
        offload_stats,
        offload_workers,
        Duration::from_secs(2),
    );

    let frontdoor = Frontdoor::new(runtime.router().clone());
    let servers = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("front-door runtime");
    servers.block_on(async {
        tokio::select! {
            result = accounts::frontdoor::serve_http(frontdoor.clone(), 9000) => {
                result.expect("HTTP server")
            }
            result = accounts::frontdoor::serve_grpc(frontdoor, 9001) => {
                result.expect("gRPC server")
            }
        }
    });
}
