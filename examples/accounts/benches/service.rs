//! Baselines that keep unlike costs separate, so a regression has an owner.
//!
//! No benchmark opens a PostgreSQL or Redis connection. The reactor group runs
//! the production router, scheduler and shard runtime, substituting only those
//! two ports, so network and server latency cannot hide the CPU, scheduling,
//! mailbox or fairness costs under investigation.

use accounts::domain::{Account, Op, Reply, RequestId, RevalueParams, heavy_revalue};
use accounts::processor::{AccountCall, Request};
use accounts::sim::SimWorld;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use futures::stream::{FuturesUnordered, StreamExt};
use grommet::metrics::ShardStats;
use grommet::{ManualClock, Router, ShardConfig, mix, shard};
use grommet_core::{Admit, Completion, Config, Disposition, Scheduler};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

type AccountRouter = Router<AccountCall, ManualClock>;

/// The pure, single-threaded costs: routing, id creation, the domain
/// transition, and the compute kernel itself.
fn domain(c: &mut Criterion) {
    let mut group = c.benchmark_group("domain");

    group.bench_function("route/16_shards", |b| {
        let mut key = 0u64;
        b.iter(|| {
            key = key.wrapping_add(1);
            black_box(mix(black_box(key)) % 16)
        });
    });

    group.bench_function("request_id/new", |b| b.iter(|| black_box(RequestId::new())));

    group.bench_function("apply_mutation", |b| {
        let account = Account { balance: 1_000, version: 7 };
        b.iter(|| {
            black_box(accounts::domain::apply_mutation(
                black_box(&account),
                accounts::domain::Mutation::Delta(1),
            ))
        });
    });

    let scenarios = 1;
    group.throughput(Throughput::Elements(u64::from(scenarios) * 200_000)).bench_function(
        "revalue/one_scenario",
        |b| {
            let account = Account { balance: 3, version: 1 };
            let params = RevalueParams { scenarios };
            b.iter(|| black_box(heavy_revalue(black_box(&account), black_box(&params))));
        },
    );
    group.finish();
}

/// The scheduler alone, with no runtime under it: admit, dispatch, complete.
fn scheduler(c: &mut Criterion) {
    let mut group = c.benchmark_group("scheduler");
    group.throughput(Throughput::Elements(1));

    for (name, keys) in [("hot_key", 1u64), ("cold_keys", 100_000)] {
        group.bench_function(name, |b| {
            let mut book: Scheduler<u64, u64, Account> = Scheduler::new(Config::new([2048, 64]));
            let mut key = 0u64;
            b.iter(|| {
                key = key.wrapping_add(1) % keys.max(1);
                book.admit(Admit { key, class: 0, expires_at: None, payload: key });
                let dispatch = book.next(0, Duration::ZERO).expect("just admitted");
                book.complete(
                    Completion {
                        key: dispatch.key,
                        class: dispatch.class,
                        state: Disposition::Keep(Account::default()),
                    },
                    Duration::ZERO,
                );
            });
        });
    }
    group.finish();
}

/// A live single-shard reactor with deterministic ports, driven through the
/// real router. Everything measured here is the runtime's own cost.
struct Harness {
    router: Arc<AccountRouter>,
    runtime: tokio::runtime::Runtime,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl Harness {
    fn new(admit_batch: usize) -> Self {
        let clock = ManualClock::new();
        let (tx, rx) = grommet::channel(4096);
        let router = Arc::new(AccountRouter::new(vec![tx], clock.clone()));
        let mut cfg = ShardConfig::new([2048, 64]);
        cfg.admit_batch = admit_batch;
        cfg.scheduler.queue_reserve = cfg.scheduler.max_pending;

        let worker = std::thread::Builder::new()
            .name("bench-shard".to_owned())
            .spawn(move || {
                let runtime =
                    tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
                let world = SimWorld::healthy();
                runtime.block_on(shard::run(
                    rx,
                    world.processor(),
                    clock,
                    Arc::new(ShardStats::default()),
                    cfg,
                ));
            })
            .unwrap();

        Self {
            router,
            runtime: tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap(),
            worker: Some(worker),
        }
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        // Dropping the last router handle closes the mailbox and drains.
        self.router = Arc::new(AccountRouter::new(vec![grommet::channel(1).0], ManualClock::new()));
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn reactor(c: &mut Criterion) {
    const CALLS: u64 = 64;
    let mut group = c.benchmark_group("reactor");
    group.sample_size(20);

    // How much a bigger admission batch amortizes cross-thread wakeups.
    group.throughput(Throughput::Elements(CALLS));
    for admit_batch in [1, 8, 64] {
        let harness = Harness::new(admit_batch);
        let router = harness.router.clone();
        group.bench_with_input(
            BenchmarkId::new("balance/64_keys", admit_batch),
            &admit_batch,
            |b, _| {
                b.to_async(&harness.runtime).iter(|| {
                    let router = router.clone();
                    async move {
                        let calls = (0..CALLS).map(|key| {
                            router.call(Request {
                                req_id: RequestId::from(u128::from(key) + 1),
                                account: key,
                                op: Op::Balance,
                            })
                        });
                        let replies: Vec<_> =
                            calls.collect::<FuturesUnordered<_>>().collect().await;
                        black_box(replies)
                    }
                });
            },
        );
    }

    // One key, so every request serializes behind the last: this is the
    // per-request reactor cost with no concurrency to hide it.
    group.throughput(Throughput::Elements(1));
    let harness = Harness::new(64);
    let router = harness.router.clone();
    group.bench_function("balance/one_hot_key", |b| {
        b.to_async(&harness.runtime).iter(|| {
            let router = router.clone();
            async move {
                let reply = router
                    .call(Request { req_id: RequestId::from(1u128), account: 7, op: Op::Balance })
                    .await;
                debug_assert!(matches!(reply, Ok(Reply::Ok(_))));
                black_box(reply)
            }
        });
    });
    group.finish();
}

criterion_group!(benches, domain, scheduler, reactor);
criterion_main!(benches);
