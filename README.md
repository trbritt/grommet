<p align="center">
  <img src="https://raw.githubusercontent.com/trbritt/grommet/main/assets/grommet-hero-panel.svg" alt="grommet" width="680">
</p>

<p align="center">
  Thread-per-core, key-affine work scheduling for Rust.
</p>

<p align="center">
  <a href="https://github.com/trbritt/grommet/actions/workflows/ci.yml"><img src="https://github.com/trbritt/grommet/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://crates.io/crates/grommet"><img src="https://img.shields.io/crates/v/grommet.svg" alt="crates.io"></a>
  <a href="https://docs.rs/grommet"><img src="https://docs.rs/grommet/badge.svg" alt="docs.rs"></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg" alt="license"></a>
</p>

## Why "grommet"

A grommet is the little metal ring you press into a hole so the rope can pass
through without sawing the canvas in half, and so the hole doesn't tear itself
wider every time you pull.

That is the whole job here. One reinforced opening per core; work goes through it
exactly once, from whoever submitted it to the shard that owns its key. On the
far side nothing needs to be `Send`, nothing needs a lock, and a dispatched item
holds the only copy of its state. Then you pull harder. The mailbox fills,
backpressure walks back to the submitter, the ring holds its shape and the sheet
stays intact.

Keeps things tight. Keeps things locked. Doesn't chafe.

## How it works

Work carries an affine key. Every item for one key is handled by one shard, in
submission order, one at a time. The state behind that key needs no locking and
no atomics, because while an item is being processed it holds the only copy.
Shards are pinned to cores, each running a single-threaded runtime.

Within a shard, keys are dispatched round-robin from per-class ready rings. That
puts a *strict* bound on starvation: a key at position `k` runs within `k`
dispatches, no matter how much work a busier key has queued. Each class has its
own in-flight budget, so saturating one class with CPU-bound work or a slow
dependency leaves the others free to dispatch.

```rust
impl Work for Job {
    type Key = u64;
    type Id = u128;
    fn key(&self) -> u64 { self.account }
    fn class(&self) -> ClassId { IO }
    fn request_id(&self) -> Option<u128> { Some(self.attempt) }
    fn time_to_live(&self) -> Option<Duration> { Some(Duration::from_millis(50)) }
}

impl Processor for Ledger {
    type Work = Call<Job, i64>;
    type State = i64;
    type Error = LedgerError;

    async fn process(&self, key: u64, state: Option<i64>, call: Call<Job, i64>)
        -> Result<Disposition<i64>, LedgerError>
    { /* you hold the only copy of this key's state */ }
}

// SystemClock and the two-class IO + CPU split are the defaults.
let runtime = Scheduler::<Ledger>::builder(shards, [2048, 64])
    .pin(PinPolicy::Require)
    .coalesce_duplicates(true)
    .spawn(|shard| Ledger::new(shard))?;

let balance = runtime.router().call(job).await?;
```

## Crates

| Crate | What it is |
|---|---|
| `grommet-core` | The scheduler as a pure data structure. No async, no clock, no IO. |
| `grommet` | The runtime: clock, traits, router, shard reactor, metrics. |
| `grommet-topology` | Reads the machine (NUMA, SMT, P/E cores, cgroup quota) and plans where shards and offload workers go. |
| `grommet-offload` | Pinned, bounded Rayon pools for CPU-bound work, one per memory node. |
| `grommet-testkit` | Fault injection and conformance checks for your processor. |
| `grommet-macros` | The assertion macros the scheduler's invariant checks are written in. |
| `examples/accounts` | A worked example, and the proof the abstractions survive use. |

`grommet-core` depends on `ahash` and `crossbeam-utils`. `grommet` adds `tokio`,
`futures`, `parking_lot` and `hdrhistogram`. Reading the machine lives in
`grommet-topology`, which wraps `hwlocality`. Rayon is a separate crate you opt
into.

## Features

| Feature | Default | What it does |
|---|---|---|
| `topology` | on | Read the machine through libhwloc and bind threads to it. |
| `vendored` | on | Build libhwloc from source rather than linking a system one. Implies `topology`. |

The default builds libhwloc from source, which needs a C toolchain, autotools
and network access at build time. That is the default because hardware-aware
placement is what this runtime is *for*, and because linking a system libhwloc
is the less reliable path: `pkg-config` finds nothing on a stock macOS install,
and distribution packages lag the hwloc 2.8 API floor.

Three configurations, and what each is for:

```toml
# Default. Plans and binds against the real machine.
grommet = "0.1"

# System libhwloc instead of a vendored build. For distribution packaging and
# reproducible builds, where a build-time download is unacceptable.
grommet = { version = "0.1", default-features = false, features = ["topology"] }

# No hwloc at all: pure Rust, builds in seconds, works on musl and air-gapped.
grommet = { version = "0.1", default-features = false }
```

The last one still plans a layout and still starts a runtime: every type stays
where it was, but it plans from `available_parallelism` alone, knows nothing
about SMT siblings, memory nodes or core kinds, and binds no threads. It says
all of that in `Plan::notes`, and `Scheduler::topology()` reports nothing pinned.

`PinPolicy::Require` does not exist in that build. Demanding placement from a
configuration that cannot perform it is a contradiction rather than a strict
setting, so it is a compile error at the call site rather than a process that
starts unpinned and quietly measures the OS scheduler instead of this one.

## What the runtime guarantees

- **One owner per key.** A dispatched item receives the key's state by value.
  No other item for that key can be in flight, so there is nothing to lock.
- **Bounded starvation.** Arrivals and completions both go to the back of a
  ring; dispatch pops the front. `grommet-testkit` measures the resulting wait
  and fails if it ever exceeds one rotation.
- **Independent class budgets.** A flood of compute work fills the compute
  budget and ring only, leaving IO free to keep dispatching.
- **End-to-end backpressure, first come first served.** At the pending cap a
  shard stops admitting, its bounded mailbox fills, and `Router::submit`
  suspends the caller. Suspended submitters are let back in in the order they
  arrived, so a steady stream of new arrivals cannot starve one that has been
  waiting. `try_submit` sheds instead, handing the work back so you can answer
  your client.
- **Deadlines cost nothing to miss.** Work past its deadline is discarded at
  dispatch, before it spends a turn.
- **Contained panics.** A panicking `process` future cannot unwind the reactor.
  It is caught, counted, and the key's state is discarded so it reloads.
- **Safe eviction.** A key is quiesced while `on_evict` flushes its state, so a
  write-back cannot race a reload of the same key.
- **Nothing is dropped on the way out.** Closing the mailbox drains what is
  queued, and then every key still holding state is handed to `on_evict` before
  the shard exits. Resident state is a write-back cache; a shutdown that skipped
  it would lose writes the processor was told it could keep.

## What grommet is, and is not

Grommet is a scheduler, not a runtime. It reads the machine, decides how many
shards it will carry and which cores they belong on, and gives each shard
key-affine dispatch with a strict starvation bound and its own compute offload.
What it does not do is drive futures: polling IO, waking a socket, owning the
thread is the host runtime's job, and grommet runs on top of one.

That division is the point. Tokio, monoio and glommio already solve driving
futures well; none of them knows that two items sharing an affine key must
never run at once, or that a long computation belongs on a different core from
the reactor that dispatched it. Grommet supplies that and borrows the rest, so
adopting it does not mean leaving the ecosystem your database, cache and HTTP
clients are written against.

Concretely: a Cargo feature selects the host. `driver-tokio` builds one
current-thread tokio runtime per shard thread, pinned where the topology plan
said, and runs a grommet scheduler on it. Your processor keeps awaiting
`tokio-postgres` exactly as before.

## What it deliberately does not do

- **No work stealing.** It would break the one-owner guarantee, which is the
  whole point. Rebalancing a skewed workload means migrating a key, which the
  router's slot table is built for.
- **No `Send` futures.** Work is `Send` because it crosses once from submitter
  to shard. Nothing after that is. `Rc` and `Cell` are correct here, and code
  written against `Send` futures and work stealing will not fit. If you want
  that, use an ordinary multi-threaded executor.
- **No IO of its own.** No sockets, no timers you can await, no file API. The
  host runtime provides those and grommet schedules around them. An io_uring
  fast path is monoio's business, not something to reimplement here.
- **No durable deduplication.** In-flight coalescing suppresses a retry while
  its original is still outstanding. Once the original completes its id leaves
  the index, because answering a later retry correctly needs the original's
  recorded outcome, which only your store has. An in-memory dedup table would
  be lost on exactly the restart it exists to survive.
- **No replies unless you ask.** Submission reports whether work was accepted,
  not what it produced. Wrap work in a `Call` for request/response; a reply
  channel costs an allocation and two atomics, and ingestion pipelines have no
  caller to answer.

## Correctness contract for a processor

- Returning `Err` always discards the key's resident state; classify it with
  `Fallout::InDoubt` when the durable outcome is unknown and `Untouched` when
  it definitely did not apply. A failure that leaves your state intact is not
  an error; return `Ok(Disposition::Keep(state))`.
- Every mutation should carry a caller-stable request id, and a retry must
  reuse it.
- `Work::key`, `class` and `request_id` are read once, at submission. The
  scheduler never asks again, so an inconsistent implementation cannot corrupt
  its rings.

## Observability

Each shard publishes its counters and gauges once per tick, from its own thread,
into a `ShardStats` that an exporter reads. Nothing on the dispatch path is
atomic: the shard writes plain cells and copies them across once a tick.

Queue wait and processing time are HDR histograms rather than sums, because a
mean hides exactly the distribution thread-per-core exists to control. A runtime
that sells a starvation bound should be able to show its own tail. Recording is
a bucket increment and never allocates.

Percentiles do not average, so what is published is the distribution rather than
precomputed quantiles. Merge the shards, then ask once:

```rust
let mut all = grommet::metrics::histogram();
for shard in runtime.stats() {
    shard.merge_queue_wait_into(&mut all);
}
println!("p99 {}ns", all.value_at_quantile(0.99));
```

`started`, `completed` and `expired` are also split by class, so one class
starving behind another is visible rather than averaged away. Every other number
describes work that finished, which is where a wedged future would hide, so
`inflight_age_max_nanos` reports how long the oldest running dispatch has been
running.

## Gates

```justfile
just test       # the fast gate: formatting, strict Clippy, tests, doctests
just doc        # rustdoc with warnings denied, as docs.rs will build it
just package    # every crate is packageable for crates.io
just sim        # optimized deterministic simulation with fault injection
just miri       # undefined-behaviour check over the unsafe modules
just loom       # exhaustive interleavings of the ring, mailbox and wake protocol
just coverage   # MC/DC instrumentation and a decision-layer line gate
just mutants    # assertion-strength check under the simulation configuration
just fuzz-list  # exact Bolero target names (run before fuzzing)
just fuzz TARGET        # coverage-guided model/fault fuzzing with ASan
just reduce TARGET      # replay and minimize one saved failure
just deny       # advisories, bans, and licenses
just bench      # domain, scheduler and full-reactor baselines
```

### Unsafe code

Unsafe code lives in exactly one crate, `grommet-core`, so that there is one
place to audit rather than a policy with exceptions. Every other crate in the
workspace is `#![deny(unsafe_code)]`, and inside `grommet-core` it is confined
to modules that opt back in one at a time.

The **queue slab** indexes without bounds checks. Every index it dereferences is
one it allocated itself, never caller data.

The **waker slot** guards an `Option<Waker>` with a two-bit lock, so a
notification from another core costs an atomic rather than a lock, and a
notifier never blocks behind the shard it is notifying. It is
`futures::task::AtomicWaker` in algorithm, written out here because that one is
built on `core::sync::atomic`, which loom cannot instrument.

The **ring** is the bounded MPSC a shard's mailbox drains. One consumer means
its read position is a plain field rather than a contended cursor, so draining
costs no read-modify-write at all. A slot a producer has claimed but not yet
published reads as nothing yet, rather than as something to spin on: a reactor
has other work and comes back next turn.

Each states its invariant at the top of its file, `debug_assert!`s it at every
use, and is checked by a model test that runs under Miri in CI. The two
synchronization primitives are model-checked by loom as well, which is the point
of owning them: a slot read that escaped its stamp fails the model as a
causality violation rather than passing as undefined behaviour that happens not
to bite.

### Evidence domains

PostgreSQL and Redis adapters are outside mutation scoring and coverage until a
conformance suite runs the same contract against real services. A green
simulator cannot prove SQL, wire-protocol or pool-configuration truth. HTTP and
gRPC are proven by Turmoil instead, across a simulated network that can be
partitioned mid-request.

`grommet-topology` is outside both for a different reason. Its branches are
selected by the machine underneath: hybrid P/E cores, a second socket, a cgroup
bandwidth limit. No single CI runner has all of those at once, so a surviving
mutant there usually records the absence of that hardware. It is tested against
synthetic hwloc topologies instead, which describe a two-socket server or a
throttled container on whatever machine is to hand.

## The example is part of the design

`examples/accounts` is a real account service: durable state behind an
idempotency key, a non-authoritative cache, CPU-bound work that must not stall
the reactor, and a commit whose acknowledgement can be lost. It exists because
an abstract scheduler with no concrete user is how these designs acquire traits
nobody can implement.

It is also where the interesting proofs live:

- `every_single_failure_position_reconciles_under_replay` runs the whole stack
  once per injectable operation, 23 of them, and insists each one converges
  on the same durable state under replay.
- `a_partition_and_an_in_doubt_retry_cross_the_whole_stack` loses a commit's
  acknowledgement, partitions the client mid-retry, repairs the network, and
  checks the replay is recognised as a duplicate rather than applied twice.
- `an_old_duplicate_cannot_regress_newer_state` covers the subtle one: a
  duplicate carries the balance recorded for *its* id, which may be older than
  what is resident. Letting it overwrite would livelock a genuinely missing
  request behind version conflicts forever.

## Performance

Machine-local evidence from 2026-08-13, not portable SLOs. Use Criterion's named
baselines on the same quiet machine for optimization decisions.

| Workload | Median |
|---|---:|
| Routing hash | 1.15 ns |
| ULID creation | 41.2 ns |
| Scheduler admit + dispatch + complete, hot key | 14.7 ns |
| Scheduler admit + dispatch + complete, 100k keys | 21.9 ns |
| Pure revalue kernel, one 200k-iteration scenario | 429.6 µs |
| One shard, 64 concurrent reads, `admit_batch = 1` | 38.2 µs |
| One shard, 64 concurrent reads, `admit_batch = 64` | 33.3 µs |
| One shard, sequential reads on one hot key | 5.82 µs |

Batching admission amortizes cross-thread wakeups, which dominate at high rates:
64 is where the curve flattens, and it is the default. The scheduler triple was
27.4 ns before per-key queues became intrusive lists over a shared slab; that is
the same machine and the same operation, but it was not a controlled A/B, so
treat it as suggestive.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
