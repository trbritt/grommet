# Fast, deterministic gate. Fault-specific tests never pretend to run here.
#
# Placement needs libhwloc, and the default builds it from source: the first run
# after a clean checkout downloads the hwloc release and compiles it, which takes
# a couple of minutes and needs network access. See `test-system-hwloc` for the
# way out.
default: test

# The whole fast gate. CI runs the four pieces as separate jobs so a formatting
# slip does not hide a test failure behind it; this is the local equivalent.
test: fmt-check lint unit doc-test

fmt-check:
    cargo fmt --all -- --check

lint:
    cargo clippy --workspace --all-targets -- -D warnings

# `--all-targets` covers lib, bins, tests and benches — and deliberately excludes
# doctests, which is why `doc-test` exists separately rather than as a flag here.
unit:
    cargo test --workspace --all-targets

# The examples in the public documentation are compiled and run. They are the
# first thing a reader copies, so they are gated like any other test.
doc-test:
    cargo test --workspace --doc

# Rustdoc as docs.rs will run it: warnings are errors, so a broken intra-doc
# link fails here rather than silently shipping a dead link on docs.rs.
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

# Documentation exactly as docs.rs will build it.
#
# `doc` above builds with default features, which is not what gets published:
# docs.rs reads `[package.metadata.docs.rs]`, and that turns defaults off to
# avoid the vendored libhwloc download. A feature the crate cannot compile
# without is therefore invisible to every other gate here, and shows up as a
# failed build on the release. This reads the same metadata rather than
# restating it, so the check cannot drift from what docs.rs actually does.
doc-docsrs:
    cargo docs-rs -p grommet
    cargo docs-rs -p grommet-core
    cargo docs-rs -p grommet-topology
    cargo docs-rs -p grommet-offload
    cargo docs-rs -p grommet-testkit

# Every driver the shard can be hosted on.
#
# The loop is identical across drivers and only the host shim differs, so what
# this catches is a shim that stopped compiling or stopped agreeing with the
# loop — which nothing else would notice, since the default build only ever
# exercises one of them.
test-drivers:
    cargo test -p grommet --lib --no-default-features --features driver-tokio

# The pure-Rust configuration: hwloc opted out entirely.
#
# This is what a musl image, an air-gapped build, or a distribution package that
# refuses a build-time download actually compiles. It has no C toolchain, no
# network fetch and no libhwloc, so nothing else in this file would notice it
# breaking — and what breaks it is usually a `#[cfg]` that drifted rather than
# anything anyone would think to test.
#
# The example is deliberately absent: it demonstrates hardware placement, so
# building it without hwloc would be demonstrating the wrong thing.
# `driver-tokio` is named explicitly: the opt-out is about hwloc, and a shard
# still needs a host to wait on.
test-no-topology:
    cargo test -p grommet -p grommet-core -p grommet-topology -p grommet-offload \
        -p grommet-testkit --no-default-features --all-targets \
        --features grommet/driver-tokio,grommet-offload/driver-tokio,grommet-testkit/driver-tokio
    cargo test -p grommet -p grommet-core -p grommet-topology -p grommet-offload \
        -p grommet-testkit --no-default-features --doc \
        --features grommet/driver-tokio,grommet-offload/driver-tokio,grommet-testkit/driver-tokio

# The same gate against a system libhwloc instead of a vendored one.
#
# This is the path for distribution packaging, reproducible builds, and anywhere
# the build-time download is unacceptable. It needs libhwloc 2.8 or newer
# installed and findable by pkg-config. CI runs it so the opt-out cannot rot.
# `topology` without `vendored` is what names the system library: the first
# turns hwloc on, the second is what would have built it from source.
test-system-hwloc:
    cargo test --workspace --all-targets --no-default-features \
        --features accounts/gen,accounts/topology,accounts/driver-tokio
    cargo test --workspace --doc --no-default-features \
        --features accounts/gen,accounts/topology,accounts/driver-tokio

# A publish cannot be taken back, so every manifest is checked for the metadata
# and the packageability crates.io demands before a tag exists, not after.
package:
    cargo package --workspace --no-verify

# Optimized simulation with assertions and fault hooks left in. `cfg` travels
# through RUSTFLAGS; Cargo features are deliberately not used for safety-relevant
# modes, because features unify across a dependency graph and a simulation build
# must never be reachable by accident.
sim:
    RUSTFLAGS="--cfg sim --cfg fault_injection" cargo test --workspace --all-targets \
        --features accounts/sim --profile sim

# Exhaustive interleaving checks on the reactor's wake protocol — the ready
# bits and the register-early discipline the parked loop depends on. Release,
# because loom explores every ordering and a debug build multiplies that.
loom:
    RUSTFLAGS="--cfg loom" cargo test -p grommet --lib --release loom_

# The scheduler slab is the only unsafe code in the workspace. Miri runs its
# randomized model test against reference queues, which is what makes the
# unchecked indexing there worth anything.
miri:
    cargo miri test -p grommet-core --lib

# MC/DC instrumentation is unstable and needs the pinned nightly. Adapters and
# the front door are a different evidence domain: PostgreSQL and Redis need
# real-service conformance tests, and HTTP/gRPC are proven by Turmoil.
coverage:
    RUSTC_WRAPPER=./scripts/rustc-condition-wrapper.sh \
        RUSTFLAGS="--cfg coverage --cfg sim --cfg fault_injection" \
        cargo llvm-cov --package grommet --package grommet-core \
        --package grommet-testkit --package accounts --all-targets \
        --features accounts/sim --mcdc \
        --ignore-filename-regex '(examples/accounts/src/(prod|frontdoor|net|main)\.rs|crates/grommet-macros/)' \
        --fail-under-lines 95

# Mutation testing runs the whole workspace suite for every production mutant.
mutants:
    RUSTFLAGS="--cfg sim --cfg fault_injection" cargo mutants --workspace \
        --features accounts/sim

fuzz-list:
    RUSTFLAGS="--cfg sim --cfg fault_injection" cargo bolero list -p accounts --features sim

fuzz TARGET:
    RUSTFLAGS="--cfg sim --cfg fault_injection" cargo bolero test -p accounts \
        --features sim '{{TARGET}}' --sanitizer address -T 5min

reduce TARGET:
    RUSTFLAGS="--cfg sim --cfg fault_injection" cargo bolero reduce -p accounts \
        --features sim '{{TARGET}}'

deny:
    cargo deny check

# The simulation feature must not reach a shipping build.
check-release-hygiene:
    ! cargo tree -e features | rg -q 'feature "sim"'

# Domain, scheduler and full-reactor baselines. No benchmark opens a PostgreSQL
# or Redis connection.
bench:
    cargo bench -p accounts --bench service

bench-one FILTER:
    cargo bench -p accounts --bench service -- '{{FILTER}}'

bench-save NAME:
    cargo bench -p accounts --bench service -- --save-baseline '{{NAME}}'

bench-compare NAME:
    cargo bench -p accounts --bench service -- --baseline '{{NAME}}'
