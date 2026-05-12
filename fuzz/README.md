<!-- SPDX-License-Identifier: Apache-2.0 OR MIT -->
# Tantivy CHT fuzz harness

Wave Z-7 #6 #4 — long-running fuzz coverage for the 3-tier Compressed
Hot Tier (CHT) in the tantivy fork.

> This crate is intentionally **not** a member of the parent tantivy
> workspace: cargo-fuzz requires nightly-only compiler flags that would
> taint the stable workspace build. Always invoke it via
> `cd fuzz && cargo +nightly fuzz …`.

## Why this exists

Three honest limitations carry forward from the Wave Z-7 #5 e2e
closure that this harness is built to attack empirically:

1. **Multi-tenant ChtKey collision probability** — theoretically
   negligible (UUID-backed `SegmentId` + 32-bit `field_id` + 64-bit
   `FxHash(term)`) but never measured under realistic tenant churn.
2. **Concurrent insert / evict / dump_by_segments / load_from_path /
   v1→v2→v3 promotion invariants** — covered by unit tests at known
   cardinalities, never stress-tested over long fuzz runs.
3. **Multi-chunk (Z-6 series) wire format** — `MAGIC_V4` byte-identical
   round-trip is pinned by `dump_by_segments_byte_invariant_with_oracle`
   on a single fixture, never exercised on arbitrary post-eviction
   states.

The two targets in this crate exercise the operation grammar shared
across all three tiers (`Insert` / `Get` / `EvictBySegments` /
`DumpRoundTrip` / `Reset` / `Stats`) and assert per-op invariants
after **every** step.

## Targets

| Name | Tier | CUDA needed | Always built |
|------|------|-------------|--------------|
| `cht_v1_concurrent_churn` | v1 host CHT (`src/postings/roaring/cht.rs`) | no | yes |
| `cht_v3_multichunk_invariants` | v3 VRAM Bitcomp-compressed CHT (`src/postings/roaring/vram_cht_v3.rs`) | yes (`cuda-bitmap-kernel`) | only with `--features cuda-bitmap-kernel` |

The v2 (VRAM uncompressed) tier shares the same key + budget invariants
as v3; a dedicated v2 target is deferred until the v3 target surfaces
its first finding (the multi-chunk admission path is the higher-risk
surface).

### Invariants asserted

Both targets check after every op:

- **Counter consistency.** `stats.entries` matches the shadow
  cardinality (v1) / the live cache state (v3, accepted-loose since
  CUDA OOM can drop inserts asymmetrically).
- **Budget compliance.** `stats.current_bytes <= stats.budget_bytes`.
- **Monotonic counters.** `inserts`, `evictions`, `hits`, `misses` are
  non-decreasing across consecutive `stats()` observations (modulo
  `Reset`, which resets the baseline).
- **No double-counting.** Cumulative `evictions <= inserts`.

The v3 target additionally checks:

- **Multi-chunk admission cap.** `entry.chunk_count() in 1 ..=
  MAX_CHUNKS_PER_ENTRY (= 64)`; every chunk's `compressed_bytes > 0`.
- **Wire round-trip.** `dump_to_path` + `reset` + `load_from_path`
  preserves entry count AND total uncompressed bytes (the Z-6 #4
  MAGIC_V4 wire format invariant).
- **Per-segment dump symmetry.** `dump_by_segments(all_segments)`
  returns the same entry count as `dump_to_path` (the oracle
  invariant from `vram_cht_v3::tests::dump_by_segments_byte_invariant_with_oracle`).
- **Evict-by-segments accounting.** `evict_by_segments` returns no
  more than the shadow's segment-filtered cardinality.

## One-time setup

```bash
rustup toolchain install nightly
cargo install cargo-fuzz --locked
```

## Quickstart — local 30s smoke

```bash
cd /home/y1/git/tantivy-gpu-compress/fuzz
cargo +nightly fuzz run cht_v1_concurrent_churn -- -max_total_time=30
```

This builds the v1 target with libFuzzer instrumentation and runs for
30 seconds. The harness is wired correctly if no panic surfaces (no
`crash-<hash>` artefact under `artifacts/cht_v1_concurrent_churn/`).

For the v3 target (CUDA-gated):

```bash
cargo +nightly fuzz run cht_v3_multichunk_invariants \
    --features cuda-bitmap-kernel \
    -- -max_total_time=30
```

If no CUDA toolchain is present, `--features cuda-bitmap-kernel`
fails the build at link time — drop the flag and only the v1 target
will be available.

## 24h soak protocol

Per `~/.claude/projects/-home-y1-git-ferroSearchProjects/memory/feedback_fuzz_lightweight_recipe_2026_05_08.md`,
24h fuzz on a shared host should run under a CPU-quota slice so the
fuzz fleet doesn't peg the box for foreground work. Recommended
invocation:

```bash
systemd-run --user --slice=fuzz.slice --unit=cht-v1-fuzz-soak \
    --working-directory=/home/y1/git/tantivy-gpu-compress/fuzz \
    -- env FUZZ_MAX_TOTAL_TIME=86400 \
    cargo +nightly fuzz run cht_v1_concurrent_churn \
        -- -max_total_time=86400 -timeout=60
```

…with `~/.config/systemd/user/fuzz.slice` set to `CPUQuota=400%` +
`CPUWeight=10` (see the feedback file for the full template). For a
clean host with no co-tenant workload, drop the `systemd-run` wrapper
and run `cargo +nightly fuzz run` directly.

### Analysis methodology

Per `feedback_24h_soak_analysis_methodology.md`, a 24h soak that
reports "no crash" is not automatically a PASS — low-frequency
poison events can hide behind 60-second sampling intervals. Required
checks at soak end:

1. **Panic log scan.** `cargo fuzz` writes panics to
   `artifacts/<target>/crash-<hash>`. Zero artefacts under
   `artifacts/cht_v1_concurrent_churn/` AND `artifacts/cht_v3_multichunk_invariants/`
   is the bar.
2. **libFuzzer stat scan.** Tail the run log for `cov: <N>` plateau
   — if coverage stalls at the same value for the last 6h+, the
   fuzzer is stuck on a saturated frontier and the soak should
   probably be terminated.
3. **OOM / `cudaMalloc` failure scan.** v3 specifically — if the
   run log shows repeated `Cuda { code: 2 }` (= cudaErrorOutOfMemory)
   that's a budget mismatch or device-pool leak, NOT a CHT
   invariant violation, but worth reporting.

## Coverage report

```bash
cd /home/y1/git/tantivy-gpu-compress/fuzz
cargo +nightly fuzz coverage cht_v1_concurrent_churn
cargo +nightly fuzz coverage cht_v3_multichunk_invariants --features cuda-bitmap-kernel

# Render to text (requires `llvm-cov` from rustup's component install
# of `llvm-tools-preview`):
llvm-cov show \
    target/<host-triple>/coverage/<host-triple>/release/cht_v1_concurrent_churn \
    --instr-profile=coverage/cht_v1_concurrent_churn/coverage.profdata \
    -Xdemangler=rustfilt
```

## KVM CI runner integration

The Ryzen 9 9950X KVM runner (`project_kvm_ci_runner.md`) is the
canonical host for nightly soaks. The template below installs a
systemd-user unit that fires at 03:00 JST and runs for 7200 s
(`FUZZ_MAX_TOTAL_TIME=7200`, the lightweight-recipe budget) — **do
not** enable this unit as part of this PR; the orchestrator opts it
in only after the smoke run has been seen green on the runner.

`~/.config/systemd/user/cht-fuzz-soak.service` (template, NOT
shipped in-tree — operator-installed):

```ini
[Unit]
Description=Tantivy CHT fuzz soak (Wave Z-7 #6 #4)

[Service]
Type=oneshot
Slice=fuzz.slice
WorkingDirectory=/home/y1/git/tantivy-gpu-compress/fuzz
Environment=FUZZ_MAX_TOTAL_TIME=7200
TimeoutStartSec=2h10min
ExecStart=/bin/bash -lc 'cargo +nightly fuzz run cht_v1_concurrent_churn -- -max_total_time=${FUZZ_MAX_TOTAL_TIME} -timeout=60'
StandardOutput=append:%h/cht-fuzz-soak.log
StandardError=append:%h/cht-fuzz-soak.log
```

`~/.config/systemd/user/cht-fuzz-soak.timer`:

```ini
[Unit]
Description=Tantivy CHT fuzz soak — nightly at 03:00 JST

[Timer]
OnCalendar=*-*-* 03:00:00 Asia/Tokyo
Persistent=true

[Install]
WantedBy=timers.target
```

Enable after smoke is green:

```bash
systemctl --user daemon-reload
systemctl --user enable --now cht-fuzz-soak.timer
```

## Adding a new target

1. Drop a new file under `fuzz_targets/` starting with `#![no_main]`
   and using `libfuzzer_sys::fuzz_target!`.
2. Add a matching `[[bin]]` section to `Cargo.toml`.
3. If the target needs CUDA, add
   `required-features = ["cuda-bitmap-kernel"]` to the `[[bin]]`
   block and `#![cfg(feature = "cuda-bitmap-kernel")]` to the
   source file.
4. Seed `corpus/<target>/` with at least one well-formed sample
   (optional — libFuzzer generates its own corpus on first run).

## Corpus & artefact layout

```
fuzz/
├── Cargo.toml
├── README.md
├── fuzz_targets/
│   ├── cht_v1_concurrent_churn.rs
│   └── cht_v3_multichunk_invariants.rs
├── corpus/            # seed inputs per target (gitignored except `regression-*`)
└── artifacts/         # libFuzzer crash artefacts (gitignored)
```

Permanent regression seeds (crashes that a past fuzz run surfaced and
a commit has since fixed) live in `corpus/<target>/regression-<topic>-<YYYY-MM-DD>`
— these ARE committed so every future run re-checks the exact input.

## Honest limitations of this harness

- **Single-threaded.** Each fuzz iteration drives the CHT from one
  thread. Multi-threaded interleaving hazards (concurrent insert +
  evict_by_segments racing) are covered by the unit-test Rayon
  drivers in `src/postings/roaring/cht.rs` /
  `src/postings/roaring/vram_cht_v3.rs`. A future `cht_concurrent_threads`
  target could spawn N workers and serialise the assertion under a
  `parking_lot::Mutex`, but multi-threaded libFuzzer corpus
  minimisation is non-deterministic and we deferred it.
- **Operation grammar is finite-state.** The seven Op variants cover
  the trait surface but not every internal branch — e.g. the v3
  `promote_v2_to_v3` cross-tier device→device path is unreachable
  through this harness (no v2 cache is constructed). A follow-up
  target with all three tiers wired together is Wave Z-7 #6 #5
  scope.
- **8 segments fixed.** `NUM_PRESEEDED_SEGMENTS = 8` so
  `EvictBySegments` can use a `u8` bitmask. Multi-tenant collision
  probing at realistic tenant churn (1000+ segments) is a follow-up.
- **v3 needs a live GPU.** The v3 target's `with_budget` call fails
  cleanly if cudart isn't initialised, so the harness silently
  skips iterations on a GPU-less host — but in that mode it's
  exercising zero v3 code. The CUDA-gate compile check in CI is
  the floor.
- **Corpus is empty at start.** No seed inputs ship with this
  harness — libFuzzer mutates from scratch. First-night runs spend
  ~10 minutes building structural awareness; coverage stabilises
  after that.
