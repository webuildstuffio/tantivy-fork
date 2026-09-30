# AGENTS.md — tantivy-fork

Fork of [Tantivy](https://github.com/quickwit-oss/tantivy) 0.26.0 adding configurable BM25
scoring (per-workspace k1/b, BM25+ delta) and hybrid MaxScore/Block-Max WAND top-k pruning.
Consumed by **MegaMem** via `[patch.crates-io]` — the fork ships no binary of its own; its only
product is the patched `tantivy` crate. Part of the webuildstuffio org.

## Stack

- Rust, edition 2021, MSRV 1.86. Stable toolchain for build/test/clippy.
- **rustfmt runs on pinned nightly `nightly-2026-09-23`** — fmt output drifts with nightly
  releases, so a floating nightly re-breaks `check` on unrelated PRs. Re-pin deliberately
  (rationale in `.github/workflows/test.yml`).
- Tests run under [cargo-nextest](https://nexte.st); CI uses three feature matrices:
  all-features, `quickwit`, and `--no-default-features`.
- `Cargo.toml` is the **crates.io-normalized manifest** (auto-generated header): this repo tracks
  the published crate tarball, so sibling crates (`tantivy-columnar`, `common`, `stacker`,
  `query-grammar`, …) resolve from **crates.io, not local paths**. `Cargo.toml.orig` is vestigial
  (it references workspace members that don't exist here) — edit `Cargo.toml`, never "fix" `.orig`.

## Fork delta (keep it minimal)

Everything except the two files below is vendored upstream 0.26.0. A minimal diff vs upstream is
the fork's strategy for absorbing future upstream releases — keep changes out of other files, and
mark touched lines with the fork's ticket tags (`INF-135:`, `SCR-310:`, `SPD-107:`), as existing
code does.

1. `src/query/bm25.rs` — `Bm25Params { k1, b, delta }`; thread-local
   `set_thread_bm25_params()` / `reset_thread_bm25_params()` (INF-135); BM25+ delta lower bound
   in `tf_factor` (SCR-310, Lü & Callan 2011).
2. `src/query/boolean_query/block_wand.rs` — hybrid top-k pruning (SPD-107): term-centric
   **MaxScore** when there are ≥ `MAXSCORE_MIN_TERMS` (4) scorers, Block-Max WAND below that.

Upstream architecture overview: `ARCHITECTURE.md` (unmodified upstream doc).

## Commands

- `make test` — `cargo test --tests --lib` (no examples; they need fetched fixtures).
- `make fmt` — nightly rustfmt. CI gates on `cargo +nightly-2026-09-23 fmt --all -- --check`.
  Style: width 120, module-granularity imports, std-external-crate grouping (`rustfmt.toml`).
- `cargo +stable clippy --locked --tests` — CI gate.
- Full CI = `cargo nextest run --locked` × feature matrix + doctests + bench compile.
- Bench fixtures (`benches/hdfs.json`, `gh.json`, `wiki.json`, `alice.txt`) are **not committed**;
  CI fetches them from upstream — bench compilation fails locally without them.
- Failpoint tests are a separate binary (`tests/failpoints/`, requires `--features failpoints`)
  because the `fail` crate is incompatible with multithreading.

## Non-negotiables

1. **Default-parameter scoring must stay identical to upstream** (k1=1.2, b=0.75, delta=0.0).
   Defaults are the zero-diff baseline MegaMem relies on; an accidental default change silently
   reranks every consumer's search results.
2. **k1/b/delta are query-time only.** The index-serializer path (`for_one_term_without_explain`
   → `new_without_explain`) must keep using default params.
3. **The fork's public API is load-bearing** for MegaMem: `Bm25Params`,
   `set_thread_bm25_params`, `reset_thread_bm25_params`, `Bm25Weight::for_terms_with_params`,
   `Bm25Weight::for_one_term_with_params`. Removing/renaming any of these breaks the consumer.
   Crate name/version must stay `tantivy 0.26.0` or `[patch.crates-io]` stops resolving.
4. **Top-k equivalence**: MaxScore and Block-Max WAND must return the same docs and scores as the
   brute-force union scanner. Enforced by proptest equivalence tests in `block_wand.rs` — a
   failure there is a real bug, never a flake to retry.

## Known traps

- Thread-local BM25 params leak into later searches on the same thread if not reset — callers
  must `set_thread_bm25_params(...)` → search → `reset_thread_bm25_params()` (invariant
  documented at `src/query/bm25.rs:9`).
- `block_wand.rs` scorer invariants are load-bearing: the WAND path needs scorers sorted by
  `doc()`, the MaxScore path sorted **ascending by `max_score`**; seeking backwards panics
  (`maxscore_loop` guards `curr > pivot_doc`); the `swap_remove` + `restore_ordering` dance keeps
  the sort valid. Pruning thresholds are strict `>` comparisons — tie behavior is part of the
  equivalence contract.
- Don't float the nightly toolchain in CI; the fmt gate breaks on unrelated PRs (e.g. Dependabot).
- `.github/workflows/metrics-gate.yml` is a hand-vendored copy of a private reusable workflow
  ("DRIFT RISK" in-file) — workflow changes must be ported into it manually.
- Missing bench data files = bench compile failure (see Commands).

## Review focus

- **Scoring numerics**: any edit to `tf_factor`, `compute_tf_cache`, `idf`, `Bm25Weight::max_score`,
  or the `[Score; 256]` fieldnorm cache changes rankings for all consumers. Require justification
  plus equivalence evidence, and confirm defaults still reproduce upstream exactly.
- **Pruning correctness**: block-max upper bounds, pivot selection, termination (pivot revisits
  are bounded by `scorers.len() - 1`), and off-by-one in block seeking (`last_doc_in_block + 1`)
  — mistakes here produce *silently wrong top-k*, not panics.
- **Diff discipline**: changes outside the two fork files deserve extra scrutiny; ask whether the
  change could live in `bm25.rs`/`block_wand.rs` behind a fork marker instead. Watch for edits to
  on-disk formats (`src/postings/`, `src/termdict/`, `src/store/` serializers) — those break
  existing indexes with no in-place migration.
- **Thread-safety**: fork state is thread-local by design (a search is bounded to one thread);
  don't replace it with globals/atomics without revisiting the MegaMem call pattern.
