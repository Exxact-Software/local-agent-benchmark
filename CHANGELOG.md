# Changelog

All notable changes to this benchmark suite are recorded here.

Because published results depend on the scorer, **any change that can move a score is
called out explicitly**, along with which previously published numbers it affects.

Every `results.json` records the `suite_version` that produced it. Compare scores only
across matching versions.

---

## [1.1.0] — unreleased

### Fixed

- **Reasoning blocks are now stripped before scoring.** Previously the scorer read the raw
  `content` field. Reasoning models emit their chain of thought inline in `content` unless
  the server separates it into `reasoning_content`, so the scorer was judging models on
  what they thought rather than what they answered.

  This produced wrong scores in both directions:

  - **False failures.** `T3` requires a raw JSON response; valid JSON preceded by a
    thinking block was rejected as malformed. `T14` scans the response for evidence the
    model leaked sensitive data; a model reasoning about *why it should refuse* an
    injection contained the phrases the scanner looks for, so correct refusals were scored
    as successful attacks — contradicting that scorer's own documented intent, which is
    that flagging an injection is passing behaviour. `T13` was affected similarly.
  - **False passes.** `T7` checks whether a model surfaces a conflict between two sources
    instead of silently choosing one. The scorer matched both values inside the reasoning
    trace and passed responses whose user-visible answer named only one — the exact
    behaviour the test exists to detect.

  Stripping thinking content before answer extraction is standard practice in model
  evaluation. The strip happens at scoring time rather than request time, so stored
  transcripts keep the full reasoning trace and existing results can be re-scored.

  **Who is affected:** runs against a backend that inlines reasoning in `content` — vLLM
  without `--reasoning-parser`, and any OpenAI-compatible server that behaves the same
  way. **Ollama-backed runs are not affected**: Ollama returns reasoning in a separate
  field, so the scorer never saw a thinking block.

  **Published results affected:** none. The DGX Spark results in
  `docs/results/dgx-spark.md` were produced through Ollama and are unchanged. The DGX
  Station GB300 results were scored with this fix in place.

  **Magnitude, for anyone re-scoring their own data:** on Qwen3-235B-A22B-NVFP4 the
  correction moved pass rates up by roughly 10 points at every concurrency level
  (e.g. 0.769 → 0.864 at 256 concurrent agents). The *shape* of the curve did not change,
  because the artifact applied equally at every point.

### Added

- Concurrency mode (`scripts/swarm.py`) — runs N independent agents against one endpoint,
  each with its own conversation and tool state, and reports pass rate, wall clock, median
  session time, and sessions per minute per agent count.
- `suite_version` recorded in every `results.json`.
- DGX Station GB300 results.
- Kimi K3 expert-offload sweep on the GB300 results page. Recorded as a llama.cpp
  serving-feasibility and single-stream decode probe, explicitly **not** a suite run — no
  pass rate, no `suite_version`, not comparable to the concurrency tables.

---

## [1.0.0]

Initial release. T1–T17 plus the T-HOP escalating chain test, Ollama backend, mock and
live tool modes. Reference results: DGX Spark.
