# Methodology

How this benchmark scores a model, what the resulting numbers mean, and the places where
the measurement can mislead you if you don't know how it works.

---

## What a score is

Each run executes 17 tests (`T1`–`T17`). Every test is scored pass/fail by
`score_result()` in `tests/harness.py` — deterministic rules, no model-as-judge, no
human in the loop. A model's score is the number of tests it passed.

`T-HOP` is scored separately and is not part of the T1–T17 total. It records how many
sequential tool calls an agent completed before the chain broke, capped at 50.

In concurrency mode the pass rate is pooled across all agents: 64 agents × 17 tests is
1,088 test executions, and the pass rate is passes ÷ executions. A pooled rate is not the
same as a per-agent score, and the distribution matters — see *Reading a pass rate* below.

---

## Reasoning models

**The scorer strips `<think>…</think>` blocks before evaluating a response.**

Reasoning models emit their chain of thought inline in the `content` field unless the
server separates it into `reasoning_content`. Scoring the raw string judges a model on
what it thought rather than what it answered, and that breaks tests in both directions:
a test demanding raw JSON sees a paragraph of deliberation and calls valid JSON malformed;
a test scanning for leaked secrets finds the model's own reasoning about why it should
*refuse* to leak them.

Stripping thinking before answer extraction is standard practice in model evaluation.
We strip at scoring time rather than at request time, so stored transcripts keep the
complete reasoning trace and old results remain re-scorable.

### Do not enable the server's reasoning parser for benchmarking

vLLM's `--reasoning-parser` routes thinking into a separate `reasoning_content` field,
which sounds like the tidier fix. Don't use it here:

- `vllm bench serve` does not count `reasoning_content` toward output tokens, so enabling
  the parser **understates throughput** — a reasoning model can appear several times slower
  than it is.
- The field is not part of the OpenAI-compatible response shape, so clients that read
  `content` see an empty answer.

Stripping in the scorer leaves the server's output stream untouched, so throughput
measurements stay honest and scoring is correct regardless of backend.

---

## Backends

The suite talks to any OpenAI-compatible `/v1` endpoint. Two behaviours differ in ways
that affect results:

| | Ollama | vLLM |
| --- | --- | --- |
| Reasoning content | returned in a separate field | inline in `content` unless `--reasoning-parser` is set |
| Tool calling | native | requires `--tool-call-parser` matched to the model |

**A mismatched tool-call parser makes a working model look broken.** The failure mode is
a model that answers correctly in prose but registers zero tool calls, which scores as a
capability failure rather than the configuration error it is. Run a single agent and
confirm tool calls are being parsed before launching a sweep.

---

## Reading a pass rate

A pass rate below 1.0 is not automatically a capability finding. Check three things first.

**Is a test structurally impossible for this model?** `T11` sends a roughly 50,000-token
tool result. Against a model with a smaller context window the request is rejected with
`HTTP 400` before the model sees it. That is a fixed deduction of 1/17 at every
concurrency level, and it says something about the context window rather than the model's
agentic ability.

**Are the failures the same tests every time?** Failures that concentrate in a fixed
handful of tests, reproducing with a single agent on an idle GPU, are model behaviour.
Failures that appear only under load are a load effect. These are different findings and
the distinction is easy to lose in a pooled average.

**Does the pooled rate hide a tail?** A flat median can coexist with a growing worst case.
In the GB300 runs, median multi-hop depth was unchanged from 1 to 64 concurrent agents
while the number of agents completing one hop or fewer grew with concurrency. If your
workload needs *every* agent to succeed rather than most, the tail is the number that
matters.

---

## Comparing across runs

- **Compare only matching `suite_version` values.** Every `results.json` records the
  version that scored it, and scorer changes are documented in `CHANGELOG.md` with the
  magnitude of their effect.
- **Compare only matching concurrency.** A synthetic figure at one concurrency tells you
  little about agentic capacity at another.
- **Discard the first measurement at a new request shape.** The first point pays for CUDA
  graph capture and prefix-cache warmup, and a handful of warmup iterations may not absorb
  it. The tell is throughput that fails to rise with concurrency — if a lower concurrency
  reports higher throughput than a higher one, the lower point is cold, not fast.
- **Repeat runs.** Single-run scores compare broad behaviour; flakiness only shows up
  across repeats.
