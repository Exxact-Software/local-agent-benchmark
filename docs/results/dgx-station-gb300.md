# NVIDIA DGX Station GB300

Two models run through the same suite at the same concurrency levels — the point being to
separate what the *box* does from what a *model* does — plus a third model run only as a
serving-feasibility probe.

**Suite version:** 1.1.0 · **Backend:** vLLM · **Mode:** concurrency (`scripts/swarm.py`)

The Kimi K3 section at the bottom is **not** a suite result. It uses llama.cpp and measures
single-stream decode only; the 17-test suite was never run against that model. It is on this
page because it is a result about the machine's memory hierarchy, and it is fenced off so it
cannot be mistaken for a comparable score.

**Write-ups.** This page is the raw result set. The analysis lives in three posts:

- *Benchmarking 256 Concurrent AI Agents on NVIDIA DGX Station GB300* — the Qwen3-235B
  sweep, the context wall, and the hour-long soak.
  <!-- TODO: link when published -->
- *Ten Simultaneous Million-Token Agents on NVIDIA DGX Station GB300* — the DeepSeek-V4-Flash
  comparison, which findings generalised, and the context-scaling result.
  <!-- TODO: link when published -->
- *Running a 2.8-Trillion-Parameter Model on One Desk* — Kimi K3 at ~1.9-bit, the expert
  offload sweep, and where the memory cliff actually sits.
  <!-- TODO: link when published -->

If you're here to reproduce something, both posts carry a *Reproduce This Benchmark*
section with the exact serving and runner commands.

---

## Hardware

Measured, not from a spec sheet.

| | |
| --- | --- |
| GPU | 1 × GB300, 250.7 GiB HBM (256,703 MiB), no tensor parallelism |
| CPU | aarch64 Grace, 72 cores |
| Memory | 798.9 GB total coherent |
| Idle draw | 213 W |

Everything below is `--tensor-parallel-size 1`. There is one GPU.

---

## Models

| | Qwen3-235B-A22B | DeepSeek-V4-Flash-0731 |
| --- | --- | --- |
| Parameters | 235B total / 22B active | 284B total / 13B active |
| Quantization | NVFP4 | FP8 |
| Weights in HBM | ~125 GiB | 146.18 GiB |
| Native context | 40,960 | 1,048,576 |
| Context as served | 40,960 | **40,960** (capped to match) |
| Attention backend | standard | `DEEPSEEK_SPARSE_SWA` |
| vLLM image | `26.01-py3` | `26.06-py3` |
| Tool-call parser | `hermes` | `deepseek_v4` |

DeepSeek is deliberately capped at Qwen3's context window. Left at its native 1M the
comparison would be confounded by KV geometry rather than measuring the models.

Serving commands: [`docs/backends/vllm.md`](../backends/vllm.md).

---

## Reliability under concurrency

Pass rate is pooled: agents × 17 tests. Every run completed with **zero agent failures**.

### Qwen3-235B-A22B-NVFP4

| agents | pass rate | wall clock | median session | sessions/min | failed |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.824 | 201.8 s | 201.8 s | 0.30 | 0 |
| 4 | 0.897 | 288.8 s | 213.4 s | 0.83 | 0 |
| 8 | 0.875 | 387.5 s | 322.8 s | 1.24 | 0 |
| 16 | 0.864 | 430.3 s | 317.5 s | 2.23 | 0 |
| 32 | 0.890 | 463.9 s | 396.6 s | 4.14 | 0 |
| 64 | 0.867 | 654.8 s | 470.4 s | 5.86 | 0 |
| 128 | 0.871 | 761.1 s | 556.5 s | 10.09 | 0 |
| 256 | 0.864 | 1,019.3 s | 772.3 s | 15.07 | 0 |

### DeepSeek-V4-Flash-0731

| agents | pass rate | passed/total | wall clock | median session | sessions/min | failed |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.882 | 15/17 | 41.4 s | 41.4 s | 1.45 | 0 |
| 16 | 0.820 | 223/272 | 137.9 s | 130.5 s | 6.96 | 0 |
| 64 | 0.819 | 891/1088 | 196.9 s | 187.2 s | 19.50 | 0 |
| 128 | 0.821 | 1787/2176 | 246.0 s | 232.2 s | 31.22 | 0 |

**Neither model degrades with concurrency.** Qwen3 holds 0.824–0.897 across a 256× range.
DeepSeek holds 0.819–0.821 from 16 to 128 agents — a spread of 0.002 across an 8× range.
Zero agents failed at any point on either model.

**DeepSeek is 3–5× faster in wall clock** at every matched point (4.9× at 1 agent, 3.1×
at 128), which tracks with 13B active parameters against 22B plus sparse attention.

---

## KV pool — the capacity ceiling

| | KV pool | concurrency at 40,960 tokens/request |
| --- | --- | --- |
| Qwen3-235B-A22B-NVFP4 | 1,034,736 tokens | 25.3× |
| DeepSeek-V4-Flash-0731 | 997,753 tokens | 24.36× |

Two unrelated attention architectures, 21 GiB apart in weights, land within **3.6%** of
each other. **Pool size is a property of the machine** — total HBM minus weights.

**Turning that pool into a session count is not.** A long-context probe (30,000-token
synthetic sessions, rising concurrency, identical method on both models) shows the
arithmetic holding on one model and failing on the other:

| concurrency | Qwen3 output tok/s | DeepSeek output tok/s |
| --- | --- | --- |
| 8 | 100.7 | 275.1 |
| 16 | 172.5 | 315.9 |
| 24 | **230.8** ← peak | 320.0 |
| 32 | 116.7 ← collapse | 329.2 |
| 40 | 113.8 | 336.1 |
| 48 | 114.9 | **345.7** |

Qwen3's wall was predicted at 34.2 from pool ÷ 30,256 and measured at 34 — 0.6% error.
The same arithmetic predicts 33 for DeepSeek. No wall appeared through 48, where 48
resident 30k sessions would need ~1.44M tokens against a 997,753-token pool. Throughput
rose monotonically and TTFT climbed linearly.

The likely cause is the attention backend: vLLM selects `DEEPSEEK_SPARSE_SWA`, so a
30,000-token request does not hold 30,000 tokens of KV resident. The pool is the same
size; sessions cost less of it.

**Rule of thumb, qualified:** measure the pool at server startup and use pool ÷ context
per agent as a *lower bound*. It is accurate for dense attention and understates capacity
for sparse or windowed attention — here by at least 45%.

### Pool size is configuration-dependent for this model

Same model, same box, three context windows, re-served each time:

| `--max-model-len` | KV pool | stated max concurrency | total live context |
| --- | --- | --- | --- |
| 40,960 | 997,753 tokens | 24.36× | ~1.0M tokens |
| 262,144 | 4,987,104 tokens | 19.02× | ~5.0M tokens |
| 1,048,576 | 11,086,409 tokens | 10.57× | ~11.1M tokens |

For a dense-attention model the pool in tokens is fixed — KV bytes per token is constant.
Here it grows 11× as the window grows 25.6×, consistent with sparse attention making long
sequences cheaper per token to cache. Concurrency falls only 2.3× while context per
session rises 25.6×.

**The GB300 serves ten simultaneous million-token sessions with this model.** A prediction
made from the 40,960 configuration — that a 997,753-token pool could not hold even one
1M-token session, so the server would refuse to start — was wrong, because the pool is not
a constant.

*Caveats:* these are vLLM's stated concurrency figures at load, not measured walls. Where
one such figure was probed empirically (24.36× at 40,960) it proved conservative. No
million-token agent workload was run; this establishes admission capacity only.

*Method caveat:* `num_requests_waiting` was sampled before and after each point rather
than during, so the queue had drained by sampling time and those counters are
inconclusive. The throughput and TTFT curves are the evidence.

---

## Token mix — what does *not* generalize

Prefill/generation split from vLLM `/metrics` deltas across each run:

| agents | Qwen3 | DeepSeek |
| --- | --- | --- |
| 16 | — | 80/20 |
| 64 | — | 81/19 |
| 128 | 52/48 | **81/19** |
| 256 | 50/50 | — |

These are the same tests driving both models, and the mix is completely different.

The cause is generation volume, not workload shape. Qwen3 emits long inline
chain-of-thought on these prompts; DeepSeek-0731 answered them without thinking blocks.
At 128 agents the two generate at almost the same rate — 2,184 tok/s for Qwen3 against
2,136 for DeepSeek — while DeepSeek pushes **8,885 tok/s of prefill** against Qwen3's
2,360, because terser answers finish sessions faster and more turns get re-read.

Total throughput at 128 agents: Qwen3 4,543.6 tok/s, DeepSeek 11,021.2 tok/s.

**Do not generalize a prefill/generation ratio from one model to "agentic workloads."**
A verbose reasoning model looks balanced; a terse one looks as prefill-heavy as a
synthetic 1024-in/256-out benchmark.

---

## Agentic vs synthetic throughput

Synthetic arm: random 1024-in / 256-out, 512 prompts, 3 warmups, `/v1/completions` —
identical for both models.

| | agentic generation | synthetic output | agents achieve |
| --- | --- | --- | --- |
| Qwen3 @ 128 | 2,183.9 tok/s | 3,037.3 tok/s | 72% |
| Qwen3 @ 256 | 3,412.2 tok/s | 4,924.6 tok/s | 69% |
| DeepSeek @ 128 | 2,136.3 tok/s | 3,639.0 tok/s | **59%** |

Agentic load underperforms a synthetic benchmark at matched concurrency on both models,
so the gap itself is real and reproducible. Its **size is not** — 59% against 72% at the
same concurrency on the same machine. Use the gap to argue that synthetic tokens/sec
overstates agentic capacity; do not carry a specific percentage across models.

A note on method: the first measurement at a new request shape can be cold. The Qwen3
c128 synthetic point was contaminated this way and re-run — it reported *lower* throughput
than c256, which is impossible. The DeepSeek arm ran a throwaway c128 point first as a
precaution; it came back at 3,635.4 tok/s against the clean run's 3,639.0, so `26.06-py3`
did not show the effect that bit `26.01-py3`. Cheap insurance, and the sanity check is the
same either way: throughput must rise with concurrency.

---

## Reading the pass rates

Failures concentrate in a fixed set of tests at every concurrency level, including a
single agent on an idle GPU. They are model behaviour, not load effects.

**T11 cannot pass on either model as served.** It sends a ~50k-token payload; both are
served at a 40,960-token window, so the request is rejected with `HTTP 400` before
inference. That is a fixed 1/17 deduction and a statement about the serving
configuration. DeepSeek's native 1M window would pass it — that comparison is not run
here, because raising its window would break the matched setup.

Both models were scored with suite 1.1.0, which strips reasoning blocks before scoring.
Qwen3 scored roughly ten points lower under 1.0.0; see [`CHANGELOG.md`](../../CHANGELOG.md).

---

## Kimi K3 at 2.8T — expert offload sweep

**Not a suite result.** Different backend (llama.cpp, not vLLM), different measurement
(single-stream decode, not pooled pass rate), and the 17-test suite was not run. Do not
compare the tok/s here against the concurrency tables above, which are aggregate rates
across many simultaneous agents.

The question was whether a 2.8-trillion-parameter model runs on one desk at all, and how the
expert-offload dial behaves.

| | |
| --- | --- |
| Model | Kimi K3, UD-IQ2_XXS (~1.9-bit) |
| Parameters | 2.8T total, 896 experts, 16 active per token |
| Blocks | 93 — 1 dense + **92 MoE** |
| On disk | 664 GB across 17 GGUF shards |
| Backend | llama.cpp (`llama-server`), `--n-gpu-layers 999` |
| Context as served | 8,192 |

Quantization was not a free choice. At 3-bit the weights need roughly 1,050 GB against
~1,068 GB of total memory — 98% of every byte in the machine, leaving nothing for KV cache,
activations or the OS. 4-bit exceeds total memory by a third. ~1.9-bit is the largest
quantization that fits with headroom.

### The sweep

`--n-cpu-moe N` keeps the expert weights of N layers in CPU memory. With 92 MoE layers,
`--n-cpu-moe 92` puts every expert on the CPU. Sweeping downward moves experts onto the GPU.
Three runs per point, 128 generated tokens each.

| `--n-cpu-moe` | expert layers on GPU | decode | vs baseline | gain over previous | HBM (MiB) | load | power |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 92 | 0 | 7.637 tok/s | — | — | 252,642 | 730 s | 311 W |
| 88 | 4 | 7.916 tok/s | +3.7% | +3.7% | 252,655 | 727 s | 318 W |
| 80 | 12 | 8.681 tok/s | +13.7% | +9.7% | 252,823 | 726 s | 329 W |
| **76** | **16** | **8.698 tok/s** | **+13.9%** | **+0.2%** | 252,647 | 713 s | 326 W |
| 72 | 20 | **fails to load** | — | — | — | died at 696 s | — |
| 64 | 28 | **fails to load** | — | — | — | died at 726 s | — |

**14% total, flat at the top, then a cliff.** The dial is not really a dial: moving 16 expert
layers onto the GPU buys 13.9%, and the last four of those buy 0.2%. Everything useful
happens between 92 and 80.

### Where the cliff sits

At `--n-cpu-moe 72` every expert weight loaded successfully. llama.cpp then failed
allocating a **fixed 4,085.50 MiB compute buffer** — `failed to allocate compute pp buffers`
— 696 seconds in. Same failure mode at 64, 726 seconds in.

That is the operationally important part: the failure is not gradual and not early. You pay
the full ~12-minute model load before discovering the configuration never worked, and the
thing that runs out is not the expert weights you were tuning but a fixed-size compute
buffer competing for the same HBM.

### Memory hierarchy

At `--n-cpu-moe 92` the model runs with **602 GiB in CPU RAM and 247 GiB in HBM at once**,
with roughly 18 GiB spare. That is the actual finding about the box: coherent memory lets a
2.8T model load and serve, and the cost is single-stream decode about **eight times slower**
than a model that fits entirely in HBM.

| model | params | precision | GPU memory | CPU memory | single-stream decode |
| --- | --- | --- | --- | --- | --- |
| Qwen3-235B-A22B | 235B | NVFP4 (4-bit) | ~125 GiB | — | 66.4 tok/s |
| **Kimi K3** | **2.8T** | **UD-IQ2_XXS (~1.9-bit)** | **247 GiB** | **602 GiB** | **7.6–8.7 tok/s** |

### Caveats

**No prefill rates are published here, deliberately.** The sweep logs contain prefill
figures of 9.4–11.4 tok/s. They are invalid: `prompt_n = 4` on the timed runs, because prompt
caching meant only four tokens were actually prefilled after the warmup. Those numbers
measure request overhead, not prefill. Decode is unaffected — `predicted_n` was a full 128
on every run.

**No output-quality evaluation was run.** Everything above is a rate. Whether ~1.9-bit
quantization of a 2.8T model produces better answers than a 4-bit quantization of a much
smaller one is the obvious question and it is not answered here.

**Single context window.** All points were served at 8,192 tokens. The compute buffer that
caused the failures scales with context, so the cliff position is specific to this window.

Raw logs: per-point `kimi-n*.log` (client), `serve-n*.log` (server), `resp-n*-r*.json`
(responses), and `summary.csv`.
