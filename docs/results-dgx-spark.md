# DGX Spark Reference Results

These are the reference results for the benchmark, run on an NVIDIA DGX Spark. They are intended as a comparison point for anyone running the benchmark on their own hardware.

The full write-up (motivation, methodology, model selection, and analysis) is in the accompanying blog post.

---

## Hardware

- **System:** NVIDIA DGX Spark
- **Architecture:** Grace Blackwell GB10
- **Unified memory:** 128 GB
- **OS:** DGX OS (Ubuntu 24.04, aarch64)

---

## Models Under Test

| Model | Type | Notes |
|---|---|---|
| `nemotron-3-nano:4b` | Dense | Small baseline / sanity check |
| `nemotron-3-nano:30b` | MoE | Mid-sized speed-oriented model |
| `qwen3.5:35b-a3b` | MoE (3B active) | Strong fast MoE baseline |
| `qwen3.5:27b` | Dense | Larger dense comparison |
| `qwen3.5:122b-a10b` | MoE (10B active) | Large memory-stressing candidate |
| `nemotron-3-super:120b-a12b` | MoE (12B active) | Flagship Spark-native candidate |
| `gemma4:e4b` | Dense | Small/fast Gemma baseline |
| `gemma4:26b` | Dense | Practical mid-size local-agent candidate |
| `gemma4:31b` | Dense | Larger Gemma comparison |

---

## Results — Original Benchmark Pass

| Model | T1–T17 | T-HOP | Avg Tok/s | Interpretation |
|---|---|---|---|---|
| `nemotron-3-nano:4b` | `16/17` | `4` | `64.2` | Fast, but one noisy full-run miss |
| `nemotron-3-nano:30b` | `15/17` | `5` | `64.7` | Fast, but flaky on `T13` |
| `qwen3.5:35b-a3b` | `17/17` | `4` | `48.2` | Clean and reliable |
| `qwen3.5:27b` | `17/17` | `5` | `10.4` | Clean and reliable |
| `qwen3.5:122b-a10b` | `17/17` | `3` | `20.1` | Clean after `T11` timeout fix |
| `nemotron-3-super:120b-a12b` | `17/17` | `6` | `16.4` | Strongest overall agent profile |

## Results — Gemma Family

| Model | T1–T17 | T-HOP | Avg Tok/s | Interpretation |
|---|---|---|---|---|
| `gemma4:e4b` | `17/17` | `2` | `52.6` | Fast and clean, but shallow multi-hop behavior |
| `gemma4:26b` | `17/17` | `4` | `52.7` | Best Gemma result; strong speed/reliability balance |
| `gemma4:31b` | `17/17` | `2` | `9.7` | Clean, but slow and shallow vs. the others |

---

## How To Read These Results

A single number does not tell the whole story. The benchmark is designed to make tradeoffs visible across three axes:

- **`T1–T17`** — structured agent behavior: tool selection, argument handling, malformed-output handling, formatting discipline, injection resistance, state tracking.
- **`T-HOP`** — multi-hop chain depth. Each result becomes the next call's input, so the model has to preserve context. Real agents rarely stop after one tool call.
- **`Avg Tok/s`** — generation speed. Important for an always-on agent, but not the same thing as reliability.

A clean `17/17` does not automatically mean a model is the right choice. A model can pass T1–T17 cleanly and still stop early in T-HOP. Another model can be slower but hold the chain together longer. The best result is the model that balances all three.

Repeat runs matter. Single-run scores are good enough to compare broad behavior, but flakiness only shows up across multiple runs.

---

## Headline Takeaways

- **Strongest overall agent profile:** `nemotron-3-super:120b-a12b` — clean `17/17`, deepest T-HOP at 6 hops.
- **Practical mid-size surprise:** `gemma4:26b` — clean `17/17`, 4 T-HOP hops, ~52.7 tok/s.
- **Strong fast MoE baseline:** `qwen3.5:35b-a3b` — clean `17/17`, 4 T-HOP hops, ~48 tok/s.
- **Speed ≠ reliability.** Smaller Nemotron models were the fastest but had the most flakiness. The largest Gemma (`gemma4:31b`) was slower than `gemma4:26b` and shallower in T-HOP.

For a local agent, reliability over the action loop usually matters more than raw tokens per second.

---

## Reproducing

```bash
git clone https://github.com/Exxact-Software/local-agent-benchmark
cd local-agent-benchmark
cp .env.example .env

# Install Ollama natively, then pull the models you want, then:
./scripts/run_all.sh
python3 scripts/summarize.py --machine dgx-spark
```

See [`setup.md`](setup.md) for the full setup walkthrough.
