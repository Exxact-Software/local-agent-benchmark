# Results

Published runs of this suite, by machine.

| Machine | Backend | Models | Mode | Suite | Write-up |
| --- | --- | --- | --- | --- | --- |
| [DGX Spark](dgx-spark.md) | Ollama | 9 (Nemotron, Qwen3.5, Gemma4) | sequential | 1.0.0 | [Benchmarking Local AI Agents on NVIDIA DGX Spark](https://www.exxactcorp.com/blog/benchmarks/benchmarking-local-ai-agents-on-nvidia-dgx-spark) |
| [DGX Station GB300](dgx-station-gb300.md) | vLLM (+ llama.cpp) | Qwen3-235B-A22B-NVFP4, DeepSeek-V4-Flash-0731, Kimi K3 UD-IQ2_XXS † | sequential + concurrency | 1.1.0 | 3 posts — see the results page |

† Kimi K3 was **not** run through the suite. It appears on the GB300 page as a
serving-feasibility and single-stream decode probe under llama.cpp, with no pass rate and no
`suite_version`. It is listed here so the model is discoverable, not as a comparable score.

Each results page is the raw data. The write-ups carry the analysis, the caveats, and a
*Reproduce This Benchmark* section with the exact commands used.

---

## Reading across machines

**Check the suite version first.** Scores are only comparable across matching
`suite_version` values. `CHANGELOG.md` records what changed and by how much. The 1.0.0 →
1.1.0 change moved reasoning-model scores by roughly ten points on affected backends, and
left Ollama-backed results untouched.

**Check the backend.** Ollama and vLLM differ in tool-call handling and in whether
reasoning content arrives inline. See [backends](../backends/).

**Check the mode.** Sequential runs score one agent at a time. Concurrency runs pool
across N simultaneous agents, so a pass rate there is passes ÷ (agents × 17) and can hide
a distribution — a flat median with a growing tail is a different result from uniform
behaviour, and pooling makes them look identical.

**Check the context window.** Capacity numbers are a function of the KV pool divided by
context per session. A model served with a smaller `--max-model-len` than its maximum will
report different capacity than the same model served wide open. Where a comparison caps a
model deliberately, the results page says so.

**Do not compare scores across different models' context windows without saying so.**
`T11` sends a ~50k-token payload; against a model with a smaller window it is rejected
before inference and costs a fixed 1/17 of the score. That is a statement about the
serving configuration, not the model's agentic ability.
