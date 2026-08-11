# Running against Ollama

Ollama is the lowest-friction backend for this suite and the one the DGX Spark reference
results were produced with. Tool calling is native — there is no parser to configure — and
reasoning content is returned in a separate field rather than inline, so the scorer never
sees a thinking block.

```bash
export BENCH_BASE_URL=http://localhost:11434   # OLLAMA_HOST still works
ollama pull gemma4:26b
python -m tests.harness --model gemma4:26b --machine dgx-spark
```

Run Ollama natively on the host rather than in a container: it uses the host NVIDIA driver
directly, and putting a container runtime between the weights and the accelerator is the
opposite of what you want when benchmarking local inference.

---

## Things worth knowing

**Model tags are the model identity.** `--model` is passed through verbatim and recorded
in `results.json`. Quantization differences hide behind tags, so record the exact tag you
pulled — `gemma4:26b` today may not be byte-identical to `gemma4:26b` in six months.

**Context window is a server-side setting.** Ollama applies a default context length that
is often smaller than the model's maximum, which silently changes how `T11` (the ~50k-token
payload test) behaves and caps concurrency in swarm runs. Set it explicitly when it
matters, and record what you set.

**Concurrency needs configuring.** Ollama serializes requests by default. For `swarm.py`
runs, raise the parallel-request limit or you will measure the queue rather than the
model — the signature is wall clock scaling linearly with agent count while GPU
utilization stays low.

**No cold-start caveat.** Unlike vLLM there is no CUDA-graph capture penalty on the first
request at a new shape, so the "discard the first point" rule in
[methodology](../methodology.md) does not apply here.
