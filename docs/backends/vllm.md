# Running against vLLM

The suite talks to any OpenAI-compatible `/v1` endpoint, so vLLM works without code
changes — point `BENCH_BASE_URL` at the server. What follows is the configuration detail
that actually affects results.

```bash
export BENCH_BASE_URL=http://127.0.0.1:8000
python -m tests.harness --model <served-model-name> --machine <tag>
```

---

## Tool calling must be configured explicitly

Unlike Ollama, vLLM needs `--tool-call-parser` matched to the model, plus
`--enable-auto-tool-choice`. Get this wrong and the model answers correctly in prose while
registering zero tool calls — which scores as a capability failure rather than the
configuration error it is.

**Verify with a single agent before launching a sweep.** One request with a tool defined;
confirm `finish_reason` is `tool_calls` and that the parsed arguments are sane. A sweep
launched against a mismatched parser produces a full set of numbers that are all wrong in
the same believable direction.

Parser names are registered in `vllm/tool_parsers/__init__.py` in your image — check there
rather than guessing, since names differ across versions and models often have a
version-specific parser (`deepseek_v3`, `deepseek_v31`, `deepseek_v32`, `deepseek_v4` are
all distinct).

---

## Do not set `--reasoning-parser` for benchmarking

It routes chain-of-thought into a separate `reasoning_content` field, which sounds like
the right thing. It is not, for this use:

- `vllm bench serve` does not count `reasoning_content` toward output tokens, so enabling
  it **understates throughput** — a reasoning model can look several times slower than it
  is.
- `reasoning_content` is not part of the OpenAI-compatible response shape, so clients
  reading `content` see an empty answer.

The scorer strips `<think>…</think>` itself (see [methodology](../methodology.md)), so
correct scoring does not depend on the server. Leave the output stream alone and let
throughput measurements stay honest.

---

## The first measurement at a new request shape is cold

The first point at a new shape pays for CUDA graph capture and prefix-cache warmup, and a
few warmup iterations may not absorb it. The tell is throughput that fails to rise with
concurrency: if a lower concurrency reports *higher* throughput than a higher one, the low
point is cold, not fast. Discard and re-run it.

---

## Reference configurations

These are the exact serving commands behind the published results.

### Qwen3-235B-A22B-NVFP4 — DGX Station GB300

```bash
docker run -d --name bench-qwen3-235b --gpus all --ipc=host --network host \
  -v /path/to/qwen3-235b-a22b-nvfp4:/model \
  nvcr.io/nvidia/vllm:26.01-py3 \
  vllm serve /model \
    --served-model-name nvidia/Qwen3-235B-A22B-NVFP4 \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.90 \
    --quantization modelopt_fp4 \
    --kv-cache-dtype fp8_e4m3 \
    --tool-call-parser hermes \
    --enable-auto-tool-choice \
    --trust-remote-code
```

Reported KV pool: **1,034,736 tokens** (25.3× concurrency at a 40,960-token window).

### DeepSeek-V4-Flash-0731 — DGX Station GB300

```bash
docker run -d --name bench-dsv4 --gpus all --ipc=host --network host \
  -v /path/to/deepseek-v4-flash-0731:/model \
  nvcr.io/nvidia/vllm:26.06-py3 \
  vllm serve /model \
    --served-model-name deepseek-ai/DeepSeek-V4-Flash-0731 \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.90 \
    --kv-cache-dtype fp8_e4m3 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --trust-remote-code
```

Reported KV pool: **997,753 tokens** (24.36× concurrency at a 40,960-token window).

Notes on this one:

- The model's native context window is 1M tokens. It is capped at 40,960 here **on
  purpose**, to match Qwen3 so the comparison is not confounded by KV geometry. Raise it
  for context-scaling work; that changes every capacity number.
- FP8 quantization is read from the checkpoint — no `--quantization` flag.
- No `--speculative-config`, which leaves MTP disabled. The checkpoint ships
  `num_nextn_predict_layers: 1`, so MTP is available; enable it deliberately if you want
  it, and re-verify stability at high concurrency.
- The engine selects a `DEEPSEEK_SPARSE_SWA` attention backend automatically.
- Image is `26.06-py3`, not `26.01-py3`: `DeepseekV4ForCausalLM` support is needed. Check
  your driver branch matches the image before assuming a newer tag will work.
