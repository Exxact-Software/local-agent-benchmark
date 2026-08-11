# Local Agent Benchmark

A reproducible benchmark for **local AI agent behavior** — running structured tests against any model served over an OpenAI-compatible API. Measures whether a model can act as a reliable agent: calling tools, handling bad outputs, following instructions under load, and sustaining multi-step chains.

Two things it answers that a tokens-per-second number cannot: *does this model behave like an agent*, and *how many of them does this machine hold before it stops behaving like one*.

Published results are in [`docs/results/`](docs/results/) — currently the NVIDIA DGX Spark and the NVIDIA DGX Station GB300. Backend setup is in [`docs/backends/`](docs/backends/) (Ollama and vLLM), and how scoring works is in [`docs/methodology.md`](docs/methodology.md).

> **Comparing numbers?** Every `results.json` records the `suite_version` that scored it, and scores are only comparable across matching versions. [`CHANGELOG.md`](CHANGELOG.md) records what each version changed and how much it moved.

---

## What This Tests

Most local LLM benchmarks measure raw throughput: tokens per second, prompt-processing speed, single-turn answer quality. Those metrics do not tell you whether a model can survive a real agent loop.

This benchmark focuses on **agent behavior**:

- correct tool selection and argument passing
- handling 404s, malformed data, timeouts, large payloads
- JSON discipline and instruction adherence
- prompt-injection resistance
- state mutation across multiple operations
- multi-hop chains where each tool result becomes the next call

---

## Test Suite

**T1–T17** — 17 structured tests across 6 categories:

| Category | Tests | What it measures |
|---|---|---|
| Basic Tool Calling | T1–T4 | Tool selection, param handling, hallucination resistance |
| Parallel Tool Calls | T5–T7 | Multi-tool turns, attribution, conflict handling |
| Stress Inputs | T8–T11 | 404s, null fields, timeouts, ~50k-token payloads |
| Instruction Adherence | T12–T15 | One-call constraints, JSON discipline, injection resistance |
| Edge Cases | T16–T17 | Knowing when not to call a tool, state mutation tracking |

**T-HOP** — Escalating multi-hop chain capped at 50 hops. Each tool result becomes the next step's input. Records the hop where the chain breaks or the model stops calling tools.

**Concurrency mode** — `scripts/swarm.py` runs N independent agents against one endpoint, each in its own process with its own conversation and tool state, and reports pass rate, wall clock, median session time and sessions per minute at each agent count. This is what answers "how many agents does this machine hold", and it is a different question from throughput.

```bash
# One point
python3 scripts/swarm.py --model <served-model-name> --agents 64 --skip-hop

# Sweep — the curve is the point
python3 scripts/swarm.py --model <served-model-name> --sweep 1,4,8,16,32,64 --skip-hop
```

`--skip-hop` is usually right for sweeps: T-HOP's 50-hop cap dominates wall clock at high agent counts. Measure chain depth in a separate, smaller run.

---

## Architecture

This benchmark is split between **host-native** and **containerized** components:

- **Ollama runs natively on the host** for direct GPU access — containerizing it would add an abstraction layer between the model weights and the accelerator, which is the opposite of what you want for a local-agent benchmark.
- **The benchmark runner is a container** so users don't need to install Python dependencies on their host.
- **SearXNG is a container** because it's a stateless web service used only for the optional live-tool mode.

The benchmark runner uses `network_mode: host` to talk to host Ollama at `localhost:11434` without any port-mapping or DNS layer in between.

---

## Quickstart

### Requirements

- Docker (with the modern `docker compose` v2 plugin — included by default in recent Docker installs)
- [Ollama](https://ollama.com/) installed natively on the host (`curl -fsSL https://ollama.com/install.sh | sh`). Ollama uses the host's NVIDIA driver directly — no container runtime needed.
- Sufficient unified memory or VRAM for the models you want to test
- Optional: a Google Maps API key for live location-tool mode (mock mode is the default and needs no keys)

### Setup

```bash
git clone https://github.com/Exxact-Software/local-agent-benchmark
cd local-agent-benchmark
cp .env.example .env

# Optional: only needed if you plan to use --live-tools
docker compose up -d searxng

# Pull the models you want to benchmark — example:
ollama pull gemma4:26b
ollama pull qwen3.5:35b-a3b
```

### Run the Benchmark

```bash
# Single model
docker compose run --rm benchmark python -m tests.harness \
  --model gemma4:26b --machine dgx-spark --run 1

# All configured models
./scripts/run_all.sh

# Subset of tests
docker compose run --rm benchmark python -m tests.harness \
  --model gemma4:26b --tests T2,T13,T-HOP
```

Results land at `results/<machine>/<model>/run<N>/results.json`.

> **A note on runtime.** A full pass — 9 models × 3 runs × T1–T17 + T-HOP — is a multi-hour workload, sometimes a full day. The 100B-class models can hold T-HOP open for a long time. Start `./scripts/run_all.sh` and walk away. For a quick first try, run a single small model with `--tests T1,T2 --skip-hop` to confirm everything is wired up before committing to the full run.

### Summarize

```bash
python3 scripts/summarize.py --machine dgx-spark
```

Prints a markdown table with pass rate, hard-break hop, average tok/s, and any tests that flaked across runs.

---

## What You'll See

The harness prints a per-test pass/fail table while it runs:

```
──────── Benchmark — gemma4:26b ────────
Machine: dgx-spark  |  Ollama: http://localhost:11434
Tools: mock
Scope: full suite
Mode: Autonomous

Smoke test...
✓ Smoke test passed: Washington

Running T1 — Single Tool, Simple String Param...
Running T2 — Required vs Optional Params...
...
                T1–T17 Results
┏━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━┓
┃ ID ┃ Test                      ┃ Tools Called┃ Tok/s ┃ Latency ┃ Pass ┃
┡━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━┩
│ T1 │ Single Tool, Simple Param │ get_weather │  56.4 │   3.7s  │  ✓   │
│ T2 │ Required vs Optional      │ search_pl…  │  54.7 │   7.6s  │  ✓   │
...
└────┴───────────────────────────┴─────────────┴───────┴─────────┴──────┘

T1–T17 auto-score: 17 pass  0 fail  0 human-review
```

After running multiple models, `summarize.py` aggregates everything into a markdown table:

```
## Benchmark Results — dgx-spark

| Model                        | Runs | Pass (avg) | Fail (avg) | Flaky | Hard Break Hop | Avg Tok/s |
| ---------------------------- | ---- | ---------- | ---------- | ----- | -------------- | --------- |
| `gemma4:26b`                 | 1    | 17.0       | 0.0        | —     | 4              | 52.7      |
| `qwen3.5:35b-a3b`            | 1    | 17.0       | 0.0        | —     | 4              | 48.2      |
| `nemotron-3-super:120b-a12b` | 1    | 17.0       | 0.0        | —     | 6              | 16.4      |
```

---

## Repo Structure

```
local-agent-benchmark/
├── README.md
├── LICENSE
├── docker-compose.yml          # SearXNG + benchmark runner
├── Dockerfile                  # Benchmark runner container
├── requirements.txt
├── .env.example
├── .gitignore
├── tests/
│   ├── harness.py              # Main test runner
│   ├── tests.py                # T1–T17 + T-HOP definitions
│   ├── tools.py                # Mock tool simulator with scenario support
│   └── tools_live.py           # Live tool implementations (SearXNG + Open-Meteo + Google Maps)
├── scripts/
│   ├── run_all.sh              # Run all configured models in sequence
│   ├── swarm.py                # Concurrency mode — N agents against one endpoint
│   └── summarize.py            # Aggregate results into a markdown table
├── searxng/
│   └── settings.yml            # SearXNG config (JSON API enabled, no rate limits)
└── docs/
    ├── setup.md                # Full setup guide from scratch
    ├── methodology.md          # How scoring works; reasoning models; reading a pass rate
    ├── backends/
    │   ├── ollama.md           # Native tool calling, no parser config
    │   └── vllm.md             # Parser config, reference serving commands
    └── results/
        ├── README.md           # Index + how to compare across machines
        ├── dgx-spark.md        # DGX Spark reference results
        └── dgx-station-gb300.md
```

---

## Services

| Service | Container | Port | Purpose |
|---|---|---|---|
| Ollama | (native, not containerized) | 11434 | Local LLM inference |
| SearXNG | `benchmark-searxng` | 8080 | Self-hosted web search (live tool mode) |

---

## Live Tool Mode

By default the benchmark uses deterministic mock responses for all tools, so results are fully reproducible without API keys.

Pass `--live-tools` to route supported tools to real services:

- `web_search` → local SearXNG (no key required)
- `get_weather` → Open-Meteo (no key required)
- `geocode`, `search_places`, `get_place_details`, `get_directions`, `search_parking`, `get_reviews` → Google Maps (requires `GOOGLE_MAPS_API_KEY` in `.env`)

Live tools are only used in default scenarios. Failure-scenario tests (404, timeout, prompt injection, etc.) always use mocks regardless of this flag, so they remain controlled and reproducible. Mock and live results are saved to separate paths and never overwrite each other.

---

## Issues / Feedback

Bug reports, feature requests, and questions welcome at the [issue tracker](https://github.com/Exxact-Software/local-agent-benchmark/issues).

---

## License

MIT — see [`LICENSE`](LICENSE).
