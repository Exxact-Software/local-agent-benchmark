# Setup Guide

Complete setup from a fresh machine to a running benchmark. The reference target is an NVIDIA DGX Spark on DGX OS (Ubuntu 24.04, aarch64), but anything with Docker, an NVIDIA driver, and enough memory should work.

---

## Prerequisites

- Linux with Docker installed (the modern `docker compose` v2 plugin must be available — `docker compose version` should print v2.x or higher)
- An NVIDIA GPU with the host driver installed (Ollama uses it directly — you don't need the NVIDIA Container Runtime, since the benchmark container itself doesn't touch the GPU)
- Enough disk space for the models you want to test (Ollama models can be large — a full model set with 100B-class models needs ~400 GB)

Verify your driver is installed and visible:

```bash
nvidia-smi
```

If your user is not yet in the `docker` group:

```bash
sudo usermod -aG docker $USER
# log out and back in for the group change to take effect
```

---

## 1. Install Ollama natively

Ollama runs on the host directly — not in a container — so it has direct access to the GPU without an extra abstraction layer. This matters for a local-agent benchmark where inference performance is the point.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify it's running and reachable:

```bash
systemctl status ollama --no-pager
curl http://localhost:11434/api/tags
```

---

## 2. Clone and configure

```bash
git clone https://github.com/Exxact-Software/local-agent-benchmark
cd local-agent-benchmark
cp .env.example .env
```

The default `.env` works out of the box for mock mode. Set `GOOGLE_MAPS_API_KEY` only if you plan to run with `--live-tools`.

---

## 3. Pull benchmark models

Use the host Ollama directly. Smallest first means you can start running tests sooner:

```bash
ollama pull nemotron-3-nano:4b           # ~2.8 GB
ollama pull gemma4:e4b                   # ~8 GB
ollama pull gemma4:26b                   # ~17 GB
ollama pull qwen3.5:27b                  # ~17 GB
ollama pull gemma4:31b                   # ~20 GB
ollama pull qwen3.5:35b-a3b              # ~23 GB
ollama pull nemotron-3-nano:30b          # ~24 GB
ollama pull qwen3.5:122b-a10b            # ~81 GB
ollama pull nemotron-3-super:120b-a12b   # ~86 GB
```

Verify:

```bash
ollama list
```

---

## 4. (Optional) Start SearXNG for live-tool mode

Skip this step if you only plan to run mock-tool benchmarks (the default).

```bash
docker compose up -d searxng
docker compose ps
```

---

## 5. Run a benchmark

Single model:

```bash
docker compose run --rm benchmark python -m tests.harness \
  --model gemma4:26b --machine dgx-spark --run 1
```

All configured models:

```bash
./scripts/run_all.sh
```

Override defaults via env vars:

```bash
MACHINE=my-rig RUNS=1 ./scripts/run_all.sh           # single run, custom machine label
LIVE_TOOLS=1 ./scripts/run_all.sh gemma4:26b         # live mode, model name filter
```

---

## 6. Summarize

```bash
python3 scripts/summarize.py --machine dgx-spark
```

Or render every machine you have results for:

```bash
python3 scripts/summarize.py --all
```

---

## Notes

- **Architecture:** if your CPU is aarch64 (ARM, e.g. DGX Spark / Grace Blackwell), make sure any binary you install is the ARM build.
- **Networking:** the benchmark runner uses `network_mode: host`, so it talks to host Ollama at `localhost:11434` directly. If you've changed Ollama's port or are running it remotely, set `OLLAMA_HOST` in `.env`.
- **File ownership:** the benchmark container runs as `${UID:-1000}:${GID:-1000}` so result files are owned by your host user, not root. If your UID/GID are not 1000, either `export UID GID` in your shell or write them into a project `.env`.
- **Mock vs live:** mock results and `--live-tools` results write to separate paths so the canonical mock baseline is never overwritten.

---

## Troubleshooting

**`permission denied while trying to connect to the Docker daemon socket`**
Your user isn't in the `docker` group. Fix:
```bash
sudo usermod -aG docker $USER
# disconnect SSH and reconnect (or fully log out and back in)
```
For a quick session-only fix without reconnecting: `sudo chmod 666 /var/run/docker.sock`.

**`Cannot connect to Ollama` / connection refused on `localhost:11434`**
Verify the host Ollama service is up:
```bash
systemctl status ollama
curl http://localhost:11434/api/tags
```
If the service isn't running: `sudo systemctl start ollama`. If you've moved Ollama to a different host or port, set `OLLAMA_HOST` in `.env`.

**`Error: model 'X' not found`**
Pull it first:
```bash
ollama pull <model-name>
ollama list   # confirm it's there
```
`scripts/run_all.sh` automatically skips models that aren't pulled, but the harness invoked directly will fail.

**`bind: address already in use` when starting SearXNG**
Another service on your host is using port 8080. Either stop that service, or change the host-side port in `docker-compose.yml` (e.g. `"8081:8080"`) and update `SEARXNG_URL` in `.env` to match.

**Result files end up owned by root**
Your shell didn't export `UID`/`GID` for `docker compose` to pick up. Either export them once:
```bash
export UID=$(id -u) GID=$(id -g)
```
or add them to your project `.env`.

**`docker compose` says "unknown command"**
You have the legacy `docker-compose` v1 instead of the v2 plugin. Either install Docker's [compose plugin](https://docs.docker.com/compose/install/), or run the equivalent `docker-compose run ...` commands manually.

**Running on macOS (Docker Desktop)**
The benchmark runner uses `network_mode: host`, which on Linux means the container shares the host's network. On macOS, Docker Desktop runs containers inside a VM, so `network_mode: host` only refers to the VM — the container can't see the host's `localhost`. If you're on a Mac, replace `network_mode: host` in `docker-compose.yml` with:
```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```
…and set `OLLAMA_HOST=http://host.docker.internal:11434` (and the same for `SEARXNG_URL`) in your `.env`. The reference platform for this benchmark is Linux, so this path is less tested.
