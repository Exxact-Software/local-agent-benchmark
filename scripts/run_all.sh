#!/usr/bin/env bash
# Run the full benchmark against all configured models sequentially, N runs each.
# Results are saved to ./results/<machine>/<model>/run<N>/results.json
#
# Usage:
#   ./scripts/run_all.sh                              # all models, machine=dgx-spark, 3 runs
#   ./scripts/run_all.sh 4b 30b                       # only models matching these patterns
#   MACHINE=my-rig ./scripts/run_all.sh               # run on a different machine
#   RUNS=1 ./scripts/run_all.sh                       # single run (no averaging)
#   LIVE_TOOLS=1 ./scripts/run_all.sh                 # use live tool implementations

set -euo pipefail

MACHINE="${MACHINE:-dgx-spark}"
LIVE_TOOLS="${LIVE_TOOLS:-}"
RUNS="${RUNS:-3}"

MODELS=(
  "nemotron-3-nano:4b"            # ~2.8 GB
  "gemma4:e4b"                    # ~8 GB
  "gemma4:26b"                    # ~17 GB
  "qwen3.5:27b"                   # ~17 GB
  "gemma4:31b"                    # ~20 GB
  "qwen3.5:35b-a3b"               # ~23 GB
  "nemotron-3-nano:30b"           # ~24 GB
  "qwen3.5:122b-a10b"             # ~81 GB
  "nemotron-3-super:120b-a12b"    # ~86 GB
)

# If args provided, filter to only matching models
if [[ $# -gt 0 ]]; then
  FILTERED=()
  for model in "${MODELS[@]}"; do
    for pattern in "$@"; do
      if [[ "$model" == *"$pattern"* ]]; then
        FILTERED+=("$model")
        break
      fi
    done
  done
  MODELS=("${FILTERED[@]}")
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "No models matched. Available models:"
  printf '  %s\n' "nemotron-3-nano:4b" "gemma4:e4b" "gemma4:26b" \
                   "qwen3.5:27b" "gemma4:31b" "qwen3.5:35b-a3b" \
                   "nemotron-3-nano:30b" "qwen3.5:122b-a10b" "nemotron-3-super:120b-a12b"
  exit 1
fi

# Skip models the host Ollama hasn't pulled — saves long timeouts on missing models.
if command -v ollama >/dev/null 2>&1; then
  PULLED=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}')
  AVAILABLE=()
  SKIPPED=()
  for model in "${MODELS[@]}"; do
    if grep -qx "$model" <<< "$PULLED"; then
      AVAILABLE+=("$model")
    else
      SKIPPED+=("$model")
    fi
  done
  if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo "Skipping (not pulled — run 'ollama pull <name>' to add):"
    printf '  - %s\n' "${SKIPPED[@]}"
    echo ""
  fi
  if [[ ${#AVAILABLE[@]} -eq 0 ]]; then
    echo "No matching models are pulled. Pull at least one with: ollama pull <model>"
    exit 1
  fi
  MODELS=("${AVAILABLE[@]}")
fi

echo "Machine: $MACHINE"
[[ -n "$LIVE_TOOLS" ]] && echo "Tools: live (SearXNG)"
echo "Runs per model: $RUNS"
echo "Models to run: ${#MODELS[@]}"
printf '  - %s\n' "${MODELS[@]}"
echo ""

PASS=0
FAIL=0
START=$(date +%s)

# Build harness args
HARNESS_ARGS="--machine $MACHINE"
[[ -n "$LIVE_TOOLS" ]] && HARNESS_ARGS="$HARNESS_ARGS --live-tools"

for model in "${MODELS[@]}"; do
  for run in $(seq 1 "$RUNS"); do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶  $model  (run $run/$RUNS)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # shellcheck disable=SC2086
    if docker compose run --rm benchmark python -m tests.harness --model "$model" $HARNESS_ARGS --run "$run"; then
      echo "✓  $model run $run — done"
      PASS=$((PASS + 1))
    else
      echo "✗  $model run $run — failed (exit $?)"
      FAIL=$((FAIL + 1))
    fi
    echo ""
  done
done

END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Done. ${PASS} succeeded, ${FAIL} failed. Total time: ${ELAPSED}m"
echo "Results: ./results/${MACHINE}/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
