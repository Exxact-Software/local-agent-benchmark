#!/usr/bin/env python3
"""Run the T1-T17 agent benchmark as a concurrent swarm.

The upstream harness is sequential by design: it answers "is this model a
reliable agent?" one agent at a time. This driver answers a different question —
"how many of those agents can one box sustain before latency or reliability
breaks?" — by running N independent agent sessions simultaneously against a
single endpoint.

Concurrency is at the *process* level on purpose. Each agent gets its own
interpreter, so the stateful tools (T17 tracks state mutation) cannot interfere
with each other, and the scoring path is byte-for-byte the published one. That
keeps swarm results directly comparable to the sequential DGX Spark numbers
rather than "similar but re-implemented".

Usage:
    # single point
    python scripts/swarm.py --model qwen3-235b --agents 8

    # full sweep, the shape used for the capacity curve
    python scripts/swarm.py --model qwen3-235b --sweep 1,4,8,16,32,64

Endpoint is taken from OLLAMA_HOST (the harness appends /v1), so any
OpenAI-compatible server works — Ollama, vLLM, anything else.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_one_agent(idx: int, model: str, machine: str, results_root: Path,
                  endpoint: str, extra_args: list[str]) -> dict:
    """Run one agent session in its own process. Never raises — a crashed agent
    is data (it is exactly the reliability degradation we are looking for), not
    a reason to abort the sweep."""
    agent_dir = results_root / f"agent-{idx:03d}"
    agent_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["RESULTS_DIR"] = str(agent_dir)
    env["BENCH_BASE_URL"] = endpoint
    env["OLLAMA_HOST"] = endpoint  # deprecated alias, kept for older harness copies

    cmd = [
        sys.executable, "-m", "tests.harness",
        "--model", model,
        "--machine", machine,
        "--run", str(idx),
        *extra_args,
    ]

    started = time.monotonic()
    out = ""
    try:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, env=env,
            capture_output=True, text=True, timeout=3600,
        )
        rc, err, out = proc.returncode, proc.stderr[-2000:], proc.stdout
    except subprocess.TimeoutExpired as exc:
        # A timed-out agent still produced output up to the point it stalled, and that
        # partial log is usually the only evidence of where it stalled — keep it.
        rc, err = -1, "timeout after 3600s"
        out = exc.stdout if isinstance(exc.stdout, str) else ""
    elapsed = time.monotonic() - started

    (agent_dir / "stdout.log").write_text(out)

    return {
        "agent": idx,
        "returncode": rc,
        "elapsed_s": round(elapsed, 2),
        "results_dir": str(agent_dir),
        "stderr_tail": err if rc != 0 else "",
    }


def score_agent(agent_dir: Path) -> dict:
    """Pull pass/fail out of whatever results.json the harness wrote.

    Deliberately tolerant: a swarm run that degrades will produce partial or
    malformed output, and silently reporting 0 would look like a clean failure
    when it is actually missing data. Unreadable results are counted as
    'unscored' and surfaced separately.
    """
    found = list(agent_dir.rglob("results.json"))
    if not found:
        return {"scored": False, "reason": "no results.json"}

    try:
        data = json.loads(found[0].read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return {"scored": False, "reason": f"unreadable: {exc}"}

    rows = data.get("tests") or data.get("results") or []
    if not isinstance(rows, list):
        return {"scored": False, "reason": "unexpected schema"}

    # T-HOP is human-scored — its `pass` is always null and the actual result lives in
    # max_hops_completed. Counting `pass is True` over it reports a pass rate of 0.0,
    # which reads as total failure when the chains in fact ran fine. Split it out.
    hop_rows = [r for r in rows if r.get("id") == "T-HOP"]
    scored_rows = [r for r in rows if r.get("id") != "T-HOP"]

    result = {
        "scored": True,
        "passed": sum(1 for r in scored_rows if r.get("pass") is True),
        "failed": sum(1 for r in scored_rows if r.get("pass") is False),
        "human_scored": sum(1 for r in scored_rows if r.get("pass") is None),
        "total": len(scored_rows),
    }
    if hop_rows:
        h = hop_rows[0]
        result["hop"] = {
            "max_hops_completed": h.get("max_hops_completed"),
            "quality_drop_hop": h.get("quality_drop_hop"),
            "hard_break_hop": h.get("hard_break_hop"),
            "hit_cap": h.get("hit_cap"),
            "tok_per_s": h.get("tok_per_s"),
        }
    return result


def run_swarm(n: int, model: str, endpoint: str, machine: str,
              out_root: Path, extra_args: list[str]) -> dict:
    results_root = out_root / f"agents-{n:03d}"
    results_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}\n  {n} concurrent agent(s) — {model}\n{'=' * 60}", flush=True)

    wall_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as pool:
        runs = list(pool.map(
            lambda i: run_one_agent(i, model, f"{machine}-c{n}", results_root,
                                    endpoint, extra_args),
            range(1, n + 1),
        ))
    wall = time.monotonic() - wall_start

    for r in runs:
        r["score"] = score_agent(Path(r["results_dir"]))

    scored = [r["score"] for r in runs if r["score"].get("scored")]
    total_pass = sum(s["passed"] for s in scored)
    total_tests = sum(s["total"] for s in scored)
    durations = [r["elapsed_s"] for r in runs]

    hops = [s["hop"]["max_hops_completed"] for s in scored
            if s.get("hop") and s["hop"].get("max_hops_completed") is not None]
    hard_breaks = sum(1 for s in scored
                      if s.get("hop") and s["hop"].get("hard_break_hop"))
    hit_caps = sum(1 for s in scored if s.get("hop") and s["hop"].get("hit_cap"))

    summary = {
        "agents": n,
        "model": model,
        "wall_clock_s": round(wall, 2),
        "agents_completed": sum(1 for r in runs if r["returncode"] == 0),
        "agents_failed": sum(1 for r in runs if r["returncode"] != 0),
        "agents_unscored": sum(1 for r in runs if not r["score"].get("scored")),
        "tests_passed": total_pass,
        "tests_total": total_tests,
        "pass_rate": round(total_pass / total_tests, 4) if total_tests else None,
        "agent_elapsed_s": {
            "min": round(min(durations), 2) if durations else None,
            "median": round(statistics.median(durations), 2) if durations else None,
            "max": round(max(durations), 2) if durations else None,
        },
        "sessions_per_min": round(n / (wall / 60), 2) if wall > 0 else None,
        "hop_depth": {
            "runs": len(hops),
            "median": statistics.median(hops),
            "min": min(hops),
            "max": max(hops),
            "hard_breaks": hard_breaks,
            "hit_cap": hit_caps,
        } if hops else None,
        "runs": runs,
    }

    print(f"  completed {summary['agents_completed']}/{n} · "
          f"pass rate {summary['pass_rate']} · "
          f"wall {summary['wall_clock_s']}s · "
          f"median agent {summary['agent_elapsed_s']['median']}s", flush=True)
    if summary["hop_depth"]:
        hd = summary["hop_depth"]
        print(f"  hop depth: median {hd['median']} (min {hd['min']}, max {hd['max']}) · "
              f"{hd['hard_breaks']} hard breaks · {hd['hit_cap']} hit cap", flush=True)
    if summary["agents_unscored"]:
        print(f"  !! {summary['agents_unscored']} agent(s) produced no readable "
              f"results — investigate before trusting this point", flush=True)

    (results_root / "swarm-summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="served-model-name on the endpoint (e.g. qwen3-235b)")
    ap.add_argument("--agents", type=int, help="single concurrency point")
    ap.add_argument("--sweep", help="comma-separated agent counts, e.g. 1,4,8,16,32,64")
    ap.add_argument("--endpoint",
                    default=os.getenv("BENCH_BASE_URL",
                                      os.getenv("OLLAMA_HOST", "http://localhost:11434")),
                    help="OpenAI-compatible base URL; /v1 is appended by the harness")
    ap.add_argument("--machine", default="gb300", help="machine tag for results paths")
    ap.add_argument("--out", default="./results/swarm", help="output root")
    ap.add_argument("--tests", help="restrict to specific test IDs (e.g. T2,T10)")
    ap.add_argument("--skip-hop", action="store_true",
                    help="skip T-HOP; it is long and dominates wall clock at high N")
    args = ap.parse_args()

    if not args.agents and not args.sweep:
        ap.error("give --agents N or --sweep 1,4,8,...")

    counts = ([int(x) for x in args.sweep.split(",")] if args.sweep else [args.agents])

    extra: list[str] = []
    if args.tests:
        extra += ["--tests", args.tests]
    if args.skip_hop:
        extra.append("--skip-hop")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(args.out) / f"{stamp}-{args.machine}"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"endpoint : {args.endpoint}")
    print(f"model    : {args.model}")
    print(f"sweep    : {counts}")
    print(f"output   : {out_root}")

    summaries = [run_swarm(n, args.model, args.endpoint, args.machine, out_root, extra)
                 for n in counts]

    print(f"\n{'=' * 60}\n  SWEEP SUMMARY\n{'=' * 60}")
    print(f"{'agents':>7} {'completed':>10} {'pass rate':>10} {'wall s':>9} "
          f"{'median s':>9} {'sess/min':>9}")
    for s in summaries:
        print(f"{s['agents']:>7} {s['agents_completed']:>10} "
              f"{str(s['pass_rate']):>10} {s['wall_clock_s']:>9} "
              f"{str(s['agent_elapsed_s']['median']):>9} {str(s['sessions_per_min']):>9}")

    (out_root / "sweep-summary.json").write_text(json.dumps(
        [{k: v for k, v in s.items() if k != "runs"} for s in summaries], indent=2))
    print(f"\nwritten: {out_root}/sweep-summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
