#!/usr/bin/env python3
"""
Summarize benchmark results across all models for a given machine.
Supports multi-run averaging: reads results/<machine>/<model>/run<N>/results.json
Falls back to results/<machine>/<model>/results.json for single-run results.

Usage:
  python3 scripts/summarize.py                    # default: dgx-spark
  python3 scripts/summarize.py --machine my-rig
  python3 scripts/summarize.py --all              # all machines side by side
"""

import argparse
import json
from pathlib import Path


def load_runs_for_model(model_dir: Path) -> list[dict]:
    """Load all runs for a model directory. Returns list of result dicts."""
    runs = []

    # Check for run subdirectories (run1, run2, run3...)
    run_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("run")])
    for run_dir in run_dirs:
        f = run_dir / "results.json"
        if f.exists():
            with open(f) as fp:
                runs.append(json.load(fp))

    # Fall back to flat results.json if no run dirs found
    if not runs:
        f = model_dir / "results.json"
        if f.exists():
            with open(f) as fp:
                runs.append(json.load(fp))

    return runs


def load_results(results_dir: Path, machine: str) -> dict:
    """Returns {model_name: [run1_data, run2_data, ...]}"""
    machine_dir = results_dir / machine
    if not machine_dir.exists():
        return {}
    models = {}
    for model_dir in sorted(machine_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        runs = load_runs_for_model(model_dir)
        if runs:
            model_name = runs[0].get("model", model_dir.name)
            models[model_name] = runs
    return models


def avg(vals):
    return round(sum(vals) / len(vals), 1) if vals else None


def summarize_machine(results_dir: Path, machine: str):
    models = load_results(results_dir, machine)

    if not models:
        print(f"No results found in {results_dir / machine}")
        return

    print(f"\n## Benchmark Results — {machine}\n")

    headers = ["Model", "Runs", "Pass (avg)", "Fail (avg)", "Flaky", "Hard Break Hop", "Avg Tok/s"]
    rows = []

    for model_name, runs in models.items():
        per_run_pass = []
        per_run_fail = []
        per_run_tok_s = []
        flaky_ids = set()

        # Track per-test pass/fail across runs to identify flaky tests
        test_results: dict[str, list] = {}

        for data in runs:
            tests = data.get("tests", [])
            t1_17 = [t for t in tests if t["id"] != "T-HOP"]

            per_run_pass.append(sum(1 for t in t1_17 if t.get("pass") is True))
            per_run_fail.append(sum(1 for t in t1_17 if t.get("pass") is False))

            tok_s_vals = [t["tok_per_s"] for t in t1_17 if t.get("tok_per_s", 0) > 0]
            if tok_s_vals:
                per_run_tok_s.append(sum(tok_s_vals) / len(tok_s_vals))

            for t in t1_17:
                test_results.setdefault(t["id"], []).append(t.get("pass"))

        # Flaky = not all runs agree on pass/fail
        for tid, outcomes in test_results.items():
            definite = [o for o in outcomes if o is not None]
            if definite and len(set(definite)) > 1:
                flaky_ids.add(tid)

        # Hop test — use last run's data
        last_data = runs[-1]
        last_tests = last_data.get("tests", [])
        t_hop = next((t for t in last_tests if t["id"] == "T-HOP"), None)
        if t_hop:
            hard_break = t_hop.get("hard_break_hop")
            max_hops = t_hop.get("max_hops_completed", "-")
            hit_cap = t_hop.get("hit_cap", False)
            if hit_cap:
                hop_str = f"{max_hops} (cap)"
            elif hard_break:
                hop_str = f"{max_hops} (break: {hard_break})"
            else:
                hop_str = str(max_hops)
        else:
            hop_str = "-"

        n_runs = len(runs)
        avg_pass = avg(per_run_pass)
        avg_fail = avg(per_run_fail)
        avg_tok_s = avg(per_run_tok_s) if per_run_tok_s else "-"
        flaky_str = ", ".join(sorted(flaky_ids)) if flaky_ids else "—"

        rows.append([
            f"`{model_name}`",
            str(n_runs),
            str(avg_pass),
            str(avg_fail),
            flaky_str,
            hop_str,
            str(avg_tok_s),
        ])

    _print_table(headers, rows)


def _print_table(headers, rows):
    col_widths = [max(len(str(cell)) for cell in [headers[i]] + [r[i] for r in rows]) for i in range(len(headers))]

    def row_str(row):
        return "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"

    print(row_str(headers))
    print("| " + " | ".join("-" * w for w in col_widths) + " |")
    for row in rows:
        print(row_str(row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize benchmark results")
    parser.add_argument("--machine", default="dgx-spark", help="Machine to summarize (e.g. dgx-spark, my-rig)")
    parser.add_argument("--all", action="store_true", help="Summarize all machines found in results dir")
    parser.add_argument("--results", default="./results", help="Path to results directory")
    args = parser.parse_args()

    results_dir = Path(args.results)

    if args.all:
        if results_dir.exists():
            for machine_dir in sorted(results_dir.iterdir()):
                if machine_dir.is_dir() and not machine_dir.name.startswith("."):
                    summarize_machine(results_dir, machine_dir.name)
        else:
            print(f"Results directory not found: {results_dir}")
    else:
        summarize_machine(results_dir, args.machine)
