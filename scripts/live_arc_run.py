"""Headless run of the whole ARC-AGI benchmark against the live Groq API.

Does what "Run N Gen" does in the Streamlit UI with the ARC benchmark + ARC evaluator suite
selected, but prints everything so a run can be inspected: per-generation mutation, active
evaluator subset, fitness, dF, cost, averaged metrics and per-task pass@2, then the best
agent's program for every task.

    uv run python scripts/live_arc_run.py                 # 10 generations, openai/gpt-oss-120b, 30 RPM
    uv run python scripts/live_arc_run.py --generations 5 --strategy no_pruning --rpm 20 --model openai/gpt-oss-20b

Rate limiting: `--rpm` is a hard sliding-window cap on LLM calls per minute (every call - agent, mutator,
LLM judge - goes through it), `--delay` a floor between consecutive calls. Groq's own 429 backoff in
GroqLLMClient still applies on top; the pool falls back to openai/gpt-oss-20b only if the model 404s.

Reads GROQ_API_KEY from .env (falls back to the deterministic mock if unset). Two CSVs are
written (both git-ignored via *_results.csv), rewritten after every generation:

    --csv       live_arc_results.csv       archive export, one row per candidate (averaged metrics)
    --task-csv  live_arc_task_results.csv  one row per (generation, task): every metric + solved flag

Point the loader at a full Kaggle download with env vars set before Python starts:

    ARC_DATA_ROOT=/path/arc-prize-2025 ARC_TASK_FILE=evaluation ARC_TASK_LIMIT=20 \
        uv run python scripts/live_arc_run.py --generations 3
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.arc_challenge import ARC_BENCHMARK_ID, build_arc_evaluator_pool  # noqa: E402
from controller import EvolutionController  # noqa: E402
from core.groq_client import GroqLLMClient  # noqa: E402


class RateLimiter:
    """Blocks so that at most `rpm` calls start within any rolling 60 s window."""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self.starts: deque = deque()

    def wait(self) -> None:
        now = time.monotonic()
        while self.starts and now - self.starts[0] >= 60.0:
            self.starts.popleft()
        if len(self.starts) >= self.rpm:
            pause = 60.0 - (now - self.starts[0])
            print(f"   [rate limit] {self.rpm} calls in the last minute, sleeping {pause:.1f}s", flush=True)
            time.sleep(pause)
            self.starts.popleft()
        self.starts.append(time.monotonic())


def install_rate_limiter(rpm: int) -> None:
    """Gates every live GroqLLMClient.generate() in this process (agent, mutator and judge share one client).

    Patched on the class because the gen-0 seed is evaluated inside EvolutionController.__init__, before
    any instance is reachable. Mock calls are not throttled.
    """
    limiter = RateLimiter(rpm)
    original = GroqLLMClient.generate

    def gated(self, *args, **kwargs):
        if not self.is_mock:
            limiter.wait()
        return original(self, *args, **kwargs)

    GroqLLMClient.generate = gated


def write_task_csv(path: str, records: list, metric_names: list) -> None:
    """One row per (generation, task). Metrics pruned by the selective search are left blank."""
    fields = ["generation", "child_id", "mutation_type", "task_id", *metric_names, "solved"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            for tid, m in r["task_metrics"].items():
                row = {"generation": r["generation"], "child_id": r["child_id"],
                       "mutation_type": r["mutation_type"], "task_id": tid}
                row.update({k: m[k] for k in metric_names if k in m})
                row["solved"] = int(m.get("arc_pass_at_2_test", 0.0) >= 1.0) if "arc_pass_at_2_test" in m else ""
                w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Groq model id")
    parser.add_argument("--rpm", type=int, default=30, help="max LLM calls per rolling minute (0 = no cap)")
    parser.add_argument("--strategy", default="full_adaptive",
                        choices=["full_adaptive", "no_pruning", "ucb1_bandit", "static_cascade", "single_metric"])
    parser.add_argument("--delay", type=float, default=1.5, help="minimum seconds between consecutive LLM calls")
    parser.add_argument("--archetype", default="General Balanced Assistant")
    parser.add_argument("--csv", default="live_arc_results.csv", help="archive export (one row per candidate)")
    parser.add_argument("--task-csv", default="live_arc_task_results.csv",
                        help="per-task metrics, one row per (generation, task)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if args.rpm > 0:
        install_rate_limiter(args.rpm)

    ctrl = EvolutionController(
        selected_task_id=ARC_BENCHMARK_ID,
        evaluator_pool_factory=build_arc_evaluator_pool,
        model=args.model,
        inter_call_delay=args.delay,
        default_strategy=args.strategy,
        initial_archetype=args.archetype,
    )
    ctrl.csv_filepath = args.csv
    print("mock mode:", ctrl.llm_client.is_mock, "| model:", ctrl.llm_client.model,
          f"| rpm cap: {args.rpm or 'none'} | delay: {args.delay}s",
          "| tasks:", len(ctrl.history_records[0]["task_metrics"]), flush=True)
    write_task_csv(args.task_csv, ctrl.history_records, ctrl.metric_names)

    seed = ctrl.history_records[0]
    print(f"gen 0 seed  fitness={seed['fitness']:.4f} metrics={seed['metrics']}", flush=True)
    print("   per task pass@2 test:", {k: v.get("arc_pass_at_2_test") for k, v in seed["task_metrics"].items()}, flush=True)

    for _ in range(args.generations):
        r = ctrl.run_generation()
        print(
            f"gen {r['generation']} {r['mutation_type']:<22} active={len(r['active_evaluators'])} "
            f"fitness={r['fitness']:.4f} dF={r['delta_f']:+.4f} cost=${r['cost_spent']:.4f} metrics={r['metrics']}",
            flush=True,
        )
        print("   per task pass@2 train:", {k: v.get("arc_pass_at_2_train") for k, v in r["task_metrics"].items()}, flush=True)
        write_task_csv(args.task_csv, ctrl.history_records, ctrl.metric_names)

    best = ctrl.archive.get_best_agent()
    print("\nBEST:", best.id, f"fitness={best.fitness:.4f}", best.metrics, flush=True)
    print("telemetry:", json.dumps(ctrl.get_telemetry_summary(), indent=1, default=str), flush=True)
    print(f"\ncsv: {args.csv}  task csv: {args.task_csv}", flush=True)
    for tid, sol in best.task_solutions.items():
        print(f"\n----- {tid} -----\n{sol[:700]}", flush=True)


if __name__ == "__main__":
    main()
