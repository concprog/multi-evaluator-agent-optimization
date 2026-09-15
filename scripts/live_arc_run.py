"""Headless run of the whole ARC-AGI benchmark against the live Groq API.

Does what "Run N Gen" does in the Streamlit UI with the ARC benchmark + ARC evaluator suite
selected, but prints everything so a run can be inspected: per-generation mutation, active
evaluator subset, fitness, dF, cost, averaged metrics and per-task pass@2, then the best
agent's program for every task.

    uv run python scripts/live_arc_run.py                 # 10 generations, openai/gpt-oss-120b, 30 RPM / 8k TPM
    uv run python scripts/live_arc_run.py --generations 5 --strategy no_pruning --tpm 30000 --model openai/gpt-oss-20b

Rate limiting: `--rpm` and `--tpm` are sliding-window caps on calls and on tokens (prompt, counted with
the gpt-oss tokenizer, + Groq's fixed ~500-token reply estimate) per minute; every call - agent, mutator,
LLM judge - goes through them. Set --tpm to your account's limit (console.groq.com/settings/limits;
on-demand tier: 8000 TPM and a 200k tokens/DAY cap on gpt-oss-120b, which is the real ceiling - a full
120-task generation is ~2 days of quota). `--delay` is a floor between calls. If Groq still 429s,
GroqLLMClient sleeps for the exact "try again in Ns" it reports; a 413 (single request over TPM) fails fast.

Reasoning models: `--reasoning-effort low` (default) and `--max-tokens 3000` are what make gpt-oss answer
at all - at default effort it burns a 1024-token budget on hidden reasoning and returns empty content.
qwen/qwen3.8-27b also accepts `--reasoning-effort none`, but its on-demand tier caps output at 1000
tokens/min so it truncates mid-answer on ARC prompts; not usable there without a paid tier.

Reads GROQ_API_KEY from .env (falls back to the deterministic mock if unset). Two CSVs are
written (both git-ignored via *_results.csv), rewritten after every generation:

    --csv       live_arc_results.csv       archive export, one row per candidate (averaged metrics)
    --task-csv  live_arc_task_results.csv  one row per (generation, task): every metric + solved flag

The ARC pool is flat (every evaluator core tier, $0 cost - none of them calls an LLM), so no strategy
prunes `arc_pass_at_2_test` and it is present on every row. `arc_pass_at_2_test_reported` is the same
score recomputed out-of-band as a cross-check (disable with --no-report-test); --deep-threshold /
--explore-prob only matter if LLM judges are added to the pool.

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

import tiktoken

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.arc_challenge import (  # noqa: E402
    ARC_BENCHMARK_ID,
    build_arc_evaluator_pool,
    get_arc_task,
    score_arc_cases,
)
from controller import EvolutionController  # noqa: E402
from core.groq_client import GroqLLMClient  # noqa: E402


CHARS_PER_TOKEN = 3.5  # fallback only, if the tokenizer cannot be loaded (first use downloads it)
PER_MESSAGE_OVERHEAD = 4  # harmony role/start/end tokens per chat message, plus a few for the reply header
_encoder = None


def _get_encoder():
    """gpt-oss uses the o200k_harmony tokenizer, shipped with tiktoken>=0.12."""
    global _encoder
    if _encoder is None:
        try:
            _encoder = tiktoken.get_encoding("o200k_harmony")
        except Exception as e:  # no network for the first download, or old tiktoken
            print(f"   [rate limit] tiktoken o200k_harmony unavailable ({e}); using {CHARS_PER_TOKEN} chars/token", flush=True)
            _encoder = False
    return _encoder or None


ADMISSION_COMPLETION_ESTIMATE = 500  # what Groq adds for the reply when admitting against TPM (seen: +430..470)


def estimate_tokens(system_prompt: str, user_prompt: str) -> int:
    """Tokens Groq charges against TPM at admission: prompt + a fixed reply estimate, NOT max_tokens.

    ("Requested 1194" for a 727-token prompt with max_tokens=2048; "Requested 9841" for ~9400 with 1024.)
    """
    enc = _get_encoder()
    if enc is None:
        prompt = int((len(system_prompt) + len(user_prompt)) / CHARS_PER_TOKEN)
    else:
        prompt = len(enc.encode(system_prompt)) + len(enc.encode(user_prompt)) + 2 * PER_MESSAGE_OVERHEAD + 3
    return prompt + ADMISSION_COMPLETION_ESTIMATE


class RateLimiter:
    """Blocks so that no more than `rpm` calls start and `tpm` tokens are requested in any rolling 60 s."""

    def __init__(self, rpm: int, tpm: int):
        self.rpm = rpm
        self.tpm = tpm
        self.log: deque = deque()  # (start_time, tokens)

    def _prune(self, now: float) -> None:
        while self.log and now - self.log[0][0] >= 60.0:
            self.log.popleft()

    def wait(self, tokens: int) -> None:
        while True:
            now = time.monotonic()
            self._prune(now)
            used = sum(t for _, t in self.log)
            over_rpm = self.rpm > 0 and len(self.log) >= self.rpm
            over_tpm = self.tpm > 0 and self.log and used + tokens > self.tpm
            if not (over_rpm or over_tpm):
                break
            pause = 60.0 - (now - self.log[0][0]) + 0.1
            why = f"{len(self.log)} calls" if over_rpm else f"{used}+{tokens} tokens > {self.tpm} TPM"
            print(f"   [rate limit] {why} in the last minute, sleeping {pause:.1f}s", flush=True)
            time.sleep(pause)
        if self.tpm > 0 and tokens > self.tpm:
            print(f"   [rate limit] single call needs ~{tokens} tokens > {self.tpm} TPM; it will likely 429", flush=True)
        self.log.append((time.monotonic(), tokens))


def install_rate_limiter(rpm: int, tpm: int) -> None:
    """Gates every live GroqLLMClient.generate() in this process (agent, mutator and judge share one client).

    Patched on the class because the gen-0 seed is evaluated inside EvolutionController.__init__, before
    any instance is reachable. Mock calls are not throttled.
    """
    limiter = RateLimiter(rpm, tpm)
    original = GroqLLMClient.generate

    def gated(self, system_prompt, user_prompt, *args, **kwargs):
        if not self.is_mock:
            limiter.wait(estimate_tokens(system_prompt, user_prompt))
        return original(self, system_prompt, user_prompt, *args, **kwargs)

    GroqLLMClient.generate = gated


def report_held_out(ctrl: EvolutionController, record: dict) -> dict:
    """Official held-out pass@2 per task for the record's agent, computed outside the evaluator loop."""
    agent = ctrl.archive.agents[record["child_id"]]
    out = {}
    for tid, program in agent.task_solutions.items():
        task = get_arc_task(tid)
        if task is None:
            continue
        m = score_arc_cases(task.edge_cases, program, task.entry_point, prefix="test_example")
        out[tid] = round(float(m["combined_score"]), 4)
    record["held_out_reported"] = out
    return out


def write_task_csv(path: str, records: list, metric_names: list, report_test: bool) -> None:
    """One row per (generation, task). Metrics pruned by the selective search are left blank."""
    fields = ["generation", "child_id", "mutation_type", "task_id", *metric_names]
    if report_test:
        fields.append("arc_pass_at_2_test_reported")
    fields.append("solved")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            reported = r.get("held_out_reported", {})
            for tid, m in r["task_metrics"].items():
                row = {"generation": r["generation"], "child_id": r["child_id"],
                       "mutation_type": r["mutation_type"], "task_id": tid}
                row.update({k: m[k] for k in metric_names if k in m})
                # solved uses the out-of-band score when available, else the in-loop one (blank if pruned)
                score = reported.get(tid, m.get("arc_pass_at_2_test"))
                if report_test:
                    row["arc_pass_at_2_test_reported"] = reported.get(tid, "")
                row["solved"] = int(score >= 1.0) if score is not None else ""
                w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Groq model id")
    parser.add_argument("--reasoning-effort", default="low",
                        help="gpt-oss: low|medium|high; qwen also accepts none. '' = don't send. At the default "
                             "effort gpt-oss-120b spends ~1020 tokens of hidden reasoning on an ARC prompt and "
                             "returns EMPTY content within a 1024 budget")
    parser.add_argument("--max-tokens", type=int, default=3000,
                        help="floor on max_tokens for every call (agent knobs default to 1024). Does not count "
                             "against TPM admission, but qwen's on-demand tier caps output at 1000/min")
    parser.add_argument("--rpm", type=int, default=30, help="max LLM calls per rolling minute (0 = no cap)")
    parser.add_argument("--tpm", type=int, default=8000,
                        help="max tokens (prompt estimate + max_tokens) requested per rolling minute; "
                             "Groq free tier gpt-oss-120b is 8000 (0 = no cap)")
    parser.add_argument("--strategy", default="full_adaptive",
                        choices=["full_adaptive", "no_pruning", "ucb1_bandit", "static_cascade", "single_metric"])
    parser.add_argument("--delay", type=float, default=1.5, help="minimum seconds between consecutive LLM calls")
    parser.add_argument("--archetype", default="General Balanced Assistant")
    parser.add_argument("--deep-threshold", type=float, default=0.65,
                        help="full_adaptive/static_cascade: core-preview score needed to run the deep tier "
                             "(arc_pass_at_2_test, arc_color_palette); 0.0 = always run it")
    parser.add_argument("--explore-prob", type=float, default=0.30,
                        help="full_adaptive/static_cascade: chance to run the deep tier anyway")
    parser.add_argument("--lambda-penalty", type=float, default=0.5,
                        help="lambda in fitness = sum(w*mu) - lambda*sum(cost)")
    parser.add_argument("--csv", default="live_arc_results.csv", help="archive export (one row per candidate)")
    parser.add_argument("--task-csv", default="live_arc_task_results.csv",
                        help="per-task metrics, one row per (generation, task)")
    parser.add_argument("--no-report-test", dest="report_test", action="store_false",
                        help="do not score held-out pairs out-of-band every generation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if args.rpm > 0 or args.tpm > 0:
        install_rate_limiter(args.rpm, args.tpm)

    ctrl = EvolutionController(
        selected_task_id=ARC_BENCHMARK_ID,
        evaluator_pool_factory=build_arc_evaluator_pool,
        model=args.model,
        inter_call_delay=args.delay,
        default_strategy=args.strategy,
        initial_archetype=args.archetype,
        lambda_penalty=args.lambda_penalty,
        reasoning_effort=args.reasoning_effort or None,
        min_max_tokens=args.max_tokens,
    )
    ctrl.csv_filepath = args.csv
    # The seed (gen 0) always runs every evaluator, so setting these after construction loses nothing.
    ctrl.sampler.deep_tier_threshold = args.deep_threshold
    ctrl.sampler.exploration_prob = args.explore_prob
    print("mock mode:", ctrl.llm_client.is_mock, "| model:", ctrl.llm_client.model,
          f"| reasoning_effort: {args.reasoning_effort or 'default'} | max_tokens>={args.max_tokens}",
          f"| rpm cap: {args.rpm or 'none'} | tpm cap: {args.tpm or 'none'} | delay: {args.delay}s",
          f"| strategy: {args.strategy} deep_threshold={args.deep_threshold} explore={args.explore_prob} "
          f"lambda={args.lambda_penalty}",
          "| tasks:", len(ctrl.history_records[0]["task_metrics"]), flush=True)

    seed = ctrl.history_records[0]
    if args.report_test:
        report_held_out(ctrl, seed)
    write_task_csv(args.task_csv, ctrl.history_records, ctrl.metric_names, args.report_test)
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
        if args.report_test:
            rep = report_held_out(ctrl, r)
            solved = sum(1 for v in rep.values() if v >= 1.0)
            print(f"   held-out (reported): {solved}/{len(rep)} solved = {solved / max(1, len(rep)):.4f}", flush=True)
        write_task_csv(args.task_csv, ctrl.history_records, ctrl.metric_names, args.report_test)

    best = ctrl.archive.get_best_agent()
    print("\nBEST:", best.id, f"fitness={best.fitness:.4f}", best.metrics, flush=True)
    print("telemetry:", json.dumps(ctrl.get_telemetry_summary(), indent=1, default=str), flush=True)
    print(f"\ncsv: {args.csv}  task csv: {args.task_csv}", flush=True)
    for tid, sol in best.task_solutions.items():
        print(f"\n----- {tid} -----\n{sol[:700]}", flush=True)


if __name__ == "__main__":
    main()
