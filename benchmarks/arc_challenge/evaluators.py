"""
Real ARC-AGI evaluators: the official scoring semantics (pixel-perfect match,
pass@2) as used by the ARC Prize leaderboard and OpenEvolve's arc_benchmark
evaluator. These produce the binary/task-level signals a real ARC submission
is judged on. Partial-credit signals live in `partial_evaluators.py`.
"""

import time
from typing import Any, Dict, Optional

from core.agent import Agent
from evaluation.base import BaseEvaluator, EvaluatorResult

from .scoring import score_arc_cases


def _strip_outputs(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in metrics.items() if k != "outputs"}


class ArcRunsSuccessfullyEvaluator(BaseEvaluator):
    """arc_runs_successfully: OpenEvolve's `runs_successfully` — program loads and exposes
    `solve` or `transform_grid_attempt_1/2` and every attempt returns without raising ($0.000)."""

    def __init__(self):
        super().__init__(name="arc_runs_successfully", cost=0.000, tier="core")

    def evaluate(
        self,
        agent: Agent,
        task: Dict[str, Any],
        agent_output: str,
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> EvaluatorResult:
        start = time.perf_counter()
        metrics = score_arc_cases(task.get("test_cases", []), agent_output, task.get("entry_point", "solve"))
        score = metrics["runs_successfully"] if not metrics.get("errors") else 0.0
        return EvaluatorResult(
            evaluator_name=self.name,
            score=float(score),
            cost=self.cost,
            execution_time_ms=(time.perf_counter() - start) * 1000.0,
            details={"error": metrics.get("error"), "errors": metrics.get("errors")},
        )


class ArcPassAt2Evaluator(BaseEvaluator):
    """arc_pass_at_2_train: mean pass@2 over the demonstration (train) pairs —
    OpenEvolve's `combined_score` during evolution ($0.001)."""

    def __init__(self):
        super().__init__(name="arc_pass_at_2_train", cost=0.000, tier="core")

    def evaluate(
        self,
        agent: Agent,
        task: Dict[str, Any],
        agent_output: str,
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> EvaluatorResult:
        start = time.perf_counter()
        metrics = score_arc_cases(
            task.get("test_cases", []), agent_output, task.get("entry_point", "solve"), prefix="train_example"
        )
        return EvaluatorResult(
            evaluator_name=self.name,
            score=round(float(metrics["combined_score"]), 4),
            cost=self.cost,
            execution_time_ms=(time.perf_counter() - start) * 1000.0,
            details=_strip_outputs(metrics),
        )


class ArcHeldOutPassAt2Evaluator(BaseEvaluator):
    """arc_pass_at_2_test: mean pass@2 over the held-out (test) pairs — the official ARC
    task score / OpenEvolve's post-evolution evaluation."""

    def __init__(self):
        super().__init__(name="arc_pass_at_2_test", cost=0.000, tier="core")

    def evaluate(
        self,
        agent: Agent,
        task: Dict[str, Any],
        agent_output: str,
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> EvaluatorResult:
        start = time.perf_counter()
        metrics = score_arc_cases(
            task.get("edge_cases", []), agent_output, task.get("entry_point", "solve"), prefix="test_example"
        )
        return EvaluatorResult(
            evaluator_name=self.name,
            score=round(float(metrics["combined_score"]), 4),
            cost=self.cost,
            execution_time_ms=(time.perf_counter() - start) * 1000.0,
            details=_strip_outputs(metrics),
        )


REAL_ARC_EVALUATORS = (ArcRunsSuccessfullyEvaluator, ArcPassAt2Evaluator, ArcHeldOutPassAt2Evaluator)
