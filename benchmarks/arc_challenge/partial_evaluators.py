# AI-GENERATED FILE: the evaluators below were written by Claude (Anthropic), not the ARC Prize.
# They are PARTIAL-CREDIT heuristics for shaping the evolutionary search on ARC grid tasks
# and are NOT part of the official ARC-AGI metric (which is pixel-perfect pass@2 only).
# See `evaluators.py` for the real metrics; treat these scores as diagnostics, never as results.
"""
Partial-credit ARC evaluators (AI-generated, see header). Each rewards a
different sub-skill of a grid transformation so the GP-weighted fitness has a
gradient before a candidate reaches an exact match.
"""

import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from core.agent import Agent
from evaluation.base import BaseEvaluator, EvaluatorResult

from .scoring import pixel_accuracy, score_arc_cases, to_grid

AI_GENERATED_HEADER = (
    "AI-GENERATED FILE: the evaluators below were written by Claude (Anthropic), not the ARC Prize.\n"
    "They are PARTIAL-CREDIT heuristics for shaping the evolutionary search on ARC grid tasks\n"
    "and are NOT part of the official ARC-AGI metric (which is pixel-perfect pass@2 only).\n"
    "See `evaluators.py` for the real metrics; treat these scores as diagnostics, never as results."
)


def _best_attempt_score(attempts: Sequence[Any], truth: Any, fn) -> float:
    """Applies `fn(pred, truth)` to each attempt and keeps the best (pass@2-style leniency)."""
    scores = [fn(a, truth) for a in attempts] or [0.0]
    return max(scores)


def _shape_match(pred: Any, truth: Any) -> float:
    g, t = to_grid(pred), to_grid(truth)
    if g is None or t is None:
        return 0.0
    return 1.0 if (len(g), len(g[0])) == (len(t), len(t[0])) else 0.0


def _palette_similarity(pred: Any, truth: Any) -> float:
    """Histogram intersection over colours 0-9, normalised by the truth cell count."""
    g, t = to_grid(pred), to_grid(truth)
    if g is None or t is None:
        return 0.0
    hg = Counter(v for row in g for v in row)
    ht = Counter(v for row in t for v in row)
    total = sum(ht.values())
    if total == 0:
        return 0.0
    return sum(min(hg[c], ht[c]) for c in ht) / total


class _PartialArcEvaluator(BaseEvaluator):
    """Shared plumbing: run the candidate on the demonstration pairs, score each with a heuristic."""

    heuristic = staticmethod(lambda pred, truth: 0.0)

    def evaluate(
        self,
        agent: Agent,
        task: Dict[str, Any],
        agent_output: str,
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> EvaluatorResult:
        start = time.perf_counter()
        cases: List[Dict[str, Any]] = task.get("test_cases", [])
        metrics = score_arc_cases(cases, agent_output, task.get("entry_point", "solve"))
        outputs = metrics.get("outputs", [])

        per_case = [
            _best_attempt_score(attempts, case["expected"], self.heuristic)
            for attempts, case in zip(outputs, cases)
        ]
        score = sum(per_case) / len(cases) if cases else 0.0
        return EvaluatorResult(
            evaluator_name=self.name,
            score=round(float(score), 4),
            cost=self.cost,
            execution_time_ms=(time.perf_counter() - start) * 1000.0,
            details={
                "ai_generated": True,
                "per_example": [round(s, 4) for s in per_case],
                "error": metrics.get("error"),
            },
        )


class ArcPixelAccuracyEvaluator(_PartialArcEvaluator):
    """arc_pixel_accuracy: mean fraction of correct cells over demonstration pairs (0 on shape mismatch) ($0.000)."""

    heuristic = staticmethod(pixel_accuracy)

    def __init__(self):
        super().__init__(name="arc_pixel_accuracy", cost=0.000, tier="core")


class ArcShapeMatchEvaluator(_PartialArcEvaluator):
    """arc_shape_match: fraction of demonstration outputs with the correct (rows, cols) ($0.000)."""

    heuristic = staticmethod(_shape_match)

    def __init__(self):
        super().__init__(name="arc_shape_match", cost=0.000, tier="core")


class ArcColorPaletteEvaluator(_PartialArcEvaluator):
    """arc_color_palette: colour-histogram intersection with the expected output, shape-agnostic ($0.002)."""

    heuristic = staticmethod(_palette_similarity)

    def __init__(self):
        super().__init__(name="arc_color_palette", cost=0.000, tier="core")


PARTIAL_ARC_EVALUATORS = (ArcPixelAccuracyEvaluator, ArcShapeMatchEvaluator, ArcColorPaletteEvaluator)
