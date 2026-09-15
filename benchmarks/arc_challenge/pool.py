"""ARC-specific EvaluatorPool: 3 real (official-metric) evaluators + 3 AI-generated partial-credit ones."""

from typing import Optional

from core.groq_client import GroqLLMClient
from evaluation.metrics import LLMReasoningQualityEvaluator, LLMSafetyHallucinationEvaluator
from evaluation.pool import EvaluatorPool

from .evaluators import REAL_ARC_EVALUATORS
from .partial_evaluators import PARTIAL_ARC_EVALUATORS


def build_arc_evaluator_pool(
    llm_client: GroqLLMClient,
    include_partial: bool = True,
    include_llm_judges: bool = False,
) -> EvaluatorPool:
    """Builds the ARC evaluator suite.

    The pool is flat: every ARC evaluator is core tier at $0 cost, since each just executes `solve()`
    on grids (no LLM call) and there is nothing worth pruning - so every candidate is scored on the
    official held-out metric every generation and fitness reduces to sum(w * mu). Only the optional
    mu_4/mu_6 LLM judges are deep tier, because they do cost a call.
    Pass as `EvolutionController(evaluator_pool_factory=build_arc_evaluator_pool)`.
    """
    pool = EvaluatorPool(llm_client=llm_client, register_defaults=False)
    for cls in REAL_ARC_EVALUATORS:
        pool.register(cls())
    if include_partial:
        for cls in PARTIAL_ARC_EVALUATORS:
            pool.register(cls())
    if include_llm_judges:
        pool.register(LLMReasoningQualityEvaluator(llm_client))
        pool.register(LLMSafetyHallucinationEvaluator(llm_client))
    return pool
