import os
import logging
import re
from typing import Optional, List
from dotenv import load_dotenv
load_dotenv()

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

logger = logging.getLogger(__name__)

import time

# Budget used when a reasoning model returns empty content because max_tokens ran out mid-reasoning.
TRUNCATION_RETRY_MIN_TOKENS = 4096

# Tried in order after the configured model. Keep to models that currently exist on Groq: each dead entry
# costs a round-trip 404 on every rate-limited call (llama3-70b-8192, mixtral-8x7b-32768, gemma2-9b-it and
# llama-3.3-70b-versatile all returned 404/decommissioned in Sept 2026).
FALLBACK_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

# "try again in 12.3s", "850ms", "1m2.5s"
_RETRY_IN_RE = re.compile(r"try again in\s*((?:\d+(?:\.\d+)?(?:ms|s|m)\s*)+)", re.IGNORECASE)
_DURATION_PART_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m)", re.IGNORECASE)
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0}


def retry_after_seconds(error: Exception, max_wait: float = 120.0) -> Optional[float]:
    """Seconds Groq asked us to wait on a 429, from the retry-after header or the message text.

    Returns None if the error carries no hint. Capped at `max_wait`, plus a small margin so the
    retry lands after the window has actually rolled over.
    """
    wait: Optional[float] = None
    headers = getattr(getattr(error, "response", None), "headers", None)
    if headers:
        try:
            wait = float(headers.get("retry-after", ""))
        except (TypeError, ValueError):
            wait = None
    if wait is None:
        m = _RETRY_IN_RE.search(str(error))
        if m:
            wait = sum(float(v) * _UNIT_SECONDS[u.lower()] for v, u in _DURATION_PART_RE.findall(m.group(1)))
    if wait is None:
        return None
    return min(max_wait, wait + 0.5)


class GroqLLMClient:
    """Groq LLM Client supporting primary models, active fallbacks, rate limit backoff, and offline simulation mode."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-oss-120b",
        inter_call_delay: float = 1.5,
        is_mock: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        min_max_tokens: int = 0,
    ):
        # reasoning_effort ("low" | "medium" | "high") is sent for openai/gpt-oss-* only. Those models spend
        # max_tokens on hidden reasoning first: at the default effort a 1024-token budget is routinely used up
        # with EMPTY content (ARC prompts: ~1020 reasoning tokens, finish_reason="length"). "low" plus a
        # bigger floor (`min_max_tokens`, applied on top of the per-agent knob) gets a full answer in one call.
        self.reasoning_effort = reasoning_effort
        self.min_max_tokens = min_max_tokens
        if is_mock is not None:
            self.is_mock = is_mock
            self.api_key = api_key or ""
        else:
            self.api_key = api_key if api_key is not None else os.getenv("GROQ_API_KEY", "")
            self.is_mock = False

        self.model = model
        self.inter_call_delay = inter_call_delay
        self.client = None
        self.failed_calls = 0  # live calls that exhausted every model and returned ""

        if not self.is_mock and GROQ_AVAILABLE and self.api_key and self.api_key.strip():
            try:
                self.client = Groq(api_key=self.api_key.strip())
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}. Falling back to mock mode.")
                self.is_mock = True
        else:
            self.is_mock = True

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1024,
    ) -> str:
        """Generates response using Groq API with automatic TPM rate-limit retry backoff and fallbacks."""
        if self.is_mock or not self.client:
            return self._mock_generate(system_prompt, user_prompt, temperature)

        candidate_models = [self.model] + [m for m in FALLBACK_MODELS if m != self.model]
        last_error = None
        max_tokens = max(max_tokens, self.min_max_tokens)
        budget = max_tokens
        budget_raised = False
        too_large = False

        for model_name in candidate_models:
            if too_large:
                break
            # Attempt up to 3 retries with exponential backoff on TPM rate limits
            for attempt in range(3):
                try:
                    if self.inter_call_delay > 0:
                        time.sleep(self.inter_call_delay)

                    extra = {}
                    if self.reasoning_effort:
                        extra["reasoning_effort"] = self.reasoning_effort
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=max(0.0, min(2.0, temperature)),
                        top_p=max(0.0, min(1.0, top_p)),
                        max_tokens=budget,
                        **extra,
                    )
                    choice = response.choices[0]
                    content = choice.message.content or ""
                    # Reasoning models (openai/gpt-oss-*) spend the token budget on hidden reasoning first.
                    # A small max_tokens then yields finish_reason="length" with EMPTY content - indistinguishable
                    # from a broken answer downstream. Retry once with a larger budget instead of returning "".
                    if not content.strip() and choice.finish_reason == "length" and not budget_raised:
                        budget = max(TRUNCATION_RETRY_MIN_TOKENS, budget * 4)
                        budget_raised = True
                        logger.warning(
                            f"{model_name}: budget of {max_tokens} tokens exhausted by reasoning with no content. "
                            f"Retrying once with max_tokens={budget}."
                        )
                        continue
                    return content
                except Exception as e:
                    err_str = str(e).lower()
                    last_error = e
                    # Model rejects the reasoning_effort value (gpt-oss: low|medium|high; qwen also "none"):
                    # drop it for the rest of this process rather than retrying the same 400.
                    if "reasoning_effort" in err_str and self.reasoning_effort:
                        logger.warning(f"{model_name} rejected reasoning_effort={self.reasoning_effort!r}; sending without it. {e}")
                        self.reasoning_effort = None
                        continue
                    # 413: the single request exceeds the org's TPM limit. No wait or fallback model
                    # (same org limit) can fix that, so fail fast instead of burning 6 backoffs.
                    if "413" in err_str or "request too large" in err_str or "reduce your message size" in err_str:
                        logger.error(f"Request too large for the TPM limit on {model_name}; not retrying. {e}")
                        too_large = True
                        break
                    # Check for rate limit / TPM / RPM error
                    if "429" in err_str or "rate limit" in err_str or "tokens per minute" in err_str:
                        # Groq says how long the window needs ("Please try again in 12.3s" / retry-after
                        # header); waiting less than that just burns an attempt on a guaranteed 429.
                        backoff = retry_after_seconds(e) or (attempt + 1) * 3.5
                        logger.warning(f"Rate limit hit on {model_name} (attempt {attempt+1}/3). Backing off {backoff:.1f}s...")
                        time.sleep(backoff)
                        continue
                    elif "404" in err_str or "model_not_found" in err_str or "decommissioned" in err_str:
                        logger.warning(f"Model '{model_name}' inaccessible (404/decommissioned). Switching to next fallback.")
                        break  # Try next candidate model
                    else:
                        logger.warning(f"Generation error on '{model_name}': {e}. Retrying.")
                        time.sleep(1.0)

        # Never substitute a mock answer for a failed LIVE call: the mock knows the canonical and bundled ARC
        # solutions, so doing so would fake successes in real results. An empty answer scores 0 honestly.
        self.failed_calls += 1
        logger.error(f"All Groq models failed; returning empty answer (failed live calls so far: {self.failed_calls}). Error: {last_error}")
        return ""

    def _mock_generate(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        """High-quality deterministic mock generation for offline testing and demo mode."""
        lower_sys = system_prompt.lower()
        lower_user = user_prompt.lower()

        # 1. Check for Judge / Evaluation calls (mu_4 and mu_6)
        if "evaluate" in lower_sys or "judge" in lower_sys or "score" in lower_sys or "safety and hallucination" in lower_sys:
            return (
                "Score: 0.92\n"
                "Reasoning: The candidate solution demonstrates strong logical structure, correctly handles "
                "problem constraints, and produces verified algorithmic logic without hallucinated APIs."
            )

        # 2. Check for Prompt Mutation / Rewriting calls
        if "prompt" in lower_sys or "mutat" in lower_sys or "optimization goal" in lower_user or "rewrite" in lower_user:
            if "edge_case" in lower_user:
                return (
                    "You are a defensive AI software engineer. Before implementing, rigorously inspect and trap "
                    "all corner cases (empty collections, zero values, negative indices, and boundary overflows). "
                    "Ensure every edge case is handled cleanly with explicit validations."
                )
            elif "chain_of_thought" in lower_user:
                return (
                    "You are an analytical AI reasoning specialist. Solve problems by decomposing them into clear logical steps: "
                    "1) Analyze mathematical invariants and boundary conditions, 2) Design optimal data structures, "
                    "3) Synthesize verified, robust Python code."
                )
            elif "conciseness" in lower_user:
                return (
                    "You are a hyper-concise AI code synthesizer. Deliver pure, production-grade Python implementations. "
                    "Omit all conversational filler, greetings, comments, and postamble text. Return only optimal code."
                )
            elif "safety" in lower_user:
                return (
                    "You are a security and constraint-focused AI assistant. Strictly enforce input types, bounds, "
                    "and safety requirements. Prevent any hallucination of external dependencies and maintain deterministic behavior."
                )
            elif "algorithmic" in lower_user or "efficiency" in lower_user:
                return (
                    "You are an algorithmic optimization specialist. Prioritize optimal asymptotic complexity: "
                    "target O(N) or O(log N) runtime and O(1) auxiliary memory. Use optimal two-pointer, hashing, or dynamic programming techniques."
                )
            elif "retry" in lower_user or "robust" in lower_user:
                return (
                    "You are a fault-tolerant AI programming agent. Build defensive logic with explicit exception handling, "
                    "type guards, and boundary checks to guarantee 100% execution reliability."
                )
            else:
                return (
                    "You are an advanced AI coding specialist. Solve the problem with maximum functional accuracy, "
                    "defensive input validation, and optimal asymptotic complexity."
                )

        # 3. Check for Benchmark Task Execution (Specific matches first)
        # ARC-AGI grid tasks (benchmarks/arc_challenge) - matched before generic keywords
        # like "target" so grid prompts never fall into the coding-task branches.
        if "arc-agi task" in lower_user:
            return self._mock_arc_solution(lower_user)

        if "parenthes" in lower_user or "bracket" in lower_user or "task_valid_parentheses" in lower_user:
            return (
                "```python\n"
                "def solve(s: str) -> bool:\n"
                "    mapping = {')': '(', '}': '{', ']': '['}\n"
                "    stack = []\n"
                "    for char in s:\n"
                "        if char in mapping.values():\n"
                "            stack.append(char)\n"
                "        elif char in mapping:\n"
                "            if not stack or stack.pop() != mapping[char]:\n"
                "                return False\n"
                "    return len(stack) == 0\n"
                "```"
            )
        elif "he_000_has_close_elements" in lower_user or "close_elements" in lower_user or "has_close_elements" in lower_user:
            return (
                "```python\n"
                "def solve(numbers: list, threshold: float) -> bool:\n"
                "    for i in range(len(numbers)):\n"
                "        for j in range(i + 1, len(numbers)):\n"
                "            if abs(numbers[i] - numbers[j]) < threshold:\n"
                "                return True\n"
                "    return False\n"
                "```"
            )
        elif "he_003_below_zero" in lower_user or "below_zero" in lower_user or "bank account" in lower_user:
            return (
                "```python\n"
                "def solve(operations: list) -> bool:\n"
                "    balance = 0\n"
                "    for op in operations:\n"
                "        balance += op\n"
                "        if balance < 0:\n"
                "            return True\n"
                "    return False\n"
                "```"
            )
        elif "he_055_fib" in lower_user or "fibonacci" in lower_user or "fib" in lower_user or "fibonacci" in lower_sys:
            return (
                "```python\n"
                "def solve(n: int) -> int:\n"
                "    if n < 0:\n"
                "        raise ValueError('n must be non-negative')\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    a, b = 0, 1\n"
                "    for _ in range(2, n + 1):\n"
                "        a, b = b, a + b\n"
                "    return b\n"
                "```"
            )
        elif "mbpp_006_differ_bits" in lower_user or "differ_bits" in lower_user or "differ at one" in lower_user or "differ" in lower_user:
            return (
                "```python\n"
                "def solve(a: int, b: int) -> bool:\n"
                "    val = a ^ b\n"
                "    return val > 0 and (val & (val - 1)) == 0\n"
                "```"
            )
        elif "mbpp_274_stock_exchange" in lower_user or "stock" in lower_user or "buy & sell" in lower_user:
            return (
                "```python\n"
                "def solve(prices: list) -> int:\n"
                "    if not prices or len(prices) < 2:\n"
                "        return 0\n"
                "    min_price = float('inf')\n"
                "    max_profit = 0\n"
                "    for price in prices:\n"
                "        if price < min_price:\n"
                "            min_price = price\n"
                "        elif price - min_price > max_profit:\n"
                "            max_profit = price - min_price\n"
                "    return max_profit\n"
                "```"
            )
        elif "task_two_sum" in lower_user or "two_sum" in lower_user or "two sum" in lower_user or "target" in lower_user:
            return (
                "```python\n"
                "def solve(nums: list, target: int) -> list:\n"
                "    seen = {}\n"
                "    for i, num in enumerate(nums):\n"
                "        comp = target - num\n"
                "        if comp in seen:\n"
                "            return sorted([seen[comp], i])\n"
                "        seen[num] = i\n"
                "    return []\n"
                "```"
            )
        else:
            return (
                "```python\n"
                "def solve(*args, **kwargs):\n"
                "    return True\n"
                "```"
            )

    @staticmethod
    def _mock_arc_solution(lower_user: str) -> str:
        """Deterministic solutions for the bundled ARC-AGI tasks (offline demo mode)."""
        if "arc-agi task 00576224" in lower_user:
            body = (
                "def solve(grid):\n"
                "    flipped = [row[::-1] for row in grid]\n"
                "    out = []\n"
                "    for block in (grid, flipped, grid):\n"
                "        for row in block:\n"
                "            out.append(row * 3)\n"
                "    return out\n"
            )
        elif "arc-agi task 25ff71a9" in lower_user:
            body = (
                "def solve(grid):\n"
                "    width = len(grid[0])\n"
                "    return [[0] * width] + [list(row) for row in grid[:-1]]\n"
            )
        elif "arc-agi task 3c9b0459" in lower_user:
            body = (
                "def solve(grid):\n"
                "    return [row[::-1] for row in grid[::-1]]\n"
            )
        elif "arc-agi task c8f0f002" in lower_user:
            body = (
                "def solve(grid):\n"
                "    return [[5 if v == 7 else v for v in row] for row in grid]\n"
            )
        else:
            # Unknown ARC task: identity transform (valid grid, will not pass).
            body = (
                "def solve(grid):\n"
                "    return [list(row) for row in grid]\n"
            )
        return "```python\n" + body + "```"
