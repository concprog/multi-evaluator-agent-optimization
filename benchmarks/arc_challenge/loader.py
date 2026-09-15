"""
Loads ARC-AGI tasks and adapts them to the project's BenchmarkTask structure.

Data layout (Kaggle arc-prize-20xx / OpenEvolve arc_benchmark): one pair of
multi-task files per split under a data root:

    arc-agi_{split}_challenges.json  {task_id: {"train": [{input, output}, ...],
                                                "test":  [{input}, ...]}}
    arc-agi_{split}_solutions.json   {task_id: [output_grid, ...]}   # test outputs

A Grid is a rectangular list-of-lists of ints 0-9, from 1x1 up to 30x30.
The bundled root `benchmarks/arc_challenge/data/` holds a small sample in this
exact layout; point ARC_DATA_ROOT at a full Kaggle download to use all tasks and
ARC_TASK_FILE at a single split (training | evaluation | test) to restrict it.

Mapping onto BenchmarkTask:
    train pairs -> test_cases  (mu_1 Functional Correctness: fit the demonstrations)
    test  pairs -> edge_cases  (mu_5 Edge-Case Stress: generalise to the held-out input)
    entry_point -> solve(grid: list[list[int]]) -> list[list[int]]
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from benchmarks.benchmark_tasks import BenchmarkTask

Grid = List[List[int]]

ARC_ID_PREFIX = "arc_"
# The whole loaded ARC set run as ONE benchmark (every task per generation, scores averaged).
ARC_BENCHMARK_ID = "arc_benchmark"
ARC_BENCHMARK_NAME = "ARC-AGI Benchmark (all loaded tasks)"
ARC_SPLITS = ("training", "evaluation", "test")
DEFAULT_DATA_ROOT = Path(__file__).parent / "data"
ARC_DATA_ROOT = Path(os.environ.get("ARC_DATA_ROOT", DEFAULT_DATA_ROOT))
ARC_TASK_FILE: Optional[str] = os.environ.get("ARC_TASK_FILE") or None  # None = every split present
_limit_env = os.environ.get("ARC_TASK_LIMIT")
ARC_TASK_LIMIT: Optional[int] = int(_limit_env) if _limit_env else None  # cap tasks per split (cost control)

# Human-readable labels for the bundled tasks (ARC-AGI-1 public data).
ARC_TASK_LABELS: Dict[str, str] = {
    "00576224": "Tile 2x2 into 6x6 with alternating mirrored rows",
    "25ff71a9": "Shift pattern down by one row",
    "3c9b0459": "Rotate grid 180 degrees",
    "c8f0f002": "Recolour 7 to 5",
}


def render_grid(grid: Grid) -> str:
    """Renders a grid as rows of digits with no separators, e.g. '86\\n64'.

    Cells are single digits so no separator is needed, and it matters for cost: the o200k tokenizer
    (gpt-oss) packs runs of up to 3 digits into one token but spends a token per cell on '8 6 0 4'.
    On the ARC-AGI evaluation split this cuts the median prompt from ~5k to ~1.1k tokens (max 16k -> 3.5k),
    which is what makes every task fit under Groq's 8k-TPM on-demand tier.
    """
    return "\n".join("".join(str(v) for v in row) for row in grid)


def _grid_shape(grid: Grid) -> str:
    return f"{len(grid)}x{len(grid[0]) if grid else 0}"


def _build_description(task_id: str, train: List[Dict[str, Grid]], test: List[Dict[str, Grid]]) -> str:
    # NOTE: wording deliberately avoids keywords the offline mock client matches on
    # ("target", "differ", "stock", "fib", ...) so ARC prompts route to the ARC mock branch.
    lines = [
        f"ARC-AGI task {task_id}. Write a Python function `solve(grid: list[list[int]]) -> list[list[int]]` "
        "that applies the single hidden transformation shown by the demonstration pairs below to any new input grid. "
        "Grids are lists of rows of ints 0-9 (0 = background); below each row is written as a string of digits, "
        "one digit per cell. The returned grid must match the expected output exactly in shape and every cell. "
        "Return a plain list of lists (not a numpy array).",
        "",
    ]
    for i, pair in enumerate(train):
        lines.append(f"Demonstration {i + 1} input ({_grid_shape(pair['input'])}):")
        lines.append(render_grid(pair["input"]))
        lines.append(f"Demonstration {i + 1} output ({_grid_shape(pair['output'])}):")
        lines.append(render_grid(pair["output"]))
        lines.append("")
    for i, pair in enumerate(test):
        lines.append(f"Held-out input {i + 1} ({_grid_shape(pair['input'])}):")
        lines.append(render_grid(pair["input"]))
        lines.append("")
    return "\n".join(lines).rstrip()


def arc_task_to_benchmark(
    arc_id: str,
    data: Dict[str, Any],
    source: str = "arc-agi-1",
    test_solutions: Optional[List[Grid]] = None,
) -> BenchmarkTask:
    """Converts a raw ARC task dict into a BenchmarkTask.

    `test_solutions` supplies held-out outputs when they live in a separate
    solutions file (Kaggle layout). Test pairs without an output are skipped
    from edge_cases since they cannot be scored.
    """
    train = data.get("train", [])
    test = list(data.get("test", []))
    if test_solutions:
        test = [
            {**pair, "output": test_solutions[i]}
            for i, pair in enumerate(test)
            if i < len(test_solutions)
        ]

    test_cases = [{"args": (pair["input"],), "expected": pair["output"]} for pair in train]
    edge_cases = [
        {"args": (pair["input"],), "expected": pair["output"]}
        for pair in test
        if "output" in pair
    ]

    label = ARC_TASK_LABELS.get(arc_id, "Grid transformation")
    return BenchmarkTask(
        id=f"{ARC_ID_PREFIX}{arc_id}",
        name=f"ARC {arc_id}: {label}",
        category=f"ARC-AGI Abstract Reasoning ({source})",
        description=_build_description(arc_id, train, test),
        entry_point="solve",
        test_cases=test_cases,
        edge_cases=edge_cases,
        constraints=(
            "Output must be pixel-perfect: identical shape and cell values (ints 0-9) to the expected grid. "
            "Return list[list[int]]. Must generalise from the demonstration pairs to the held-out input; "
            "no hard-coding of the expected outputs. Pure Python / numpy only."
        ),
    )


def load_arc_task_file(path: os.PathLike, source: str = "arc-agi-1") -> BenchmarkTask:
    """Loads one `<task_id>.json` file (fchollet/ARC-AGI layout)."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return arc_task_to_benchmark(path.stem, data, source=source)


def load_kaggle_split(
    data_root: os.PathLike,
    split: str = "evaluation",
    limit: Optional[int] = None,
) -> List[BenchmarkTask]:
    """Loads one split from `arc-agi_{split}_challenges.json` (+ optional `_solutions.json`).

    Task order follows the challenges file, matching the numbering on
    https://arcprize.org/tasks/ (and OpenEvolve's TASK_NUM index).
    """
    data_root = Path(data_root)
    with open(data_root / f"arc-agi_{split}_challenges.json", "r", encoding="utf-8") as f:
        challenges: Dict[str, Any] = json.load(f)

    solutions: Dict[str, List[Grid]] = {}
    sol_path = data_root / f"arc-agi_{split}_solutions.json"
    if sol_path.exists():
        with open(sol_path, "r", encoding="utf-8") as f:
            solutions = json.load(f)

    tasks = []
    for i, (arc_id, data) in enumerate(challenges.items()):
        if limit is not None and i >= limit:
            break
        tasks.append(arc_task_to_benchmark(arc_id, data, source=f"arc-prize {split}", test_solutions=solutions.get(arc_id)))
    return tasks


def available_splits(data_root: os.PathLike = ARC_DATA_ROOT) -> List[str]:
    """Splits for which a challenges file exists under data_root, in canonical order."""
    root = Path(data_root)
    return [s for s in ARC_SPLITS if (root / f"arc-agi_{s}_challenges.json").exists()]


def load_arc_tasks(
    data_root: os.PathLike = ARC_DATA_ROOT,
    splits: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> List[BenchmarkTask]:
    """Loads every task from the given splits (default: all splits present under data_root)."""
    tasks: List[BenchmarkTask] = []
    for split in splits or available_splits(data_root):
        tasks.extend(load_kaggle_split(data_root, split, limit=limit))
    return tasks


ARC_TASKS: List[BenchmarkTask] = load_arc_tasks(
    ARC_DATA_ROOT, splits=[ARC_TASK_FILE] if ARC_TASK_FILE else None, limit=ARC_TASK_LIMIT
)


def is_arc_benchmark(task_id: Optional[str]) -> bool:
    """True for the whole-benchmark selector (by id or display name)."""
    return task_id in (ARC_BENCHMARK_ID, ARC_BENCHMARK_NAME)


def get_arc_task(task_id: str) -> Optional[BenchmarkTask]:
    """Looks up a loaded ARC task by BenchmarkTask id, raw ARC id, or display name."""
    for t in ARC_TASKS:
        if task_id in (t.id, t.name) or t.id == f"{ARC_ID_PREFIX}{task_id}":
            return t
    return None
