"""Re-measurement (补测) plans for samples whose optimum is not unique.

Given a non-unique sample from a fresh report, this module answers: which
candidate elements — at most two — should be re-measured so that the
sample's set of optimal blends collapses to a single one?

Assumption (stated on the page): the re-measured values are exact and
follow the original mixing model.  Formally, the sample's true blend
``w*`` is one of the current optimal blends Ω, and re-measuring element
``e`` returns exactly ``c_e · w*``.  The new objective then is

    f_new(w) = ||A w - y||² + Σ_e (c_e·w − c_e·w*)²
             = ||A w - y||² + Σ_e (c_e·(w − w*))² .

For ``w ∈ Ω`` the first term equals the old minimal SSE, so
``f_new(w) ≥ f_new(w*)`` with equality iff every re-measured element
agrees with ``w*``; for ``w ∉ Ω`` the first term is already strictly
larger.  Hence the new optimal set is ``{w ∈ Ω : c_e·w = c_e·w* ∀e}``,
which is the singleton ``{w*}`` exactly when the chosen elements separate
Ω — i.e. no two distinct optimal blends agree on all of them.  That
separation property is checked exactly, per candidate subset, by
:func:`app.solver.separating_subsets`.

The plan never fabricates a blend: the current optimal set is reported
through its extreme points (each of them a real attainable optimal
blend), never by splicing per-ore intervals into a recipe.
"""

from __future__ import annotations

from typing import Any

from .inputs import (
    BaselineInput,
    CandidatesInput,
    SamplesInput,
    fraction_to_text,
)
from .solver import (
    difference_basis,
    matrix_rank,
    optimal_set_extremes,
    separating_subsets,
)

MAX_PLAN_ELEMENTS = 2


class PlanError(ValueError):
    """A user-fixable problem with the requested re-measurement plan."""


def _direction_text(direction, ores) -> str:
    positive = [ores[i] for i, value in enumerate(direction) if value > 0]
    negative = [ores[i] for i, value in enumerate(direction) if value < 0]
    return (
        "（" + "、".join(negative) + "）与（" + "、".join(positive) + "）此消彼长"
    )


def compute_plan(
    baseline: BaselineInput,
    samples: SamplesInput,
    candidates: CandidatesInput,
    sample_id: str,
) -> dict[str, Any]:
    """Compute the re-measurement plan for one non-unique sample.

    The inputs must come from one consistent snapshot; the caller binds
    the resulting plan to their version ids.
    """

    if tuple(candidates.ores) != tuple(baseline.ores):
        raise PlanError(
            "候选元素文件的原矿名称与当前基准不一致，请重新上传候选元素 CSV。"
        )
    missing = [name for name in baseline.elements if name not in samples.elements]
    extra = [name for name in samples.elements if name not in baseline.elements]
    if missing or extra:
        raise PlanError("基准与成品的元素列不一致，无法重新定位该样本。")
    if sample_id not in samples.sample_ids:
        raise PlanError(f"成品输入中没有样本“{sample_id}”。")

    row_index = samples.sample_ids.index(sample_id)
    column = [samples.elements.index(name) for name in baseline.elements]
    y = [samples.rows[row_index][j] for j in column]
    a = [[baseline.rows[ore][e] for ore in range(3)] for e in range(4)]

    _best_sse, extremes = optimal_set_extremes(a, y)
    if len(extremes) < 2:
        raise PlanError(f"样本“{sample_id}”的最优占比本就唯一，无需补测。")

    basis = difference_basis(extremes)
    ores = list(baseline.ores)
    element_rows = [
        [candidates.rows[ore][e] for ore in range(3)]
        for e in range(len(candidates.elements))
    ]
    size, combos = separating_subsets(
        extremes, element_rows, max_size=MAX_PLAN_ELEMENTS
    )

    weight_ranges = [
        (
            fraction_to_text(min(point[i] for point in extremes)),
            fraction_to_text(max(point[i] for point in extremes)),
        )
        for i in range(3)
    ]

    payload: dict[str, Any] = {
        "sample_id": sample_id,
        "ores": ores,
        "baseline_elements": list(baseline.elements),
        "candidate_elements": list(candidates.elements),
        "optimal_dimension": len(basis),
        "optimal_extremes": [
            [fraction_to_text(value) for value in point] for point in extremes
        ],
        "weight_ranges": [[low, high] for low, high in weight_ranges],
        "max_elements": MAX_PLAN_ELEMENTS,
    }

    if size is not None:
        payload.update(
            solved=True,
            min_elements=size,
            combos=[
                [candidates.elements[i] for i in combo] for combo in combos
            ],
            reason=None,
        )
        return payload

    # No subset of at most MAX_PLAN_ELEMENTS separates the optimal set.
    # Separation is monotone in the subset, and the difference space here
    # is at most 2-dimensional, so if the full candidate set separated, a
    # subset of at most 2 would too.  Check anyway to explain precisely.
    full = [
        [sum(element_rows[e][j] * d[j] for j in range(3)) for d in basis]
        for e in range(len(element_rows))
    ]
    if matrix_rank(full) == len(basis):  # pragma: no cover - defensive
        reason = (
            f"需要同时补测超过 {MAX_PLAN_ELEMENTS} 项元素才能区分全部最优占比，"
            f"超出最多 {MAX_PLAN_ELEMENTS} 项的限制。"
        )
    else:
        directions = "；".join(_direction_text(d, ores) for d in basis)
        reason = (
            "即使补测全部候选元素（"
            + "、".join(candidates.elements)
            + "），它们在最优占比集合的差异方向"
            + directions
            + "上的含量差都为 0：沿这些方向变化的所有最优占比在每项候选元素"
            "上的取值完全相同，补测结果无法区分它们。需要更换或增加能区分"
            "这些占比的候选元素。"
        )
    payload.update(
        solved=False,
        min_elements=None,
        combos=[],
        reason=reason,
    )
    return payload
