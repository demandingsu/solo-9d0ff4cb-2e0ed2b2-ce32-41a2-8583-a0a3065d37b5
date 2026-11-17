"""Tests for candidate-element parsing and re-measurement (补测) plans."""

from __future__ import annotations

import sqlite3
from fractions import Fraction as F
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db import Store, StoreError
from app.inputs import (
    BaselineInput,
    CandidatesInput,
    InputValidationError,
    SamplesInput,
    parse_baseline,
    parse_candidates,
    parse_samples,
)
from app.main import create_app
from app.plans import PlanError, compute_plan
from app.solver import (
    difference_basis,
    optimal_set_extremes,
    separating_subsets,
    solve_sample,
)

# Baseline where ores A and B are indistinguishable: the optimum of a
# sample between "A/B" and C is a segment (w3 fixed, w1/w2 trade off).
DUP_BASE_CSV = """ore,Fe,SiO2,Al2O3,CaO
A,60,4,2,1
B,60,4,2,1
C,50,6,3,2
"""

# AMBIG sits exactly halfway between the A/B column and the C column, so
# the optimal set is the segment (t, 1/2 - t, 1/2); UNIQ is exactly ore C.
DUP_SAMPLES_CSV = """sample_id,Fe,SiO2,Al2O3,CaO
AMBIG,55,5,2.5,1.5
UNIQ,50,6,3,2
"""

# P separates A from B, Ti does not (same content on A and B).
DUP_CANDIDATES_CSV = """ore,P,Ti
A,1,3
B,0,3
C,5,1
"""

# All three ores identical: the optimal set is the whole simplex.
FLAT_BASE_CSV = """ore,Fe,SiO2,Al2O3,CaO
A,60,4,2,1
B,60,4,2,1
C,60,4,2,1
"""


def parse_base(csv_text: str) -> BaselineInput:
    return parse_baseline(csv_text.encode())


# ---------------------------------------------------------------------------
# Candidate CSV parsing
# ---------------------------------------------------------------------------


def test_parse_candidates_ok_and_reorders_ores():
    baseline = parse_base(DUP_BASE_CSV)
    raw = "ore,P,Ti\nC,5,1\nA,1,3\nB,0,3\n"
    parsed = parse_candidates(raw.encode(), baseline)
    assert parsed.elements == ("P", "Ti")
    assert parsed.ores == ("A", "B", "C")
    # Rows follow the baseline ore order, not the file order.
    assert parsed.rows[0] == (F(1), F(3))
    assert parsed.rows[1] == (F(0), F(3))
    assert parsed.rows[2] == (F(5), F(1))
    # Payload round-trip.
    again = CandidatesInput.from_payload(parsed.to_payload())
    assert again == parsed


def test_candidates_require_existing_baseline_names():
    baseline = parse_base(DUP_BASE_CSV)
    raw = "ore,P\nA,1\nB,2\nD,3\n"  # D instead of C
    with pytest.raises(InputValidationError, match="多出"):
        parse_candidates(raw.encode(), baseline)
    raw = "ore,P\nA,1\nB,2\nB,3\n"
    with pytest.raises(InputValidationError, match="重复"):
        parse_candidates(raw.encode(), baseline)


def test_candidates_reject_baseline_element_names():
    baseline = parse_base(DUP_BASE_CSV)
    raw = "ore,Fe,X\nA,1,1\nB,2,2\nC,3,3\n"
    with pytest.raises(InputValidationError, match="重复"):
        parse_candidates(raw.encode(), baseline)


def test_candidates_element_count_limits():
    baseline = parse_base(DUP_BASE_CSV)
    nine = "ore," + ",".join(f"E{i}" for i in range(9)) + "\n"
    nine += "A," + ",".join(["1"] * 9) + "\n"
    nine += "B," + ",".join(["2"] * 9) + "\n"
    nine += "C," + ",".join(["3"] * 9) + "\n"
    with pytest.raises(InputValidationError, match="1~8"):
        parse_candidates(nine.encode(), baseline)
    with pytest.raises(InputValidationError):
        parse_candidates("ore\nA\nB\nC\n".encode(), baseline)
    eight = "ore," + ",".join(f"E{i}" for i in range(8)) + "\n"
    eight += "A," + ",".join(["1"] * 8) + "\n"
    eight += "B," + ",".join(["2"] * 8) + "\n"
    eight += "C," + ",".join(["3"] * 8) + "\n"
    assert len(parse_candidates(eight.encode(), baseline).elements) == 8


def test_candidates_reject_bad_cells_and_row_counts():
    baseline = parse_base(DUP_BASE_CSV)
    with pytest.raises(InputValidationError):
        parse_candidates("ore,P\nA,-1\nB,2\nC,3\n".encode(), baseline)
    with pytest.raises(InputValidationError):
        parse_candidates("ore,P\nA,1\nB,2\n".encode(), baseline)
    with pytest.raises(InputValidationError):
        parse_candidates("ore,P\nA,1\nB,2\nC,3\nD,4\n".encode(), baseline)
    with pytest.raises(InputValidationError):
        parse_candidates("ore,P,Q\nA,1\nB,2\nC,3\n".encode(), baseline)


# ---------------------------------------------------------------------------
# Optimal-set extremes and separation (solver level)
# ---------------------------------------------------------------------------


def _dup_matrix():
    return [
        [F(60), F(60), F(50)],
        [F(4), F(4), F(6)],
        [F(2), F(2), F(3)],
        [F(1), F(1), F(2)],
    ]


def test_optimal_set_extremes_segment():
    _sse, extremes = optimal_set_extremes(_dup_matrix(), [F(55), F(5), F("2.5"), F("1.5")])
    assert sorted(extremes) == [
        (F(0), F(1, 2), F(1, 2)),
        (F(1, 2), F(0), F(1, 2)),
    ]
    basis = difference_basis(extremes)
    assert len(basis) == 1


def test_optimal_set_extremes_triangle_and_unique():
    flat = [[F(60)] * 3, [F(4)] * 3, [F(2)] * 3, [F(1)] * 3]
    _sse, extremes = optimal_set_extremes(flat, [F(55), F(5), F(2), F(1)])
    assert len(extremes) == 3
    assert len(difference_basis(extremes)) == 2
    # A unique optimum has exactly one extreme point.
    _sse, single = optimal_set_extremes(_dup_matrix(), [F(50), F(6), F(3), F(2)])
    assert single == [(F(0), F(0), F(1))]


def test_separating_subsets_segment():
    points = [(F(1, 2), F(0), F(1, 2)), (F(0), F(1, 2), F(1, 2))]
    rows = [[F(1), F(0), F(5)], [F(3), F(3), F(1)], [F(1), F(2), F(0)]]
    size, combos = separating_subsets(points, rows)
    # P (row 0) and the third row separate A from B; Ti (row 1) does not.
    assert size == 1
    assert set(combos) == {(0,), (2,)}


def test_separating_subsets_triangle_needs_two():
    points = [(F(1), F(0), F(0)), (F(0), F(1), F(0)), (F(0), F(0), F(1))]
    rows = [[F(1), F(0), F(0)], [F(0), F(1), F(0)], [F(5), F(5), F(5)]]
    size, combos = separating_subsets(points, rows)
    assert size == 2
    assert combos == [(0, 1)]


def test_separating_subsets_unsolvable():
    points = [(F(1, 2), F(0), F(1, 2)), (F(0), F(1, 2), F(1, 2))]
    rows = [[F(3), F(3), F(1)], [F(7), F(7), F(0)]]
    size, combos = separating_subsets(points, rows)
    assert size is None
    assert combos == []


def test_separating_subsets_requires_non_singleton():
    with pytest.raises(ValueError):
        separating_subsets([(F(1), F(0), F(0))], [[F(1), F(2), F(3)]])


def test_optimal_set_extremes_random_cross_check():
    """Every extreme attains the best SSE, and blends between extremes of
    a segment optimum attain it too (the set really is the whole segment)."""

    import random

    rng = random.Random(23)
    checked = 0
    for _trial in range(40):
        column = [F(str(round(rng.uniform(1, 100), 2))) for _ in range(4)]
        other = [F(str(round(rng.uniform(1, 100), 2))) for _ in range(4)]
        a = [[column[e], column[e], other[e]] for e in range(4)]
        y = [F(str(round(rng.uniform(1, 100), 2))) for _ in range(4)]
        best, extremes = optimal_set_extremes(a, y)
        assert 1 <= len(extremes) <= 2  # duplicate columns: point or segment

        def sse(w):
            return sum(
                (sum(a[e][j] * w[j] for j in range(3)) - y[e]) ** 2
                for e in range(4)
            )

        for point in extremes:
            assert sse(point) == best
            assert sum(point) == 1
            assert all(value >= 0 for value in point)
        if len(extremes) == 2:
            checked += 1
            p, q = extremes
            for num in (1, 1, 3):  # 1/4, 1/2, 3/4 along the segment
                mid = tuple(
                    (p[i] * (4 - num) + q[i] * num) / 4 for i in range(3)
                )
                assert sse(mid) == best
    assert checked > 0


def test_remeasured_sample_becomes_unique_at_true_blend():
    """End-to-end check of the planning assumption.

    Pick a true blend inside the current optimal set, produce the exact
    model-consistent re-measurement, and solve the augmented problem:
    the optimum must collapse to exactly that blend — for every optimal
    blend and every separating combination.
    """

    a = _dup_matrix()
    y = [F(55), F(5), F("2.5"), F("1.5")]
    _sse, extremes = optimal_set_extremes(a, y)
    rows = [[F(1), F(0), F(5)], [F(3), F(3), F(1)]]
    size, combos = separating_subsets(extremes, rows)
    assert size == 1 and combos == [(0,)]

    for t in (F(0), F(1, 4), F(1, 2)):
        true = (t, F(1, 2) - t, F(1, 2))
        z = sum(rows[0][j] * true[j] for j in range(3))
        augmented = a + [rows[0]]
        sol = solve_sample(augmented, y + [z])
        assert sol.unique
        assert sol.weights == true

    # A non-separating element cannot distinguish two different optima.
    for t in (F(0), F(1, 2)):
        true = (t, F(1, 2) - t, F(1, 2))
        z = sum(rows[1][j] * true[j] for j in range(3))
        sol = solve_sample(a + [rows[1]], y + [z])
        assert not sol.unique


# ---------------------------------------------------------------------------
# compute_plan
# ---------------------------------------------------------------------------


def _dup_inputs():
    baseline = parse_base(DUP_BASE_CSV)
    samples = parse_samples(DUP_SAMPLES_CSV.encode())
    candidates = parse_candidates(DUP_CANDIDATES_CSV.encode(), baseline)
    return baseline, samples, candidates


def test_compute_plan_segment_singleton():
    baseline, samples, candidates = _dup_inputs()
    payload = compute_plan(baseline, samples, candidates, "AMBIG")
    assert payload["solved"] is True
    assert payload["min_elements"] == 1
    assert payload["combos"] == [["P"]]
    assert payload["reason"] is None
    assert payload["optimal_dimension"] == 1
    assert len(payload["optimal_extremes"]) == 2
    # Every extreme is a genuine blend: non-negative and sums to 1.
    for point in payload["optimal_extremes"]:
        weights = [F(value) for value in point]
        assert sum(weights) == 1
        assert all(value >= 0 for value in weights)


def test_compute_plan_triangle_pair():
    baseline = parse_base(FLAT_BASE_CSV)
    samples = parse_samples(
        "sample_id,Fe,SiO2,Al2O3,CaO\nFLAT,58,4.5,2.2,1.1\n".encode()
    )
    candidates = parse_candidates(
        "ore,P,Q\nA,1,0\nB,0,1\nC,0,0\n".encode(), baseline
    )
    payload = compute_plan(baseline, samples, candidates, "FLAT")
    assert payload["solved"] is True
    assert payload["min_elements"] == 2
    assert payload["combos"] == [["P", "Q"]]
    assert payload["optimal_dimension"] == 2


def test_compute_plan_unsolvable_explains():
    baseline = parse_base(DUP_BASE_CSV)
    samples = parse_samples(DUP_SAMPLES_CSV.encode())
    candidates = parse_candidates("ore,Ti\nA,3\nB,3\nC,1\n".encode(), baseline)
    payload = compute_plan(baseline, samples, candidates, "AMBIG")
    assert payload["solved"] is False
    assert payload["min_elements"] is None
    assert payload["combos"] == []
    assert "即使补测全部候选元素" in payload["reason"]
    assert "Ti" in payload["reason"]


def test_compute_plan_rejects_unique_or_unknown_sample():
    baseline, samples, candidates = _dup_inputs()
    with pytest.raises(PlanError, match="无需补测"):
        compute_plan(baseline, samples, candidates, "UNIQ")
    with pytest.raises(PlanError, match="没有样本"):
        compute_plan(baseline, samples, candidates, "NOPE")


def test_compute_plan_rejects_ore_mismatch():
    baseline, samples, candidates = _dup_inputs()
    other = BaselineInput(
        elements=baseline.elements,
        ores=("A", "B", "D"),
        rows=baseline.rows,
    )
    with pytest.raises(PlanError, match="不一致"):
        compute_plan(other, samples, candidates, "AMBIG")


# ---------------------------------------------------------------------------
# Web application flow
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path: Path):
    app = create_app(str(tmp_path / "test.db"))
    with TestClient(app) as test_client:
        test_client.app_state = app.state  # type: ignore[attr-defined]
        yield test_client


def upload_all(client: TestClient, candidates: bool = True):
    files = {
        "baseline_file": ("b.csv", DUP_BASE_CSV.encode(), "text/csv"),
        "samples_file": ("s.csv", DUP_SAMPLES_CSV.encode(), "text/csv"),
    }
    if candidates:
        files["candidates_file"] = ("c.csv", DUP_CANDIDATES_CSV.encode(), "text/csv")
    return client.post("/inputs/upload", files=files, follow_redirects=False)


def make_plan(client: TestClient, sample_id: str = "AMBIG"):
    return client.post(
        "/plans/compute", data={"sample_id": sample_id}, follow_redirects=True
    )


def plans(client: TestClient) -> list[dict]:
    return client.get("/api/plans").json()["plans"]


def test_candidates_upload_requires_baseline(client: TestClient):
    response = client.post(
        "/inputs/upload",
        files={"candidates_file": ("c.csv", DUP_CANDIDATES_CSV.encode(), "text/csv")},
    )
    assert response.status_code == 400
    assert "需先上传基准" in response.text


def test_candidates_upload_ore_mismatch_keeps_state(client: TestClient):
    upload_all(client)
    store: Store = client.app_state.store
    before = store.current_inputs()["candidates"].version_id
    bad = DUP_CANDIDATES_CSV.replace("C,5,1", "D,5,1")
    response = client.post(
        "/inputs/upload",
        files={"candidates_file": ("c.csv", bad.encode(), "text/csv")},
    )
    assert response.status_code == 400
    assert "多出" in response.text
    assert store.current_inputs()["candidates"].version_id == before


def test_candidates_checked_against_new_baseline_in_same_upload(client: TestClient):
    """A baseline + candidates pair is validated together and fails atomically."""

    upload_all(client)
    store: Store = client.app_state.store
    before = {
        kind: value.version_id
        for kind, value in store.current_inputs().items()
        if value is not None
    }
    # The candidates match the *old* baseline but not the new one.
    new_base = DUP_BASE_CSV.replace("C,50,6,3,2", "D,50,6,3,2")
    response = client.post(
        "/inputs/upload",
        files={
            "baseline_file": ("b2.csv", new_base.encode(), "text/csv"),
            "candidates_file": ("c.csv", DUP_CANDIDATES_CSV.encode(), "text/csv"),
        },
    )
    assert response.status_code == 400
    assert "多出" in response.text
    after = {
        kind: value.version_id
        for kind, value in store.current_inputs().items()
        if value is not None
    }
    assert after == before
    # And a matching pair succeeds together.
    ok_candidates = DUP_CANDIDATES_CSV.replace("C,5,1", "D,5,1")
    response = client.post(
        "/inputs/upload",
        files={
            "baseline_file": ("b2.csv", new_base.encode(), "text/csv"),
            "candidates_file": ("c.csv", ok_candidates.encode(), "text/csv"),
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    current = store.current_inputs()
    assert current["baseline"].payload["ores"] == ["A", "B", "D"]
    assert current["candidates"].payload["ores"] == ["A", "B", "D"]


def test_candidates_export_roundtrip(client: TestClient):
    upload_all(client)
    response = client.get("/inputs/candidates/export.csv")
    assert response.status_code == 200
    text = response.content.decode("utf-8-sig")
    assert text.splitlines()[0] == "ore,P,Ti"
    assert text.splitlines()[1].startswith("A,1,3")


def test_candidates_upload_does_not_stale_report(client: TestClient):
    upload_all(client, candidates=False)
    client.post("/compute")
    assert client.get("/api/latest").json()["report"]["stale"] is False
    client.post(
        "/inputs/upload",
        files={"candidates_file": ("c.csv", DUP_CANDIDATES_CSV.encode(), "text/csv")},
    )
    # The report still matches its inputs; only plans would expire.
    assert client.get("/api/latest").json()["report"]["stale"] is False


def test_full_plan_flow(client: TestClient):
    upload_all(client)
    client.post("/compute")
    response = make_plan(client)
    assert response.status_code == 200
    assert "已保存" in response.text

    all_plans = plans(client)
    assert len(all_plans) == 1
    plan = all_plans[0]
    assert plan["expired"] is False
    assert plan["sample_id"] == "AMBIG"
    assert plan["payload"]["solved"] is True
    assert plan["payload"]["combos"] == [["P"]]

    html = client.get("/").text
    assert "补测计划" in html
    assert "计划 #1" in html
    assert "有效" in html
    assert "P" in html


def test_plan_requires_fresh_report_and_candidates(client: TestClient):
    upload_all(client, candidates=False)
    client.post("/compute")
    response = make_plan(client)
    assert "请先上传候选元素" in response.text

    # Replace the samples: the report becomes stale and planning stops.
    upload_all(client)
    client.post("/compute")
    changed = DUP_SAMPLES_CSV.replace("UNIQ,50,6,3,2", "UNIQ,50,6,3,2.1")
    client.post(
        "/inputs/upload",
        files={"samples_file": ("s2.csv", changed.encode(), "text/csv")},
    )
    response = make_plan(client)
    assert "已过期" in response.text
    assert plans(client) == []


def test_plan_rejects_unique_sample(client: TestClient):
    upload_all(client)
    client.post("/compute")
    response = make_plan(client, "UNIQ")
    assert "无需补测" in response.text
    assert plans(client) == []


def test_plan_unsolvable_is_saved_with_reason(client: TestClient):
    upload_all(client, candidates=False)
    client.post(
        "/inputs/upload",
        files={
            "candidates_file": (
                "c.csv",
                "ore,Ti\nA,3\nB,3\nC,1\n".encode(),
                "text/csv",
            )
        },
    )
    client.post("/compute")
    response = make_plan(client)
    assert response.status_code == 200
    assert "已保存" in response.text
    (plan,) = plans(client)
    assert plan["payload"]["solved"] is False
    assert "即使补测全部候选元素" in plan["payload"]["reason"]
    html = client.get("/").text
    assert "无法唯一确定" in html


def test_plan_expires_when_sources_change(client: TestClient):
    upload_all(client)
    client.post("/compute")
    make_plan(client)
    assert plans(client)[0]["expired"] is False

    # New candidates version -> expired but still listed.
    client.post(
        "/inputs/upload",
        files={"candidates_file": ("c2.csv", DUP_CANDIDATES_CSV.encode(), "text/csv")},
    )
    assert plans(client)[0]["expired"] is True
    assert "已过期" in client.get("/").text

    # Fresh plan, then a baseline replacement expires it too.
    make_plan(client)
    assert plans(client)[0]["expired"] is False
    client.post(
        "/inputs/upload",
        files={"baseline_file": ("b2.csv", DUP_BASE_CSV.encode(), "text/csv")},
    )
    assert all(plan["expired"] for plan in plans(client))

    # A recomputed report also expires plans bound to the older report.
    client.post("/compute")
    make_plan(client)
    assert plans(client)[0]["expired"] is False
    client.post("/compute")
    assert plans(client)[0]["expired"] is True


def test_plan_publish_race_is_rejected(tmp_path: Path):
    store = Store(str(tmp_path / "race.db"))
    baseline = parse_base(DUP_BASE_CSV)
    samples = parse_samples(DUP_SAMPLES_CSV.encode())
    candidates = parse_candidates(DUP_CANDIDATES_CSV.encode(), baseline)
    store.save_inputs(
        {
            "baseline": (baseline.to_payload(), "b.csv"),
            "samples": (samples.to_payload(), "s.csv"),
            "candidates": (candidates.to_payload(), "c.csv"),
        }
    )
    snapshot_inputs = store.snapshot()
    from app.compute import compute_report

    report = compute_report(
        BaselineInput.from_payload(snapshot_inputs.baseline.payload),
        SamplesInput.from_payload(snapshot_inputs.samples.payload),
    )
    assert store.publish_report(snapshot_inputs, report) is not None

    snapshot = store.plan_snapshot()
    assert snapshot is not None
    # A candidates upload landing mid-computation blocks the publish.
    store.save_input("candidates", candidates.to_payload(), "c2.csv")
    assert store.publish_plan(snapshot, "AMBIG", {"marker": 1}) is None
    assert store.list_plans() == []

    # So does a report published in between.
    snapshot = store.plan_snapshot()
    snapshot_inputs = store.snapshot()
    assert store.publish_report(snapshot_inputs, report) is not None
    assert store.publish_plan(snapshot, "AMBIG", {"marker": 1}) is None
    assert store.list_plans() == []


def test_plan_write_failure_keeps_existing_plans(client: TestClient, monkeypatch):
    upload_all(client)
    client.post("/compute")
    make_plan(client)
    assert len(plans(client)) == 1

    store: Store = client.app_state.store

    def broken_publish(*args, **kwargs):
        raise StoreError("磁盘只读（测试注入）")

    monkeypatch.setattr(store, "publish_plan", broken_publish)
    response = client.post(
        "/plans/compute", data={"sample_id": "AMBIG"}, follow_redirects=True
    )
    assert response.status_code == 200
    assert "写入失败" in response.text
    # The old plan is untouched and still shown.
    assert len(plans(client)) == 1
    assert "计划 #1" in response.text


def test_plan_endpoint_discards_when_sources_change_midway(
    client: TestClient, monkeypatch
):
    """An upload landing inside /plans/compute must prevent publishing."""

    import app.main as main_module
    from app.plans import compute_plan as real_compute_plan

    upload_all(client)
    client.post("/compute")
    store: Store = client.app_state.store

    def compute_and_upload(baseline, samples, candidates, sample_id):
        payload = real_compute_plan(baseline, samples, candidates, sample_id)
        store.save_input("candidates", candidates.to_payload(), "raced.csv")
        return payload

    monkeypatch.setattr(main_module, "compute_plan", compute_and_upload)
    response = client.post(
        "/plans/compute", data={"sample_id": "AMBIG"}, follow_redirects=True
    )
    assert response.status_code == 200
    assert "已丢弃" in response.text
    assert plans(client) == []


def test_old_database_is_migrated(tmp_path: Path):
    """Databases created before the candidates kind existed keep working."""

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE input_versions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            kind        TEXT    NOT NULL
                        CHECK (kind IN ('baseline', 'samples')),
            filename    TEXT,
            payload     TEXT    NOT NULL,
            created_at  TEXT    NOT NULL,
            is_current  INTEGER NOT NULL DEFAULT 0
                        CHECK (is_current IN (0, 1))
        );
        CREATE UNIQUE INDEX input_versions_one_current
        ON input_versions (kind) WHERE is_current = 1;
        CREATE TABLE reports (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            baseline_version  INTEGER NOT NULL,
            samples_version   INTEGER NOT NULL,
            payload           TEXT    NOT NULL,
            created_at        TEXT    NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO input_versions "
        "(kind, filename, payload, created_at, is_current) "
        "VALUES ('baseline', 'b.csv', '{\"keep\": 1}', '2026-01-01', 1)"
    )
    conn.commit()
    conn.close()

    store = Store(path)
    baseline = parse_base(DUP_BASE_CSV)
    candidates = parse_candidates(DUP_CANDIDATES_CSV.encode(), baseline)
    # The migrated table accepts the new kind and still holds the old row.
    store.save_input("candidates", candidates.to_payload(), "c.csv")
    current = store.current_inputs()
    assert current["baseline"].payload == {"keep": 1}
    assert current["candidates"].payload["elements"] == ["P", "Ti"]
    # The single-current-version invariant survives the rebuild.
    store.save_input("candidates", candidates.to_payload(), "c2.csv")
    with sqlite3.connect(path) as check:
        count = check.execute(
            "SELECT COUNT(*) FROM input_versions "
            "WHERE kind = 'candidates' AND is_current = 1"
        ).fetchone()[0]
    assert count == 1
