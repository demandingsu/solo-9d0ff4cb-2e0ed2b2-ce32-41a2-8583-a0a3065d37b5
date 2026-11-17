"""FastAPI application for ore-blend proportion reconstruction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from . import formatting
from .compute import ComputeError, compute_report
from .db import PlanRecord, ReportRecord, Store, StoreError, VersionedInput
from .inputs import (
    MAX_UPLOAD_BYTES,
    BaselineInput,
    CandidatesInput,
    InputValidationError,
    SamplesInput,
    parse_baseline,
    parse_candidates,
    parse_samples,
)
from .plans import PlanError, compute_plan

BASE_DIR = Path(__file__).resolve().parent
REPORT_ROW_LIMIT = 500


def create_app(database_path: str | None = None) -> FastAPI:
    app = FastAPI(title="矿石配料反推与异常定位")
    store = Store(database_path or os.environ.get("ORE_DB_PATH", "data/app.db"))
    app.state.store = store

    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    templates.env.filters["decimal"] = formatting.decimal
    templates.env.filters["percent"] = formatting.percent
    templates.env.filters["signed"] = formatting.signed
    templates.env.filters["ratio"] = formatting.ratio_text
    templates.env.filters["exact"] = formatting.exact

    app.mount(
        "/static",
        StaticFiles(directory=str(BASE_DIR / "static")),
        name="static",
    )

    # -- helpers ---------------------------------------------------------

    def redirect_notice(
        message: str, kind: str = "ok", anchor: str = ""
    ) -> RedirectResponse:
        query = urlencode({"notice": message, "notice_kind": kind})
        url = f"/?{query}"
        if anchor:
            url += f"#{anchor}"
        return RedirectResponse(url, status_code=303)

    def payload_to_inputs(
        snapshot,
    ) -> tuple[BaselineInput, SamplesInput]:
        return (
            BaselineInput.from_payload(snapshot.baseline.payload),
            SamplesInput.from_payload(snapshot.samples.payload),
        )

    def input_summary(value: VersionedInput | None) -> dict[str, Any] | None:
        if value is None:
            return None
        if value.kind in ("baseline", "candidates"):
            parsed = (
                BaselineInput.from_payload(value.payload)
                if value.kind == "baseline"
                else CandidatesInput.from_payload(value.payload)
            )
            return {
                "version": value.version_id,
                "filename": value.filename,
                "created_at": value.created_at,
                "elements": list(parsed.elements),
                "names": list(parsed.ores),
                "rows": [
                    [formatting.decimal(v, 8) for v in row] for row in parsed.rows
                ],
                "count": len(parsed.rows),
            }
        parsed_samples = SamplesInput.from_payload(value.payload)
        return {
            "version": value.version_id,
            "filename": value.filename,
            "created_at": value.created_at,
            "elements": list(parsed_samples.elements),
            "names": list(parsed_samples.sample_ids[:5]),
            "rows": [],
            "count": len(parsed_samples.sample_ids),
        }

    def report_view(record: ReportRecord | None) -> dict[str, Any] | None:
        if record is None:
            return None
        payload = dict(record.payload)
        rows = payload.get("results", [])
        visible = rows[:REPORT_ROW_LIMIT]
        return {
            "id": record.report_id,
            "created_at": record.created_at,
            "baseline_version": record.baseline_version,
            "samples_version": record.samples_version,
            "stale": record.is_stale,
            "baseline_current": record.baseline_is_current,
            "samples_current": record.samples_is_current,
            "payload": payload,
            "rows": visible,
            "hidden_rows": max(0, len(rows) - len(visible)),
        }

    def plan_view(record: PlanRecord) -> dict[str, Any]:
        return {
            "id": record.plan_id,
            "report_id": record.report_id,
            "sample_id": record.sample_id,
            "baseline_version": record.baseline_version,
            "samples_version": record.samples_version,
            "candidates_version": record.candidates_version,
            "created_at": record.created_at,
            "expired": record.is_expired,
            "payload": record.payload,
        }

    def context(request: Request) -> dict[str, Any]:
        current = store.current_inputs()
        record = store.latest_report()
        plans = store.list_plans()

        candidates_ores_match = True
        if current.get("baseline") and current.get("candidates"):
            candidates_ores_match = (
                CandidatesInput.from_payload(current["candidates"].payload).ores
                == BaselineInput.from_payload(current["baseline"].payload).ores
            )

        plan_samples: list[str] = []
        if record is not None and not record.is_stale:
            plan_samples = [
                row["sample_id"]
                for row in record.payload.get("results", [])
                if not row.get("unique")
            ]

        return {
            "request": request,
            "current": {
                kind: input_summary(value) for kind, value in current.items()
            },
            "report": report_view(record),
            "plans": [plan_view(item) for item in plans],
            "plan_samples": plan_samples,
            "candidates_ores_match": candidates_ores_match,
            "notice": request.query_params.get("notice"),
            "notice_kind": request.query_params.get("notice_kind", "ok"),
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        }

    # -- pages -----------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Response:
        return templates.TemplateResponse(request, "index.html", context(request))

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/inputs/upload")
    async def upload_inputs(
        request: Request,
        baseline_file: UploadFile | None = File(None),
        samples_file: UploadFile | None = File(None),
        candidates_file: UploadFile | None = File(None),
    ) -> Response:
        provided = {
            "baseline": baseline_file,
            "samples": samples_file,
            "candidates": candidates_file,
        }
        if not any(file is not None and file.filename for file in provided.values()):
            return redirect_notice("请至少选择一个 CSV 文件再上传。", "error", "upload")

        parsed: dict[str, Any] = {}
        filenames: dict[str, str] = {}
        labels = {"baseline": "基准", "samples": "成品", "candidates": "候选元素"}
        try:
            raw_files: dict[str, bytes] = {}
            for kind, file in provided.items():
                if file is None or not file.filename:
                    continue
                raw = await file.read(MAX_UPLOAD_BYTES + 1)
                if len(raw) > MAX_UPLOAD_BYTES:
                    raise _LabeledError(
                        labels[kind],
                        f"文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB 上限",
                    )
                raw_files[kind] = raw
                filenames[kind] = (
                    file.filename.replace("\\", "/").rsplit("/", 1)[-1][:200]
                    or "未命名.csv"
                )

            for kind in ("baseline", "samples"):
                if kind not in raw_files:
                    continue
                parser = parse_baseline if kind == "baseline" else parse_samples
                try:
                    parsed[kind] = await run_in_threadpool(parser, raw_files[kind])
                except InputValidationError as exc:
                    raise _LabeledError(labels[kind], str(exc)) from exc

            if "candidates" in raw_files:
                # Ore names are validated against the baseline that will be
                # in effect: the freshly uploaded one if any, otherwise the
                # current one.  A later baseline replacement can still make
                # this file inconsistent; that is re-checked when a plan is
                # computed and surfaced as plan expiry.
                if "baseline" in parsed:
                    baseline_context = parsed["baseline"]
                else:
                    current_baseline = await run_in_threadpool(
                        lambda: store.current_inputs()["baseline"]
                    )
                    if current_baseline is None:
                        raise _LabeledError(
                            labels["candidates"],
                            "需先上传基准 CSV：候选文件的原矿名称须与基准一致",
                        )
                    baseline_context = BaselineInput.from_payload(
                        current_baseline.payload
                    )
                try:
                    parsed["candidates"] = await run_in_threadpool(
                        parse_candidates, raw_files["candidates"], baseline_context
                    )
                except InputValidationError as exc:
                    raise _LabeledError(labels["candidates"], str(exc)) from exc
        except _LabeledError as exc:
            return _render_with_error(request, f"{exc.label}文件：{exc.message}")

        updates: dict[str, tuple[dict[str, Any], str]] = {}
        for kind, value in parsed.items():
            updates[kind] = (value.to_payload(), filenames[kind])

        had_report = await run_in_threadpool(
            lambda: store.latest_report() is not None
        )
        had_plans = await run_in_threadpool(lambda: bool(store.list_plans(1)))
        try:
            versions = await run_in_threadpool(store.save_inputs, updates)
        except StoreError as exc:
            return _render_with_error(
                request, f"输入写入失败，原有输入未被替换：{exc}"
            )

        replaced = []
        if "baseline" in versions:
            replaced.append(f"基准 v{versions['baseline']}")
        if "samples" in versions:
            replaced.append(f"成品 v{versions['samples']}")
        if "candidates" in versions:
            replaced.append(f"候选元素 v{versions['candidates']}")
        notices = []
        if had_report and ("baseline" in versions or "samples" in versions):
            notices.append("旧报告已标记为过期")
        if had_plans:
            notices.append("相关补测计划已标记为过期")
        suffix = "。" + "；".join(notices) + "。" if notices else "。"
        return redirect_notice(
            "已整版替换：" + "，".join(replaced) + suffix,
            "ok",
            "report",
        )

    def _render_with_error(request: Request, message: str) -> HTMLResponse:
        ctx = context(request)
        ctx["notice"] = message
        ctx["notice_kind"] = "error"
        return templates.TemplateResponse(
            request, "index.html", ctx, status_code=400
        )

    @app.post("/compute")
    async def compute(request: Request) -> Response:
        snapshot = await run_in_threadpool(store.snapshot)
        if snapshot is None:
            return redirect_notice(
                "基准与成品输入都必须先上传，才能计算。", "error", "upload"
            )

        try:
            baseline, samples = payload_to_inputs(snapshot)
            payload = await run_in_threadpool(
                compute_report, baseline, samples
            )
        except (ComputeError, InputValidationError) as exc:
            return redirect_notice(str(exc), "error", "report")
        except Exception as exc:  # pragma: no cover - defensive
            return redirect_notice(
                f"计算失败，未产生新报告：{type(exc).__name__}: {exc}",
                "error",
                "report",
            )

        try:
            report_id = await run_in_threadpool(
                store.publish_report, snapshot, payload
            )
        except StoreError as exc:
            return redirect_notice(
                f"报告写入失败，仍显示上一份报告：{exc}", "error", "report"
            )

        if report_id is None:
            return redirect_notice(
                "计算期间输入已被更新，本次结果与当前输入不符，已丢弃，"
                "未发布报告。请基于新输入重新计算。",
                "error",
                "report",
            )

        return redirect_notice(
            f"报告 #{report_id} 已生成并发布（{payload['sample_count']} 个样本，"
            f"{payload['non_unique_count']} 个无法唯一确定）。",
            "ok",
            "report",
        )

    # -- re-measurement plans ---------------------------------------------

    @app.post("/plans/compute")
    async def compute_plan_endpoint(request: Request) -> Response:
        form = await request.form()
        sample_id = str(form.get("sample_id", "")).strip()
        if not sample_id:
            return redirect_notice(
                "请选择需要制定补测计划的样本。", "error", "plans"
            )

        snapshot = await run_in_threadpool(store.plan_snapshot)
        if snapshot is None:
            return redirect_notice(
                "需要先有报告及其对应的两类输入，才能制定补测计划。",
                "error",
                "plans",
            )
        report = snapshot.report
        if not (
            report.baseline_version == snapshot.baseline.version_id
            and report.samples_version == snapshot.samples.version_id
        ):
            return redirect_notice(
                "当前报告已过期（基准或成品输入已更新）。"
                "请重新计算占比后再制定补测计划。",
                "error",
                "plans",
            )
        if snapshot.candidates is None:
            return redirect_notice(
                "请先上传候选元素 CSV，再制定补测计划。", "error", "plans"
            )

        results = {
            row["sample_id"]: row
            for row in report.payload.get("results", [])
        }
        entry = results.get(sample_id)
        if entry is None:
            return redirect_notice(
                f"报告 #{report.report_id} 中没有样本“{sample_id}”。",
                "error",
                "plans",
            )
        if entry.get("unique"):
            return redirect_notice(
                f"样本“{sample_id}”的最优占比已唯一，无需补测。",
                "error",
                "plans",
            )

        try:
            baseline = BaselineInput.from_payload(snapshot.baseline.payload)
            samples = SamplesInput.from_payload(snapshot.samples.payload)
            candidates = CandidatesInput.from_payload(
                snapshot.candidates.payload
            )
            payload = await run_in_threadpool(
                compute_plan, baseline, samples, candidates, sample_id
            )
        except PlanError as exc:
            return redirect_notice(str(exc), "error", "plans")
        except Exception as exc:  # pragma: no cover - defensive
            return redirect_notice(
                f"补测计划计算失败，未保存：{type(exc).__name__}: {exc}",
                "error",
                "plans",
            )

        try:
            plan_id = await run_in_threadpool(
                store.publish_plan, snapshot, sample_id, payload
            )
        except StoreError as exc:
            return redirect_notice(
                f"补测计划写入失败，已有计划未受影响：{exc}",
                "error",
                "plans",
            )

        if plan_id is None:
            return redirect_notice(
                "计算期间报告、基准或候选文件已被更新，本次补测计划与当前"
                "来源不符，已丢弃，未保存。请重新制定。",
                "error",
                "plans",
            )

        if payload["solved"]:
            combos_text = "、".join(
                "「" + "＋".join(combo) + "」" for combo in payload["combos"]
            )
            message = (
                f"补测计划 #{plan_id} 已保存：样本 {sample_id} 补测 "
                f"{payload['min_elements']} 项即可唯一确定——{combos_text}。"
            )
        else:
            message = (
                f"补测计划 #{plan_id} 已保存：样本 {sample_id} 在最多 "
                f"{payload['max_elements']} 项内无法唯一确定。"
                f"{payload['reason']}"
            )
        return redirect_notice(message, "ok", "plans")

    # -- exports ---------------------------------------------------------

    @app.get("/reports/{report_id}/export.csv")
    def export_report(report_id: int) -> Response:
        record = store.get_report(report_id)
        if record is None:
            return PlainTextResponse("报告不存在", status_code=404)
        payload = record.payload
        elements = payload["elements"]
        header = ["sample_id", "unique"]
        for ore in payload["ores"]:
            header.append(f"{ore}_weight")
        for ore in payload["ores"]:
            header.append(f"{ore}_weight_min")
            header.append(f"{ore}_weight_max")
        for element in elements:
            header.append(f"measured_{element}")
        for element in elements:
            header += [
                f"predicted_{element}",
                f"predicted_{element}_min",
                f"predicted_{element}_max",
            ]
        for element in elements:
            header += [
                f"residual_{element}",
                f"residual_{element}_min",
                f"residual_{element}_max",
            ]
        header += ["sse", "note"]

        def triple(point_value, interval):
            """One element's (value, min, max) cells.

            A non-unique optimum has no single value, so the value cell is
            left empty and only the exact attainable range is emitted.
            """

            low, high = interval
            return [
                "" if point_value is None else formatting.decimal(point_value, 10),
                formatting.decimal(low, 10),
                formatting.decimal(high, 10),
            ]

        rows = [",".join(_csv_cell(cell) for cell in header)]
        for result in payload["results"]:
            row = [result["sample_id"], "yes" if result["unique"] else "no"]
            if result["unique"]:
                row += [formatting.percent(v, 8) for v in result["weights"]]
            else:
                row += [""] * len(payload["ores"])
            for low, high in result["weight_ranges"]:
                row += [formatting.percent(low, 8), formatting.percent(high, 8)]
            row += [formatting.decimal(v, 10) for v in result["measured"]]
            for i in range(len(elements)):
                row += triple(
                    None if result["predicted"] is None else result["predicted"][i],
                    result["predicted_ranges"][i],
                )
            for i in range(len(elements)):
                row += triple(
                    None if result["residuals"] is None else result["residuals"][i],
                    result["residual_ranges"][i],
                )
            row.append(formatting.decimal(result["sse"], 12))
            row.append(
                ""
                if result["unique"]
                else "最优占比不唯一：占比、预测值与残差仅以下列 min/max 范围给出，"
                "不存在唯一的配料组合"
            )
            rows.append(",".join(_csv_cell(cell) for cell in row))
        return Response(
            "﻿" + "\r\n".join(rows) + "\r\n",
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="ore-report-{report_id}.csv"'
                )
            },
        )

    @app.get("/inputs/{kind}/export.csv")
    def export_input(kind: str) -> Response:
        if kind not in ("baseline", "samples", "candidates"):
            return PlainTextResponse("输入类型不存在", status_code=404)
        current = store.current_inputs().get(kind)
        if current is None:
            return PlainTextResponse("输入尚未上传", status_code=404)
        if kind in ("baseline", "candidates"):
            value = (
                BaselineInput.from_payload(current.payload)
                if kind == "baseline"
                else CandidatesInput.from_payload(current.payload)
            )
            header = ["ore", *value.elements]
            rows = [
                [ore, *[formatting.decimal(x, 12) for x in row]]
                for ore, row in zip(value.ores, value.rows)
            ]
        else:
            value = SamplesInput.from_payload(current.payload)
            header = ["sample_id", *value.elements]
            rows = [
                [sample_id, *[formatting.decimal(x, 12) for x in row]]
                for sample_id, row in zip(value.sample_ids, value.rows)
            ]
        lines = [",".join(_csv_cell(cell) for cell in header)]
        lines += [",".join(_csv_cell(cell) for cell in row) for row in rows]
        return Response(
            "﻿" + "\r\n".join(lines) + "\r\n",
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{kind}.csv"'
            },
        )

    @app.get("/api/latest")
    def api_latest() -> JSONResponse:
        record = store.latest_report()
        current = store.current_inputs()
        if record is None:
            return JSONResponse(
                {
                    "report": None,
                    "inputs": {
                        kind: (
                            {
                                "version": value.version_id,
                                "filename": value.filename,
                                "created_at": value.created_at,
                            }
                            if value
                            else None
                        )
                        for kind, value in current.items()
                    },
                }
            )
        return JSONResponse(
            {
                "report": {
                    "id": record.report_id,
                    "created_at": record.created_at,
                    "baseline_version": record.baseline_version,
                    "samples_version": record.samples_version,
                    "stale": record.is_stale,
                    "payload": record.payload,
                }
            }
        )

    @app.get("/api/plans")
    def api_plans() -> JSONResponse:
        return JSONResponse(
            {
                "plans": [
                    {
                        "id": record.plan_id,
                        "report_id": record.report_id,
                        "sample_id": record.sample_id,
                        "created_at": record.created_at,
                        "baseline_version": record.baseline_version,
                        "samples_version": record.samples_version,
                        "candidates_version": record.candidates_version,
                        "expired": record.is_expired,
                        "payload": record.payload,
                    }
                    for record in store.list_plans()
                ]
            }
        )

    return app


def _csv_cell(value: Any) -> str:
    text = str(value)
    if any(ch in text for ch in [",", '"', "\r", "\n"]):
        return '"' + text.replace('"', '""') + '"'
    return text


class _LabeledError(Exception):
    def __init__(self, label: str, message: str) -> None:
        super().__init__(message)
        self.label = label
        self.message = message


app = create_app()
