"""SQLite persistence.

Three tables carry the whole state:

* ``input_versions`` – append-only versions of each input (baseline,
  samples, candidate elements).  Exactly one row per ``kind`` is flagged
  ``is_current``.  A report names the version ids it was computed from,
  so "is this report stale?" is always a comparison against the current
  versions rather than a flag that could drift.
* ``reports`` – only fully computed reports are ever inserted.  The
  insert re-checks that the snapshot versions are still current *inside*
  the same ``BEGIN IMMEDIATE`` transaction that writes the report, which
  is what makes publishing safe when an upload interleaves with a
  calculation.
* ``plans`` – re-measurement plans for non-unique samples.  A plan binds
  the report id plus the baseline/samples/candidates version ids it was
  computed from; the same atomic re-check guards its publication, and
  expiry is derived at read time by comparing with the latest report and
  the current versions.  Plans are never deleted: expired ones stay
  queryable.

All writes from this process are serialised by a re-entrant lock; SQLite's
own locking covers other processes/workers.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

KINDS = ("baseline", "samples", "candidates")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS input_versions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT    NOT NULL
                CHECK (kind IN ('baseline', 'samples', 'candidates')),
    filename    TEXT,
    payload     TEXT    NOT NULL,
    created_at  TEXT    NOT NULL,
    is_current  INTEGER NOT NULL DEFAULT 0
                CHECK (is_current IN (0, 1))
);

CREATE UNIQUE INDEX IF NOT EXISTS
    input_versions_one_current
ON input_versions (kind)
WHERE is_current = 1;

CREATE TABLE IF NOT EXISTS reports (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    baseline_version  INTEGER NOT NULL
                      REFERENCES input_versions (id),
    samples_version   INTEGER NOT NULL
                      REFERENCES input_versions (id),
    payload           TEXT    NOT NULL,
    created_at        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS plans (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id           INTEGER NOT NULL
                        REFERENCES reports (id),
    sample_id           TEXT    NOT NULL,
    baseline_version    INTEGER NOT NULL,
    samples_version     INTEGER NOT NULL,
    candidates_version  INTEGER NOT NULL,
    payload             TEXT    NOT NULL,
    created_at          TEXT    NOT NULL
);
"""


class StoreError(RuntimeError):
    """A persistence problem surfaced to the UI."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class VersionedInput:
    version_id: int
    kind: str
    filename: str | None
    created_at: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class Snapshot:
    baseline: VersionedInput
    samples: VersionedInput


@dataclass(frozen=True)
class ReportRecord:
    report_id: int
    baseline_version: int
    samples_version: int
    created_at: str
    payload: dict[str, Any]
    baseline_is_current: bool
    samples_is_current: bool

    @property
    def is_stale(self) -> bool:
        return not (self.baseline_is_current and self.samples_is_current)


@dataclass(frozen=True)
class PlanSnapshot:
    """Consistent read of everything a plan computation depends on."""

    report: ReportRecord
    baseline: VersionedInput
    samples: VersionedInput
    candidates: VersionedInput | None


@dataclass(frozen=True)
class PlanRecord:
    plan_id: int
    report_id: int
    sample_id: str
    baseline_version: int
    samples_version: int
    candidates_version: int
    created_at: str
    payload: dict[str, Any]
    report_is_latest: bool
    baseline_is_current: bool
    samples_is_current: bool
    candidates_is_current: bool

    @property
    def is_expired(self) -> bool:
        return not (
            self.report_is_latest
            and self.baseline_is_current
            and self.samples_is_current
            and self.candidates_is_current
        )


class Store:
    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path)
        self._write_lock = threading.RLock()
        self._initialized = False

    # -- connections -----------------------------------------------------

    def _ensure_ready(self) -> None:
        """Create the directory and schema on first use.

        Deferring this keeps merely importing the app (or constructing a
        store) free of filesystem side effects.  Databases created before
        the ``candidates`` input kind existed are migrated in place.
        """

        if self._initialized:
            return
        with self._write_lock:
            if self._initialized:
                return
            if self.path.parent and str(self.path.parent) not in ("", "."):
                self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = self._connect()
            try:
                conn.execute("PRAGMA journal_mode = WAL")
                self._migrate(conn)
                conn.executescript(_SCHEMA)
            except sqlite3.Error as exc:
                raise StoreError(f"初始化数据库失败：{exc}") from exc
            finally:
                conn.close()
            self._initialized = True

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Rebuild ``input_versions`` when its CHECK predates 'candidates'.

        SQLite cannot alter a CHECK constraint, so the table is rebuilt
        under the new schema and the rows copied over.  The whole rebuild
        runs in one transaction; a fresh database skips this entirely.
        """

        row = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name = 'input_versions'"
        ).fetchone()
        if row is None or "'candidates'" in row[0]:
            return
        try:
            conn.execute("BEGIN")
            conn.execute(
                "ALTER TABLE input_versions RENAME TO input_versions_old"
            )
            conn.execute(
                """
                CREATE TABLE input_versions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind        TEXT    NOT NULL
                                CHECK (kind IN
                                      ('baseline', 'samples', 'candidates')),
                    filename    TEXT,
                    payload     TEXT    NOT NULL,
                    created_at  TEXT    NOT NULL,
                    is_current  INTEGER NOT NULL DEFAULT 0
                                CHECK (is_current IN (0, 1))
                )
                """
            )
            conn.execute(
                "INSERT INTO input_versions "
                "(id, kind, filename, payload, created_at, is_current) "
                "SELECT id, kind, filename, payload, created_at, is_current "
                "FROM input_versions_old"
            )
            # Dropping the old table also drops the old unique index,
            # freeing its name for the schema script below.
            conn.execute("DROP TABLE input_versions_old")
            conn.execute("COMMIT")
        except sqlite3.Error:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.path), timeout=30.0, isolation_level=None
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        self._ensure_ready()
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            self._ensure_ready()
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    # -- inputs ----------------------------------------------------------

    def save_inputs(
        self, updates: dict[str, tuple[dict[str, Any], str | None]]
    ) -> dict[str, int]:
        """Replace one or both inputs in a single transaction.

        Whole-file replacement plus atomicity: either every supplied input
        gets a new current version, or none of them change.
        """

        for kind in updates:
            if kind not in KINDS:
                raise ValueError(f"unknown input kind: {kind!r}")
        encoded = {
            kind: json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            for kind, (payload, _filename) in updates.items()
        }
        now = _utcnow()
        new_ids: dict[str, int] = {}
        try:
            with self._write() as conn:
                for kind in sorted(encoded):
                    _payload, filename = updates[kind]
                    conn.execute(
                        "UPDATE input_versions SET is_current = 0 "
                        "WHERE kind = ? AND is_current = 1",
                        (kind,),
                    )
                    cursor = conn.execute(
                        "INSERT INTO input_versions "
                        "(kind, filename, payload, created_at, is_current) "
                        "VALUES (?, ?, ?, ?, 1)",
                        (kind, filename, encoded[kind], now),
                    )
                    new_ids[kind] = int(cursor.lastrowid)
        except sqlite3.Error as exc:
            raise StoreError(f"保存输入失败：{exc}") from exc
        return new_ids

    def save_input(
        self, kind: str, payload: dict[str, Any], filename: str | None
    ) -> int:
        return self.save_inputs({kind: (payload, filename)})[kind]

    def _current_row(self, conn: sqlite3.Connection, kind: str):
        return conn.execute(
            "SELECT id, kind, filename, payload, created_at "
            "FROM input_versions WHERE kind = ? AND is_current = 1",
            (kind,),
        ).fetchone()

    @staticmethod
    def _row_to_input(row: sqlite3.Row) -> VersionedInput:
        return VersionedInput(
            version_id=row["id"],
            kind=row["kind"],
            filename=row["filename"],
            created_at=row["created_at"],
            payload=json.loads(row["payload"]),
        )

    def current_inputs(self) -> dict[str, VersionedInput | None]:
        with self._read() as conn:
            result: dict[str, VersionedInput | None] = {}
            for kind in KINDS:
                row = self._current_row(conn, kind)
                result[kind] = self._row_to_input(row) if row else None
            return result

    def snapshot(self) -> Snapshot | None:
        """Consistent read of both current inputs, or ``None`` if incomplete."""

        with self._read() as conn:
            rows = {kind: self._current_row(conn, kind) for kind in KINDS}
            if rows["baseline"] is None or rows["samples"] is None:
                return None
            return Snapshot(
                baseline=self._row_to_input(rows["baseline"]),
                samples=self._row_to_input(rows["samples"]),
            )

    # -- reports ---------------------------------------------------------

    _REPORT_SELECT = (
        "SELECT r.id, r.baseline_version, r.samples_version, "
        "       r.payload, r.created_at, "
        "       (SELECT id FROM input_versions "
        "         WHERE kind = 'baseline' AND is_current = 1) "
        "         AS current_baseline, "
        "       (SELECT id FROM input_versions "
        "         WHERE kind = 'samples' AND is_current = 1) "
        "         AS current_samples "
        "FROM reports r "
    )

    @staticmethod
    def _versions_are_current(
        conn: sqlite3.Connection, baseline_version: int, samples_version: int
    ) -> bool:
        rows = {
            "baseline": conn.execute(
                "SELECT id FROM input_versions "
                "WHERE kind = 'baseline' AND is_current = 1"
            ).fetchone(),
            "samples": conn.execute(
                "SELECT id FROM input_versions "
                "WHERE kind = 'samples' AND is_current = 1"
            ).fetchone(),
        }
        return (
            rows["baseline"] is not None
            and rows["samples"] is not None
            and rows["baseline"]["id"] == baseline_version
            and rows["samples"]["id"] == samples_version
        )

    def publish_report(
        self,
        snapshot: Snapshot,
        payload: dict[str, Any],
    ) -> int | None:
        """Insert a complete report if, and only if, its inputs are current.

        Returns the new report id, or ``None`` when an upload won the race.
        The check and the insert share one ``BEGIN IMMEDIATE`` transaction,
        so an upload cannot slip in between them.
        """

        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        try:
            with self._write() as conn:
                if not self._versions_are_current(
                    conn,
                    snapshot.baseline.version_id,
                    snapshot.samples.version_id,
                ):
                    return None
                cursor = conn.execute(
                    "INSERT INTO reports "
                    "(baseline_version, samples_version, payload, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        snapshot.baseline.version_id,
                        snapshot.samples.version_id,
                        encoded,
                        _utcnow(),
                    ),
                )
                return int(cursor.lastrowid)
        except sqlite3.Error as exc:
            raise StoreError(f"保存报告失败：{exc}") from exc

    def latest_report(self) -> ReportRecord | None:
        with self._read() as conn:
            row = conn.execute(
                self._REPORT_SELECT + "ORDER BY r.id DESC LIMIT 1"
            ).fetchone()
            return self._row_to_report(row) if row else None

    def get_report(self, report_id: int) -> ReportRecord | None:
        with self._read() as conn:
            row = conn.execute(
                self._REPORT_SELECT + "WHERE r.id = ?",
                (report_id,),
            ).fetchone()
            return self._row_to_report(row) if row else None

    @staticmethod
    def _row_to_report(row: sqlite3.Row) -> ReportRecord:
        return ReportRecord(
            report_id=row["id"],
            baseline_version=row["baseline_version"],
            samples_version=row["samples_version"],
            created_at=row["created_at"],
            payload=json.loads(row["payload"]),
            baseline_is_current=row["current_baseline"]
            == row["baseline_version"],
            samples_is_current=row["current_samples"] == row["samples_version"],
        )

    # -- re-measurement plans ---------------------------------------------

    def plan_snapshot(self) -> PlanSnapshot | None:
        """Consistent read of the latest report and all current inputs.

        Returns ``None`` unless a report and both report inputs exist;
        the candidates input is optional at snapshot time (the endpoint
        reports its absence with a clearer message).
        """

        with self._read() as conn:
            report_row = conn.execute(
                self._REPORT_SELECT + "ORDER BY r.id DESC LIMIT 1"
            ).fetchone()
            if report_row is None:
                return None
            current = {
                kind: self._current_row(conn, kind) for kind in KINDS
            }
            if current["baseline"] is None or current["samples"] is None:
                return None
            return PlanSnapshot(
                report=self._row_to_report(report_row),
                baseline=self._row_to_input(current["baseline"]),
                samples=self._row_to_input(current["samples"]),
                candidates=(
                    self._row_to_input(current["candidates"])
                    if current["candidates"] is not None
                    else None
                ),
            )

    def publish_plan(
        self,
        snapshot: PlanSnapshot,
        sample_id: str,
        payload: dict[str, Any],
    ) -> int | None:
        """Insert a plan iff its sources are still current.

        The plan binds the report id and the baseline / samples /
        candidates version ids from the snapshot.  The re-check and the
        insert share one ``BEGIN IMMEDIATE`` transaction, so a report or
        input change landing mid-computation makes this return ``None``
        and nothing is published.
        """

        if snapshot.candidates is None:
            raise ValueError("cannot publish a plan without candidates")
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        try:
            with self._write() as conn:
                latest = conn.execute(
                    "SELECT id FROM reports ORDER BY id DESC LIMIT 1"
                ).fetchone()
                current = {
                    kind: self._current_row(conn, kind) for kind in KINDS
                }
                still_current = (
                    latest is not None
                    and latest["id"] == snapshot.report.report_id
                    and current["baseline"] is not None
                    and current["baseline"]["id"]
                    == snapshot.baseline.version_id
                    and current["samples"] is not None
                    and current["samples"]["id"] == snapshot.samples.version_id
                    and current["candidates"] is not None
                    and current["candidates"]["id"]
                    == snapshot.candidates.version_id
                )
                if not still_current:
                    return None
                cursor = conn.execute(
                    "INSERT INTO plans "
                    "(report_id, sample_id, baseline_version, "
                    " samples_version, candidates_version, payload, "
                    " created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        snapshot.report.report_id,
                        sample_id,
                        snapshot.baseline.version_id,
                        snapshot.samples.version_id,
                        snapshot.candidates.version_id,
                        encoded,
                        _utcnow(),
                    ),
                )
                return int(cursor.lastrowid)
        except sqlite3.Error as exc:
            raise StoreError(f"保存补测计划失败：{exc}") from exc

    def list_plans(self, limit: int = 100) -> list[PlanRecord]:
        """All plans, newest first, with expiry derived at read time."""

        with self._read() as conn:
            rows = conn.execute(
                "SELECT p.id, p.report_id, p.sample_id, "
                "       p.baseline_version, p.samples_version, "
                "       p.candidates_version, p.payload, p.created_at, "
                "       (SELECT id FROM reports ORDER BY id DESC LIMIT 1) "
                "         AS latest_report, "
                "       (SELECT id FROM input_versions "
                "         WHERE kind = 'baseline' AND is_current = 1) "
                "         AS current_baseline, "
                "       (SELECT id FROM input_versions "
                "         WHERE kind = 'samples' AND is_current = 1) "
                "         AS current_samples, "
                "       (SELECT id FROM input_versions "
                "         WHERE kind = 'candidates' AND is_current = 1) "
                "         AS current_candidates "
                "FROM plans p ORDER BY p.id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_plan(row) for row in rows]

    @staticmethod
    def _row_to_plan(row: sqlite3.Row) -> PlanRecord:
        return PlanRecord(
            plan_id=row["id"],
            report_id=row["report_id"],
            sample_id=row["sample_id"],
            baseline_version=row["baseline_version"],
            samples_version=row["samples_version"],
            candidates_version=row["candidates_version"],
            created_at=row["created_at"],
            payload=json.loads(row["payload"]),
            report_is_latest=row["latest_report"] == row["report_id"],
            baseline_is_current=row["current_baseline"]
            == row["baseline_version"],
            samples_is_current=row["current_samples"]
            == row["samples_version"],
            candidates_is_current=row["current_candidates"]
            == row["candidates_version"],
        )
