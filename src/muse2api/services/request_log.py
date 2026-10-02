"""Request log: one SQLite row per ``/v1/*`` request, for the dashboard and ``/admin/stats``.

Rows are written by ``api.middleware.RequestLogMiddleware``. Code deeper in the
stack (the gateway) adds to the in-flight row through ``current_record``. Async
image/video submits return 200 straight away, so the task's outcome is written back
onto the submit row when it finishes (``finish_task``) and counts as an error. Each
attempt that failed on an account (including ones the gateway then failed over from,
which the client never sees) goes into ``attempt_errors`` as well. All
database work runs in a worker thread; a failed write is logged and dropped,
never surfaced to the client.
"""

from __future__ import annotations

import asyncio
import logging
import math
import sqlite3
import threading
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# The row being built for the current request. It holds a mutable dict rather than
# values so that writes from copied contexts (streaming body tasks) are still seen.
current_record: ContextVar[dict[str, Any] | None] = ContextVar("current_record", default=None)

COLUMNS = ("ts", "method", "path", "model", "key_id", "key_name", "account_id", "status_code",
           "latency_ms", "stream", "poll", "error", "client_ip", "user_agent", "task_id",
           "failed_attempts")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    method      TEXT    NOT NULL,
    path        TEXT    NOT NULL,
    model       TEXT,
    key_id      TEXT,
    key_name    TEXT,
    account_id  TEXT,
    status_code INTEGER NOT NULL,
    latency_ms  INTEGER NOT NULL,
    stream      INTEGER NOT NULL DEFAULT 0,
    poll        INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    client_ip   TEXT,
    user_agent  TEXT,
    task_id     TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests (ts);
CREATE INDEX IF NOT EXISTS idx_requests_key ON requests (key_id, ts);
CREATE INDEX IF NOT EXISTS idx_requests_account ON requests (account_id, ts);
CREATE INDEX IF NOT EXISTS idx_requests_task ON requests (task_id);
CREATE TABLE IF NOT EXISTS attempt_errors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    account_id  TEXT,
    error       TEXT,
    task_id     TEXT
);
CREATE INDEX IF NOT EXISTS idx_attempt_errors_ts ON attempt_errors (ts);
"""

# Columns added after the first release; created on open when missing.
_ADDED_COLUMNS = {"task_status": "TEXT", "task_ms": "INTEGER", "failed_attempts": "INTEGER"}

# A request failed if it got an HTTP error or its async task failed later.
FAILED = "(status_code >= 400 OR COALESCE(task_status = 'failed', 0))"

# window -> (length, bucket size) in seconds
WINDOWS = {"1h": (3600, 60), "24h": (86400, 1800), "7d": (7 * 86400, 3 * 3600)}

STATUS_CLASSES = ("2xx", "3xx", "4xx", "5xx")
STATUS_FILTERS = (*STATUS_CLASSES, "failed")


def note_account(account_id: str) -> None:
    """Record the account serving the current request (the last one wins on failover)."""
    record = current_record.get()
    if record is not None:
        record["account_id"] = account_id


def note_failed_attempt(account_id: str, error: BaseException) -> None:
    """Record an attempt that failed on ``account_id``, whether or not a failover follows."""
    record = current_record.get()
    if record is not None:
        record.setdefault("attempt_errors", []).append(
            (time.time(), account_id, str(error)[:300]))


def _percentile(sorted_values: list[int], q: float) -> int | None:
    if not sorted_values:
        return None
    # Nearest-rank percentile.
    idx = max(0, min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[idx]


class RequestLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn: sqlite3.Connection | None = None
        # One connection shared by worker threads; sqlite3 objects are not thread-safe.
        self._lock = threading.Lock()
        # Outcomes of tasks that finished before their submit row was written.
        self._early: dict[str, tuple] = {}

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(_SCHEMA)
            have = {r[1] for r in conn.execute("PRAGMA table_info(requests)")}
            for col, kind in _ADDED_COLUMNS.items():
                if col not in have:
                    conn.execute(f"ALTER TABLE requests ADD COLUMN {col} {kind}")
            self._conn = conn
        return self._conn

    def _call(self, fn, *args):
        with self._lock:
            return fn(self._db(), *args)

    async def open(self) -> None:
        try:
            await asyncio.to_thread(self._call, lambda db: None)
        except Exception:  # noqa: BLE001
            # Not fatal: the connection is retried on the next write.
            log.exception("failed to open request log %s", self.path)

    async def close(self) -> None:
        def _close() -> None:
            with self._lock:
                if self._conn is not None:
                    self._conn.close()
                    self._conn = None

        await asyncio.to_thread(_close)

    # ---- writes ----
    async def add(self, record: dict[str, Any]) -> None:
        task_id = None if record.get("poll") else record.get("task_id")
        # An async task's attempts happen after this row is written; finish_task logs them.
        attempts = [] if task_id else list(record.get("attempt_errors") or [])
        record = {"stream": 0, "poll": 0, **record, "failed_attempts": len(attempts)}
        row = tuple(record.get(c) for c in COLUMNS)
        sql = f"INSERT INTO requests ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})"

        def run(db: sqlite3.Connection) -> None:
            db.execute(sql, row)
            if attempts:
                db.executemany(_ATTEMPT_SQL, [(*a, None) for a in attempts])
            # Under the same lock as finish_task, so an outcome is never stashed
            # just after this row went in.
            if task_id and (early := self._early.pop(task_id, None)):
                db.execute(_FINISH_SQL, early)

        try:
            await asyncio.to_thread(self._call, run)
        except Exception:  # noqa: BLE001
            log.exception("failed to write request log row")

    async def finish_task(self, task_id: str, status: str, finished_at: float,
                          error: str | None = None, account_id: str | None = None,
                          attempt_errors: list[tuple] | None = None) -> None:
        """Write an async task's outcome onto the request that submitted it."""
        attempts = list(attempt_errors or [])
        args = (status, finished_at, error, account_id, len(attempts), task_id)

        def run(db: sqlite3.Connection) -> None:
            if attempts:
                db.executemany(_ATTEMPT_SQL, [(*a, task_id) for a in attempts])
            if not db.execute(_FINISH_SQL, args).rowcount:
                # The task beat the response out (e.g. no account available); the
                # submit row picks the outcome up when the middleware writes it.
                self._early[task_id] = args
                while len(self._early) > 200:
                    self._early.pop(next(iter(self._early)))

        try:
            await asyncio.to_thread(self._call, run)
        except Exception:  # noqa: BLE001
            log.exception("failed to record outcome of task %s", task_id)

    async def backfill_tasks(self, tasks: list[Any]) -> None:
        """Fill in outcomes for submit rows logged before outcomes were recorded."""
        rows = [(t.status.value, t.updated_at, (t.error or {}).get("message"), None, None, t.id)
                for t in tasks if t.finished]
        if not rows:
            return
        sql = _FINISH_SQL + " AND task_status IS NULL"
        try:
            await asyncio.to_thread(self._call, lambda db: db.executemany(sql, rows))
        except Exception:  # noqa: BLE001
            log.exception("failed to backfill task outcomes")

    async def prune(self, retention_days: float) -> int:
        cutoff = time.time() - retention_days * 86400

        def run(db: sqlite3.Connection) -> sqlite3.Cursor:
            db.execute("DELETE FROM attempt_errors WHERE ts < ?", (cutoff,))
            return db.execute("DELETE FROM requests WHERE ts < ?", (cutoff,))

        try:
            cur = await asyncio.to_thread(self._call, run)
        except Exception:  # noqa: BLE001
            log.exception("failed to prune request log")
            return 0
        if cur.rowcount:
            log.info("pruned %d request log rows older than %s days", cur.rowcount, retention_days)
        return cur.rowcount

    # ---- reads ----
    async def query(self, *, key_id: str | None = None, account_id: str | None = None,
                    status: str | None = None, path: str | None = None,
                    since: float | None = None, hide_polls: bool = False,
                    limit: int = 100, offset: int = 0) -> dict[str, Any]:
        where, args = ["1=1"], []
        if key_id:
            where.append("key_id = ?")
            args.append(key_id)
        if account_id:
            where.append("account_id = ?")
            args.append(account_id)
        if status == "failed":
            where.append(FAILED)
        elif status in STATUS_CLASSES:
            low = int(status[0]) * 100
            where.append("status_code >= ? AND status_code < ?")
            args += [low, low + 100]
        if path:
            where.append("path LIKE ? ESCAPE '\\'")
            escaped = path.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            args.append(escaped + "%")
        if since:
            where.append("ts >= ?")
            args.append(since)
        if hide_polls:
            where.append("poll = 0")
        cond = " AND ".join(where)

        def run(db: sqlite3.Connection) -> dict[str, Any]:
            total = db.execute(f"SELECT COUNT(*) FROM requests WHERE {cond}", args).fetchone()[0]
            rows = db.execute(
                f"SELECT * FROM requests WHERE {cond} ORDER BY id DESC LIMIT ? OFFSET ?",
                [*args, limit, offset]).fetchall()
            return {"data": [_row(r) for r in rows], "total": total}

        return await asyncio.to_thread(self._call, run)

    async def stats(self, window: str) -> dict[str, Any]:
        length, bucket = WINDOWS[window]
        now = time.time()
        # Align buckets to the bucket size so the chart does not shift on every refresh.
        start = (now - length) // bucket * bucket + bucket
        n_buckets = int((now - start) // bucket) + 1
        # Task polls are cheap and frequent; they would drown out real traffic.
        cond, args = "ts >= ? AND poll = 0", (start,)

        def grouped(db: sqlite3.Connection, cols: str) -> list[dict[str, Any]]:
            rows = db.execute(
                f"SELECT {cols}, COUNT(*) AS count, SUM({FAILED}) AS errors "
                f"FROM requests WHERE {cond} GROUP BY {cols} ORDER BY count DESC LIMIT 50",
                args).fetchall()
            return [dict(r) for r in rows]

        def run(db: sqlite3.Connection) -> dict[str, Any]:
            total, errors, server_errors = db.execute(
                f"SELECT COUNT(*), COALESCE(SUM({FAILED}), 0), "
                f"COALESCE(SUM(status_code >= 500), 0) FROM requests WHERE {cond}",
                args).fetchone()
            latencies = [r[0] for r in db.execute(
                f"SELECT latency_ms FROM requests WHERE {cond} ORDER BY latency_ms", args)]
            failed_attempts, rescued = db.execute(
                f"SELECT COALESCE(SUM(failed_attempts), 0), "
                f"COALESCE(SUM(failed_attempts > 0 AND NOT {FAILED}), 0) "
                f"FROM requests WHERE {cond}", args).fetchone()
            attempts_by_account = {r[0]: r[1] for r in db.execute(
                "SELECT account_id, COUNT(*) FROM attempt_errors WHERE ts >= ? GROUP BY account_id",
                (start,))}
            by_account = grouped(db, "account_id")
            for row in by_account:
                row["attempt_errors"] = attempts_by_account.pop(row["account_id"], 0)
            # Accounts that only failed (every request moved on to another account).
            by_account += [{"account_id": acc, "count": 0, "errors": 0, "attempt_errors": n}
                           for acc, n in attempts_by_account.items()]
            series = [{"t": start + i * bucket, "count": 0, "errors": 0} for i in range(n_buckets)]
            for idx, count, errs in db.execute(
                    f"SELECT CAST((ts - ?) / ? AS INTEGER) AS b, COUNT(*), "
                    f"SUM({FAILED}) FROM requests WHERE {cond} GROUP BY b",
                    (start, bucket, *args)):
                if 0 <= idx < n_buckets:
                    series[idx]["count"], series[idx]["errors"] = count, errs
            return {
                "window": window,
                "since": start,
                "bucket_seconds": bucket,
                "total": total,
                "errors": errors,
                "server_errors": server_errors,
                "error_rate": errors / total if total else 0.0,
                # Attempts that failed on some account; "rescued" requests still
                # succeeded on another one, so the client never saw those errors.
                "failed_attempts": failed_attempts,
                "rescued": rescued,
                "latency_ms": {"p50": _percentile(latencies, 0.50),
                               "p95": _percentile(latencies, 0.95)},
                "by_key": grouped(db, "key_id, key_name"),
                "by_account": by_account,
                "by_model": grouped(db, "model"),
                "by_status": grouped(db, "status_code"),
                "series": series,
            }

        return await asyncio.to_thread(self._call, run)

    async def key_usage(self) -> dict[str, Any]:
        """Per-key request counts in one GROUP BY: everything kept (see retention), the
        last 24h, and the failures among those (4xx/5xx or a failed task). Task polls are excluded, as in ``stats``."""
        since = time.time() - 86400

        def run(db: sqlite3.Connection) -> list[dict[str, Any]]:
            # A bare column next to MAX() takes its value from the max row in SQLite,
            # so key_name is the name used by the key's latest request.
            rows = db.execute(
                "SELECT key_id, key_name, MAX(ts) AS last_request_at, COUNT(*) AS total, "
                "COALESCE(SUM(ts >= ?), 0) AS requests_24h, "
                f"COALESCE(SUM(ts >= ? AND {FAILED}), 0) AS errors_24h "
                "FROM requests WHERE poll = 0 AND key_id IS NOT NULL "
                "GROUP BY key_id ORDER BY total DESC",
                (since, since)).fetchall()
            return [dict(r) for r in rows]

        return {"since": since, "data": await asyncio.to_thread(self._call, run)}


_FINISH_SQL = (
    "UPDATE requests SET task_status = ?, task_ms = CAST((? - ts) * 1000 AS INTEGER), "
    "error = COALESCE(error, ?), account_id = COALESCE(account_id, ?), "
    "failed_attempts = COALESCE(?, failed_attempts) "
    "WHERE task_id = ? AND poll = 0"
)
_ATTEMPT_SQL = "INSERT INTO attempt_errors (ts, account_id, error, task_id) VALUES (?, ?, ?, ?)"


def _row(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    d["stream"], d["poll"] = bool(d["stream"]), bool(d["poll"])
    return d
