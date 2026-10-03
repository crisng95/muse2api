"""Prepaid USD credit per stored client key.

Money is integer micro-USD (1e-6 USD) internally and a USD float rounded to six
decimals in APIs. Balances and the ledger live in ``data/requests.db`` on the
request log's connection, in their own tables, which retention pruning never
touches. Every balance change is one transaction: the balance update plus its
ledger row.

Only stored keys are billed; the admin key, the legacy ``api_key`` and keys marked
``unlimited`` are not, and nothing is when ``billing_enabled`` is off. Responses
still carry ``cost_usd`` (the request's price) for those, for information only;
the request log's ``cost_micro`` holds what was actually charged.

Images and videos have a known price, so it is charged up front ("reserved") and
refunded if the work fails. A reservation stays in ``reservations`` until its work
succeeds (``settle``) or it is refunded; whatever is still there at startup was cut
short by a restart and is refunded then (``reconcile``). Chat costs are only known once the reply is done, so
they are charged afterwards and may take a balance slightly below zero.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from collections.abc import Coroutine, Iterable
from typing import Any

from ..auth.keys import KeyStore
from ..config import Settings
from ..errors import InsufficientBalance
from .request_log import RequestLog, current_record
from .tasks import Task, TaskStatus

log = logging.getLogger(__name__)

MICRO = 1_000_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS balances (
    key_id        TEXT    PRIMARY KEY,
    balance_micro INTEGER NOT NULL DEFAULT 0,
    updated_at    REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    key_id        TEXT    NOT NULL,
    amount_micro  INTEGER NOT NULL,
    kind          TEXT    NOT NULL,
    ref           TEXT,
    note          TEXT,
    balance_after INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ledger_key ON ledger (key_id, id);
CREATE INDEX IF NOT EXISTS idx_ledger_ref ON ledger (ref);
-- A charge is refunded at most once.
CREATE UNIQUE INDEX IF NOT EXISTS idx_ledger_refund ON ledger (ref) WHERE kind = 'refund';
-- Up-front charges whose work has neither succeeded nor been refunded yet.
CREATE TABLE IF NOT EXISTS reservations (
    ref    TEXT PRIMARY KEY,
    key_id TEXT NOT NULL,
    ts     REAL NOT NULL
);
"""

# Task outcomes that cost nothing.
REFUNDABLE = (TaskStatus.FAILED, TaskStatus.CANCELLED)


def to_micro(amount_usd: float) -> int:
    return round(amount_usd * MICRO)


def usd(micro: int) -> float:
    return round(micro / MICRO, 6)


def task_cost_usd(task: Task) -> float | None:
    """``cost_usd`` shown on a task object: its price, or 0 once it failed."""
    cost = task.request.get("cost_usd")
    return 0.0 if cost is not None and task.status in REFUNDABLE else cost


def _note_cost(delta: int) -> None:
    record = current_record.get()
    if record is not None:
        record["cost_micro"] = (record.get("cost_micro") or 0) + delta


def _balance(db: sqlite3.Connection, key_id: str) -> int:
    row = db.execute("SELECT balance_micro FROM balances WHERE key_id = ?", (key_id,)).fetchone()
    return row[0] if row else 0


def _apply(db: sqlite3.Connection, key_id: str, amount: int, kind: str,
           ref: str | None, note: str) -> int:
    """Change a balance and write its ledger row; call inside ``_tx``. Returns the new balance."""
    now = time.time()
    db.execute(
        "INSERT INTO balances (key_id, balance_micro, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT (key_id) DO UPDATE SET balance_micro = balance_micro + excluded.balance_micro, "
        "updated_at = excluded.updated_at", (key_id, amount, now))
    after = _balance(db, key_id)
    db.execute(
        "INSERT INTO ledger (ts, key_id, amount_micro, kind, ref, note, balance_after) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)", (now, key_id, amount, kind, ref, note, after))
    return after


def _tx(db: sqlite3.Connection, fn, *args):
    # The connection is in autocommit mode; IMMEDIATE takes the write lock up front
    # so a balance check and the charge that follows it cannot interleave.
    db.execute("BEGIN IMMEDIATE")
    try:
        result = fn(db, *args)
        db.execute("COMMIT")
    except BaseException:
        # Never leave the shared connection inside a transaction: every later
        # request-log write would fail.
        try:
            db.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    return result


def _refund(db: sqlite3.Connection, ref: str) -> int:
    db.execute("DELETE FROM reservations WHERE ref = ?", (ref,))
    row = db.execute("SELECT key_id, SUM(amount_micro) FROM ledger "
                     "WHERE ref = ? AND kind = 'charge' GROUP BY key_id", (ref,)).fetchone()
    if row is None or db.execute("SELECT 1 FROM ledger WHERE ref = ? AND kind = 'refund'",
                                 (ref,)).fetchone():
        return 0
    _apply(db, row[0], -row[1], "refund", ref, "")
    return -row[1]


def _entry(r: sqlite3.Row) -> dict[str, Any]:
    return {"id": r["id"], "ts": r["ts"], "key_id": r["key_id"],
            "amount_usd": usd(r["amount_micro"]), "kind": r["kind"], "ref": r["ref"],
            "note": r["note"], "balance_after_usd": usd(r["balance_after"])}


class Billing:
    def __init__(self, settings: Settings, keys: KeyStore, requests: RequestLog) -> None:
        self.settings = settings
        self.keys = keys
        self.requests = requests
        self._ready = False
        # Balance changes that must finish even if the request that started them is gone.
        self._pending: set[asyncio.Task] = set()

    # ---- prices (micro-USD) ----
    def image_cost(self, n: int) -> int:
        return n * to_micro(self.settings.price_image_usd)

    def video_cost(self, seconds: int | None) -> int:
        seconds = seconds or self.settings.video_default_seconds
        return to_micro(seconds * self.settings.price_video_per_second_usd)

    def chat_cost(self, prompt_tokens: int, completion_tokens: int) -> int:
        s = self.settings
        # USD per million tokens is micro-USD per token.
        return round(prompt_tokens * s.price_chat_input_per_mtok_usd
                     + completion_tokens * s.price_chat_output_per_mtok_usd)

    def billed(self, key_id: str | None) -> bool:
        if not self.settings.billing_enabled or not key_id:
            return False
        key = self.keys.get(key_id)  # None for the admin and legacy identities
        return key is not None and not key.unlimited

    # ---- plumbing ----
    async def _run(self, fn, *args):
        def call(db: sqlite3.Connection):
            if not self._ready:
                db.executescript(_SCHEMA)
                self._ready = True
            return fn(db, *args)

        return await self.requests.run(call)

    async def _detached(self, coro: Coroutine):
        """Await ``coro`` in its own task, so cancelling the caller (a client that
        disconnected mid-stream) does not cancel the balance change."""
        task = asyncio.create_task(coro)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return await asyncio.shield(task)

    async def drain(self) -> None:
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)

    # ---- charging ----
    async def reserve(self, key_id: str | None, amount: int, ref: str, note: str = "") -> bool:
        """Charge a known price before the work starts; ``InsufficientBalance`` if the
        balance does not cover it. Returns whether anything was charged."""
        if not self.billed(key_id):
            return False

        def run(db: sqlite3.Connection) -> None:
            balance = _balance(db, key_id)
            if balance < amount:
                raise InsufficientBalance(
                    f"insufficient balance: this request costs ${usd(amount):.6f}, "
                    f"balance is ${usd(balance):.6f}; top up to continue")
            _apply(db, key_id, -amount, "charge", ref, note)
            db.execute("INSERT INTO reservations (ref, key_id, ts) VALUES (?, ?, ?)",
                       (ref, key_id, time.time()))

        await self._run(_tx, run)
        _note_cost(amount)
        return True

    async def check_positive(self, key_id: str | None) -> None:
        """Gate for requests priced afterwards (chat): reject only an empty balance."""
        if not self.billed(key_id):
            return
        balance = await self.balance(key_id)
        if balance <= 0:
            raise InsufficientBalance(
                f"insufficient balance: balance is ${usd(balance):.6f}; top up to continue")

    async def charge(self, key_id: str | None, amount: int, ref: str | None = None,
                     note: str = "") -> None:
        """Charge an amount known after the fact; never raises, may go below zero."""
        if amount <= 0 or not self.billed(key_id):
            return
        # Noted first: the request-log row may be written while the charge is in flight.
        _note_cost(amount)

        async def run() -> None:
            try:
                await self._run(_tx, _apply, key_id, -amount, "charge", ref, note)
            except Exception:  # noqa: BLE001
                log.exception("failed to charge %s for %s", key_id, ref)

        await self._detached(run())

    async def refund(self, ref: str | None) -> int:
        """Refund the charge made under ``ref``. Idempotent: returns the amount
        refunded, 0 if there was no charge or it was already refunded."""
        if not ref:
            return 0

        async def run() -> int:
            try:
                return await self._run(_tx, _refund, ref)
            except Exception:  # noqa: BLE001
                log.exception("failed to refund %s", ref)
                return 0

        amount = await self._detached(run())
        if amount:
            _note_cost(-amount)
        return amount

    async def settle(self, ref: str) -> None:
        """Mark the reservation under ``ref`` as earned: its work succeeded."""

        async def run() -> None:
            try:
                await self._run(lambda db: db.execute(
                    "DELETE FROM reservations WHERE ref = ?", (ref,)))
            except Exception:  # noqa: BLE001
                # Left in place, it is refunded at the next startup: the client's gain.
                log.exception("failed to settle %s", ref)

        await self._detached(run())

    async def reconcile(self, tasks: Iterable[Task]) -> int:
        """At startup, before serving: refund every reservation left unsettled by a
        restart (a sync image cut short, a task that failed or never got saved),
        except those of tasks that did succeed, which are settled instead."""
        succeeded = {t.id for t in tasks if t.status == TaskStatus.SUCCEEDED}

        def run(db: sqlite3.Connection) -> int:
            done = 0
            for (ref,) in db.execute("SELECT ref FROM reservations").fetchall():
                if ref in succeeded:
                    db.execute("DELETE FROM reservations WHERE ref = ?", (ref,))
                elif _tx(db, _refund, ref):
                    db.execute("UPDATE requests SET cost_micro = 0 WHERE task_id = ? AND poll = 0",
                               (ref,))
                    done += 1
            return done

        try:
            done = await self._run(run)
        except Exception:  # noqa: BLE001
            log.exception("failed to reconcile reservations")
            return 0
        if done:
            log.info("refunded %d reservation(s) left by a restart", done)
        return done

    # ---- admin ----
    async def credit(self, key_id: str, amount: int, note: str = "") -> int:
        """Top up (positive) or adjust (negative) a balance; returns the new balance."""
        kind = "topup" if amount > 0 else "adjust"
        return await self._run(_tx, _apply, key_id, amount, kind, None, note)

    async def balance(self, key_id: str) -> int:
        return await self._run(_balance, key_id)

    async def balances(self) -> dict[str, int]:
        return dict(await self._run(
            lambda db: db.execute("SELECT key_id, balance_micro FROM balances").fetchall()))

    async def ledger(self, key_id: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = await self._run(lambda db: db.execute(
            "SELECT * FROM ledger WHERE key_id = ? ORDER BY id DESC LIMIT ?",
            (key_id, limit)).fetchall())
        return [_entry(r) for r in rows]
