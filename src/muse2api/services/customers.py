"""Customers: they own keys and one shared wallet, and sign in to the portal.

A customer is created by the admin (with a key), by a self-serve purchase (email
not verified: anyone can type any email at checkout) or by a first Google sign-in.
Signing in finds the customer by Google subject, then by email, but only a
*verified* email (set by the admin or by an earlier Google sign-in) links. An
unverified customer with that email is never merged into the Google identity: it
would hand whoever made that purchase a key on the real owner's wallet. Its email
is freed (kept in its name for the admin) and a fresh verified customer is created;
the owner of such a key can move it over with ``claim_key``, which needs the
plaintext key. Emails are stored lowercase.

Portal sessions are random tokens kept in a cookie; the server stores only their
SHA-256, with a per-session CSRF token, so signing out revokes the session.

``migrate`` gives every key from before customers existed a customer of its own,
once (never grouped by the unverified purchase email, for the same reason; that
email is kept as unverified contact info when no other customer has it). Each key's old balance moves into its customer's wallet
in the same transaction that deletes it, with a zero-amount ledger row saying so
(the key's earlier ledger rows are attributed to the customer too, so the ledger
still adds up to the wallet). An unlimited key stays unlimited on its own; its
customer is not made unlimited. The
key -> customer choice is recorded in ``key_migration`` before keys.json is
written, so a crash in between is finished on the next start without moving any
balance twice.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import secrets
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from ..auth.keys import ApiKey, KeyStore
from ..config import Settings
from ..errors import Forbidden, InvalidRequest
from .billing import Billing, _apply, _balance, _tx, usd

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id             TEXT    PRIMARY KEY,
    email          TEXT    UNIQUE,
    google_sub     TEXT    UNIQUE,
    name           TEXT    NOT NULL DEFAULT '',
    created_at     REAL    NOT NULL,
    unlimited      INTEGER NOT NULL DEFAULT 0,
    email_verified INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash  TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    csrf        TEXT NOT NULL,
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_customer ON sessions (customer_id);
CREATE TABLE IF NOT EXISTS key_migration (
    key_id      TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL
);
"""

# Notes given to keys created by checkout (services/payments.py).
SELF_SERVE_NOTES = ("self-serve", "self-serve (webhook)", "sandbox")


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _norm(email: str | None) -> str | None:
    email = (email or "").strip().lower()
    return email or None


@dataclass
class Customer:
    id: str
    email: str | None
    google_sub: str | None
    name: str
    created_at: float
    unlimited: bool
    email_verified: bool

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> Customer:
        return cls(r["id"], r["email"], r["google_sub"], r["name"], r["created_at"],
                   bool(r["unlimited"]), bool(r["email_verified"]))

    def public(self) -> dict[str, Any]:
        """For the customer themselves and the admin; the Google subject stays inside."""
        d = asdict(self)
        d.pop("google_sub")
        d["google_linked"] = bool(self.google_sub)
        return d


def _insert(db: sqlite3.Connection, email: str | None, name: str, *, unlimited: bool = False,
            google_sub: str | None = None, verified: bool = False) -> str:
    customer_id = "cus_" + uuid.uuid4().hex[:16]
    db.execute("INSERT INTO customers (id, email, google_sub, name, created_at, unlimited, "
               "email_verified) VALUES (?, ?, ?, ?, ?, ?, ?)",
               (customer_id, _norm(email), google_sub, name, time.time(), int(unlimited),
                int(verified)))
    return customer_id


class Customers:
    def __init__(self, settings: Settings, keys: KeyStore, billing: Billing) -> None:
        self.settings = settings
        self.keys = keys
        self.billing = billing
        self._cache: dict[str, Customer] = {}
        self._ready = False
        self._claim_lock = asyncio.Lock()  # one key move at a time: checks then moves

    async def _run(self, fn, *args):
        def call(db: sqlite3.Connection):
            if not self._ready:
                db.executescript(_SCHEMA)
                self._ready = True
            return fn(db, *args)

        return await self.billing._run(call)

    async def _reload(self, customer_id: str) -> Customer:
        row = await self._run(lambda db: db.execute(
            "SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone())
        customer = Customer.from_row(row)
        self._cache[customer.id] = customer
        return customer

    async def load(self) -> None:
        rows = await self._run(lambda db: db.execute("SELECT * FROM customers").fetchall())
        self._cache = {r["id"]: Customer.from_row(r) for r in rows}

    # ---- lookups (in memory) ----
    def get(self, customer_id: str | None) -> Customer | None:
        return self._cache.get(customer_id) if customer_id else None

    def by_email(self, email: str | None) -> Customer | None:
        email = _norm(email)
        return next((c for c in self._cache.values() if email and c.email == email), None)

    def all(self) -> list[Customer]:
        return sorted(self._cache.values(), key=lambda c: c.created_at)

    def keys_of(self, customer_id: str) -> list[ApiKey]:
        return [k for k in self.keys.all() if k.customer_id == customer_id]

    # ---- changes ----
    async def create(self, email: str | None, name: str, *, unlimited: bool = False) -> Customer:
        """New customer; ``sqlite3.IntegrityError`` if the email is taken."""
        customer_id = await self._run(lambda db: _insert(db, email, name, unlimited=unlimited))
        return await self._reload(customer_id)

    async def update(self, customer_id: str, *, name: str | None = None,
                     unlimited: bool | None = None, email: str | None = None) -> Customer:
        """Admin changes. An email set here counts as verified (Google sign-in links it)."""
        if email is not None:
            try:
                await self._run(lambda db: db.execute(
                    "UPDATE customers SET email = ?, email_verified = ? WHERE id = ?",
                    (_norm(email), int(bool(_norm(email))), customer_id)))
            except sqlite3.IntegrityError as exc:
                raise InvalidRequest("another customer has this email") from exc
        if name is not None:
            await self._run(lambda db: db.execute(
                "UPDATE customers SET name = ? WHERE id = ?", (name, customer_id)))
        if unlimited is not None:
            await self._run(lambda db: db.execute(
                "UPDATE customers SET unlimited = ? WHERE id = ?", (int(unlimited), customer_id)))
        return await self._reload(customer_id)

    async def sign_in(self, google_sub: str, email: str, name: str) -> Customer:
        """The customer for a verified Google identity: found, linked or created."""
        email = _norm(email)

        def run(db: sqlite3.Connection) -> str:
            row = db.execute("SELECT id FROM customers WHERE google_sub = ?",
                             (google_sub,)).fetchone()
            if row:
                return row[0]
            row = db.execute("SELECT id, google_sub, email_verified FROM customers WHERE email = ?",
                             (email,)).fetchone()
            if row and row["email_verified"]:
                if row["google_sub"]:  # the email moved to another Google identity
                    raise Forbidden("this email is linked to a different Google sign-in; "
                                    "contact support")
                db.execute("UPDATE customers SET google_sub = ?, "
                           "name = CASE WHEN name = '' OR name = email THEN ? ELSE name END "
                           "WHERE id = ?", (google_sub, name or email, row["id"]))
                log.info("linked Google sign-in to customer %s", row["id"])
                return row["id"]
            if row:
                # Typed at checkout by whoever paid: free the email, merge nothing.
                db.execute("UPDATE customers SET email = NULL, "
                           "name = CASE WHEN name = '' THEN email ELSE name END WHERE id = ?",
                           (row["id"],))
                log.info("freed unverified email of customer %s for a Google sign-in", row["id"])
            return _insert(db, email, name or email, google_sub=google_sub, verified=True)

        customer_id = await self._run(_tx, run)
        await self.load()  # the freed email changed another customer too
        return self._cache[customer_id]

    async def claim_key(self, customer_id: str, plaintext: str) -> tuple[ApiKey, int]:
        """Move a key bought before signing in (and its wallet balance) to ``customer_id``.
        Allowed only with the plaintext key, and only from an unverified customer with
        no Google sign-in whose only key it is. Returns the key and the amount moved."""
        async with self._claim_lock:
            return await self._claim(customer_id, plaintext)

    async def _claim(self, customer_id: str, plaintext: str) -> tuple[ApiKey, int]:
        key = self.keys.verify(plaintext)
        if key is None:
            raise InvalidRequest("unknown or revoked API key", code="invalid_api_key")
        old = self.get(key.customer_id)
        if old is None or old.id == customer_id or old.email_verified or old.google_sub \
                or [k.id for k in self.keys_of(old.id)] != [key.id]:
            raise Forbidden("this key cannot be added here; contact support", code="not_claimable")

        def run(db: sqlite3.Connection) -> int:
            amount = _balance(db, old.id)
            if amount:
                note = f"key {key.name} ({key.id}) moved to customer {customer_id}"
                _apply(db, old.id, key.id, -amount, "adjust", None, note)
                _apply(db, customer_id, key.id, amount, "adjust", None,
                       f"key {key.name} ({key.id}) moved from customer {old.id}")
            return amount

        amount = await self._run(_tx, run)
        # After the wallet moved: a crash here leaves the key on an empty wallet, and
        # claiming it again (still that customer's only key) finishes the move.
        key.customer_id = customer_id
        await self.keys.save()
        log.info("customer %s claimed key %s from %s ($%s)", customer_id, key.id, old.id,
                 usd(amount))
        return key, amount

    # ---- sessions ----
    async def create_session(self, customer_id: str) -> tuple[str, str]:
        """Returns (session token for the cookie, CSRF token for the page)."""
        token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(24)
        now = time.time()

        def run(db: sqlite3.Connection) -> None:
            db.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
            db.execute("INSERT INTO sessions (token_hash, customer_id, csrf, created_at, "
                       "expires_at) VALUES (?, ?, ?, ?, ?)",
                       (_hash(token), customer_id, csrf, now,
                        now + self.settings.session_days * 86400))

        await self._run(run)
        return token, csrf

    async def session(self, token: str | None) -> tuple[Customer, str] | None:
        """(customer, CSRF token) of a live session, or None."""
        if not token:
            return None
        row = await self._run(lambda db: db.execute(
            "SELECT customer_id, csrf FROM sessions WHERE token_hash = ? AND expires_at > ?",
            (_hash(token), time.time())).fetchone())
        customer = self.get(row["customer_id"]) if row else None
        return (customer, row["csrf"]) if customer else None

    @staticmethod
    def csrf_ok(expected: str, provided: str | None) -> bool:
        return bool(provided) and hmac.compare_digest(expected, provided)

    async def end_session(self, token: str | None) -> None:
        if token:
            await self._run(lambda db: db.execute(
                "DELETE FROM sessions WHERE token_hash = ?", (_hash(token),)))

    # ---- one-time move from per-key balances ----
    async def migrate(self) -> int:
        """Give keys without a customer one (see the module docstring). Idempotent."""
        orphans = [k for k in self.keys.all() if not k.customer_id]
        if not orphans:
            return 0

        def run(db: sqlite3.Connection) -> dict[str, str]:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            purchases = "purchases" in tables
            balances = "balances" in tables
            mapping: dict[str, str] = {}
            for key in orphans:
                prior = db.execute("SELECT customer_id FROM key_migration WHERE key_id = ?",
                                   (key.id,)).fetchone()
                if prior:  # chosen by a run that crashed before writing keys.json
                    mapping[key.id] = prior[0]
                    continue
                email = None
                if key.note in SELF_SERVE_NOTES:
                    row = purchases and db.execute(
                        "SELECT email FROM purchases WHERE key_id = ? AND email IS NOT NULL "
                        "ORDER BY created_at LIMIT 1", (key.id,)).fetchone()
                    email = _norm(row[0] if row else key.name if "@" in key.name else None)
                    if email and db.execute("SELECT 1 FROM customers WHERE email = ?",
                                            (email,)).fetchone():
                        email = None  # contact info only, and only where unique
                customer_id = _insert(db, email, key.name)
                db.execute("INSERT INTO key_migration (key_id, customer_id) VALUES (?, ?)",
                           (key.id, customer_id))
                for table in ("ledger", "reservations"):
                    db.execute(f"UPDATE {table} SET customer_id = ? "
                               "WHERE key_id = ? AND customer_id IS NULL", (customer_id, key.id))
                if purchases:
                    db.execute("UPDATE purchases SET customer_id = ? "
                               "WHERE key_id = ? AND customer_id IS NULL", (customer_id, key.id))
                row = balances and db.execute("SELECT balance_micro FROM balances WHERE key_id = ?",
                                              (key.id,)).fetchone()
                if row:
                    db.execute("DELETE FROM balances WHERE key_id = ?", (key.id,))
                    now = time.time()
                    db.execute(
                        "INSERT INTO wallets (customer_id, balance_micro, updated_at) VALUES (?, ?, ?) "
                        "ON CONFLICT (customer_id) DO UPDATE SET "
                        "balance_micro = balance_micro + excluded.balance_micro, "
                        "updated_at = excluded.updated_at", (customer_id, row[0], now))
                    db.execute(
                        "INSERT INTO ledger (ts, customer_id, key_id, amount_micro, kind, ref, note, "
                        "balance_after) VALUES (?, ?, ?, 0, 'adjust', NULL, ?, ?)",
                        (now, customer_id, key.id,
                         f"wallet migration: balance ${usd(row[0])} of key {key.name} ({key.id})",
                         _balance(db, customer_id)))
                mapping[key.id] = customer_id
            return mapping

        mapping = await self._run(_tx, run)
        for key in orphans:
            key.customer_id = mapping[key.id]
        try:
            await self.keys.save()
        except OSError:
            # Finished from key_migration on the next start; in memory it is done.
            log.exception("failed to write customer ids to %s", self.keys.path)
        await self.load()
        log.info("gave %d key(s) a customer and a wallet", len(mapping))
        return len(mapping)
