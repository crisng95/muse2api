"""Self-serve credit purchases through PayPal (the ``/billing`` page).

A purchase row is written when the PayPal order is created, with the amount we
charge; the browser never gets a say in it. Completing a purchase marks the row
completed and credits the key in one transaction (ledger ``topup``, ref = PayPal
capture id), so a payment is credited at most once, whether the capture call or
the ``PAYMENT.CAPTURE.COMPLETED`` webhook gets there first.

Creating an order also returns a secret claim token (only its hash is stored).
The order id alone is not enough to capture: it shows up in PayPal's redirect
URLs, and the capture call hands out a new customer's API key.

New customers get their API key when their payment completes. The plaintext is
returned once, to the capture call that created the key; a key created from a
webhook (the capture call never finished) is never shown, so that customer has
to ask support for a replacement.

Purchase status: ``pending`` -> ``completed``, or ``failed`` (declined, amount
mismatch), ``refunded`` (refunded or reversed before it was credited) or
``needs_support`` (paid, but the key to top up was revoked or deleted meanwhile).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import re
import secrets
import sqlite3
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlparse

from ..auth.keys import KeyStore
from ..config import Settings
from ..errors import CheckoutUnavailable, InvalidRequest, NotFound, PaymentError, TooManyRequests
from .billing import MICRO, Billing, _apply, _balance, _tx, usd
from .paypal import PayPalClient

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS purchases (
    id              TEXT    PRIMARY KEY,
    paypal_order_id TEXT    NOT NULL UNIQUE,
    amount_micro    INTEGER NOT NULL,
    mode            TEXT    NOT NULL,
    email           TEXT,
    key_id          TEXT,
    status          TEXT    NOT NULL,
    capture_id      TEXT    UNIQUE,
    payer_email     TEXT,
    created_at      REAL    NOT NULL,
    completed_at    REAL
);
CREATE INDEX IF NOT EXISTS idx_purchases_created ON purchases (created_at);
"""
# Columns added after the table first shipped; created on open when missing.
_ADDED_COLUMNS = {"claim_hash": "TEXT"}

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_NAME_LEN = 100  # KeyCreate.name max length

# (requests, window in seconds) for each in-memory limit.
ORDER_LIMIT = (10, 600.0)  # order creation per client IP
ORDER_GLOBAL_LIMIT = (60, 60.0)  # order creation, everyone together
CAPTURE_LIMIT = (30, 600.0)  # capture calls per client IP
WEBHOOK_LIMIT = (120, 60.0)  # webhook deliveries per sender IP

WEBHOOK_EVENTS = ("PAYMENT.CAPTURE.COMPLETED", "PAYMENT.CAPTURE.REFUNDED",
                  "PAYMENT.CAPTURE.REVERSED")
WEBHOOK_MAX_BYTES = 64 * 1024
_WEBHOOK_HEADERS = ("paypal-auth-algo", "paypal-cert-url", "paypal-transmission-id",
                    "paypal-transmission-sig", "paypal-transmission-time")
_CERT_HOSTS = ("api.paypal.com", "api.sandbox.paypal.com")
_MAX_CLOCK_SKEW = 300.0

_NEEDS_SUPPORT = {
    "status": "needs_support",
    "message": "the payment went through, but the key it was for is no longer active; "
               "contact support with your PayPal receipt to have the credit applied",
}


class Throttle:
    """Sliding-window request counter per key, kept in memory."""

    def __init__(self, limit: int, window: float) -> None:
        self.limit, self.window = limit, window
        self._hits: dict[str, deque[float]] = {}

    def hit(self, key: str) -> bool:
        """Count one request for ``key``; False once it is over the limit."""
        now = time.monotonic()
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        if len(self._hits) > 10_000:  # forget idle clients
            for k in [k for k, h in self._hits.items() if not h or now - h[-1] > self.window]:
                del self._hits[k]
        return True


def _hash(claim: str) -> str:
    return hashlib.sha256(claim.encode()).hexdigest()


def _micro(value: Any) -> int | None:
    try:
        amount = Decimal(str(value)) * MICRO
    except (InvalidOperation, ValueError):
        return None
    return int(amount) if amount.is_finite() and amount == amount.to_integral_value() else None


def _capture_of(order: dict[str, Any]) -> dict[str, Any] | None:
    for unit in order.get("purchase_units") or []:
        for capture in (unit.get("payments") or {}).get("captures") or []:
            return capture
    return None


def _related_order(resource: dict[str, Any]) -> str | None:
    """The order a capture, refund or reversal event belongs to, if it says."""
    related = (resource.get("supplementary_data") or {}).get("related_ids") or {}
    if related.get("order_id"):
        return related["order_id"]
    for link in resource.get("links") or []:
        href = link.get("href", "")
        if "/checkout/orders/" in href:
            return href.split("/checkout/orders/", 1)[1].split("/")[0]
    return None


def signature_headers_problem(headers: dict[str, str], now: float | None = None) -> str:
    """Cheap checks on a webhook's signature headers, before asking PayPal to verify."""
    if any(not headers.get(h) for h in _WEBHOOK_HEADERS):
        return "missing PayPal signature headers"
    cert = urlparse(headers["paypal-cert-url"])
    if cert.scheme != "https" or cert.hostname not in _CERT_HOSTS:
        return "unexpected certificate URL"
    try:
        sent = datetime.fromisoformat(headers["paypal-transmission-time"].replace("Z", "+00:00"))
        if sent.tzinfo is None:
            sent = sent.replace(tzinfo=timezone.utc)
    except ValueError:
        return "invalid transmission time"
    if abs((now or time.time()) - sent.timestamp()) > _MAX_CLOCK_SKEW:
        return "stale transmission time"
    return ""


def _complete_tx(db: sqlite3.Connection, purchase: dict[str, Any], key_id: str,
                 capture_id: str, payer_email: str | None, note: str) -> int | None:
    """Mark the purchase completed and credit the key; None unless it was still pending."""
    status = db.execute("SELECT status FROM purchases WHERE id = ?", (purchase["id"],)).fetchone()[0]
    if status != "pending":
        return None
    db.execute("UPDATE purchases SET status = 'completed', key_id = ?, capture_id = ?, "
               "payer_email = ?, completed_at = ? WHERE id = ?",
               (key_id, capture_id, payer_email, time.time(), purchase["id"]))
    return _apply(db, key_id, purchase["amount_micro"], "topup", capture_id, note)


def _debit_tx(db: sqlite3.Connection, key_id: str, amount: int, ref: str, note: str) -> bool:
    if db.execute("SELECT 1 FROM ledger WHERE ref = ? AND kind = 'adjust'", (ref,)).fetchone():
        return False
    _apply(db, key_id, -amount, "adjust", ref, note)
    return True


class Payments:
    def __init__(self, settings: Settings, keys: KeyStore, billing: Billing,
                 paypal: PayPalClient) -> None:
        self.settings = settings
        self.keys = keys
        self.billing = billing
        self.paypal = paypal
        self._ready = False
        # Serialises the capture call and the webhook for one order.
        self._locks: dict[str, asyncio.Lock] = {}
        self.order_limit = Throttle(*ORDER_LIMIT)
        self.order_global_limit = Throttle(*ORDER_GLOBAL_LIMIT)
        self.capture_limit = Throttle(*CAPTURE_LIMIT)
        self.webhook_limit = Throttle(*WEBHOOK_LIMIT)

    async def _run(self, fn, *args):
        def call(db: sqlite3.Connection):
            if not self._ready:
                db.executescript(_SCHEMA)
                have = {r[1] for r in db.execute("PRAGMA table_info(purchases)")}
                for col, kind in _ADDED_COLUMNS.items():
                    if col not in have:
                        db.execute(f"ALTER TABLE purchases ADD COLUMN {col} {kind}")
                self._ready = True
            return fn(db, *args)

        return await self.billing._run(call)

    def _require_enabled(self) -> None:
        if not self.settings.checkout_enabled:
            raise CheckoutUnavailable("online checkout is not available; contact support to top up")

    async def _get(self, order_id: str) -> dict[str, Any] | None:
        row = await self._run(lambda db: db.execute(
            "SELECT * FROM purchases WHERE paypal_order_id = ?", (order_id,)).fetchone())
        return dict(row) if row else None

    async def _find(self, column: str, value: str | None) -> dict[str, Any] | None:
        if not value:
            return None
        row = await self._run(lambda db: db.execute(
            f"SELECT * FROM purchases WHERE {column} = ?", (value,)).fetchone())
        return dict(row) if row else None

    async def _mark(self, purchase: dict[str, Any], status: str, capture_id: str | None) -> None:
        """Close a purchase that was not credited (it must not be credited later)."""
        await self._run(lambda db: db.execute(
            "UPDATE purchases SET status = ?, capture_id = COALESCE(capture_id, ?), "
            "completed_at = ? WHERE id = ? AND status != 'completed'",
            (status, capture_id, time.time(), purchase["id"])))

    # ---- checkout ----
    async def create_order(self, amount_usd: float, email: str | None, api_key: str | None,
                           ip: str) -> dict[str, str]:
        """Validate a purchase, create its PayPal order and record it.
        Returns the order id and the claim token that ``capture`` asks for."""
        s = self.settings
        self._require_enabled()
        if not self.order_limit.hit(ip) or not self.order_global_limit.hit("*"):
            raise TooManyRequests("too many checkout attempts; try again in a few minutes")
        cents = round(amount_usd * 100)
        if abs(amount_usd * 100 - cents) > 1e-6:
            raise InvalidRequest("amount_usd must have at most 2 decimals")
        if not s.topup_min_usd <= amount_usd <= s.topup_max_usd:
            raise InvalidRequest(f"amount_usd must be between ${s.topup_min_usd:g} "
                                 f"and ${s.topup_max_usd:g}")
        api_key, email = (api_key or "").strip(), (email or "").strip()
        if api_key:
            if api_key in {s.api_key, s.admin_key} - {""}:
                raise InvalidRequest("this key cannot be topped up")
            key = self.keys.verify(api_key)
            if key is None:
                raise InvalidRequest("unknown or revoked API key", code="invalid_api_key")
            mode, key_id, email = "topup", key.id, None
        else:
            if len(email) > 254 or not _EMAIL.match(email):
                raise InvalidRequest("a valid email is required", code="invalid_email")
            mode, key_id = "new", None
        purchase_id = "pur_" + uuid.uuid4().hex[:16]
        claim = secrets.token_urlsafe(24)
        order_id = await self.paypal.create_order(purchase_id, f"{cents / 100:.2f}")
        await self._run(lambda db: db.execute(
            "INSERT INTO purchases (id, paypal_order_id, amount_micro, mode, email, key_id, "
            "status, created_at, claim_hash) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
            (purchase_id, order_id, cents * 10_000, mode, email, key_id, time.time(),
             _hash(claim))))
        log.info("checkout %s: order %s, $%.2f, %s", purchase_id, order_id, cents / 100, mode)
        return {"id": order_id, "claim": claim}

    async def capture(self, order_id: str, claim: str | None, ip: str) -> dict[str, Any]:
        """Capture an approved order and credit it. Repeating the call is harmless."""
        self._require_enabled()
        if not self.capture_limit.hit(ip):
            raise TooManyRequests("too many attempts; try again in a few minutes")
        purchase = await self._get(order_id)
        # A wrong claim looks exactly like an unknown order.
        if (purchase is None or not purchase["claim_hash"] or not claim
                or not hmac.compare_digest(_hash(claim), purchase["claim_hash"])):
            raise NotFound("unknown order")
        async with self._locks.setdefault(order_id, asyncio.Lock()):
            purchase = await self._get(order_id)
            if purchase["status"] == "completed":
                return await self._already(purchase)
            if purchase["status"] == "needs_support":
                return dict(_NEEDS_SUPPORT)
            if purchase["status"] != "pending":
                raise PaymentError("this payment cannot be completed; contact support")
            order = await self.paypal.capture_order(order_id)
            capture = _capture_of(order) or {}
            state = capture.get("status")
            if state in ("DECLINED", "FAILED"):
                await self._mark(purchase, "failed", capture.get("id"))
                raise PaymentError("the payment was declined; try another card or method",
                                   code="payment_declined")
            if state != "COMPLETED":
                # e.g. PENDING under review: the webhook credits it once it clears.
                return {"status": "pending",
                        "message": "the payment is still being processed; the credit is added "
                                   "as soon as it clears"}
            await self._check(purchase, capture)
            payer = (order.get("payer") or {}).get("email_address")
            return await self._complete(purchase, capture["id"], payer, reveal=True)

    async def _check(self, purchase: dict[str, Any], capture: dict[str, Any]) -> None:
        amount = capture.get("amount") or {}
        if amount.get("currency_code") != "USD" or _micro(amount.get("value")) != purchase["amount_micro"]:
            await self._mark(purchase, "failed", capture.get("id"))
            log.error("purchase %s: captured %s does not match $%s", purchase["id"], amount,
                      usd(purchase["amount_micro"]))
            raise PaymentError("the payment does not match the order; contact support",
                               code="payment_mismatch")

    async def _complete(self, purchase: dict[str, Any], capture_id: str,
                        payer_email: str | None, *, reveal: bool) -> dict[str, Any]:
        key_id, plaintext = purchase["key_id"], None
        if purchase["mode"] == "topup":
            key = self.keys.get(key_id)
            if key is None or key.revoked:
                # Paid, but crediting a dead key would strand the money: support decides.
                await self._mark(purchase, "needs_support", capture_id)
                log.error("purchase %s paid for revoked or deleted key %s", purchase["id"], key_id)
                return dict(_NEEDS_SUPPORT)
        else:
            if self.settings.paypal_env == "sandbox":
                note = "sandbox"
            else:
                note = "self-serve" if reveal else "self-serve (webhook)"
            key, plaintext = await self.keys.create(purchase["email"][:_NAME_LEN], note)
            key_id = key.id
        note = f"PayPal {purchase['paypal_order_id']} {payer_email or purchase['email'] or ''}".strip()
        try:
            balance = await self._run(_tx, _complete_tx, purchase, key_id, capture_id,
                                      payer_email, note)
        except BaseException:
            if plaintext:
                await self.keys.remove(key_id)
            raise
        if balance is None:
            if plaintext:
                await self.keys.remove(key_id)
            return await self._already(await self._get(purchase["paypal_order_id"]))
        log.info("purchase %s completed: $%s to %s", purchase["id"], usd(purchase["amount_micro"]),
                 key_id)
        result: dict[str, Any] = {"status": "completed", "balance_usd": usd(balance)}
        if plaintext and reveal:
            result["api_key"] = plaintext
        return result

    async def _already(self, purchase: dict[str, Any]) -> dict[str, Any]:
        """Answer to a repeated capture, made with the right claim token."""
        if purchase["status"] != "completed":
            return {"status": purchase["status"]}
        balance = await self._run(_balance, purchase["key_id"])
        message = "this payment was already credited"
        if purchase["mode"] == "new":
            message += ("; the API key is shown only once, when the payment completes. "
                        "If you did not save it, contact support")
        return {"status": "completed", "balance_usd": usd(balance), "message": message}

    # ---- webhook ----
    async def handle_webhook(self, headers: dict[str, str], raw: bytes,
                             event: dict[str, Any], ip: str) -> dict[str, Any]:
        """``raw`` is the body exactly as received (PayPal verifies those bytes)."""
        if not (self.settings.checkout_enabled and self.settings.paypal_webhook_id):
            return {"ignored": "webhooks are not configured"}
        if not self.webhook_limit.hit(ip):
            raise TooManyRequests("too many webhook deliveries")
        kind, resource = event.get("event_type"), event.get("resource") or {}
        if kind not in WEBHOOK_EVENTS:
            return {"ignored": "event type not handled"}
        headers = {k.lower(): v for k, v in headers.items()}
        if problem := signature_headers_problem(headers):
            log.warning("rejected PayPal webhook %s: %s", event.get("id"), problem)
            raise InvalidRequest(problem, code="invalid_signature")
        if not await self.paypal.verify_webhook(headers, raw):
            log.warning("rejected PayPal webhook %s: bad signature", event.get("id"))
            raise InvalidRequest("invalid webhook signature", code="invalid_signature")
        if kind == "PAYMENT.CAPTURE.COMPLETED":
            await self._capture_completed(resource)
        else:
            await self._capture_refunded(kind, resource)
        return {"ok": True}

    async def _capture_completed(self, capture: dict[str, Any]) -> None:
        """Credit a payment whose capture call never finished on our side."""
        order_id = _related_order(capture)
        if not order_id or await self._get(order_id) is None:
            log.warning("webhook for unknown order %s (capture %s)", order_id, capture.get("id"))
            return
        async with self._locks.setdefault(order_id, asyncio.Lock()):
            purchase = await self._get(order_id)
            if purchase["status"] != "pending" or capture.get("status") != "COMPLETED":
                return
            try:
                await self._check(purchase, capture)
            except PaymentError:
                return  # logged; a 2xx stops PayPal from redelivering
            await self._complete(purchase, capture["id"], None, reveal=False)

    async def _capture_refunded(self, kind: str, resource: dict[str, Any]) -> None:
        """Take back the credit of a refunded or reversed payment (may go below zero), or
        close the purchase so that it is never credited if it was not yet."""
        up = next((link.get("href", "") for link in resource.get("links") or []
                   if link.get("rel") == "up"), "")
        if "/captures/" in up:  # a refund object, linked to its capture
            capture_id, ref = up.rstrip("/").rsplit("/", 1)[-1], resource.get("id")
        else:  # the capture itself
            capture_id, ref = resource.get("id"), f"reversal-{resource.get('id')}"
        # A purchase that was never credited has no capture id yet.
        purchase = (await self._find("capture_id", capture_id)
                    or await self._find("paypal_order_id", _related_order(resource))
                    or await self._find("id", resource.get("custom_id")))
        label = kind.rsplit(".", 1)[-1].lower()
        if purchase is None or not ref:
            log.warning("ignored %s for unknown capture %s", kind, capture_id)
            return
        async with self._locks.setdefault(purchase["paypal_order_id"], asyncio.Lock()):
            purchase = await self._get(purchase["paypal_order_id"])
            if purchase["status"] != "completed" or not purchase["key_id"]:
                await self._mark(purchase, "refunded", capture_id)
                log.warning("purchase %s (%s) %s before it was credited", purchase["id"],
                            purchase["status"], label)
                return
            amount = resource.get("amount") or {}
            micro = _micro(amount.get("value"))
            if not micro or micro < 0 or amount.get("currency_code") != "USD":
                log.warning("ignored %s for capture %s (%s)", kind, capture_id, amount)
                return
            key_id = purchase["key_id"]
            if await self._run(_tx, _debit_tx, key_id, micro, ref, f"PayPal {label} {capture_id}"):
                log.info("PayPal %s: -$%s from %s", label, usd(micro), key_id)

    # ---- admin ----
    async def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = await self._run(lambda db: db.execute(
            "SELECT * FROM purchases ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall())
        return [{**{k: r[k] for k in r.keys() if k not in ("amount_micro", "claim_hash")},
                 "amount_usd": usd(r["amount_micro"])} for r in rows]
