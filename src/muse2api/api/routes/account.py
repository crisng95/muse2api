"""Customer portal: Google sign-in, sessions and the JSON API behind ``/account``.

Every ``/account/api/*`` endpoint is scoped to the session's customer: keys,
requests and wallet of other customers are invisible (a foreign key id is 404).
The session is a cookie, so every state-changing call also needs the page's CSRF
token in ``X-CSRF-Token`` and, when the browser sends one, a same-site Origin.

Request-log rows reach customers without the raw ``error`` text (it can hold
upstream details), the serving account or client details: the page explains a
failure from its status code alone.
"""

from __future__ import annotations

import logging
from typing import Literal
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import RedirectResponse, Response

from ...errors import (
    Forbidden,
    InvalidRequest,
    Muse2APIError,
    NotFound,
    TooManyRequests,
    Unauthorized,
)
from ...services.billing import usd
from ...services.container import Services
from ...services.customers import Customer
from ...services.google import LoginFailed
from ..deps import billing_ip, get_services, public_base
from ..schemas import PortalKeyClaim, PortalKeyCreate, PortalKeyUpdate

log = logging.getLogger(__name__)
router = APIRouter(tags=["account"])

# __Host-: only ever set by this host, over HTTPS, for the whole site.
SESSION_COOKIE = "__Host-m2a_session"
_STATE_COOKIE = "__Host-m2a_oauth"
# Where a finished sign-in may send the browser back to.
_NEXT_PATHS = ("/account", "/billing")
# Request-log fields a customer may see.
_REQUEST_FIELDS = ("id", "ts", "method", "path", "model", "key_id", "status_code", "latency_ms",
                   "stream", "task_status", "task_ms")


class PortalUnavailable(Muse2APIError):
    status_code = 503
    error_type = "server_error"
    code = "portal_unavailable"


def _redirect_uri(request: Request, svc: Services) -> str:
    return (svc.settings.google_redirect_uri
            or f"{public_base(request)}/account/auth/google/callback")


def _cookie(resp: Response, name: str, value: str, max_age: int, path: str = "/") -> None:
    resp.set_cookie(name, value, max_age=max_age, path=path, httponly=True, secure=True,
                    samesite="lax")


async def portal_session(request: Request,
                         svc: Services = Depends(get_services)) -> tuple[Customer, str]:
    """(customer, CSRF token) of the signed-in browser; 401 otherwise."""
    if not svc.settings.portal_enabled:
        raise PortalUnavailable("the customer portal is not available")
    found = await svc.customers.session(request.cookies.get(SESSION_COOKIE))
    if found is None:
        raise Unauthorized("not signed in", code="not_signed_in")
    return found


def check_csrf(request: Request, csrf: str, svc: Services) -> None:
    origin = request.headers.get("origin")
    if origin and urlparse(origin).netloc != request.headers.get("host"):
        raise Forbidden("cross-site request refused")
    if not svc.customers.csrf_ok(csrf, request.headers.get("x-csrf-token")):
        raise Forbidden("missing or invalid CSRF token", code="csrf_failed")


async def portal_writer(request: Request, found: tuple[Customer, str] = Depends(portal_session),
                        svc: Services = Depends(get_services)) -> Customer:
    """The signed-in customer, for a state-changing call (CSRF-checked)."""
    check_csrf(request, found[1], svc)
    return found[0]


# ---- sign-in ----
@router.get("/account/login", include_in_schema=False)
async def login(request: Request, next: str = "/account",
                svc: Services = Depends(get_services)) -> Response:
    if not svc.settings.portal_enabled:
        return RedirectResponse("/account", status_code=303)
    if not svc.google.login_limit.hit(billing_ip(request)):
        raise TooManyRequests("too many sign-in attempts; try again in a few minutes")
    url, cookie = svc.google.start(_redirect_uri(request, svc),
                                   next if next in _NEXT_PATHS else "/account")
    resp = RedirectResponse(url, status_code=303)
    _cookie(resp, _STATE_COOKIE, cookie, 600)
    return resp


@router.get("/account/auth/google/callback", include_in_schema=False)
async def google_callback(request: Request, code: str = "", state: str = "", error: str = "",
                          svc: Services = Depends(get_services)) -> Response:
    if not svc.settings.portal_enabled:
        return RedirectResponse("/account", status_code=303)
    failed = RedirectResponse("/account?login=failed", status_code=303)
    failed.delete_cookie(_STATE_COOKIE, path="/", secure=True, httponly=True, samesite="lax")
    try:
        pending = svc.google.take(state, request.cookies.get(_STATE_COOKIE))
        if error or not code:
            raise LoginFailed(error or "no authorization code")
        claims = await svc.google.exchange(code, _redirect_uri(request, svc), pending)
        customer = await svc.customers.sign_in(claims["sub"], claims["email"],
                                               str(claims.get("name") or ""))
    except (LoginFailed, Forbidden) as exc:
        log.warning("Google sign-in failed: %s", exc.message)
        return failed
    token, _ = await svc.customers.create_session(customer.id)
    resp = RedirectResponse(pending.next, status_code=303)
    resp.delete_cookie(_STATE_COOKIE, path="/", secure=True, httponly=True, samesite="lax")
    _cookie(resp, SESSION_COOKIE, token, int(svc.settings.session_days * 86400))
    log.info("customer %s signed in", customer.id)
    return resp


@router.post("/account/logout")
async def logout(request: Request, found: tuple[Customer, str] = Depends(portal_session),
                 svc: Services = Depends(get_services)) -> Response:
    check_csrf(request, found[1], svc)
    await svc.customers.end_session(request.cookies.get(SESSION_COOKIE))
    resp = Response('{"ok":true}', media_type="application/json")
    resp.delete_cookie(SESSION_COOKIE, path="/", secure=True, httponly=True, samesite="lax")
    return resp


# ---- portal API ----
def _own_key(svc: Services, customer: Customer, key_id: str):
    key = svc.keys.get(key_id)
    if key is None or key.customer_id != customer.id:
        raise NotFound("key not found")
    return key


def _key_view(key, spend: dict) -> dict:
    return {"id": key.id, "name": key.name, "prefix": key.prefix, "created_at": key.created_at,
            "last_used_at": key.last_used_at or None, "revoked": key.revoked,
            "spend_usd": usd(spend.get(key.id) or 0)}


@router.get("/account/api/me")
async def me(found: tuple[Customer, str] = Depends(portal_session),
             svc: Services = Depends(get_services)) -> dict:
    customer, csrf = found
    return {"customer": {"id": customer.id, "email": customer.email, "name": customer.name,
                         "unlimited": customer.unlimited},
            "balance_usd": usd(await svc.billing.balance(customer.id)),
            "csrf": csrf, "checkout": svc.settings.checkout_enabled}


@router.get("/account/api/usage")
async def usage(days: int = 7, found: tuple[Customer, str] = Depends(portal_session),
                svc: Services = Depends(get_services)) -> dict:
    if days not in (7, 30):
        raise InvalidRequest("days must be 7 or 30")
    customer = found[0]
    keys = {k.id: k.name for k in svc.customers.keys_of(customer.id)}
    data = await svc.requests.usage(list(keys), days)

    def money(rows: list[dict]) -> list[dict]:
        return [{**{k: v for k, v in r.items() if k != "cost_micro"},
                 "cost_usd": usd(r["cost_micro"])} for r in rows]

    return {"days": days, "since": data["since"], "series": money(data["series"]),
            "by_model": money(data["by_model"]),
            "by_key": [{**r, "key_name": keys.get(r["key_id"])} for r in money(data["by_key"])]}


@router.get("/account/api/keys")
async def list_keys(found: tuple[Customer, str] = Depends(portal_session),
                    svc: Services = Depends(get_services)) -> dict:
    customer = found[0]
    spend = await svc.billing.spend(customer.id)
    keys = sorted(svc.customers.keys_of(customer.id), key=lambda k: k.created_at, reverse=True)
    return {"data": [_key_view(k, spend) for k in keys]}


@router.post("/account/api/keys")
async def create_key(body: PortalKeyCreate, customer: Customer = Depends(portal_writer),
                     svc: Services = Depends(get_services)) -> dict:
    if sum(not k.revoked for k in svc.customers.keys_of(customer.id)) >= 50:
        raise Forbidden("key limit reached; revoke unused keys or contact support")
    key, plaintext = await svc.keys.create(body.name.strip(), "portal", customer.id)
    # The only time the plaintext key is ever returned.
    return {"key": _key_view(key, {}), "api_key": plaintext}


@router.post("/account/api/keys/claim")
async def claim_key(body: PortalKeyClaim, customer: Customer = Depends(portal_writer),
                    svc: Services = Depends(get_services)) -> dict:
    """Add a key bought before signing in (see Customers.claim_key)."""
    key, moved = await svc.customers.claim_key(customer.id, body.api_key.strip())
    return {"key": _key_view(key, await svc.billing.spend(customer.id)),
            "moved_usd": usd(moved), "balance_usd": usd(await svc.billing.balance(customer.id))}


@router.patch("/account/api/keys/{key_id}")
async def rename_key(key_id: str, body: PortalKeyUpdate, customer: Customer = Depends(portal_writer),
                     svc: Services = Depends(get_services)) -> dict:
    key = _own_key(svc, customer, key_id)
    key.name = body.name.strip()
    await svc.keys.save()
    return {"key": _key_view(key, await svc.billing.spend(customer.id))}


@router.post("/account/api/keys/{key_id}/revoke")
async def revoke_key(key_id: str, customer: Customer = Depends(portal_writer),
                     svc: Services = Depends(get_services)) -> dict:
    key = _own_key(svc, customer, key_id)
    key.revoked = True
    await svc.keys.save()
    return {"key": _key_view(key, await svc.billing.spend(customer.id))}


@router.get("/account/api/requests")
async def list_requests(
    key_id: str | None = None,
    status: Literal["2xx", "4xx", "5xx", "failed"] | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    found: tuple[Customer, str] = Depends(portal_session),
    svc: Services = Depends(get_services),
) -> dict:
    customer = found[0]
    names = {k.id: k.name for k in svc.customers.keys_of(customer.id)}
    if key_id:
        _own_key(svc, customer, key_id)
    res = await svc.requests.query(key_ids=[key_id] if key_id else list(names), status=status,
                                   hide_polls=True, limit=limit, offset=offset)
    rows = [{**{f: r.get(f) for f in _REQUEST_FIELDS}, "key_name": names.get(r["key_id"]),
             "task": bool(r.get("task_id")),
             "cost_usd": usd(r["cost_micro"]) if r.get("cost_micro") is not None else None}
            for r in res["data"]]
    return {"data": rows, "total": res["total"]}


@router.get("/account/api/billing")
async def billing_history(found: tuple[Customer, str] = Depends(portal_session),
                          svc: Services = Depends(get_services)) -> dict:
    """Top-ups, refunds and adjustments of the wallet (spending is under requests),
    and the customer's PayPal purchases."""
    customer = found[0]
    ledger = await svc.billing.ledger(customer.id, 200, kinds=("topup", "refund", "adjust"))
    purchases = [{k: p[k] for k in ("id", "created_at", "completed_at", "amount_usd", "status",
                                     "paypal_order_id", "mode")}
                 for p in await svc.payments.recent(200, customer_id=customer.id)
                 if p["status"] != "pending"]  # abandoned checkouts are noise
    # Notes stay out: they are written for the admin (payer emails, invoices).
    return {"ledger": [{k: e[k] for k in ("id", "ts", "kind", "amount_usd", "balance_after_usd",
                                          "ref")} for e in ledger],
            "purchases": purchases}
