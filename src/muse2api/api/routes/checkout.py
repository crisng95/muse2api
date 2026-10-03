"""Public self-serve checkout behind the ``/billing`` page (see services/payments.py).

No API key: a new customer has none yet, and a top-up names its key in the body.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Request

from ...errors import InvalidRequest
from ...services.container import Services
from ...services.payments import WEBHOOK_MAX_BYTES
from ..deps import billing_ip, get_services
from ..schemas import CheckoutCapture, CheckoutOrder
from .account import SESSION_COOKIE, check_csrf

router = APIRouter(tags=["checkout"])


@router.post("/billing/orders")
async def create_order(body: CheckoutOrder, request: Request,
                       svc: Services = Depends(get_services)) -> dict:
    # Signed in to the portal: top up that customer's wallet (the cookie makes this
    # a state change, hence the CSRF check). Signed out: the email or key decides.
    customer = None
    found = (await svc.customers.session(request.cookies.get(SESSION_COOKIE))
             if svc.settings.portal_enabled else None)
    if found is not None:
        check_csrf(request, found[1], svc)
        customer = found[0]
    return await svc.payments.create_order(body.amount_usd, body.email, body.api_key,
                                           billing_ip(request), customer)


@router.post("/billing/orders/{order_id}/capture")
async def capture_order(order_id: str, request: Request, body: CheckoutCapture | None = None,
                        svc: Services = Depends(get_services)) -> dict:
    return await svc.payments.capture(order_id, body.claim if body else "", billing_ip(request))


@router.post("/billing/paypal/webhook")
async def paypal_webhook(request: Request, svc: Services = Depends(get_services)) -> dict:
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > WEBHOOK_MAX_BYTES:
        raise InvalidRequest("payload too large")
    raw = b""
    async for chunk in request.stream():
        raw += chunk
        if len(raw) > WEBHOOK_MAX_BYTES:
            raise InvalidRequest("payload too large")
    try:
        event = json.loads(raw)
    except ValueError as exc:
        raise InvalidRequest("invalid JSON") from exc
    if not isinstance(event, dict):
        raise InvalidRequest("invalid event")
    return await svc.payments.handle_webhook(dict(request.headers), raw, event, billing_ip(request))
