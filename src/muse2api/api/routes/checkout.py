"""Public self-serve checkout behind the ``/billing`` page (see services/payments.py).

No API key: a new customer has none yet, and a top-up names its key in the body.
"""

from __future__ import annotations

import ipaddress
import json

from fastapi import APIRouter, Depends, Request

from ...errors import InvalidRequest
from ...services.container import Services
from ...services.payments import WEBHOOK_MAX_BYTES
from ..deps import get_services
from ..schemas import CheckoutCapture, CheckoutOrder

router = APIRouter(tags=["checkout"])


def billing_ip(request: Request) -> str:
    """Client address for rate limits. Cloudflare's header is trusted only when the
    connection comes from this machine (the tunnel); any other forwarding header is
    ignored, since a client could set it. IPv6 clients are bucketed by /64."""
    peer = request.client.host if request.client else ""
    ip = peer
    try:
        if ipaddress.ip_address(peer).is_loopback:
            ip = request.headers.get("cf-connecting-ip", "").strip() or peer
    except ValueError:
        pass
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return ip or "?"
    if addr.version == 6:
        return str(ipaddress.ip_network(f"{addr}/64", strict=False))
    return str(addr)


@router.post("/billing/orders")
async def create_order(body: CheckoutOrder, request: Request,
                       svc: Services = Depends(get_services)) -> dict:
    return await svc.payments.create_order(body.amount_usd, body.email, body.api_key,
                                           billing_ip(request))


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
