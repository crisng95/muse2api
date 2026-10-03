"""Minimal PayPal REST client: OAuth token, Orders API v2 and webhook verification.

Amounts always come from our own purchase rows, never from the browser. Every call
that changes state carries a ``PayPal-Request-Id`` so that a retry is idempotent
on PayPal's side too.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx

from ..config import Settings
from ..errors import CheckoutUnavailable, PaymentError

log = logging.getLogger(__name__)

API_BASE = {"sandbox": "https://api-m.sandbox.paypal.com", "live": "https://api-m.paypal.com"}
_TIMEOUT = 30.0
# Webhook headers PayPal signs, as named by the verify-webhook-signature API.
_SIGNATURE_HEADERS = {
    "auth_algo": "paypal-auth-algo",
    "cert_url": "paypal-cert-url",
    "transmission_id": "paypal-transmission-id",
    "transmission_sig": "paypal-transmission-sig",
    "transmission_time": "paypal-transmission-time",
}


class PayPalClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        # Tests pass an httpx.MockTransport.
        self.transport = transport
        self._token = ""
        self._token_expires = 0.0
        self._token_lock = asyncio.Lock()

    @property
    def base(self) -> str:
        return API_BASE[self.settings.paypal_env]

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base, timeout=_TIMEOUT, transport=self.transport)

    async def _access_token(self) -> str:
        s = self.settings
        if not s.checkout_enabled:
            raise CheckoutUnavailable("checkout is not configured")
        async with self._token_lock:
            if self._token and time.time() < self._token_expires:
                return self._token
            try:
                async with self._client() as client:
                    r = await client.post("/v1/oauth2/token", data={"grant_type": "client_credentials"},
                                          auth=(s.paypal_client_id, s.paypal_client_secret))
                r.raise_for_status()
                data = r.json()
            except (httpx.HTTPError, ValueError) as exc:
                log.error("PayPal token request failed: %s", exc)
                raise PaymentError("payment provider unavailable, try again shortly") from exc
            self._token = data["access_token"]
            # Renew a minute early so a token never expires mid-request.
            self._token_expires = time.time() + float(data.get("expires_in", 300)) - 60
            return self._token

    async def _call(self, method: str, path: str, *, body: dict | bytes | None = None,
                    request_id: str | None = None, ok: tuple[int, ...] = (200, 201)) -> httpx.Response:
        headers = {"Authorization": f"Bearer {await self._access_token()}",
                   "Prefer": "return=representation", "Content-Type": "application/json"}
        if request_id:
            headers["PayPal-Request-Id"] = request_id
        content = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
        try:
            async with self._client() as client:
                r = await client.request(method, path, content=content, headers=headers)
        except httpx.HTTPError as exc:
            log.error("PayPal %s %s failed: %s", method, path, exc)
            raise PaymentError("payment provider unavailable, try again shortly") from exc
        if r.status_code == 401:
            self._token = ""  # revoked or expired early; the next call fetches a new one
        if r.status_code not in ok:
            log.error("PayPal %s %s -> %s %s", method, path, r.status_code, r.text[:500])
        return r

    async def create_order(self, purchase_id: str, amount: str) -> str:
        """Create a USD order for ``amount`` (``"10.00"``); returns PayPal's order id."""
        body = {
            "intent": "CAPTURE",
            "purchase_units": [{
                "reference_id": purchase_id,
                "custom_id": purchase_id,
                "description": "Muse API credit",
                "amount": {"currency_code": "USD", "value": amount},
            }],
            "application_context": {"brand_name": "Muse API", "shipping_preference": "NO_SHIPPING",
                                    "user_action": "PAY_NOW"},
        }
        r = await self._call("POST", "/v2/checkout/orders", body=body, request_id=purchase_id)
        if r.status_code not in (200, 201):
            raise PaymentError("could not create the payment, try again shortly")
        return r.json()["id"]

    async def capture_order(self, order_id: str) -> dict[str, Any]:
        """Capture an approved order; an order captured before is fetched instead."""
        r = await self._call("POST", f"/v2/checkout/orders/{order_id}/capture", body={},
                             request_id=f"capture-{order_id}", ok=(200, 201, 422))
        if r.status_code in (200, 201):
            return r.json()
        issue = ""
        if r.status_code == 422:
            details = r.json().get("details") or [{}]
            issue = details[0].get("issue", "")
            if issue == "ORDER_ALREADY_CAPTURED":
                return await self.get_order(order_id)
        if issue == "INSTRUMENT_DECLINED":
            raise PaymentError("the payment was declined; try another card or method",
                               code="payment_declined")
        raise PaymentError("the payment could not be completed")

    async def get_order(self, order_id: str) -> dict[str, Any]:
        r = await self._call("GET", f"/v2/checkout/orders/{order_id}")
        if r.status_code != 200:
            raise PaymentError("could not look up the payment")
        return r.json()

    async def verify_webhook(self, headers: dict[str, str], raw_event: bytes) -> bool:
        """``raw_event`` is the webhook body as received: PayPal signed those exact bytes,
        so it is spliced in rather than parsed and re-serialised."""
        fields = {key: headers.get(name, "") for key, name in _SIGNATURE_HEADERS.items()}
        fields["webhook_id"] = self.settings.paypal_webhook_id
        body = json.dumps(fields)[:-1].encode() + b', "webhook_event": ' + raw_event + b"}"
        try:
            r = await self._call("POST", "/v1/notifications/verify-webhook-signature", body=body)
            return r.status_code == 200 and r.json().get("verification_status") == "SUCCESS"
        except (PaymentError, ValueError):
            return False
