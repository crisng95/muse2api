"""Self-serve PayPal checkout (/billing), against a fake PayPal API."""

from __future__ import annotations

import itertools
import json
import re
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from muse2api.api.routes import pages
from muse2api.api.routes.checkout import billing_ip
from muse2api.app import create_app
from muse2api.drivers.mock import MockDriver
from muse2api.services import payments

# Customers now have an account (/account); the upstream pool's accounts must not show.
_INTERNAL = re.compile(r"cookie|muse\.ai|muse2api|browser|chromium|account[_ ]?ids?\b|"
                       r"accounts\.json|account pool", re.I)


class FakePayPal:
    def __init__(self) -> None:
        self.orders: dict[str, dict] = {}
        self.ids = itertools.count(1)
        self.calls: list[tuple[str, str]] = []
        self.verify = "SUCCESS"
        self.verified: list[bytes] = []
        # Overrides for what a capture reports.
        self.capture_amount: str | None = None
        self.capture_currency = "USD"
        self.capture_status = "COMPLETED"

    def order_body(self, order_id: str) -> dict:
        o = self.orders[order_id]
        capture = {"id": f"CAP-{order_id}", "status": self.capture_status,
                   "amount": {"currency_code": self.capture_currency,
                              "value": self.capture_amount or o["amount"]}}
        return {"id": order_id, "status": "COMPLETED" if o["captured"] else "APPROVED",
                "payer": {"email_address": "payer@example.com"},
                "purchase_units": [{"reference_id": o["purchase"],
                                    "payments": {"captures": [capture] if o["captured"] else []}}]}

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((request.method, path))
        if path == "/v1/oauth2/token":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        assert request.headers["authorization"] == "Bearer tok"
        if path == "/v2/checkout/orders" and request.method == "POST":
            unit = json.loads(request.content)["purchase_units"][0]
            order_id = f"ORDER{next(self.ids)}"
            self.orders[order_id] = {"amount": unit["amount"]["value"], "purchase": unit["custom_id"],
                                     "currency": unit["amount"]["currency_code"], "captured": False}
            return httpx.Response(201, json={"id": order_id, "status": "CREATED"})
        if m := re.fullmatch(r"/v2/checkout/orders/(\w+)/capture", path):
            order = self.orders[m.group(1)]
            if order["captured"]:
                return httpx.Response(422, json={"details": [{"issue": "ORDER_ALREADY_CAPTURED"}]})
            order["captured"] = True
            return httpx.Response(201, json=self.order_body(m.group(1)))
        if m := re.fullmatch(r"/v2/checkout/orders/(\w+)", path):
            return httpx.Response(200, json=self.order_body(m.group(1)))
        if path == "/v1/notifications/verify-webhook-signature":
            self.verified.append(request.content)
            return httpx.Response(200, json={"verification_status": self.verify})
        return httpx.Response(404)


@pytest.fixture
def paypal() -> FakePayPal:
    return FakePayPal()


def _shop(settings, paypal, **overrides):
    settings.paypal_client_id, settings.paypal_client_secret = "cid", "secret"
    settings.paypal_env, settings.paypal_webhook_id = "live", "WH-1"
    for k, v in overrides.items():
        setattr(settings, k, v)
    app = create_app(settings, MockDriver(delay=0))
    app.state.services.payments.paypal.transport = httpx.MockTransport(paypal.handle)
    return TestClient(app)


@pytest.fixture
def shop(settings, paypal, monkeypatch):
    for name in ("ORDER_LIMIT", "ORDER_GLOBAL_LIMIT", "CAPTURE_LIMIT", "WEBHOOK_LIMIT"):
        monkeypatch.setattr(payments, name, (1000, 600.0))
    with _shop(settings, paypal) as c:
        yield c


def _order(c, **body):
    return c.post("/billing/orders", json={"amount_usd": 25, **body})


def _capture(c, order: dict, claim: str | None = None):
    return c.post(f"/billing/orders/{order['id']}/capture",
                  json={"claim": order["claim"] if claim is None else claim})


def _buy(c, **body) -> dict:
    order = _order(c, **body)
    assert order.status_code == 200, order.text
    r = _capture(c, order.json())
    assert r.status_code == 200, r.text
    return {"order_id": order.json()["id"], "claim": order.json()["claim"], **r.json()}


def _signed() -> dict:
    return {"paypal-auth-algo": "SHA256withRSA",
            "paypal-cert-url": "https://api.paypal.com/v1/notifications/certs/CERT-1",
            "paypal-transmission-id": "tid", "paypal-transmission-sig": "sig",
            "paypal-transmission-time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def _hook(c, event: dict, **headers):
    return c.post("/billing/paypal/webhook", content=json.dumps(event, ensure_ascii=False).encode(),
                  headers={"content-type": "application/json", **_signed(), **headers})


def _new_key(c, admin, **extra) -> tuple[str, str]:
    body = c.post("/admin/keys", headers=admin, json={"name": "client"}).json()
    if extra:
        c.patch(f"/admin/keys/{body['key']['id']}", headers=admin, json=extra)
    return body["key"]["id"], body["api_key"]


def _ledger(c, admin, key_id) -> list[dict]:
    return c.get(f"/admin/keys/{key_id}/ledger", headers=admin).json()["data"]


def _event(kind: str, resource: dict) -> dict:
    return {"id": "WH-EVT", "event_type": kind, "resource": resource}


def test_order_validation(shop, admin, auth, paypal):
    for amount in (4.99, 1000.01, 10.001, 0, -5):
        assert _order(shop, amount_usd=amount, email="a@b.co").status_code == 400
    for literal in ("NaN", "Infinity", "-Infinity"):
        r = shop.post("/billing/orders", content=f'{{"amount_usd": {literal}, "email": "a@b.co"}}',
                      headers={"content-type": "application/json"})
        assert r.status_code == 400, literal
    r = _order(shop)
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_email"
    assert _order(shop, email="not-an-email").status_code == 400
    r = _order(shop, api_key="m2a-unknown")
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_api_key"
    key_id, plaintext = _new_key(shop, admin, revoked=True)
    assert _order(shop, api_key=plaintext).status_code == 400
    for builtin in (admin, auth):
        assert _order(shop, api_key=builtin["Authorization"][7:]).status_code == 400
    assert paypal.orders == {}  # nothing invalid reached PayPal

    r = _order(shop, amount_usd=10.5, email="a@b.co")
    assert r.status_code == 200 and set(r.json()) == {"id", "claim"}
    (order,) = paypal.orders.values()
    assert order["amount"] == "10.50" and order["currency"] == "USD"
    assert order["purchase"].startswith("pur_")


def test_new_customer_gets_a_billed_key_once(shop, admin, paypal):
    bought = _buy(shop, email="new@example.com")
    assert bought["status"] == "completed" and bought["balance_usd"] == 25.0
    plaintext = bought["api_key"]
    h = {"Authorization": f"Bearer {plaintext}"}
    assert shop.get("/v1/balance", headers=h).json() == {
        "object": "balance", "balance_usd": 25.0, "currency": "USD", "unlimited": False}
    assert shop.post("/v1/images/generations", headers=h, json={"prompt": "x"}).status_code == 200
    assert shop.get("/v1/balance", headers=h).json()["balance_usd"] == 24.985

    (key,) = shop.get("/admin/keys", headers=admin).json()["data"]
    assert (key["name"], key["note"], key["unlimited"]) == ("new@example.com", "self-serve", False)
    topup = _ledger(shop, admin, key["id"])[-1]
    assert (topup["kind"], topup["amount_usd"], topup["ref"]) == ("topup", 25.0,
                                                                 f"CAP-{bought['order_id']}")
    assert topup["note"] == f"PayPal {bought['order_id']} payer@example.com"

    # Repeating the capture neither credits again nor shows the key again.
    order = {"id": bought["order_id"], "claim": bought["claim"]}
    again = _capture(shop, order).json()
    assert again["status"] == "completed" and "api_key" not in again and "message" in again
    assert len(shop.get("/admin/keys", headers=admin).json()["data"]) == 1
    assert [e["kind"] for e in _ledger(shop, admin, key["id"])] == ["charge", "topup"]

    (purchase,) = shop.get("/admin/purchases", headers=admin).json()["data"]
    assert purchase["status"] == "completed" and purchase["amount_usd"] == 25.0
    assert purchase["key_name"] == "new@example.com"
    assert purchase["payer_email"] == "payer@example.com"
    assert "claim_hash" not in purchase


def test_capture_needs_the_claim(shop, admin):
    key_id, plaintext = _new_key(shop, admin)
    order = _order(shop, email="new@example.com").json()
    # The order id leaks through PayPal's redirect URL; on its own it gets nothing.
    assert shop.post(f"/billing/orders/{order['id']}/capture").status_code == 404
    for claim in ("", "wrong", order["claim"][:-1]):
        r = _capture(shop, order, claim)
        assert r.status_code == 404 and r.json()["error"]["message"] == "unknown order"
    assert len(shop.get("/admin/keys", headers=admin).json()["data"]) == 1  # no key handed out
    assert _capture(shop, order).json()["api_key"].startswith("m2a-")
    # After completion, still no balance (or anything else) without the claim.
    assert _capture(shop, order, "wrong").status_code == 404
    assert "balance_usd" in _capture(shop, order).json()
    # Only a hash of the claim is stored.
    db = shop.app.state.services.payments
    rows = shop.portal.call(db._run, lambda d: d.execute("SELECT claim_hash FROM purchases").fetchall())
    assert rows[0][0] and order["claim"] not in rows[0][0]


def test_top_up_existing_key(shop, admin):
    key_id, plaintext = _new_key(shop, admin)
    bought = _buy(shop, api_key=plaintext, amount_usd=10)
    assert {k: bought[k] for k in ("status", "balance_usd")} == {"status": "completed",
                                                                 "balance_usd": 10.0}
    assert "api_key" not in bought
    assert len(shop.get("/admin/keys", headers=admin).json()["data"]) == 1
    # A second capture (e.g. the browser retrying after a timeout) changes nothing.
    _capture(shop, {"id": bought["order_id"], "claim": bought["claim"]})
    assert [e["kind"] for e in _ledger(shop, admin, key_id)] == ["topup"]


@pytest.mark.parametrize("change", ["revoked", "deleted"])
def test_dead_key_is_not_credited(shop, admin, change):
    key_id, plaintext = _new_key(shop, admin)
    order = _order(shop, api_key=plaintext, amount_usd=10).json()
    if change == "revoked":
        shop.patch(f"/admin/keys/{key_id}", headers=admin, json={"revoked": True})
    else:
        shop.delete(f"/admin/keys/{key_id}", headers=admin)
    r = _capture(shop, order).json()
    assert r["status"] == "needs_support" and "support" in r["message"]
    (purchase,) = shop.get("/admin/purchases", headers=admin).json()["data"]
    assert purchase["status"] == "needs_support" and purchase["capture_id"]
    assert _capture(shop, order).json()["status"] == "needs_support"  # and stays that way
    db = shop.app.state.services.billing
    assert shop.portal.call(db.balance, key_id) == 0


@pytest.mark.parametrize("field,value", [("capture_amount", "1.00"), ("capture_currency", "EUR")])
def test_capture_must_match_the_order(shop, admin, paypal, field, value):
    key_id, plaintext = _new_key(shop, admin)
    setattr(paypal, field, value)
    order = _order(shop, api_key=plaintext).json()
    r = _capture(shop, order)
    assert r.status_code == 502 and r.json()["error"]["code"] == "payment_mismatch"
    assert _ledger(shop, admin, key_id) == []
    assert shop.get("/admin/purchases", headers=admin).json()["data"][0]["status"] == "failed"


def test_pending_capture_is_not_credited(shop, admin, paypal):
    key_id, plaintext = _new_key(shop, admin)
    paypal.capture_status = "PENDING"
    order = _order(shop, api_key=plaintext).json()
    assert _capture(shop, order).json()["status"] == "pending"
    assert _ledger(shop, admin, key_id) == []


def test_unknown_order(shop):
    assert _capture(shop, {"id": "NOPE", "claim": "x"}).status_code == 404


def test_disabled_without_credentials(client):
    assert _order(client, email="a@b.co").status_code == 503
    assert client.post("/billing/orders/X/capture").status_code == 503
    page = client.get("/billing", params={"lang": "en"}).text
    assert 'data-state="off"' in page and "paypal.com/sdk" not in page
    assert "company@isemi.io" in page


def test_sandbox_needs_an_explicit_opt_in(settings, paypal, admin):
    with _shop(settings, paypal, paypal_env="sandbox") as c:
        assert _order(c, email="a@b.co").status_code == 503
        page = c.get("/billing", params={"lang": "en"}).text
        assert 'data-state="off"' in page and "SANDBOX" not in page
    with _shop(settings, paypal, paypal_env="sandbox", paypal_allow_sandbox=True) as c:
        page = c.get("/billing", params={"lang": "en"}).text
        assert 'data-state="on"' in page and 'class="sandbox"' in page and "SANDBOX" in page
        _buy(c, email="tester@example.com")
        (key,) = c.get("/admin/keys", headers=admin).json()["data"]
        assert key["note"] == "sandbox"
    with _shop(settings, paypal) as c:  # live: no banner
        assert 'class="sandbox"' not in c.get("/billing", params={"lang": "en"}).text


def test_rate_limits(shop):
    p = shop.app.state.services.payments
    p.order_limit = payments.Throttle(2, 600)
    assert _order(shop, email="a@b.co").status_code == 200
    assert _order(shop, email="a@b.co").status_code == 200
    r = _order(shop, email="a@b.co")
    assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limited"
    # The global cap holds however many addresses a client rotates through.
    p.order_limit, p.order_global_limit = payments.Throttle(100, 600), payments.Throttle(1, 60)
    assert _order(shop, email="a@b.co").status_code == 200
    assert _order(shop, email="a@b.co").status_code == 429
    p.capture_limit = payments.Throttle(1, 600)
    assert _capture(shop, {"id": "NOPE", "claim": "x"}).status_code == 404
    assert _capture(shop, {"id": "NOPE", "claim": "x"}).status_code == 429


def _req(peer: str, **headers) -> SimpleNamespace:
    return SimpleNamespace(client=SimpleNamespace(host=peer), headers=headers)


def test_billing_ip():
    # Cloudflare's header counts only on connections from the local tunnel.
    assert billing_ip(_req("127.0.0.1", **{"cf-connecting-ip": "203.0.113.9"})) == "203.0.113.9"
    assert billing_ip(_req("::1", **{"cf-connecting-ip": "203.0.113.9"})) == "203.0.113.9"
    assert billing_ip(_req("198.51.100.4", **{"cf-connecting-ip": "203.0.113.9"})) == "198.51.100.4"
    assert billing_ip(_req("127.0.0.1", **{"x-forwarded-for": "203.0.113.9",
                                           "x-real-ip": "203.0.113.9"})) == "127.0.0.1"
    # One IPv6 /64 is one client.
    a = billing_ip(_req("127.0.0.1", **{"cf-connecting-ip": "2001:db8:1:2::1"}))
    b = billing_ip(_req("2001:db8:1:2:ffff::9"))
    assert a == b == "2001:db8:1:2::/64"


def _capture_event(order_id: str, value: str = "25.00") -> dict:
    return _event("PAYMENT.CAPTURE.COMPLETED", {
        "id": f"CAP-{order_id}", "status": "COMPLETED",
        "amount": {"currency_code": "USD", "value": value},
        "supplementary_data": {"related_ids": {"order_id": order_id}}})


def test_webhook_credits_a_payment_the_capture_call_missed(shop, admin, paypal):
    key_id, plaintext = _new_key(shop, admin)
    order = _order(shop, api_key=plaintext).json()
    paypal.orders[order["id"]]["captured"] = True  # captured, but our capture call never finished
    for _ in range(2):
        assert _hook(shop, _capture_event(order["id"])).json() == {"ok": True}
    assert [e["kind"] for e in _ledger(shop, admin, key_id)] == ["topup"]
    # The browser's late capture call then finds the order already captured and credited.
    assert _capture(shop, order).json()["status"] == "completed"
    assert [e["kind"] for e in _ledger(shop, admin, key_id)] == ["topup"]


def test_webhook_verifies_the_raw_body(shop, admin, paypal):
    key_id, plaintext = _new_key(shop, admin)
    order = _order(shop, api_key=plaintext).json()
    event = _capture_event(order["id"])
    event["resource"]["note"] = "Señor café ☕ — 支付"
    raw = json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode()
    r = shop.post("/billing/paypal/webhook", content=raw,
                  headers={"content-type": "application/json", **_signed()})
    assert r.status_code == 200
    (sent,) = paypal.verified
    assert raw in sent  # the exact bytes, not a re-serialisation
    body = json.loads(sent)
    assert body["webhook_event"] == event and body["webhook_id"] == "WH-1"
    assert body["transmission_id"] == "tid"


def test_webhook_with_a_bad_signature_is_ignored(shop, admin, paypal):
    key_id, plaintext = _new_key(shop, admin)
    order = _order(shop, api_key=plaintext).json()
    paypal.verify = "FAILURE"
    r = _hook(shop, _capture_event(order["id"]))
    assert r.status_code == 400
    assert _ledger(shop, admin, key_id) == []


def test_webhook_rejected_before_calling_paypal(shop, paypal):
    event = _capture_event("ORDER1")
    old = datetime.fromtimestamp(time.time() - 600, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for headers in ({"paypal-transmission-sig": ""},
                    {"paypal-cert-url": "https://evil.example/certs/x"},
                    {"paypal-cert-url": "http://api.paypal.com/v1/notifications/certs/x"},
                    {"paypal-transmission-time": old},
                    {"paypal-transmission-time": "yesterday"}):
        assert _hook(shop, event, **headers).status_code == 400, headers
    r = _hook(shop, _event("CHECKOUT.ORDER.APPROVED", {}))
    assert r.status_code == 200 and "ignored" in r.json()
    big = {**event, "padding": "x" * 70_000}
    assert _hook(shop, big).status_code == 400
    assert paypal.verified == []
    shop.app.state.services.payments.webhook_limit = payments.Throttle(1, 60)
    _hook(shop, event)
    assert _hook(shop, event).status_code == 429


def test_webhook_ignored_without_webhook_id(shop, settings, admin):
    settings.paypal_webhook_id = ""
    r = _hook(shop, _capture_event("ORDER1"))
    assert r.status_code == 200 and "ignored" in r.json()


def test_refund_webhook_debits_once(shop, admin):
    key_id, plaintext = _new_key(shop, admin)
    bought = _buy(shop, api_key=plaintext, amount_usd=10)
    capture = f"CAP-{bought['order_id']}"
    refund = _event("PAYMENT.CAPTURE.REFUNDED", {
        "id": "REF-1", "status": "COMPLETED", "amount": {"currency_code": "USD", "value": "4.00"},
        "links": [{"rel": "up", "href": f"https://api-m.paypal.com/v2/payments/captures/{capture}"}]})
    for _ in range(2):
        assert _hook(shop, refund).status_code == 200
    entries = _ledger(shop, admin, key_id)
    assert [(e["kind"], e["amount_usd"], e["ref"]) for e in entries] == [
        ("adjust", -4.0, "REF-1"), ("topup", 10.0, capture)]
    # A reversal (chargeback) takes back the full amount; the balance may go negative.
    reversal = _event("PAYMENT.CAPTURE.REVERSED", {
        "id": capture, "status": "REVERSED", "amount": {"currency_code": "USD", "value": "10.00"}})
    _hook(shop, reversal)
    _hook(shop, reversal)
    assert _ledger(shop, admin, key_id)[0]["balance_after_usd"] == -4.0


@pytest.mark.parametrize("kind", ["PAYMENT.CAPTURE.REFUNDED", "PAYMENT.CAPTURE.REVERSED"])
def test_refund_before_credit_closes_the_purchase(shop, admin, kind):
    # A new-customer purchase, still pending: no capture id and no key yet.
    order = _order(shop, email="new@example.com").json()
    refund = _event(kind, {
        "id": "REF-9", "status": "COMPLETED", "amount": {"currency_code": "USD", "value": "25.00"},
        "supplementary_data": {"related_ids": {"order_id": order["id"]}},
        "links": [{"rel": "up", "href": "https://api-m.paypal.com/v2/payments/captures/CAP-X"}]})
    assert _hook(shop, refund).status_code == 200
    (purchase,) = shop.get("/admin/purchases", headers=admin).json()["data"]
    assert purchase["status"] == "refunded"
    # A COMPLETED event arriving late must not credit it now.
    assert _hook(shop, _capture_event(order["id"])).status_code == 200
    assert shop.get("/admin/keys", headers=admin).json()["data"] == []
    assert shop.get("/admin/purchases", headers=admin).json()["data"][0]["status"] == "refunded"
    assert _capture(shop, order).status_code == 502


def test_token_is_cached(shop, paypal):
    _order(shop, email="a@b.co")
    _order(shop, email="a@b.co")
    assert paypal.calls.count(("POST", "/v1/oauth2/token")) == 1


@pytest.mark.parametrize("lang", pages.LANGUAGES)
def test_billing_page_in_every_language(shop, lang):
    r = shop.get("/billing", params={"lang": lang})
    assert r.status_code == 200
    assert r.headers["content-language"] == lang
    assert r.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert r.headers["x-frame-options"] == "DENY"
    assert f'<html lang="{lang}">' in r.text
    assert "{{" not in r.text and "}}" not in r.text
    assert not _INTERNAL.search(r.text)
    assert 'data-state="on"' in r.text
    assert "https://www.paypal.com/sdk/js?client-id=cid&amp;currency=USD" in r.text


@pytest.mark.parametrize("name", ["landing.html", "docs.html", "billing.html"])
def test_host_header_is_escaped(settings, name):
    # Without public_base, the base URL comes from the request's Host header.
    text = pages.render(name, "en", settings, 'https://evil"><script>x</script>', "/p")
    assert "<script>x</script>" not in text
    assert 'href="https://evil&quot;&gt;&lt;script&gt;x&lt;/script&gt;/p?lang=en"' in text


def test_landing_sends_buyers_to_billing(client):
    text = client.get("/", params={"lang": "en"}).text
    assert 'href="#contact"' not in text
    assert text.count('href="/billing"') >= 4
