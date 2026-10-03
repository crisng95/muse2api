"""Customers, shared wallets, Google sign-in and the /account portal."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import sqlite3
import time
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi.testclient import TestClient

from muse2api.api.routes import pages
from muse2api.api.routes.account import SESSION_COOKIE
from muse2api.app import create_app
from muse2api.drivers.mock import MockDriver
from muse2api.errors import UpstreamRefused
from muse2api.services import google as google_mod
from muse2api.services import payments
from test_checkout import FakePayPal

_INTERNAL = re.compile(r"cookie|muse\.ai|muse2api|browser|chromium|account[_ ]?ids?\b|"
                       r"accounts\.json|account pool", re.I)


def _b64(data: dict | bytes) -> str:
    raw = data if isinstance(data, bytes) else json.dumps(data).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


class FakeGoogle:
    """Google's token endpoint: checks PKCE and returns an ID token for ``self.user``."""

    def __init__(self) -> None:
        self.user = {"sub": "g-1", "email": "Alice@Example.com", "email_verified": True,
                     "name": "Alice"}
        self.claims: dict = {}  # overrides for the next token
        self.challenges: dict[str, str] = {}  # code -> code_challenge the browser was given
        self.calls = 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        form = parse_qs(request.content.decode())
        verifier = form["code_verifier"][0]
        challenge = _b64(hashlib.sha256(verifier.encode()).digest())
        if challenge != self.challenges.get(form["code"][0]):
            return httpx.Response(400, json={"error": "invalid_grant"})
        claims = {"iss": "https://accounts.google.com", "aud": "gid", "exp": time.time() + 600,
                  "nonce": self.nonce, **self.user, **self.claims}
        token = f"{_b64({'alg': 'RS256'})}.{_b64(claims)}.sig"
        return httpx.Response(200, json={"id_token": token, "access_token": "at"})


@pytest.fixture
def google() -> FakeGoogle:
    return FakeGoogle()


def _app(settings, google, **overrides):
    settings.google_client_id, settings.google_client_secret = "gid", "gsecret"
    for k, v in overrides.items():
        setattr(settings, k, v)
    app = create_app(settings, MockDriver(delay=0))
    app.state.services.google.transport = httpx.MockTransport(google.handle)
    return app


@pytest.fixture
def portal(settings, google):
    # https, so that the Secure session cookie is stored and sent back.
    with TestClient(_app(settings, google), base_url="https://testserver") as c:
        yield c


def _second(c) -> TestClient:
    """Another browser on the same app."""
    return TestClient(c.app, base_url="https://testserver")


def _begin(c, google, code="code-1") -> str:
    r = c.get("/account/login", follow_redirects=False)
    assert r.status_code == 303
    url = urlparse(r.headers["location"])
    assert url.netloc == "accounts.google.com"
    q = {k: v[0] for k, v in parse_qs(url.query).items()}
    assert q["code_challenge_method"] == "S256" and q["client_id"] == "gid"
    assert q["redirect_uri"] == "https://testserver/account/auth/google/callback"
    google.challenges[code] = q["code_challenge"]
    google.nonce = q["nonce"]
    return q["state"]


def _login(c, google, code="code-1") -> dict:
    state = _begin(c, google, code)
    r = c.get("/account/auth/google/callback", params={"code": code, "state": state},
              follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/account", r.headers
    me = c.get("/account/api/me")
    assert me.status_code == 200
    return me.json()


def _csrf(c) -> dict:
    return {"X-CSRF-Token": c.get("/account/api/me").json()["csrf"]}


def _failed(c, r) -> None:
    assert r.status_code == 303 and r.headers["location"] == "/account?login=failed"
    assert SESSION_COOKIE not in r.headers.get("set-cookie", "")
    assert c.get("/account/api/me").status_code == 401


# ---- sign-in ----
def test_login_creates_a_customer_and_a_secure_session(portal, google):
    state = _begin(portal, google)
    r = portal.get("/account/auth/google/callback", params={"code": "code-1", "state": state},
                   follow_redirects=False)
    cookies = r.headers.get_list("set-cookie")
    session = next(h for h in cookies if h.startswith(f"{SESSION_COOKIE}="))
    assert SESSION_COOKIE.startswith("__Host-") and "domain" not in session.lower()
    for flag in ("HttpOnly", "Secure", "SameSite=lax", "Max-Age=2592000", "Path=/"):
        assert flag in session, flag
    assert r.headers["cache-control"] == "no-store"
    me = portal.get("/account/api/me").json()
    assert me["customer"]["email"] == "alice@example.com" and me["customer"]["name"] == "Alice"
    assert me["balance_usd"] == 0.0 and me["csrf"]
    # Only a hash of the session token is stored.
    token = session.split(";")[0].split("=", 1)[1]
    customers = portal.app.state.services.customers
    rows = portal.portal.call(customers._run, lambda db: db.execute(
        "SELECT token_hash FROM sessions").fetchall())
    assert rows[0][0] == hashlib.sha256(token.encode()).hexdigest()
    # Signing in again finds the same customer.
    portal.cookies.clear()
    assert _login(portal, google, "code-2")["customer"]["id"] == me["customer"]["id"]


def test_state_must_match_the_browser_and_is_used_once(portal, google):
    state = _begin(portal, google)
    _failed(portal, portal.get("/account/auth/google/callback",
                               params={"code": "code-1", "state": "forged"}, follow_redirects=False))
    # A callback URL planted on another browser (no state cookie) is refused.
    other = _second(portal)
    _failed(other, other.get("/account/auth/google/callback",
                             params={"code": "code-1", "state": state}, follow_redirects=False))
    assert google.calls == 0
    state = _begin(portal, google)
    cookie = portal.cookies.get("__Host-m2a_oauth")
    # The pending sign-in lives in a signed cookie: tampering with it is refused.
    payload, sig = cookie.split(".")
    portal.cookies.set("__Host-m2a_oauth", payload + "x." + sig)
    _failed(portal, portal.get("/account/auth/google/callback",
                               params={"code": "code-1", "state": state}, follow_redirects=False))
    portal.cookies.set("__Host-m2a_oauth", cookie)
    ok = portal.get("/account/auth/google/callback", params={"code": "code-1", "state": state},
                    follow_redirects=False)
    assert ok.headers["location"] == "/account"
    # Replaying the same cookie and callback works only once.
    portal.cookies.set("__Host-m2a_oauth", cookie)
    replay = portal.get("/account/auth/google/callback", params={"code": "code-1", "state": state},
                        follow_redirects=False)
    assert replay.headers["location"] == "/account?login=failed"


def test_login_is_rate_limited(portal, google, monkeypatch):
    portal.app.state.services.google.login_limit = google_mod.Throttle(2, 600)
    for _ in range(2):
        assert portal.get("/account/login", follow_redirects=False).status_code == 303
    assert portal.get("/account/login", follow_redirects=False).status_code == 429


@pytest.mark.parametrize("claims", [
    {"nonce": "other"},
    {"email_verified": False},
    {"aud": "someone-else"},
    {"iss": "https://evil.example"},
    {"exp": time.time() - 10},
])
def test_bad_id_tokens_are_refused(portal, google, claims):
    state = _begin(portal, google)
    google.claims = claims
    _failed(portal, portal.get("/account/auth/google/callback",
                               params={"code": "code-1", "state": state}, follow_redirects=False))
    assert portal.app.state.services.customers.all() == []


def test_pkce_verifier_mismatch_is_refused(portal, google):
    state = _begin(portal, google)
    google.challenges["code-1"] = "challenge-of-another-browser"
    _failed(portal, portal.get("/account/auth/google/callback",
                               params={"code": "code-1", "state": state}, follow_redirects=False))


def test_login_links_a_customer_only_by_verified_email(portal, google, admin):
    svc = portal.app.state.services
    customer = portal.portal.call(svc.customers.create, None, "Alice Ltd")
    key, _ = portal.portal.call(svc.keys.create, "prod", "", customer.id)
    # The admin sets the email: that counts as verified, so Google sign-in links it.
    r = portal.patch(f"/admin/customers/{customer.id}", headers=admin,
                     json={"email": "ALICE@example.com"})
    assert r.json()["customer"]["email_verified"] is True
    me = _login(portal, google)
    assert me["customer"]["id"] == customer.id
    assert [k["id"] for k in portal.get("/account/api/keys").json()["data"]] == [key.id]


def test_checkout_email_cannot_pre_hijack_a_google_sign_in(shop, google, admin):
    """An attacker buys with the victim's email before the victim ever signs in."""
    order = shop.post("/billing/orders", json={"amount_usd": 5, "email": "alice@example.com"}).json()
    attacker = shop.post(f"/billing/orders/{order['id']}/capture",
                         json={"claim": order["claim"]}).json()["api_key"]
    victim = _login(shop, google)  # Google: alice@example.com, verified
    assert shop.get("/account/api/keys").json()["data"] == []  # nothing merged
    order = shop.post("/billing/orders", json={"amount_usd": 20}, headers=_csrf(shop)).json()
    shop.post(f"/billing/orders/{order['id']}/capture", json={"claim": order["claim"]})
    assert shop.get("/account/api/me").json()["balance_usd"] == 20.0
    # The attacker's key still only sees the attacker's own $5.
    h = {"Authorization": f"Bearer {attacker}"}
    assert shop.get("/v1/balance", headers=h).json()["balance_usd"] == 5.0
    customers = {c["id"]: c for c in shop.get("/admin/customers", headers=admin).json()["data"]}
    assert customers[victim["customer"]["id"]]["email"] == "alice@example.com"
    old = next(c for c in customers.values() if c["id"] != victim["customer"]["id"])
    assert old["email"] is None and old["name"] == "alice@example.com"  # freed, still visible
    # And the attacker's key cannot be claimed without... being the attacker: the
    # victim does not have it. Claiming needs the plaintext key.
    r = shop.post("/account/api/keys/claim", json={"api_key": "m2a-guess"}, headers=_csrf(shop))
    assert r.status_code == 400


def test_claim_a_key_bought_before_signing_in(shop, google, admin):
    order = shop.post("/billing/orders", json={"amount_usd": 10, "email": "me@example.com"}).json()
    key = shop.post(f"/billing/orders/{order['id']}/capture",
                    json={"claim": order["claim"]}).json()["api_key"]
    h = {"Authorization": f"Bearer {key}"}
    shop.post("/v1/images/generations", headers=h, json={"prompt": "x"})  # spends $0.015
    google.user = {**google.user, "email": "someone.else@example.com", "sub": "g-9"}
    me = _login(shop, google)
    assert shop.post("/account/api/keys/claim", json={"api_key": key}).status_code == 403  # CSRF
    r = shop.post("/account/api/keys/claim", json={"api_key": key}, headers=_csrf(shop))
    assert r.status_code == 200, r.text
    assert r.json()["moved_usd"] == 9.985 and r.json()["balance_usd"] == 9.985
    assert [k["name"] for k in shop.get("/account/api/keys").json()["data"]] == ["me@example.com"]
    assert shop.get("/v1/balance", headers=h).json()["balance_usd"] == 9.985
    # Auditable: an adjust out of the old wallet and one into the new.
    ledger = shop.get(f"/admin/customers/{me['customer']['id']}/ledger", headers=admin).json()["data"]
    assert (ledger[0]["kind"], ledger[0]["amount_usd"]) == ("adjust", 9.985)
    old = next(c for c in shop.get("/admin/customers", headers=admin).json()["data"]
               if c["id"] != me["customer"]["id"])
    assert old["balance_usd"] == 0.0 and old["keys"] == []
    # Claiming it again (now ours) is refused.
    assert shop.post("/account/api/keys/claim", json={"api_key": key},
                     headers=_csrf(shop)).status_code == 403


def test_claim_is_refused_for_verified_or_shared_customers(portal, google, admin):
    svc = portal.app.state.services
    verified = portal.portal.call(svc.customers.create, None, "acme")
    portal.patch(f"/admin/customers/{verified.id}", headers=admin, json={"email": "ops@acme.example"})
    _, k_verified = portal.portal.call(svc.keys.create, "a", "", verified.id)
    shared = portal.portal.call(svc.customers.create, None, "two keys")
    _, k_shared = portal.portal.call(svc.keys.create, "b", "", shared.id)
    portal.portal.call(svc.keys.create, "c", "", shared.id)
    _login(portal, google)
    for key in (k_verified, k_shared):
        r = portal.post("/account/api/keys/claim", json={"api_key": key}, headers=_csrf(portal))
        assert r.status_code == 403 and r.json()["error"]["code"] == "not_claimable"
    assert portal.get("/account/api/keys").json()["data"] == []


def test_portal_disabled_without_google_credentials(client):
    r = client.get("/account/login", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/account"
    assert client.get("/account/api/me").status_code == 503
    assert 'data-state="off"' in client.get("/account", params={"lang": "en"}).text


# ---- sessions and CSRF ----
def test_logout_revokes_the_session(portal, google):
    _login(portal, google)
    token = portal.cookies.get(SESSION_COOKIE)
    assert portal.post("/account/logout").status_code == 403  # no CSRF token
    r = portal.post("/account/logout", headers=_csrf(portal))
    assert r.status_code == 200 and f"{SESSION_COOKIE}=" in r.headers["set-cookie"]
    # Replaying the old cookie does not work: the server forgot it.
    replay = _second(portal)
    replay.cookies.set(SESSION_COOKIE, token)
    assert replay.get("/account/api/me").status_code == 401


def test_state_changes_need_the_csrf_token(portal, google):
    _login(portal, google)
    body = {"name": "prod"}
    assert portal.post("/account/api/keys", json=body).status_code == 403
    assert portal.post("/account/api/keys", json=body,
                       headers={"X-CSRF-Token": "wrong"}).status_code == 403
    cross = {**_csrf(portal), "Origin": "https://evil.example"}
    assert portal.post("/account/api/keys", json=body, headers=cross).status_code == 403
    r = portal.post("/account/api/keys", json=body,
                    headers={**_csrf(portal), "Origin": "https://testserver"})
    assert r.status_code == 200 and r.json()["api_key"].startswith("m2a-")


# ---- portal API ----
def test_keys_lifecycle_and_shared_wallet(portal, google, admin):
    me = _login(portal, google)
    h = _csrf(portal)
    k1 = portal.post("/account/api/keys", json={"name": "one"}, headers=h).json()
    k2 = portal.post("/account/api/keys", json={"name": "two"}, headers=h).json()
    assert "hash" not in k1["key"]
    r = portal.post(f"/admin/customers/{me['customer']['id']}/credit", headers=admin,
                    json={"amount_usd": 1})
    assert r.json()["balance_usd"] == 1.0
    a = {"Authorization": f"Bearer {k1['api_key']}"}
    b = {"Authorization": f"Bearer {k2['api_key']}"}
    assert portal.post("/v1/images/generations", headers=a, json={"prompt": "x"}).status_code == 200
    # Both keys see (and spend) the same wallet.
    assert portal.get("/v1/balance", headers=b).json()["balance_usd"] == 0.985
    assert portal.get("/account/api/me").json()["balance_usd"] == 0.985
    keys = {k["name"]: k for k in portal.get("/account/api/keys").json()["data"]}
    assert keys["one"]["spend_usd"] == 0.015 and keys["two"]["spend_usd"] == 0.0

    r = portal.patch(f"/account/api/keys/{k2['key']['id']}", json={"name": "renamed"}, headers=h)
    assert r.json()["key"]["name"] == "renamed"
    assert portal.post(f"/account/api/keys/{k2['key']['id']}/revoke", headers=h).status_code == 200
    assert portal.get("/v1/balance", headers=b).status_code == 401

    usage = portal.get("/account/api/usage", params={"days": 7}).json()
    assert len(usage["series"]) == 7 and usage["series"][-1]["cost_usd"] == 0.015
    assert usage["by_key"][0]["key_name"] == "one"
    assert portal.get("/account/api/usage", params={"days": 3}).status_code == 400

    billing = portal.get("/account/api/billing").json()
    assert [(e["kind"], e["amount_usd"]) for e in billing["ledger"]] == [("topup", 1.0)]
    assert "note" not in billing["ledger"][0]


def test_customers_are_isolated(portal, google, admin):
    _login(portal, google)
    ka = portal.post("/account/api/keys", json={"name": "a"}, headers=_csrf(portal)).json()
    bob = _second(portal)
    google.user = {"sub": "g-2", "email": "bob@example.com", "email_verified": True, "name": "Bob"}
    bob_me = _login(bob, google, "code-b")
    kb = bob.post("/account/api/keys", json={"name": "b"}, headers=_csrf(bob)).json()
    bob.post(f"/admin/customers/{bob_me['customer']['id']}/credit", headers=admin,
             json={"amount_usd": 2})
    bob.post("/v1/images/generations", headers={"Authorization": f"Bearer {kb['api_key']}"},
             json={"prompt": "bob's secret"})

    b_id = kb["key"]["id"]
    h = _csrf(portal)
    assert portal.patch(f"/account/api/keys/{b_id}", json={"name": "x"}, headers=h).status_code == 404
    assert portal.post(f"/account/api/keys/{b_id}/revoke", headers=h).status_code == 404
    assert portal.get("/account/api/requests", params={"key_id": b_id}).status_code == 404
    assert [k["id"] for k in portal.get("/account/api/keys").json()["data"]] == [ka["key"]["id"]]
    assert portal.get("/account/api/requests").json()["total"] == 0
    assert portal.get("/account/api/me").json()["balance_usd"] == 0.0
    assert portal.get("/account/api/billing").json() == {"ledger": [], "purchases": []}
    assert portal.get("/account/api/usage").json()["by_key"] == []
    assert bob.get("/account/api/requests").json()["total"] == 1
    assert bob.app.state.services.keys.get(b_id).name == "b"


def test_requests_never_show_upstream_error_text(portal, google, admin, monkeypatch):
    me = _login(portal, google)
    key = portal.post("/account/api/keys", json={"name": "k"}, headers=_csrf(portal)).json()
    portal.post(f"/admin/customers/{me['customer']['id']}/credit", headers=admin,
                json={"amount_usd": 1})

    async def boom(account, req):
        raise UpstreamRefused("acc-17 cookie expired at muse.ai (internal detail XYZ)")

    monkeypatch.setattr(portal.app.state.services.driver, "generate_image", boom)
    h = {"Authorization": f"Bearer {key['api_key']}"}
    assert portal.post("/v1/images/generations", headers=h, json={"prompt": "x"}).status_code == 502
    r = portal.get("/account/api/requests")
    assert "XYZ" not in r.text and "acc-17" not in r.text and "cookie" not in r.text
    (row,) = r.json()["data"]
    assert row["status_code"] == 502 and row["cost_usd"] == 0.0
    for field in ("error", "account_id", "client_ip", "user_agent", "failed_attempts"):
        assert field not in row
    failed = portal.get("/account/api/requests", params={"status": "failed"}).json()
    assert failed["total"] == 1


# ---- checkout ----
@pytest.fixture
def shop(settings, google, monkeypatch):
    for name in ("ORDER_LIMIT", "ORDER_GLOBAL_LIMIT", "CAPTURE_LIMIT", "WEBHOOK_LIMIT"):
        monkeypatch.setattr(payments, name, (1000, 600.0))
    settings.paypal_client_id, settings.paypal_client_secret = "cid", "secret"
    settings.paypal_env = "live"
    fake = FakePayPal()
    app = _app(settings, google)
    app.state.services.payments.paypal.transport = httpx.MockTransport(fake.handle)
    with TestClient(app, base_url="https://testserver") as c:
        yield c


def test_checkout_while_signed_in_tops_up_the_wallet(shop, google, admin):
    me = _login(shop, google)
    assert shop.post("/billing/orders", json={"amount_usd": 10}).status_code == 403  # no CSRF
    order = shop.post("/billing/orders", json={"amount_usd": 10}, headers=_csrf(shop)).json()
    r = shop.post(f"/billing/orders/{order['id']}/capture", json={"claim": order["claim"]})
    assert r.json()["status"] == "completed" and r.json()["balance_usd"] == 10.0
    assert "api_key" not in r.json()
    assert shop.get("/account/api/me").json()["balance_usd"] == 10.0
    (purchase,) = shop.get("/account/api/billing").json()["purchases"]
    assert purchase["mode"] == "wallet" and purchase["status"] == "completed"
    assert shop.get("/admin/customers", headers=admin).json()["data"][0]["id"] == me["customer"]["id"]


def test_signed_out_checkout_still_works(shop, admin):
    order = shop.post("/billing/orders", json={"amount_usd": 10, "email": "new@example.com"}).json()
    r = shop.post(f"/billing/orders/{order['id']}/capture", json={"claim": order["claim"]}).json()
    assert r["api_key"].startswith("m2a-") and r["balance_usd"] == 10.0
    (customer,) = shop.get("/admin/customers", headers=admin).json()["data"]
    assert customer["email"] == "new@example.com" and not customer["email_verified"]
    assert customer["balance_usd"] == 10.0 and len(customer["keys"]) == 1
    # Top up through the key: the same wallet.
    order = shop.post("/billing/orders", json={"amount_usd": 5, "api_key": r["api_key"]}).json()
    shop.post(f"/billing/orders/{order['id']}/capture", json={"claim": order["claim"]})
    assert shop.get("/v1/balance", headers={"Authorization": f"Bearer {r['api_key']}"}).json()[
        "balance_usd"] == 15.0
    # A registered email is accepted (checkout does not reveal registrations) but never
    # attached: the new key gets a customer of its own, without that email.
    order = shop.post("/billing/orders", json={"amount_usd": 10, "email": "NEW@example.com"})
    assert order.status_code == 200
    order = order.json()
    other = shop.post(f"/billing/orders/{order['id']}/capture",
                      json={"claim": order["claim"]}).json()["api_key"]
    assert shop.get("/v1/balance", headers={"Authorization": f"Bearer {other}"}).json()[
        "balance_usd"] == 10.0
    customers = shop.get("/admin/customers", headers=admin).json()["data"]
    assert sorted(str(c["email"]) for c in customers) == ["None", "new@example.com"]


# ---- admin ----
def test_admin_customers(client, admin):
    key = client.post("/admin/keys", headers=admin, json={"name": "acme"}).json()
    customer_id = key["key"]["customer_id"]
    extra = client.post("/admin/keys", headers=admin,
                        json={"name": "acme-2", "customer_id": customer_id}).json()
    assert extra["key"]["customer_id"] == customer_id
    assert client.post("/admin/keys", headers=admin,
                       json={"name": "x", "customer_id": "cus_nope"}).status_code == 404
    # The key's "credit" credits its customer's wallet, shared with the other key.
    client.post(f"/admin/keys/{key['key']['id']}/credit", headers=admin, json={"amount_usd": 3})
    r = client.post(f"/admin/customers/{customer_id}/credit", headers=admin,
                    json={"amount_usd": -1, "note": "fix"})
    assert r.json() == {"customer_id": customer_id, "kind": "adjust", "amount_usd": -1.0,
                        "balance_usd": 2.0}
    (c,) = client.get("/admin/customers", headers=admin).json()["data"]
    assert c["balance_usd"] == 2.0 and c["name"] == "acme" and len(c["keys"]) == 2
    keys = client.get("/admin/keys", headers=admin).json()["data"]
    assert {k["balance_usd"] for k in keys} == {2.0}
    r = client.patch(f"/admin/customers/{customer_id}", headers=admin, json={"unlimited": True})
    assert r.json()["customer"]["unlimited"] is True
    h = {"Authorization": f"Bearer {extra['api_key']}"}
    assert client.get("/v1/balance", headers=h).json()["unlimited"] is True
    ledger = client.get(f"/admin/customers/{customer_id}/ledger", headers=admin).json()["data"]
    assert [e["kind"] for e in ledger] == ["adjust", "topup"]
    assert client.get("/admin/customers", headers={"Authorization": "Bearer test-key"}).status_code == 401


# ---- migration from per-key balances ----
_OLD_SCHEMA = """
CREATE TABLE balances (key_id TEXT PRIMARY KEY, balance_micro INTEGER NOT NULL DEFAULT 0,
                       updated_at REAL NOT NULL);
CREATE TABLE ledger (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, key_id TEXT NOT NULL,
                     amount_micro INTEGER NOT NULL, kind TEXT NOT NULL, ref TEXT, note TEXT,
                     balance_after INTEGER NOT NULL);
CREATE INDEX idx_ledger_key ON ledger (key_id, id);
CREATE INDEX idx_ledger_ref ON ledger (ref);
CREATE UNIQUE INDEX idx_ledger_refund ON ledger (ref) WHERE kind = 'refund';
CREATE TABLE reservations (ref TEXT PRIMARY KEY, key_id TEXT NOT NULL, ts REAL NOT NULL);
"""


def _key(key_id, name, note="", unlimited=False):
    return {"id": key_id, "name": name, "prefix": "m2a-abcd", "hash": key_id.ljust(64, "0"),
            "created_at": 1.0, "note": note, "unlimited": unlimited}


def test_migration_gives_each_key_a_customer_and_moves_balances(settings, admin):
    settings.ensure_dirs()
    settings.keys_file.write_text(json.dumps([
        _key("k_ss1", "pat@example.com", "self-serve"),
        _key("k_ss2", "pat@example.com", "self-serve (webhook)"),
        _key("k_admin", "acme"),
        _key("k_free", "postforge", unlimited=True),
    ]))
    (settings.data_dir / "keys.version").write_text("2\n")
    db = sqlite3.connect(settings.requests_db)
    db.executescript(_OLD_SCHEMA)
    for key_id, amounts in {"k_ss1": [10_000_000, -15_000], "k_ss2": [5_000_000],
                            "k_admin": [2_000_000, -500_000]}.items():
        bal = 0
        for i, amount in enumerate(amounts):
            bal += amount
            db.execute("INSERT INTO ledger (ts, key_id, amount_micro, kind, ref, note, balance_after) "
                       "VALUES (?, ?, ?, ?, ?, '', ?)",
                       (1.0, key_id, amount, "topup" if amount > 0 else "charge",
                        f"{key_id}-{i}", bal))
        db.execute("INSERT INTO balances VALUES (?, ?, 1.0)", (key_id, bal))
    db.commit()
    db.close()

    def snapshot():
        with TestClient(create_app(settings, MockDriver(delay=0))) as c:
            customers = c.get("/admin/customers", headers=admin).json()["data"]
            return {tuple(sorted(k["id"] for k in x["keys"])): x for x in customers}

    first = snapshot()
    # Never grouped by the unverified purchase email.
    assert set(first) == {("k_ss1",), ("k_ss2",), ("k_admin",), ("k_free",)}
    pat1, pat2 = first[("k_ss1",)], first[("k_ss2",)]
    acme, free = first[("k_admin",)], first[("k_free",)]
    # The email is kept as unverified contact info where unique; never verified.
    assert pat1["email"] == "pat@example.com" and not pat1["email_verified"]
    assert pat2["email"] is None and pat2["name"] == "pat@example.com"
    assert pat1["balance_usd"] == 9.985 and pat2["balance_usd"] == 5.0
    assert acme["email"] is None and acme["name"] == "acme" and acme["balance_usd"] == 1.5
    # An unlimited key stays unlimited by itself; its customer is not made unlimited.
    assert not any(c["unlimited"] for c in first.values())
    assert free["keys"][0]["unlimited"] is True
    assert pat1["spend_usd"] == 0.015

    db = sqlite3.connect(settings.requests_db)
    # The ledger still adds up to each wallet, and says where the money came from.
    sums = dict(db.execute("SELECT customer_id, SUM(amount_micro) FROM ledger GROUP BY customer_id"))
    wallets = dict(db.execute("SELECT customer_id, balance_micro FROM wallets"))
    for c, micro in ((pat1, 9_985_000), (pat2, 5_000_000), (acme, 1_500_000)):
        assert sums[c["id"]] == wallets[c["id"]] == micro
    notes = [n for (n,) in db.execute("SELECT note FROM ledger WHERE note LIKE 'wallet migration%'")]
    assert len(notes) == 3
    assert db.execute("SELECT COUNT(*) FROM balances").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM ledger WHERE customer_id IS NULL").fetchone()[0] == 0
    db.close()
    stored = {k["id"]: k["customer_id"] for k in json.loads(settings.keys_file.read_text())}
    assert stored["k_ss1"] == pat1["id"] and stored["k_ss2"] == pat2["id"]

    # Idempotent: a restart changes nothing, and so does a crash before keys.json was
    # written (the choice is finished from key_migration, no balance moves twice).
    assert snapshot() == first
    keys = json.loads(settings.keys_file.read_text())
    for k in keys:
        k.pop("customer_id")
    settings.keys_file.write_text(json.dumps(keys))
    assert snapshot() == first


def test_refunds_still_work_after_migration(settings, admin):
    """A reservation made per key before the upgrade is refunded to the new wallet."""
    settings.ensure_dirs()
    settings.keys_file.write_text(json.dumps([_key("k1", "acme")]))
    (settings.data_dir / "keys.version").write_text("2\n")
    db = sqlite3.connect(settings.requests_db)
    db.executescript(_OLD_SCHEMA)
    db.execute("INSERT INTO ledger (ts, key_id, amount_micro, kind, ref, note, balance_after) "
               "VALUES (1, 'k1', 1000000, 'topup', NULL, '', 1000000)")
    db.execute("INSERT INTO ledger (ts, key_id, amount_micro, kind, ref, note, balance_after) "
               "VALUES (2, 'k1', -60000, 'charge', 'img_lost', '', 940000)")
    db.execute("INSERT INTO reservations VALUES ('img_lost', 'k1', 2)")
    db.execute("INSERT INTO balances VALUES ('k1', 940000, 2)")
    db.commit()
    db.close()
    with TestClient(create_app(settings, MockDriver(delay=0))) as c:
        (customer,) = c.get("/admin/customers", headers=admin).json()["data"]
        assert customer["balance_usd"] == 1.0  # the interrupted charge came back
        ledger = c.get(f"/admin/customers/{customer['id']}/ledger", headers=admin).json()["data"]
        assert ledger[0]["kind"] == "refund" and ledger[0]["key_id"] == "k1"


# ---- pages ----
@pytest.mark.parametrize("lang", pages.LANGUAGES)
def test_account_page_in_every_language(portal, lang):
    r = portal.get("/account", params={"lang": lang})
    assert r.status_code == 200 and r.headers["content-language"] == lang
    assert "{{" not in r.text and "}}" not in r.text
    assert not _INTERNAL.search(r.text)
    assert 'data-state="on"' in r.text


@pytest.mark.parametrize("path", ["/", "/docs", "/billing"])
def test_nav_links_to_the_portal(client, path):
    text = client.get(path, params={"lang": "en"}).text
    assert '<a class="acct" href="/account">Sign in</a>' in text
    client.cookies.set(SESSION_COOKIE, "x")
    text = client.get(path, params={"lang": "en"}).text
    assert '<a class="acct" href="/account">Account</a>' in text


def test_account_api_is_never_cached(portal, google):
    assert portal.get("/account", params={"lang": "en"}).headers["cache-control"] == "no-store"
    assert portal.get("/account/api/me").headers["cache-control"] == "no-store"  # even a 401
    _login(portal, google)
    for path in ("/account/api/me", "/account/api/keys", "/account/api/requests",
                 "/account/api/billing", "/account/api/usage"):
        assert portal.get(path).headers["cache-control"] == "no-store", path
    assert portal.get("/", params={"lang": "en"}).headers["cache-control"] == "no-cache"


def test_key_limit_counts_only_active_keys(portal, google, monkeypatch):
    _login(portal, google)
    h = _csrf(portal)
    svc = portal.app.state.services
    customer_id = portal.get("/account/api/me").json()["customer"]["id"]
    for i in range(50):
        key, _ = portal.portal.call(svc.keys.create, f"k{i}", "portal", customer_id)
        key.revoked = i < 49  # 49 revoked, 1 active
    assert portal.post("/account/api/keys", json={"name": "ok"}, headers=h).status_code == 200
    for key in svc.customers.keys_of(customer_id):
        key.revoked = False
    r = portal.post("/account/api/keys", json={"name": "too many"}, headers=h)
    assert r.status_code == 403
