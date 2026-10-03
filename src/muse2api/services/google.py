"""Google sign-in for the customer portal: OpenID Connect authorization code flow
with PKCE, ``state`` and ``nonce``.

The ID token is taken from the token endpoint's own TLS response (never from the
browser), so its signature need not be checked again (OIDC Core 3.1.3.7); its
issuer, audience, expiry, nonce and ``email_verified`` still are.

A sign-in in progress (state, nonce, PKCE verifier) lives in a short-lived cookie
signed with a per-process key, not in server memory, so nobody can fill or evict
others' pending sign-ins; the server only remembers which states were used.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

from ..config import Settings
from ..errors import Muse2APIError
from .payments import Throttle

log = logging.getLogger(__name__)

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
ISSUERS = ("https://accounts.google.com", "accounts.google.com")
_PENDING_TTL = 600.0
_USED_MAX = 20_000
LOGIN_LIMIT = (20, 600.0)  # sign-ins started per client IP


class LoginFailed(Muse2APIError):
    status_code = 400
    error_type = "invalid_request_error"
    code = "login_failed"


@dataclass
class PendingLogin:
    nonce: str
    verifier: str
    next: str


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _jwt_claims(token: str) -> dict[str, Any]:
    try:
        payload = token.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (IndexError, ValueError) as exc:
        raise LoginFailed("malformed ID token") from exc


class GoogleOAuth:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        self.transport = transport  # tests pass an httpx.MockTransport
        # A restart only cancels the sign-ins in flight.
        self._key = secrets.token_bytes(32)
        self._used: dict[str, float] = {}  # state -> expiry: each cookie works once
        self.login_limit = Throttle(*LOGIN_LIMIT)

    def _sign(self, payload: str) -> str:
        return _b64url(hmac.new(self._key, payload.encode(), hashlib.sha256).digest())

    def start(self, redirect_uri: str, next_path: str) -> tuple[str, str]:
        """Begin a sign-in; returns (Google URL to redirect to, value for the cookie)."""
        state, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(3))
        payload = _b64url(json.dumps({"s": state, "n": nonce, "v": verifier, "x": next_path,
                                      "e": time.time() + _PENDING_TTL}).encode())
        cookie = f"{payload}.{self._sign(payload)}"
        query = urlencode({
            "client_id": self.settings.google_client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "nonce": nonce,
            "code_challenge": _b64url(hashlib.sha256(verifier.encode()).digest()),
            "code_challenge_method": "S256",
            "prompt": "select_account",
        })
        return f"{AUTH_URL}?{query}", cookie

    def take(self, state: str | None, cookie: str | None) -> PendingLogin:
        """The sign-in that ``state`` belongs to, once, and only in the browser that began
        it (its signed cookie carries the same state), so a callback URL cannot be
        replayed or planted on someone else."""
        payload, _, sig = (cookie or "").partition(".")
        if not state or not sig or not hmac.compare_digest(sig, self._sign(payload)):
            raise LoginFailed("sign-in state mismatch")
        try:
            data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        except ValueError as exc:
            raise LoginFailed("sign-in state mismatch") from exc
        now = time.time()
        if not secrets.compare_digest(str(data.get("s", "")), state):
            raise LoginFailed("sign-in state mismatch")
        if data.get("e", 0) < now:
            raise LoginFailed("sign-in expired; start again")
        for used in [s for s, exp in self._used.items() if exp < now]:
            del self._used[used]
        if state in self._used:
            raise LoginFailed("sign-in already used; start again")
        while len(self._used) >= _USED_MAX:  # only ever states we issued (signed)
            self._used.pop(next(iter(self._used)))
        self._used[state] = data["e"]
        return PendingLogin(data["n"], data["v"], data["x"])

    async def exchange(self, code: str, redirect_uri: str, pending: PendingLogin) -> dict[str, Any]:
        """Trade the code for an ID token and return its checked claims."""
        s = self.settings
        try:
            async with httpx.AsyncClient(timeout=20.0, transport=self.transport) as client:
                r = await client.post(TOKEN_URL, data={
                    "code": code, "client_id": s.google_client_id,
                    "client_secret": s.google_client_secret, "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code", "code_verifier": pending.verifier})
            data = r.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.error("Google token exchange failed: %s", exc)
            raise LoginFailed("could not reach Google; try again") from exc
        if r.status_code != 200 or "id_token" not in data:
            log.warning("Google token exchange refused: %s %s", r.status_code, data.get("error"))
            raise LoginFailed("Google refused the sign-in")
        claims = _jwt_claims(data["id_token"])
        if claims.get("iss") not in ISSUERS:
            raise LoginFailed("unexpected token issuer")
        aud = claims.get("aud")
        if aud != s.google_client_id and not (isinstance(aud, list) and s.google_client_id in aud):
            raise LoginFailed("token is for another client")
        if not isinstance(claims.get("exp"), (int, float)) or claims["exp"] < time.time():
            raise LoginFailed("token expired")
        if not secrets.compare_digest(str(claims.get("nonce", "")), pending.nonce):
            raise LoginFailed("nonce mismatch")
        if claims.get("email_verified") not in (True, "true") or not claims.get("email"):
            raise LoginFailed("the Google email is not verified")
        if not claims.get("sub"):
            raise LoginFailed("token has no subject")
        return claims
