"""HTML pages under ``web/``.

``/`` (landing page with models and pricing), ``/docs`` (API reference) and
``/billing`` (self-serve PayPal checkout) are for customers. They are templates: ``{{key}}`` placeholders are filled from
``web/i18n/<lang>.json`` (missing keys fall back to English) and from the live
prices in ``Settings``, so the page always quotes what billing charges.

``/dashboard`` is the internal admin console: the page itself is public, but it
asks for the admin key in the browser and calls ``/admin/*`` with it, so serving
the HTML reveals nothing. It is kept out of search engines and is not linked
from the public pages.
"""

from __future__ import annotations

import html
import json
import re
from functools import cache
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ...config import Settings
from ..deps import get_services, public_base

router = APIRouter(tags=["pages"])

_WEB = Path(__file__).resolve().parents[2] / "web"

LANGUAGES = ("en", "vi", "zh", "es", "id", "ja")
DEFAULT_LANGUAGE = "en"
SUPPORT_EMAIL = "company@isemi.io"
_EXAMPLE_CREDIT = 10.0

_HEADERS = {"Cache-Control": "no-cache", "X-Frame-Options": "DENY",
            "Referrer-Policy": "no-referrer"}
# PayPal's checkout scripts may rely on the referrer; the other pages send none.
_CHECKOUT_HEADERS = {**_HEADERS, "Referrer-Policy": "strict-origin-when-cross-origin"}
_PAYPAL_SDK = ("https://www.paypal.com/sdk/js?client-id={}&currency=USD&intent=capture"
               "&components=buttons&enable-funding=card&disable-funding=paylater,venmo")
_PLACEHOLDER = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")


@cache
def _html(name: str) -> str:
    return (_WEB / name).read_text(encoding="utf-8")


def _flatten(tree: dict, prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


@cache
def strings(lang: str) -> dict[str, str]:
    """Flattened UI strings for ``lang``; keys it lacks come from English."""
    flat = _flatten(json.loads((_WEB / "i18n" / "en.json").read_text(encoding="utf-8")))
    if lang != DEFAULT_LANGUAGE:
        path = _WEB / "i18n" / f"{lang}.json"
        flat.update(_flatten(json.loads(path.read_text(encoding="utf-8"))))
    return flat


def _usd(amount: float) -> str:
    return "$" + f"{amount:.6f}".rstrip("0").rstrip(".")


def _count(n: float) -> str:
    return f"{int(n):,}"


def _mtok(n: float) -> str:
    return f"{n:g}M"


def price_vars(s: Settings) -> dict[str, str]:
    image = s.price_image_usd
    video_s = s.price_video_per_second_usd
    chat_in, chat_out = s.price_chat_input_per_mtok_usd, s.price_chat_output_per_mtok_usd
    return {
        "price.image": _usd(image),
        "price.video_second": _usd(video_s),
        "price.video_10s": _usd(video_s * 10),
        "price.chat_input": _usd(chat_in),
        "price.chat_output": _usd(chat_out),
        "ex.amount": _usd(_EXAMPLE_CREDIT),
        "ex.images": _count(_EXAMPLE_CREDIT / image),
        "ex.video_clips": _count(_EXAMPLE_CREDIT / (video_s * 10)),
        "ex.chat_mtok": _mtok(_EXAMPLE_CREDIT / chat_in),
    }


def checkout_vars(s: Settings) -> dict[str, str]:
    sdk = ""
    if s.checkout_enabled:
        src = html.escape(_PAYPAL_SDK.format(quote(s.paypal_client_id, safe="")))
        sdk = f'<script src="{src}" defer></script>'
    sandbox = s.checkout_enabled and s.paypal_env == "sandbox"
    return {
        "checkout.state": "on" if s.checkout_enabled else "off",
        "checkout.banner": ('<div class="sandbox" role="status">{{billing.sandbox_banner}}</div>'
                            if sandbox else ""),
        "checkout.sdk": sdk,
        "topup.min": f"{s.topup_min_usd:g}",
        "topup.max": f"{s.topup_max_usd:g}",
    }


def _fill(text: str, values: dict[str, str]) -> str:
    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in values:
            raise KeyError(f"no value for placeholder {{{{{key}}}}}")
        return values[key]

    # Strings may themselves hold placeholders (prices, the support email).
    return _PLACEHOLDER.sub(sub, _PLACEHOLDER.sub(sub, text))


def _lang_menu(lang: str, names: dict[str, str], label: str) -> str:
    current = ' aria-current="true"'
    items = "".join(
        f'<li><a href="?lang={code}" hreflang="{code}" lang="{code}"'
        f'{current if code == lang else ""}>{html.escape(names[code])}</a></li>'
        for code in LANGUAGES
    )
    return (f'<details class="lang"><summary aria-label="{html.escape(label)}">'
            f'<svg class="i"><use href="#i-globe"/></svg><span>{html.escape(names[lang])}</span>'
            f"</summary><ul>{items}</ul></details>")


def render(name: str, lang: str, settings: Settings, base: str, path: str) -> str:
    names = {code: strings(code)["_language"] for code in LANGUAGES}
    values = {
        **strings(lang),
        **price_vars(settings),
        **checkout_vars(settings),
        "lang": lang,
        "support_email": SUPPORT_EMAIL,
        "base_code": '<code><span class="base">https://muse.isemi.io</span>/v1</code>',
        "lang_menu": _lang_menu(lang, names, strings(lang)["common.language"]),
        # base comes from the Host header unless public_base is set: escape it.
        "hreflang": "\n".join(
            f'<link rel="alternate" hreflang="{code}" '
            f'href="{html.escape(f"{base}{path}?lang={code}", quote=True)}">'
            for code in LANGUAGES
        ) + f'\n<link rel="alternate" hreflang="x-default" '
            f'href="{html.escape(base + path, quote=True)}">',
    }
    return _fill(_html(name), values)


def pick_language(request: Request) -> str:
    """``?lang=`` wins, then the cookie it sets, then Accept-Language, then English."""
    for candidate in (request.query_params.get("lang"), request.cookies.get("lang")):
        if candidate in LANGUAGES:
            return candidate
    for part in request.headers.get("accept-language", "").split(","):
        code = part.split(";")[0].strip().lower().split("-")[0]
        if code in LANGUAGES:
            return code
    return DEFAULT_LANGUAGE


def _localized(name: str, request: Request, headers: dict[str, str] = _HEADERS) -> HTMLResponse:
    settings = get_services(request).settings
    lang = pick_language(request)
    base = public_base(request)
    resp = HTMLResponse(render(name, lang, settings, base, request.url.path),
                        headers={**headers, "Vary": "Accept-Language, Cookie",
                                 "Content-Language": lang})
    if request.query_params.get("lang") == lang:
        resp.set_cookie("lang", lang, max_age=365 * 86400, samesite="lax")
    return resp


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing(request: Request) -> HTMLResponse:
    return _localized("landing.html", request)


@router.get("/docs", response_class=HTMLResponse, include_in_schema=False)
async def docs(request: Request) -> HTMLResponse:
    return _localized("docs.html", request)


@router.get("/billing", response_class=HTMLResponse, include_in_schema=False)
async def billing(request: Request) -> HTMLResponse:
    return _localized("billing.html", request, _CHECKOUT_HEADERS)


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    return HTMLResponse(_html("dashboard.html"),
                        headers={**_HEADERS, "X-Robots-Tag": "noindex, nofollow"})
