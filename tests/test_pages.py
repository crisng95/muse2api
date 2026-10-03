"""The public landing and docs pages, and the internal dashboard."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from muse2api.api.routes import pages
from muse2api.core.models import ALIASES, MODELS

# Words that would tell a customer how the service is built.
_INTERNAL = re.compile(r"cookie|muse\.ai|muse2api|browser|chromium|account", re.I)
_I18N = Path(pages.__file__).resolve().parents[2] / "web" / "i18n"


def _path_shape(text: str) -> str:
    return re.sub(r"\{[^}]*\}", "{}", text)


@pytest.mark.parametrize("lang", pages.LANGUAGES)
@pytest.mark.parametrize("path", ["/", "/docs"])
def test_public_page_in_every_language(client, path, lang):
    r = client.get(path, params={"lang": lang})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert r.headers["content-language"] == lang
    assert f'<html lang="{lang}">' in r.text
    assert "{{" not in r.text
    assert "x-robots-tag" not in r.headers
    assert not _INTERNAL.search(r.text)
    assert "/dashboard" not in r.text


@pytest.mark.parametrize("lang", [code for code in pages.LANGUAGES if code != "en"])
def test_translations_have_exactly_the_english_keys(lang):
    def keys(path: Path) -> set[str]:
        return set(pages._flatten(json.loads(path.read_text(encoding="utf-8"))))

    assert keys(_I18N / f"{lang}.json") == keys(_I18N / "en.json")


def test_language_choice(client):
    assert client.get("/", headers={"accept-language": "ja-JP,ja;q=0.9"}).headers["content-language"] == "ja"
    assert client.get("/", headers={"accept-language": "fr-FR"}).headers["content-language"] == "en"
    r = client.get("/", params={"lang": "vi"}, headers={"accept-language": "ja"})
    assert r.headers["content-language"] == "vi" and "lang=vi" in r.headers["set-cookie"]
    # The cookie set by ?lang= now beats Accept-Language.
    assert client.get("/docs", headers={"accept-language": "ja"}).headers["content-language"] == "vi"


def test_pages_quote_the_configured_prices(client):
    settings = client.app.state.services.settings
    settings.price_image_usd = 0.02
    text = client.get("/", params={"lang": "en"}).text
    assert "$0.02" in text and "$0.015" not in text
    assert "$0.02" in client.get("/docs", params={"lang": "en"}).text


def test_docs_cover_every_model_and_endpoint(client):
    text = client.get("/docs", params={"lang": "en"}).text
    for name in (*MODELS, *ALIASES):
        assert f"<code>{name}</code>" in text, name
    shaped = _path_shape(text)
    paths = {r.path for r in client.app.routes if getattr(r, "path", "").startswith("/v1/")}
    paths.discard("/v1/responses")  # reserved, answers 501
    for path in paths:
        assert _path_shape(path) in shaped, path


def test_landing_lists_every_model(client):
    text = client.get("/", params={"lang": "en"}).text
    for name in MODELS:
        assert f"<h3>{name}</h3>" in text, name


def test_dashboard_is_kept_out_of_search(client):
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "<title>muse2api dashboard</title>" in r.text
    assert r.headers["x-robots-tag"] == "noindex, nofollow"


@pytest.mark.parametrize("path", ["/openapi.json", "/redoc"])
def test_generated_schema_is_not_served(client, path):
    # It would list /admin/* and its request bodies.
    assert client.get(path).status_code == 404
