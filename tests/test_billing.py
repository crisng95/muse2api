from __future__ import annotations

import json
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient

from muse2api.app import create_app
from muse2api.core.tokens import count_tokens
from muse2api.drivers.mock import MockDriver
from muse2api.errors import UpstreamGlitch, UpstreamRefused
from muse2api.services.tasks import Task, TaskStatus


def _bearer(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def _key(client, admin, credit: float | None = None, name: str = "client") -> tuple[str, dict]:
    """Create a stored key, optionally topped up; returns (key id, auth headers)."""
    body = client.post("/admin/keys", headers=admin, json={"name": name}).json()
    key_id = body["key"]["id"]
    if credit is not None:
        r = client.post(f"/admin/keys/{key_id}/credit", headers=admin, json={"amount_usd": credit})
        assert r.status_code == 200
    return key_id, _bearer(body["api_key"])


def _balance(client, headers) -> float:
    return client.get("/v1/balance", headers=headers).json()["balance_usd"]


def _ledger(client, admin, key_id) -> list[dict]:
    return client.get(f"/admin/keys/{key_id}/ledger", headers=admin).json()["data"]


def _wait(client, path, headers) -> dict:
    for _ in range(200):
        view = client.get(path, headers=headers).json()
        if view["status"] in ("succeeded", "failed", "cancelled"):
            return view
        time.sleep(0.02)
    raise AssertionError("task did not finish")


def _last_row(client, admin, path) -> dict:
    rows = client.get("/admin/requests", headers=admin, params={"path": path}).json()["data"]
    return next(r for r in rows if not r["poll"])


def _fail_images(client, monkeypatch) -> None:
    async def boom(account, req):
        raise UpstreamRefused("no image")

    monkeypatch.setattr(client.app.state.services.driver, "generate_image", boom)


def test_count_tokens():
    assert count_tokens("") == 0
    assert count_tokens("a") == 1
    assert count_tokens("abcd") == 1
    assert count_tokens("abcde") == 2
    assert count_tokens("你好世界") == 4
    assert count_tokens("こんにちは") == 5
    assert count_tokens("안녕") == 2
    assert count_tokens("hi 你好") == 2 + 1  # two ideographs + ceil(3 / 4)


def test_image_charged_on_success(client, admin):
    key_id, h = _key(client, admin, credit=1.0)
    r = client.post("/v1/images/generations", headers=h, json={"prompt": "a cat", "n": 2})
    assert r.status_code == 200
    assert r.json()["cost_usd"] == 0.03
    assert _balance(client, h) == 0.97
    charge, topup = _ledger(client, admin, key_id)
    assert (charge["kind"], charge["amount_usd"], charge["balance_after_usd"]) == ("charge", -0.03, 0.97)
    assert topup["kind"] == "topup"
    assert _last_row(client, admin, "/v1/images/generations")["cost_micro"] == 30000


def test_image_402_reserves_nothing(client, admin):
    key_id, h = _key(client, admin, credit=0.004)
    r = client.post("/v1/images/generations", headers=h, json={"prompt": "a cat"})
    assert r.status_code == 402
    err = r.json()["error"]
    assert err["type"] == "insufficient_quota" and err["code"] == "insufficient_balance"
    assert "$0.015000" in err["message"] and "$0.004000" in err["message"]
    assert _balance(client, h) == 0.004
    assert [e["kind"] for e in _ledger(client, admin, key_id)] == ["topup"]
    # Async submits are checked before a task is created too.
    r = client.post("/v1/images/generations", headers=h, json={"prompt": "a cat", "async": True})
    assert r.status_code == 402
    assert client.get("/admin/tasks", headers=admin).json()["data"] == []


def test_video_402_and_charge(client, admin):
    key_id, h = _key(client, admin, credit=0.05)
    r = client.post("/v1/videos", headers=h, json={"prompt": "a wave"})  # default 10 s = $0.06
    assert r.status_code == 402
    assert _balance(client, h) == 0.05
    task = client.post("/v1/videos", headers=h, json={"prompt": "a wave", "seconds": 5}).json()
    assert task["cost_usd"] == 0.03
    assert _wait(client, f"/v1/videos/{task['id']}", h)["status"] == "succeeded"
    assert _balance(client, h) == 0.02
    assert _ledger(client, admin, key_id)[0]["ref"] == task["id"]


def test_sync_image_failure_refunded(client, admin, monkeypatch):
    key_id, h = _key(client, admin, credit=1.0)
    _fail_images(client, monkeypatch)
    r = client.post("/v1/images/generations", headers=h, json={"prompt": "a cat"})
    assert r.status_code == 502
    assert _balance(client, h) == 1.0
    refund, charge, _ = _ledger(client, admin, key_id)
    assert (refund["kind"], refund["amount_usd"], refund["ref"]) == ("refund", 0.015, charge["ref"])
    assert _last_row(client, admin, "/v1/images/generations")["cost_micro"] == 0


def test_async_image_failure_refunded(client, admin, monkeypatch):
    key_id, h = _key(client, admin, credit=1.0)
    _fail_images(client, monkeypatch)
    task = client.post("/v1/images/generations", headers=h,
                       json={"prompt": "a cat", "async": True}).json()
    assert task["cost_usd"] == 0.015
    view = _wait(client, f"/v1/images/generations/{task['id']}", h)
    assert view["status"] == "failed" and view["cost_usd"] == 0.0
    assert _balance(client, h) == 1.0
    assert [e["kind"] for e in _ledger(client, admin, key_id)] == ["refund", "charge", "topup"]
    assert _last_row(client, admin, "/v1/images/generations")["cost_micro"] == 0


def test_refund_after_restart(settings, admin):
    with TestClient(create_app(settings, MockDriver(delay=0))) as c:
        key_id, h = _key(c, admin, credit=1.0)
        c.portal.call(c.app.state.services.billing.reserve, key_id, 60000, "task_lost", "video 10s")
        assert _balance(c, h) == 0.94
    # The server died while the task was running.
    settings.tasks_file.write_text(json.dumps([Task(
        id="task_lost", kind="video", status=TaskStatus.RUNNING).model_dump(mode="json")]))
    for _ in range(2):  # the second restart must not refund again
        with TestClient(create_app(settings, MockDriver(delay=0))) as c:
            assert _balance(c, h) == 1.0
            kinds = [e["kind"] for e in _ledger(c, admin, key_id)]
            assert kinds == ["refund", "charge", "topup"]


def _reservations(c) -> list[str]:
    billing = c.app.state.services.billing
    rows = c.portal.call(billing._run, lambda db: db.execute(
        "SELECT ref FROM reservations").fetchall())
    return [r[0] for r in rows]


def test_restart_refunds_orphaned_reservations(settings, admin):
    with TestClient(create_app(settings, MockDriver(delay=0))) as c:
        key_id, h = _key(c, admin, credit=1.0)
        reserve = c.app.state.services.billing.reserve
        # A sync image and an async submit killed mid-flight (the task never reached
        # tasks.json), plus a task that succeeded but whose settle was lost.
        c.portal.call(reserve, key_id, 15000, "img_killed")
        c.portal.call(reserve, key_id, 60000, "task_unsaved")
        c.portal.call(reserve, key_id, 15000, "task_done")
        # Work that finished normally settles its reservation.
        assert c.post("/v1/images/generations", headers=h, json={"prompt": "x"}).status_code == 200
        task = c.post("/v1/videos", headers=h, json={"prompt": "x", "seconds": 1}).json()
        assert _wait(c, f"/v1/videos/{task['id']}", h)["status"] == "succeeded"
        assert sorted(_reservations(c)) == ["img_killed", "task_done", "task_unsaved"]
        assert _balance(c, h) == round(1 - 0.09 - 0.015 - 0.006, 6)
    tasks = json.loads(settings.tasks_file.read_text())
    tasks.append(Task(id="task_done", kind="image", status=TaskStatus.SUCCEEDED)
                 .model_dump(mode="json"))
    settings.tasks_file.write_text(json.dumps(tasks))
    for _ in range(2):
        with TestClient(create_app(settings, MockDriver(delay=0))) as c:
            assert _reservations(c) == []
            assert _balance(c, h) == round(1 - 0.015 - 0.015 - 0.006, 6)
            refunds = [e["ref"] for e in _ledger(c, admin, key_id) if e["kind"] == "refund"]
            assert sorted(refunds) == ["img_killed", "task_unsaved"]


def test_tx_rolls_back_when_commit_fails():
    from muse2api.services.billing import _tx

    class DB:
        def __init__(self):
            self.calls = []

        def execute(self, sql, *args):
            self.calls.append(sql)
            if sql == "COMMIT":
                raise sqlite3.OperationalError("disk I/O error")

    db = DB()
    with pytest.raises(sqlite3.OperationalError):
        _tx(db, lambda d: None)
    assert db.calls == ["BEGIN IMMEDIATE", "COMMIT", "ROLLBACK"]


def test_chat_charged_by_tokens(client, admin):
    key_id, h = _key(client, admin, credit=1.0)
    r = client.post("/v1/chat/completions", headers=h, json={
        "model": "gpt-4o", "messages": [{"role": "user", "content": "hello there"}]})
    assert r.status_code == 200
    body = r.json()
    usage = body["usage"]
    assert usage["prompt_tokens"] == count_tokens("hello there")
    assert usage["completion_tokens"] == count_tokens(body["choices"][0]["message"]["content"])
    expected = usage["prompt_tokens"] * 1 + usage["completion_tokens"] * 3  # micro-USD
    assert usage["cost_usd"] == round(expected / 1e6, 6)
    assert _balance(client, h) == round(1 - expected / 1e6, 6)
    entry = _ledger(client, admin, key_id)[0]
    assert entry["kind"] == "charge" and entry["ref"] == body["id"]


def test_chat_stream_charged_at_end(client, admin):
    key_id, h = _key(client, admin, credit=1.0)
    with client.stream("POST", "/v1/chat/completions", headers=h, json={
        "messages": [{"role": "user", "content": "stream me"}], "stream": True,
        "stream_options": {"include_usage": True},
    }) as r:
        lines = [ln[6:] for ln in r.iter_lines() if ln.startswith("data: ")]
    usage = json.loads(lines[-2])["usage"]
    assert usage["cost_usd"] > 0
    assert _balance(client, h) == round(1 - usage["cost_usd"], 6)
    assert _last_row(client, admin, "/v1/chat/completions")["cost_micro"] == round(usage["cost_usd"] * 1e6)


def test_chat_rejected_without_balance_but_may_go_negative(client, admin):
    _, h = _key(client, admin, credit=None)
    r = client.post("/v1/chat/completions", headers=h,
                    json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 402
    _, h = _key(client, admin, credit=0.000001, name="tiny")
    r = client.post("/v1/chat/completions", headers=h,
                    json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert _balance(client, h) < 0


def test_chat_error_not_charged(client, admin, monkeypatch):
    key_id, h = _key(client, admin, credit=1.0)

    async def boom(account, req):
        raise UpstreamRefused("nope")
        yield  # pragma: no cover

    monkeypatch.setattr(client.app.state.services.driver, "chat_stream", boom)
    r = client.post("/v1/chat/completions", headers=h,
                    json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    assert [e["kind"] for e in _ledger(client, admin, key_id)] == ["topup"]


def test_chat_glitch_reply_not_charged(client, admin, monkeypatch):
    key_id, h = _key(client, admin, credit=1.0)

    async def glitch(account, req):
        yield "Sorry, I ran into a problem while responding."
        raise UpstreamGlitch("muse.ai failed while responding")

    monkeypatch.setattr(client.app.state.services.driver, "chat_stream", glitch)
    msgs = {"messages": [{"role": "user", "content": "hi"}]}
    assert client.post("/v1/chat/completions", headers=h, json=msgs).status_code == 502
    with client.stream("POST", "/v1/chat/completions", headers=h,
                       json={**msgs, "stream": True}) as r:
        body = "".join(r.iter_lines())
    assert "upstream_glitch" in body
    assert [e["kind"] for e in _ledger(client, admin, key_id)] == ["topup"]


@pytest.mark.parametrize("who", ["unlimited", "legacy", "admin"])
def test_unbilled_identities(client, admin, auth, who):
    if who == "unlimited":
        key_id, h = _key(client, admin)
        r = client.patch(f"/admin/keys/{key_id}", headers=admin, json={"unlimited": True})
        assert r.json()["key"]["unlimited"] is True
    else:
        key_id, h = None, auth if who == "legacy" else admin
    r = client.post("/v1/images/generations", headers=h, json={"prompt": "a cat"})
    assert r.status_code == 200
    assert r.json()["cost_usd"] == 0.015  # informational
    assert client.post("/v1/chat/completions", headers=h, json={
        "messages": [{"role": "user", "content": "hi"}]}).status_code == 200
    balance = client.get("/v1/balance", headers=h).json()
    assert balance["unlimited"] is True
    if key_id:
        assert balance["balance_usd"] == 0.0
        assert _ledger(client, admin, key_id) == []
    else:
        assert balance["balance_usd"] is None
    assert _last_row(client, admin, "/v1/images/generations")["cost_micro"] is None


def test_billing_disabled(settings, admin):
    settings.billing_enabled = False
    with TestClient(create_app(settings, MockDriver(delay=0))) as c:
        _, h = _key(c, admin)
        assert c.post("/v1/images/generations", headers=h, json={"prompt": "x"}).status_code == 200


def test_balance_endpoint(client, admin):
    _, h = _key(client, admin, credit=12.34)
    assert client.get("/v1/balance", headers=h).json() == {
        "object": "balance", "balance_usd": 12.34, "currency": "USD", "unlimited": False}
    assert client.get("/v1/balance").status_code == 401


def test_admin_credit_and_ledger(client, admin, auth):
    key_id, _ = _key(client, admin)
    r = client.post(f"/admin/keys/{key_id}/credit", headers=admin,
                    json={"amount_usd": 10, "note": "invoice #12"})
    customer_id = client.get("/admin/keys", headers=admin).json()["data"][0]["customer_id"]
    assert r.json() == {"key_id": key_id, "customer_id": customer_id, "kind": "topup",
                        "amount_usd": 10.0, "balance_usd": 10.0}
    r = client.post(f"/admin/keys/{key_id}/credit", headers=admin, json={"amount_usd": -2.5})
    assert r.json()["kind"] == "adjust" and r.json()["balance_usd"] == 7.5
    assert client.post(f"/admin/keys/{key_id}/credit", headers=admin,
                       json={"amount_usd": 0}).status_code == 400
    assert client.post("/admin/keys/key_nope/credit", headers=admin,
                       json={"amount_usd": 1}).status_code == 404
    assert client.post(f"/admin/keys/{key_id}/credit", headers=auth,
                       json={"amount_usd": 1}).status_code == 401

    adjust, topup = _ledger(client, admin, key_id)
    assert (adjust["kind"], adjust["amount_usd"], adjust["balance_after_usd"]) == ("adjust", -2.5, 7.5)
    assert (topup["note"], topup["balance_after_usd"]) == ("invoice #12", 10.0)
    assert len(client.get(f"/admin/keys/{key_id}/ledger", headers=admin,
                          params={"limit": 1}).json()["data"]) == 1
    (listed,) = client.get("/admin/keys", headers=admin).json()["data"]
    assert listed["balance_usd"] == 7.5 and listed["unlimited"] is False


def test_migration_marks_existing_keys_unlimited(settings, admin):
    settings.ensure_dirs()
    settings.keys_file.write_text(json.dumps([{
        "id": "key_old", "name": "postforge", "prefix": "m2a-abcd", "hash": "0" * 64,
        "created_at": 1.0, "last_used_at": 0.0, "revoked": False, "note": ""}]))
    with TestClient(create_app(settings, MockDriver(delay=0))) as c:
        old = c.get("/admin/keys", headers=admin).json()["data"][0]
        assert old["unlimited"] is True
        new = c.post("/admin/keys", headers=admin, json={"name": "new"}).json()["key"]
        assert new["unlimited"] is False
    stored = {k["id"]: k for k in json.loads(settings.keys_file.read_text())}
    assert stored["key_old"]["unlimited"] is True
    assert stored[new["id"]]["unlimited"] is False
    # Once written, the flag is honoured as stored (an admin may turn billing on later).
    stored["key_old"]["unlimited"] = False
    settings.keys_file.write_text(json.dumps(list(stored.values())))
    with TestClient(create_app(settings, MockDriver(delay=0))) as c:
        keys = {k["id"]: k for k in c.get("/admin/keys", headers=admin).json()["data"]}
        assert keys["key_old"]["unlimited"] is False
    # A rollback to pre-billing code rewrites keys.json without the field: the
    # migration does not run again, so paying keys stay billed.
    for item in stored.values():
        item.pop("unlimited")
    settings.keys_file.write_text(json.dumps(list(stored.values())))
    with TestClient(create_app(settings, MockDriver(delay=0))) as c:
        keys = c.get("/admin/keys", headers=admin).json()["data"]
        assert all(k["unlimited"] is False for k in keys)


def test_failed_migration_save_does_not_stop_startup(settings, admin, monkeypatch):
    from muse2api.auth.keys import KeyStore

    settings.ensure_dirs()
    settings.keys_file.write_text(json.dumps([{
        "id": "key_old", "name": "old", "prefix": "m2a-abcd", "hash": "0" * 64}]))

    def readonly(self, payload):
        raise PermissionError("read-only")

    monkeypatch.setattr(KeyStore, "_atomic_write", readonly)
    with TestClient(create_app(settings, MockDriver(delay=0))) as c:
        assert c.get("/admin/keys", headers=admin).json()["data"][0]["unlimited"] is True
    assert not (settings.data_dir / "keys.version").exists()


async def test_prune_keeps_ledger(settings):
    from muse2api.auth.keys import KeyStore
    from muse2api.services.billing import Billing
    from muse2api.services.request_log import RequestLog

    log = RequestLog(settings.requests_db)
    billing = Billing(settings, KeyStore(settings.keys_file), log)
    await billing.credit("key_x", 5_000_000)
    await log.prune(-1)  # cutoff in the future: everything prunable goes
    assert await billing.balance("key_x") == 5_000_000
    assert len(await billing.ledger("key_x")) == 1
    await log.close()
