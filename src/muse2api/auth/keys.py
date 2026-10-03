"""Client API keys, persisted to ``data/keys.json``.

Only a SHA-256 hash and a short display prefix are stored; the plaintext key is
returned once, when it is created. ``last_used_at`` is kept in memory and
written out by ``flush()`` so that authenticating a request never touches disk.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import tempfile
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

KEY_PREFIX = "m2a-"
_DISPLAY_LEN = 8
# Format version of keys.json, kept in a file of its own so that an older release
# rewriting keys.json (rollback) cannot drop it. 2 = keys carry "unlimited".
_VERSION = "2"


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(24)


class ApiKey(BaseModel):
    id: str = Field(default_factory=lambda: "key_" + uuid.uuid4().hex[:10])
    name: str
    prefix: str
    hash: str
    created_at: float = Field(default_factory=time.time)
    last_used_at: float = 0.0
    revoked: bool = False
    note: str = ""
    # Not charged for requests (see services/billing.py).
    unlimited: bool = False

    def public(self) -> dict:
        """Serialisable view without the hash."""
        return self.model_dump(mode="json", exclude={"hash"})


class KeyStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.version_file = path.with_name(path.stem + ".version")
        self._keys: dict[str, ApiKey] = {}
        self._lock = asyncio.Lock()
        self._dirty = False

    async def load(self) -> None:
        if not self.path.is_file():
            return
        raw = await asyncio.to_thread(self.path.read_text, encoding="utf-8")
        if not raw.strip():
            return
        items = json.loads(raw)
        # One-time billing migration, keyed on the version file rather than on the
        # field: keys from before billing stay free. Once migrated, a key without
        # the field (say, after a rollback rewrote the file) loads as billed.
        migrate = not self.version_file.is_file()
        migrated = [i for i in items if "unlimited" not in i] if migrate else []
        for item in migrated:
            item["unlimited"] = True
        self._keys = {k.id: k for k in (ApiKey.model_validate(i) for i in items)}
        if migrate:
            log.info("marked %d pre-billing key(s) unlimited", len(migrated))
            try:
                await self.save()
            except OSError:
                # Retried on the next start; until then the keys stay unlimited in memory.
                log.exception("failed to write migrated %s", self.path)

    def all(self) -> list[ApiKey]:
        return sorted(self._keys.values(), key=lambda k: k.created_at)

    def get(self, key_id: str) -> ApiKey | None:
        return self._keys.get(key_id)

    def verify(self, provided: str) -> ApiKey | None:
        """Return the active key matching ``provided``, comparing hashes in constant time."""
        if not provided:
            return None
        digest = hash_key(provided)
        found = None
        # No early exit, so timing does not reveal which (or whether a) key matched.
        for key in self._keys.values():
            if hmac.compare_digest(digest, key.hash) and not key.revoked:
                found = key
        return found

    def touch(self, key: ApiKey) -> None:
        now = time.time()
        # Second resolution is plenty for a "last used" column.
        if now - key.last_used_at >= 1:
            key.last_used_at = now
            self._dirty = True

    async def create(self, name: str, note: str = "") -> tuple[ApiKey, str]:
        plaintext = generate_key()
        key = ApiKey(name=name, note=note, prefix=plaintext[:_DISPLAY_LEN],
                     hash=hash_key(plaintext))
        self._keys[key.id] = key
        await self.save()
        return key, plaintext

    async def remove(self, key_id: str) -> bool:
        if self._keys.pop(key_id, None) is None:
            return False
        await self.save()
        return True

    async def save(self) -> None:
        self._dirty = False
        payload = json.dumps([k.model_dump(mode="json") for k in self.all()],
                             ensure_ascii=False, indent=2)
        async with self._lock:
            await asyncio.to_thread(self._atomic_write, payload)

    async def flush(self) -> None:
        """Persist pending ``last_used_at`` updates, if any."""
        if self._dirty:
            try:
                await self.save()
            except OSError:
                self._dirty = True
                log.exception("failed to write %s", self.path)

    def _atomic_write(self, payload: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".keys.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        if not self.version_file.is_file():
            self.version_file.write_text(_VERSION + "\n", encoding="utf-8")
