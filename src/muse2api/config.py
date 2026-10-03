"""Runtime configuration, loaded from environment variables (prefix ``MUSE2API_``) and ``.env``."""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DriverName = Literal["mock", "browser", "http"]
PoolStrategyName = Literal["round_robin", "lru", "affinity"]
PayPalEnv = Literal["sandbox", "live"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MUSE2API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- server ---
    host: str = "127.0.0.1"
    port: int = 18610
    log_level: str = "INFO"
    public_base: str = Field(
        default="",
        description="Externally reachable base URL used when building media links. "
        "Empty means derive from the incoming request.",
    )

    # --- auth ---
    api_key: str = Field(default="", description="Bearer key for /v1/*. Auto-generated if empty.")
    admin_key: str = Field(default="", description="Bearer key for /admin/*. Falls back to api_key.")

    # --- storage ---
    data_dir: Path = Path("data")
    request_log_retention_days: float = Field(
        default=14, description="Days of /v1/* request history kept in data/requests.db."
    )

    # --- upstream driver ---
    driver: DriverName = "mock"
    site_url: str = "https://muse.ai"

    # browser driver
    chromium_path: str = ""
    cdp_port: int = 19210
    headless: bool = True
    browser_proxy: str = Field(
        default="",
        description="Proxy for the browser only, e.g. socks5://127.0.0.1:10808 or http://10.144.1.10:8080. "
        "Python's own calls (session renewal) follow http_proxy/https_proxy instead.",
    )
    browser_profile_dir: Path | None = None
    page_ready_timeout: float = 45.0

    # --- timeouts (seconds) ---
    chat_timeout: float = 300.0
    first_token_timeout: float = 45.0
    image_timeout: float = 240.0
    video_timeout: float = 600.0

    # --- account pool ---
    pool_strategy: PoolStrategyName = "affinity"
    pool_acquire_timeout: float = 60.0
    account_max_concurrency: int = 1
    account_cooldown: float = 120.0
    max_failover: int = 2

    # --- background removal (images with background="transparent") ---
    matting_model: str = Field(
        default="birefnet-general",
        description="rembg model used to cut out the subject, e.g. birefnet-general "
        "(best) or birefnet-general-lite (faster).",
    )

    # --- billing (prepaid USD credit per stored client key) ---
    billing_enabled: bool = Field(
        default=True, description="Charge stored client keys for successful requests. "
        "The admin key, the legacy api_key and keys marked unlimited are never charged.")
    price_image_usd: float = Field(default=0.015, description="Per generated image.")
    price_video_per_second_usd: float = Field(
        default=0.006, description="Per second of requested video duration.")
    video_default_seconds: int = Field(
        default=10, description="Duration billed when a video request gives none.")
    price_chat_input_per_mtok_usd: float = 1.0
    price_chat_output_per_mtok_usd: float = 3.0

    # --- self-serve checkout (/billing, PayPal Orders API v2) ---
    # Checkout is disabled (the page shows the support email) until both are set.
    paypal_client_id: str = ""
    paypal_client_secret: str = ""
    paypal_env: PayPalEnv = "sandbox"
    paypal_allow_sandbox: bool = Field(
        default=False, description="Run checkout against the PayPal sandbox. Sandbox payments "
        "are fake but the credit is real, so only for testing.")
    paypal_webhook_id: str = Field(
        default="", description="Webhook ID from the PayPal app; empty ignores webhooks.")
    topup_min_usd: float = 5.0
    topup_max_usd: float = 1000.0

    # --- keepalive (session renewal) ---
    keepalive_enabled: bool = False
    keepalive_interval: float = 6 * 3600

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def accounts_file(self) -> Path:
        return self.data_dir / "accounts.json"

    @property
    def tasks_file(self) -> Path:
        return self.data_dir / "tasks.json"

    @property
    def key_file(self) -> Path:
        return self.data_dir / "api_key"

    @property
    def keys_file(self) -> Path:
        return self.data_dir / "keys.json"

    @property
    def requests_db(self) -> Path:
        return self.data_dir / "requests.db"

    @property
    def profile_dir(self) -> Path:
        return self.browser_profile_dir or (self.data_dir / "browser-profile")

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.media_dir):
            d.mkdir(parents=True, exist_ok=True)

    def resolve_api_key(self) -> str:
        """Return the configured key, or load/generate a persistent one under ``data_dir``."""
        if self.api_key:
            return self.api_key
        self.ensure_dirs()
        if self.key_file.is_file():
            key = self.key_file.read_text(encoding="utf-8").strip()
            if key:
                self.api_key = key
                return key
        key = "m2a-" + secrets.token_urlsafe(24)
        self.key_file.write_text(key + "\n", encoding="utf-8")
        self.key_file.chmod(0o600)
        self.api_key = key
        return key

    @property
    def checkout_enabled(self) -> bool:
        # Sandbox payments are fake money, so they need an explicit opt-in.
        return bool(self.paypal_client_id and self.paypal_client_secret) and (
            self.paypal_env == "live" or self.paypal_allow_sandbox)

    @property
    def effective_admin_key(self) -> str:
        return self.admin_key or self.api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
