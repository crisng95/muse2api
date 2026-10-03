"""Service container wired once at startup and stored on ``app.state.services``."""

from __future__ import annotations

from dataclasses import dataclass

from ..accounts.pool import AccountPool
from ..accounts.store import JsonAccountStore
from ..auth.keys import KeyStore
from ..config import Settings
from ..core.matting import Matting
from ..core.media import MediaStore
from ..drivers.base import MuseDriver
from ..drivers.registry import create_driver
from .billing import REFUNDABLE, Billing
from .customers import Customers
from .gateway import Gateway
from .google import GoogleOAuth
from .payments import Payments
from .paypal import PayPalClient
from .request_log import RequestLog, current_record
from .tasks import Task, TaskManager


@dataclass
class Services:
    settings: Settings
    driver: MuseDriver
    pool: AccountPool
    gateway: Gateway
    tasks: TaskManager
    media: MediaStore
    matting: Matting
    keys: KeyStore
    requests: RequestLog
    billing: Billing
    customers: Customers
    google: GoogleOAuth
    payments: Payments

    @classmethod
    def build(cls, settings: Settings, driver: MuseDriver | None = None) -> Services:
        settings.ensure_dirs()
        driver = driver or create_driver(settings)
        pool = AccountPool(
            JsonAccountStore(settings.accounts_file),
            strategy=settings.pool_strategy,
            max_concurrency=settings.account_max_concurrency,
            cooldown=settings.account_cooldown,
            acquire_timeout=settings.pool_acquire_timeout,
        )
        requests = RequestLog(settings.requests_db)
        keys = KeyStore(settings.keys_file)
        billing = Billing(settings, keys, requests)
        customers = Customers(settings, keys, billing)
        billing.customers = customers

        async def record_outcome(task: Task) -> None:
            record = current_record.get()
            # Failed work is free: give back what the submit reserved.
            refunded = 0
            if task.status in REFUNDABLE:
                refunded = await billing.refund(task.id)
            else:
                await billing.settle(task.id)
            await requests.finish_task(task.id, task.status.value, task.updated_at,
                                       (task.error or {}).get("message"),
                                       record.get("account_id") if record else None,
                                       record.get("attempt_errors") if record else None,
                                       cost_micro=0 if refunded else None)

        return cls(
            settings=settings,
            driver=driver,
            pool=pool,
            gateway=Gateway(pool, driver, max_failover=settings.max_failover),
            tasks=TaskManager(settings.tasks_file, on_finish=record_outcome),
            media=MediaStore(settings.media_dir),
            matting=Matting(settings.matting_model),
            keys=keys,
            requests=requests,
            billing=billing,
            customers=customers,
            google=GoogleOAuth(settings),
            payments=Payments(settings, keys, billing, customers, PayPalClient(settings)),
        )
