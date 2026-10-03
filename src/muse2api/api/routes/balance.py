from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ...services.billing import usd
from ...services.container import Services
from ..deps import get_services, require_api_key

router = APIRouter(tags=["billing"], dependencies=[Depends(require_api_key)])


@router.get("/v1/balance")
async def get_balance(request: Request, svc: Services = Depends(get_services)) -> dict:
    """Wallet balance of the calling key's customer (shared by all of its keys). The
    admin and legacy keys have none and are never billed."""
    key = svc.keys.get(request.state.key_id)
    balance = usd(await svc.billing.balance(key.customer_id)) if key else None
    return {"object": "balance", "balance_usd": balance, "currency": "USD",
            "unlimited": not svc.billing.billed(request.state.key_id)}
