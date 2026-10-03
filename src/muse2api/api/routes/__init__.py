from fastapi import APIRouter

from . import (
    account,
    admin,
    balance,
    chat,
    checkout,
    health,
    images,
    media,
    models,
    pages,
    responses,
    videos,
)


def build_router() -> APIRouter:
    router = APIRouter()
    for module in (health, models, chat, responses, images, videos, balance, media, admin,
                   checkout, account, pages):
        router.include_router(module.router)
    return router
