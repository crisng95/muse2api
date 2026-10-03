from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ...core.media import load_image_ref
from ...core.models import resolve_model
from ...core.prompt import flatten_messages, message_turns
from ...core.tokens import count_tokens
from ...drivers.base import ChatRequest, InputImage
from ...errors import InvalidRequest, Muse2APIError, UpstreamGlitch
from ...services.billing import usd
from ...services.container import Services
from ..deps import get_services, require_api_key
from ..schemas import ChatCompletionRequest

log = logging.getLogger(__name__)
router = APIRouter(tags=["chat"], dependencies=[Depends(require_api_key)])


def _sse(payload: dict | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {data}\n\n".encode()


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request,
                           svc: Services = Depends(get_services)):
    request.state.model = body.model  # logged even if it does not resolve
    spec = resolve_model(body.model, "chat")
    request.state.model = body.model or spec.id
    flat = flatten_messages([m.model_dump() for m in body.messages])
    if not flat.text and not flat.images:
        raise InvalidRequest("messages contain no text or images")
    # Priced only once the reply is done, so just turn away an empty balance here.
    key_id = request.state.key_id
    await svc.billing.check_positive(key_id)

    images = [InputImage(*(await load_image_ref(ref))) for ref in flat.images]
    cancel = asyncio.Event()
    req = ChatRequest(
        prompt=flat.text,
        model=spec.id,
        images=images,
        timeout=svc.settings.chat_timeout,
        first_token_timeout=svc.settings.first_token_timeout,
        conversation_hint=body.user,
        turns=message_turns([m.model_dump() for m in body.messages]),
        cancel=cancel,
    )
    completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    model_name = body.model or spec.id
    stream = svc.gateway.chat_stream(req)
    prompt_tokens = count_tokens(flat.text)

    def usage(reply: str) -> dict:
        # The same numbers are billed, so what the client sees is what it pays.
        completion_tokens = count_tokens(reply)
        return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost_usd": usd(svc.billing.chat_cost(prompt_tokens, completion_tokens))}

    async def charge(reply: str) -> None:
        completion_tokens = count_tokens(reply)
        await svc.billing.charge(key_id, svc.billing.chat_cost(prompt_tokens, completion_tokens),
                                 completion_id, f"chat {prompt_tokens}+{completion_tokens} tokens")

    if not body.stream:
        text = "".join([d async for d in stream])
        await charge(text)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": usage(text),
        }

    # Pull the first delta before committing to a 200 so that account/upstream
    # failures surface as proper HTTP errors instead of a broken event stream.
    try:
        first = await anext(stream)
    except StopAsyncIteration:
        first = ""

    def chunk(delta: dict, finish: str | None = None) -> dict:
        return {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    async def events() -> AsyncIterator[bytes]:
        produced = first
        try:
            yield _sse(chunk({"role": "assistant", "content": first}))
            async for delta in stream:
                produced += delta
                yield _sse(chunk({"content": delta}))
            yield _sse(chunk({}, "stop"))
            if (body.stream_options or {}).get("include_usage"):
                yield _sse({"id": completion_id, "object": "chat.completion.chunk",
                            "created": created, "model": model_name, "choices": [],
                            "usage": usage(produced)})
        except Muse2APIError as exc:
            if isinstance(exc, UpstreamGlitch):
                produced = ""  # muse.ai's own failure text, not a billable answer
            yield _sse(exc.to_body())
        except Exception:  # noqa: BLE001
            log.exception("stream crashed")
            yield _sse({"error": {"message": "internal error", "type": "server_error"}})
        finally:
            cancel.set()
            try:
                await stream.aclose()
            finally:
                # Also on a client disconnect or a mid-stream error: whatever was
                # produced is charged. Nothing produced, nothing charged.
                if produced:
                    await charge(produced)
        yield _sse("[DONE]")

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
