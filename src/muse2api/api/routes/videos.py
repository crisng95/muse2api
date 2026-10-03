from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ...core.media import load_image_ref
from ...core.models import resolve_model
from ...drivers.base import InputImage, VideoRequest
from ...errors import NotFound
from ...services.billing import task_cost_usd, usd
from ...services.container import Services
from ...services.tasks import Task, new_task_id
from ..deps import get_services, public_base, require_api_key
from ..schemas import VideoCreateRequest

router = APIRouter(tags=["videos"], dependencies=[Depends(require_api_key)])


def _task_view(task: Task) -> dict:
    return {
        "id": task.id,
        "object": "video",
        "status": task.status.value,
        "progress": task.progress,
        "created_at": int(task.created_at),
        "model": task.request.get("model"),
        "result": task.result,
        "error": task.error,
        "cost_usd": task_cost_usd(task),
    }


@router.post("/v1/videos")
async def create_video(body: VideoCreateRequest, request: Request,
                       svc: Services = Depends(get_services)) -> dict:
    request.state.model = body.model  # logged even if it does not resolve
    spec = resolve_model(body.model, "video")
    request.state.model = body.model or spec.id
    frame_ref = body.image or body.input_reference
    first_frame = InputImage(*(await load_image_ref(frame_ref))) if frame_ref else None
    base = public_base(request)

    async def runner(progress) -> dict:
        req = VideoRequest(prompt=body.prompt, model=spec.id, size=body.size,
                           duration=body.duration or body.seconds, first_frame=first_frame,
                           timeout=svc.settings.video_timeout, on_progress=progress)
        result = await svc.gateway.generate_video(req)
        name = svc.media.save(result.data, result.mime, prefix="vid")
        return {"url": f"{base}/v1/media/{name}", "mime": result.mime,
                "width": result.width, "height": result.height}

    # Billed on the requested duration, reserved before any upstream work; the
    # task's on_finish hook refunds a failure.
    seconds = body.duration or body.seconds
    cost = svc.billing.video_cost(seconds)
    request_meta = {"model": spec.id, "prompt": body.prompt, "size": body.size,
                    "duration": seconds, "has_first_frame": bool(first_frame),
                    "cost_usd": usd(cost)}
    task_id = new_task_id()
    await svc.billing.reserve(request.state.key_id, cost, task_id,
                              f"video {seconds or svc.settings.video_default_seconds}s")
    try:
        task = svc.tasks.submit("video", request_meta, runner, task_id=task_id)
    except BaseException:
        await svc.billing.refund(task_id)
        raise
    request.state.task_id = task.id
    return _task_view(task)


@router.get("/v1/videos/{task_id}")
async def get_video(task_id: str, request: Request,
                    svc: Services = Depends(get_services)) -> dict:
    task = svc.tasks.get(task_id)
    if task is None or task.kind != "video":
        raise NotFound(f"video task '{task_id}' not found")
    request.state.model = task.request.get("model")
    return _task_view(task)
