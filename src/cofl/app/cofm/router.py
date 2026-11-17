"""CoF-M HTTP routes; no model construction or planning algorithms."""

from __future__ import annotations

from functools import partial
import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from ..resources import require_same_origin
from .contracts import (
    MDock, MObservation, MObservationRequest, MPair, MPointRequest, MRandomRequest,
    MScene, MSceneSummary, MTaskRequest, MTeacher,
)


def create_router(service, require_open, online_running):
    router = APIRouter(prefix="/api/v1/cofm", tags=["cof-m"])

    async def execute(request, method, *args):
        require_same_origin(request)
        require_open()
        try:
            return await run_in_threadpool(partial(method, *args))
        except (FileNotFoundError, KeyError) as error:
            raise HTTPException(404, str(error)) from error
        except (ValueError, TypeError) as error:
            raise HTTPException(400, str(error)) from error
        except (RuntimeError, OSError) as error:
            raise HTTPException(503, str(error)) from error
        except ImportError as error:
            raise HTTPException(503, "cof-m teacher requires the generation extra (SciPy): " + str(error)) from error

    @router.get("/scenes", response_model=list[MSceneSummary])
    async def scenes(request: Request):
        return await execute(request, service.scenes)

    @router.get("/scenes/{scene_id}", response_model=MScene)
    async def scene(scene_id: str, request: Request):
        return await execute(request, service.scene, scene_id)

    @router.post("/validate", response_model=MDock)
    async def validate(body: MPointRequest, request: Request):
        return await execute(request, service.validate, body.scene_id, body.point)

    @router.post("/sample", response_model=MPair)
    async def sample(body: MRandomRequest, request: Request):
        return await execute(request, service.random_pair, body.scene_id, body.seed)

    @router.post("/teacher", response_model=MTeacher)
    async def teacher(body: MTaskRequest, request: Request):
        return await execute(request, service.teacher, body)

    @router.post("/observation", response_model=MObservation)
    async def observation(body: MObservationRequest, request: Request):
        if online_running():
            raise HTTPException(409, "End the active CoFL-S session before opening cof-m Habitat")
        return await execute(request, service.observation, body)

    @router.get("/tasks/{artifact_id}/teacher")
    async def download(artifact_id: str, request: Request):
        require_same_origin(request)
        require_open()
        if not re.fullmatch(r"[0-9a-f]{32}", artifact_id):
            raise HTTPException(404, "Unknown teacher artifact")
        path = service.output / artifact_id / "teacher.npz"
        if not path.is_file():
            raise HTTPException(404, "Unknown teacher artifact")
        return FileResponse(path, media_type="application/octet-stream", filename="teacher.npz")

    return router
