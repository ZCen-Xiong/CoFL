"""cof-m Playground service, HTTP router and request/response contracts."""

from .contracts import (
    MObstacle, MSceneSummary, MScene, MPointRequest, MDock, MRandomRequest,
    MPair, MTaskRequest, MTeacher, MObservationRequest, MObservation,
)
from .service import CofMService
from .router import create_router

__all__ = [
    "MObstacle",
    "MSceneSummary",
    "MScene",
    "MPointRequest",
    "MDock",
    "MRandomRequest",
    "MPair",
    "MTaskRequest",
    "MTeacher",
    "MObservationRequest",
    "MObservation",
    "CofMService",
    "create_router",
]
