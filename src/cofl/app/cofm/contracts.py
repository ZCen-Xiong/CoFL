"""Studio HTTP request/response contracts; these are not neural models."""

from __future__ import annotations

from typing import Literal
from pydantic import Field

from ..contracts import Contract, Finite, Point2, ResourceId


class MObstacle(Contract):
    kind: Literal["circle", "box"]
    center: Point2
    radius: float = 0
    size: Point2 = (0, 0)


class MSceneSummary(Contract):
    id: str
    label: str


class MScene(Contract):
    id: str
    label: str
    length_m: float
    width_m: float
    height_m: float
    resolution_m: float
    obstacles: list[MObstacle]
    docking: dict[str, float | list[float] | str]
    habitat_available: bool
    map_image: str | None = None
    source_kind: Literal['procedural','matterport'] = 'procedural'


class MPointRequest(Contract):
    scene_id: ResourceId
    point: Point2


class MDock(Contract):
    point: Point2
    valid: bool
    clearance_m: float
    reason: str


class MRandomRequest(Contract):
    scene_id: ResourceId
    seed: int = Field(default=0, ge=0, le=2**32-1)


class MPair(Contract):
    start: MDock
    goal: MDock


class MTaskRequest(Contract):
    scene_id: ResourceId
    start: Point2
    goal: Point2
    narrow_only: bool | None = None
    max_width_m: Finite = Field(default=1.1,gt=0,le=3)
    min_narrow_length_m: Finite = Field(default=.3,gt=0,le=5)


class MTeacher(Contract):
    source: Literal["geometry_teacher"] = "geometry_teacher"
    reachable: bool
    resolution_m: float
    path: list[tuple[float, float]]
    arrows: list[tuple[float, float, float, float]]
    min_path_clearance_m: float | None
    artifact_id: str
    message: str
    narrow: dict | None = None


class MObservationRequest(Contract):
    scene_id: ResourceId
    point: Point2


class MObservation(Contract):
    image: str
    rgb_image: str | None = None
    position: tuple[float, float, float]
    heading_deg: float = 0
    valid_fraction: float
    far_m: float
    source: Literal["habitat_depth"] = "habitat_depth"


