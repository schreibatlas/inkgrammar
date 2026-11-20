from __future__ import annotations

from math import isfinite
from typing import Literal

from pydantic import Field, field_validator, model_validator

from ..models import StrictModel
from ..programs.models import (
    ProgramConstraint,
    ProgramContact,
    ProgramLandmark,
    ProgramParameter,
    ProgramProvenance,
)
from .evidence import GestureProvenance

GestureRole = Literal[
    "entry",
    "oval",
    "ascender-loop",
    "descender-loop",
    "stem",
    "hump",
    "underturn",
    "overturn",
    "exit",
    "connector",
    "dot",
    "cross-stroke",
    "other",
]


class GesturePrimitiveSlot(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    primitive_family: str = Field(pattern=r"^curve-family-[0-9]+$")
    direction: Literal["forward", "reverse"] = "forward"
    attach_to_previous: bool = True

    from_landmark: str | None = Field(default=None, min_length=1)
    to_landmark: str | None = Field(default=None, min_length=1)


class GestureLandmark(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    kind: Literal[
        "start",
        "end",
        "turn",
        "apex",
        "baseline-contact",
        "closure",
        "crossing",
        "reference",
    ]


class GestureConstraint(StrictModel):
    id: str = Field(min_length=1)
    kind: Literal["attachment", "loop-closure", "ordering", "tangent-continuity"]
    first: str = Field(min_length=1)
    second: str = Field(min_length=1)
    hard: bool = False
    weight: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    tolerance: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)


class GestureParameter(StrictModel):
    id: str = Field(min_length=1)
    target: str = Field(min_length=1)
    category: Literal[
        "relative-length",
        "relative-turn",
        "landmark",
        "attachment",
        "loop-proportion",
        "heading-range",
    ]
    initial: float = Field(allow_inf_nan=False)
    lower: float = Field(allow_inf_nan=False)
    upper: float = Field(allow_inf_nan=False)
    shared_group: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> GestureParameter:
        if not self.lower < self.upper:
            raise ValueError("gesture parameter bounds must be increasing")
        if not self.lower <= self.initial <= self.upper:
            raise ValueError("gesture parameter initial value must be within bounds")
        return self


class GestureTemplate(StrictModel):
    id: str = Field(min_length=1)
    role: GestureRole
    primitives: list[GesturePrimitiveSlot] = Field(min_length=1)
    landmarks: dict[str, GestureLandmark] = Field(default_factory=dict)
    constraints: list[GestureConstraint] = Field(default_factory=list)
    parameters: list[GestureParameter] = Field(default_factory=list)
    provenance: GestureProvenance = Field(default_factory=GestureProvenance)

    @model_validator(mode="after")
    def validate_references(self) -> GestureTemplate:
        primitive_ids = [primitive.id for primitive in self.primitives]
        if len(primitive_ids) != len(set(primitive_ids)):
            raise ValueError("gesture primitive slot IDs must be unique")
        for key, landmark in self.landmarks.items():
            if key != landmark.id:
                raise ValueError(f"gesture landmark key does not match id: {key}")

        landmark_ids = set(self.landmarks)
        for primitive in self.primitives:
            for landmark_id in (primitive.from_landmark, primitive.to_landmark):
                if landmark_id is not None and landmark_id not in landmark_ids:
                    raise ValueError(
                        f"gesture primitive references an unknown landmark: {primitive.id}"
                    )
        slot_ids = set(primitive_ids)
        constraint_ids = [constraint.id for constraint in self.constraints]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("gesture constraint IDs must be unique")
        for constraint in self.constraints:
            reference_ids = landmark_ids if constraint.kind == "loop-closure" else slot_ids
            if constraint.first not in reference_ids or constraint.second not in reference_ids:
                raise ValueError(f"gesture constraint references an unknown item: {constraint.id}")

        parameter_ids = [parameter.id for parameter in self.parameters]
        targets = [parameter.target for parameter in self.parameters]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("gesture parameter IDs must be unique")
        if len(targets) != len(set(targets)):
            raise ValueError("gesture parameter targets must be unique")
        return self


class WritingStyleParameters(StrictModel):
    slant_deg: float = Field(allow_inf_nan=False, ge=-45.0, le=45.0)
    body_height: float = Field(gt=0.0, allow_inf_nan=False)
    stroke_width: float = Field(gt=0.0, allow_inf_nan=False)
    entry_heading_deg: float | None = Field(default=None, allow_inf_nan=False)
    exit_heading_deg: float | None = Field(default=None, allow_inf_nan=False)


class ProgramGesture(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    gesture: str = Field(min_length=1)
    from_landmark: str = Field(min_length=1)
    to_landmark: str = Field(min_length=1)
    landmark_map: dict[str, str] = Field(default_factory=dict)
    parameter_overrides: dict[str, float] = Field(default_factory=dict)

    @field_validator("parameter_overrides")
    @classmethod
    def validate_overrides(cls, values: dict[str, float]) -> dict[str, float]:
        if any(not parameter_id for parameter_id in values):
            raise ValueError("parameter override IDs must not be empty")
        if any(not isfinite(value) for value in values.values()):
            raise ValueError("parameter overrides must be finite")
        return values


class GestureStroke(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    gestures: list[str] = Field(min_length=1)


class GestureGlyphProgram(StrictModel):
    format: Literal["inkgrammar-gesture-glyph"] = "inkgrammar-gesture-glyph"
    format_version: Literal[1] = 1
    id: str = Field(min_length=1)
    character: str = Field(min_length=1)
    variant: str = Field(min_length=1)
    gestures: dict[str, ProgramGesture]
    strokes: list[GestureStroke] = Field(min_length=1)
    landmarks: dict[str, ProgramLandmark]
    contacts: list[ProgramContact] = Field(default_factory=list)
    constraints: list[ProgramConstraint] = Field(default_factory=list)
    parameters: list[ProgramParameter] = Field(default_factory=list)
    entry_landmark: str | None = None
    exit_landmark: str | None = None
    provenance: ProgramProvenance = Field(default_factory=ProgramProvenance)

    @model_validator(mode="after")
    def validate_references(self) -> GestureGlyphProgram:
        if not self.gestures or not self.landmarks:
            raise ValueError("gesture glyph programs require gestures and landmarks")
        for key, gesture in self.gestures.items():
            if key != gesture.id:
                raise ValueError(f"gesture instance key does not match id: {key}")
            if gesture.from_landmark not in self.landmarks:
                raise ValueError(f"gesture references unknown from_landmark: {gesture.id}")
            if gesture.to_landmark not in self.landmarks:
                raise ValueError(f"gesture references unknown to_landmark: {gesture.id}")
            if any(landmark not in self.landmarks for landmark in gesture.landmark_map.values()):
                raise ValueError(
                    f"gesture landmark map references an unknown landmark: {gesture.id}"
                )
        for key, landmark in self.landmarks.items():
            if key != landmark.id:
                raise ValueError(f"landmark key does not match landmark id: {key}")

        stroke_ids = [stroke.id for stroke in self.strokes]
        if len(stroke_ids) != len(set(stroke_ids)):
            raise ValueError("gesture stroke IDs must be unique")
        uses = {gesture_id: 0 for gesture_id in self.gestures}
        ordered_gestures: list[ProgramGesture] = []
        for stroke in self.strokes:
            previous: ProgramGesture | None = None
            for gesture_id in stroke.gestures:
                gesture = self.gestures.get(gesture_id)
                if gesture is None:
                    raise ValueError(f"stroke references unknown gesture: {gesture_id}")
                if previous is not None and previous.to_landmark != gesture.from_landmark:
                    raise ValueError(
                        f"gesture order is discontinuous between {previous.id} and {gesture.id}"
                    )
                uses[gesture_id] += 1
                ordered_gestures.append(gesture)
                previous = gesture
        unused = sorted(gesture_id for gesture_id, count in uses.items() if count == 0)
        repeated = sorted(gesture_id for gesture_id, count in uses.items() if count > 1)
        if unused:
            raise ValueError(f"program gestures are not assigned to a stroke: {unused}")
        if repeated:
            raise ValueError(f"program gestures must be used exactly once: {repeated}")

        if self.entry_landmark is not None:
            if self.entry_landmark not in self.landmarks:
                raise ValueError("entry_landmark is not in program landmarks")
            if ordered_gestures[0].from_landmark != self.entry_landmark:
                raise ValueError("entry_landmark must match the first gesture start")
        if self.exit_landmark is not None:
            if self.exit_landmark not in self.landmarks:
                raise ValueError("exit_landmark is not in program landmarks")
            if ordered_gestures[-1].to_landmark != self.exit_landmark:
                raise ValueError("exit_landmark must match the last gesture end")

        parameter_ids = [parameter.id for parameter in self.parameters]
        parameter_targets = [parameter.target for parameter in self.parameters]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("program parameter IDs must be unique")
        if len(parameter_targets) != len(set(parameter_targets)):
            raise ValueError("program parameter targets must be unique")
        parameters = {parameter.id: parameter for parameter in self.parameters}
        overrides: dict[str, float] = {}
        for gesture in ordered_gestures:
            for parameter_id, value in gesture.parameter_overrides.items():
                parameter = parameters.get(parameter_id)
                if parameter is None:
                    raise ValueError(f"gesture overrides unknown program parameter: {parameter_id}")
                if not parameter.lower <= value <= parameter.upper:
                    raise ValueError(
                        f"gesture parameter override is outside bounds: {parameter_id}"
                    )
                previous = overrides.setdefault(parameter_id, value)
                if previous != value:
                    raise ValueError(f"conflicting gesture parameter overrides: {parameter_id}")
        return self
