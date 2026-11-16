from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import Field, model_validator

from ..models import StrictModel

LandmarkKind = Literal[
    "start",
    "end",
    "continuation",
    "turn",
    "closure",
    "contact",
    "crossing",
    "reference",
]
MovementRole = Literal["entry", "body", "stem", "exit", "other"]
ContactMode = Literal["must-touch", "may-touch", "must-cross-without-turn", "retrace"]
ParameterCategory = Literal[
    "global-alignment",
    "style-shared",
    "movement-geometry",
    "local-occurrence",
    "width",
]


class ProgramProvenance(StrictModel):
    construction: Literal["synthetic", "inferred", "source-backed", "reviewed", "verified"] = (
        "inferred"
    )
    trajectory: Literal["synthetic", "inferred", "source-backed", "reviewed", "verified"] = (
        "inferred"
    )
    sources: list[str] = Field(default_factory=list)
    notes: str | None = None


class ProgramLandmark(StrictModel):
    id: str = Field(min_length=1)
    kind: LandmarkKind
    x: float | None = Field(default=None, allow_inf_nan=False)
    y: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_coordinates(self) -> ProgramLandmark:
        if (self.x is None) != (self.y is None):
            raise ValueError("landmark coordinates must provide both x and y")
        return self


class ProgramMovement(StrictModel):
    id: str = Field(min_length=1)
    primitive: str = Field(min_length=1)
    direction: Literal["forward", "reverse"] = "forward"
    from_landmark: str = Field(min_length=1)
    to_landmark: str = Field(min_length=1)
    role: MovementRole = "body"
    attach: bool = True
    width_profile: str | None = None


class ProgramStroke(StrictModel):
    id: str = Field(min_length=1)
    movements: list[str] = Field(min_length=1)


class ProgramContact(StrictModel):
    id: str = Field(min_length=1)
    first_movement: str = Field(min_length=1)
    first_u: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    second_movement: str = Field(min_length=1)
    second_u: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mode: ContactMode
    tolerance: float = Field(ge=0.0, allow_inf_nan=False)


class ConstraintBase(StrictModel):
    id: str = Field(min_length=1)
    hard: bool = False
    weight: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)


class PositionConstraint(ConstraintBase):
    kind: Literal["position"]
    landmark: str = Field(min_length=1)
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)


class DistanceConstraint(ConstraintBase):
    kind: Literal["distance"]
    first_landmark: str = Field(min_length=1)
    second_landmark: str = Field(min_length=1)
    distance: float = Field(ge=0.0, allow_inf_nan=False)


class AngleConstraint(ConstraintBase):
    kind: Literal["angle"]
    first_movement: str = Field(min_length=1)
    second_movement: str = Field(min_length=1)
    angle_deg: float = Field(allow_inf_nan=False)


class TangentContinuityConstraint(ConstraintBase):
    kind: Literal["tangent-continuity"]
    first_movement: str = Field(min_length=1)
    second_movement: str = Field(min_length=1)
    tolerance_deg: float = Field(default=0.0, ge=0.0, le=180.0, allow_inf_nan=False)


class CurvatureContinuityConstraint(ConstraintBase):
    kind: Literal["curvature-continuity"]
    first_movement: str = Field(min_length=1)
    second_movement: str = Field(min_length=1)
    tolerance: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)


class LoopClosureConstraint(ConstraintBase):
    kind: Literal["loop-closure"]
    start_landmark: str = Field(min_length=1)
    end_landmark: str = Field(min_length=1)
    tolerance: float = Field(ge=0.0, allow_inf_nan=False)


class ContactConstraint(ConstraintBase):
    kind: Literal["contact"]
    contact_id: str = Field(min_length=1)


class OrderingConstraint(ConstraintBase):
    kind: Literal["ordering"]
    before_movement: str = Field(min_length=1)
    after_movement: str = Field(min_length=1)


ProgramConstraint = Annotated[
    PositionConstraint
    | DistanceConstraint
    | AngleConstraint
    | TangentContinuityConstraint
    | CurvatureContinuityConstraint
    | LoopClosureConstraint
    | ContactConstraint
    | OrderingConstraint,
    Field(discriminator="kind"),
]


_MOVEMENT_TARGET = re.compile(
    r"movement\.([A-Za-z][A-Za-z0-9_-]*)\."
    r"(length|start_angle|rotation|curvature\.(start_rate|end_rate|change\[\d+\]\.(at|rate)))"
)
_OCCURRENCE_TARGET = re.compile(
    r"occurrence\.([A-Za-z][A-Za-z0-9_-]*)\."
    r"(scale|rotation|translation\.(x|y))"
)


class ProgramParameter(StrictModel):
    id: str = Field(min_length=1)
    target: str = Field(min_length=1)
    category: ParameterCategory
    initial: float = Field(allow_inf_nan=False)
    lower: float = Field(allow_inf_nan=False)
    upper: float = Field(allow_inf_nan=False)
    prior_mean: float | None = Field(default=None, allow_inf_nan=False)
    prior_sigma: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    shared_group: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_definition(self) -> ProgramParameter:
        if not self.lower < self.upper:
            raise ValueError("parameter bounds must be increasing")
        if not self.lower <= self.initial <= self.upper:
            raise ValueError("parameter initial value must be within bounds")
        if (self.prior_mean is None) != (self.prior_sigma is None):
            raise ValueError("prior_mean and prior_sigma must be provided together")
        if self.prior_mean is not None and not self.lower <= self.prior_mean <= self.upper:
            raise ValueError("parameter prior_mean must be within bounds")
        if self.shared_group is not None and not self.shared_group.strip():
            raise ValueError("shared_group must not be blank")

        category_patterns = {
            "global-alignment": r"alignment\.(scale|rotation|translation\.(x|y))",
            "style-shared": r"style\.slant",
            "movement-geometry": _MOVEMENT_TARGET.pattern,
            "local-occurrence": _OCCURRENCE_TARGET.pattern,
            "width": r"style\.stroke_width|movement\.[A-Za-z][A-Za-z0-9_-]*\.width",
        }
        if re.fullmatch(category_patterns[self.category], self.target) is None:
            raise ValueError(f"target does not match parameter category {self.category!r}")
        return self


class DirectedGlyphProgram(StrictModel):
    format: Literal["inkgrammar-directed-glyph"] = "inkgrammar-directed-glyph"
    format_version: Literal[1] = 1
    id: str = Field(min_length=1)
    character: str = Field(min_length=1)
    variant: str = Field(min_length=1)
    landmarks: dict[str, ProgramLandmark]
    movements: dict[str, ProgramMovement]
    strokes: list[ProgramStroke] = Field(min_length=1)
    contacts: list[ProgramContact] = Field(default_factory=list)
    constraints: list[ProgramConstraint] = Field(default_factory=list)
    parameters: list[ProgramParameter] = Field(default_factory=list)
    entry_landmark: str | None = None
    exit_landmark: str | None = None
    provenance: ProgramProvenance = Field(default_factory=ProgramProvenance)

    @model_validator(mode="after")
    def validate_references(self) -> DirectedGlyphProgram:
        if not self.landmarks:
            raise ValueError("program requires at least one landmark")
        if not self.movements:
            raise ValueError("program requires at least one movement")
        for key, landmark in self.landmarks.items():
            if key != landmark.id:
                raise ValueError(f"landmark key does not match landmark id: {key}")
        for key, movement in self.movements.items():
            if key != movement.id:
                raise ValueError(f"movement key does not match movement id: {key}")
            if movement.from_landmark not in self.landmarks:
                raise ValueError(f"movement references unknown from_landmark: {movement.id}")
            if movement.to_landmark not in self.landmarks:
                raise ValueError(f"movement references unknown to_landmark: {movement.id}")

        stroke_ids = [stroke.id for stroke in self.strokes]
        if len(stroke_ids) != len(set(stroke_ids)):
            raise ValueError("stroke ids must be unique")

        movement_order: dict[str, int] = {}
        uses: dict[str, int] = {movement_id: 0 for movement_id in self.movements}
        order_index = 0
        for stroke in self.strokes:
            previous: ProgramMovement | None = None
            for movement_id in stroke.movements:
                movement = self.movements.get(movement_id)
                if movement is None:
                    raise ValueError(f"stroke references unknown movement: {movement_id}")
                if previous is not None and previous.to_landmark != movement.from_landmark:
                    raise ValueError(
                        f"movement order is discontinuous between {previous.id} and {movement.id}"
                    )
                uses[movement_id] += 1
                movement_order[movement_id] = order_index
                order_index += 1
                previous = movement
        unused = sorted(movement_id for movement_id, count in uses.items() if count == 0)
        repeated = sorted(movement_id for movement_id, count in uses.items() if count > 1)
        if unused:
            raise ValueError(f"program movements are not assigned to a stroke: {unused}")
        if repeated:
            raise ValueError(f"program movements must be used exactly once: {repeated}")

        if self.entry_landmark is not None and self.entry_landmark not in self.landmarks:
            raise ValueError("entry_landmark is not in program landmarks")
        if self.exit_landmark is not None and self.exit_landmark not in self.landmarks:
            raise ValueError("exit_landmark is not in program landmarks")
        first_movement = min(movement_order, key=movement_order.__getitem__)
        last_movement = max(movement_order, key=movement_order.__getitem__)
        if (
            self.entry_landmark is not None
            and self.movements[first_movement].from_landmark != self.entry_landmark
        ):
            raise ValueError("entry_landmark must match the first movement start")
        if (
            self.exit_landmark is not None
            and self.movements[last_movement].to_landmark != self.exit_landmark
        ):
            raise ValueError("exit_landmark must match the last movement end")

        contact_ids = [contact.id for contact in self.contacts]
        if len(contact_ids) != len(set(contact_ids)):
            raise ValueError("contact ids must be unique")
        for contact in self.contacts:
            if contact.first_movement not in self.movements:
                raise ValueError(f"contact references unknown movement: {contact.first_movement}")
            if contact.second_movement not in self.movements:
                raise ValueError(f"contact references unknown movement: {contact.second_movement}")

        constraint_ids = [constraint.id for constraint in self.constraints]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("constraint ids must be unique")
        for constraint in self.constraints:
            if isinstance(constraint, PositionConstraint) and (
                constraint.landmark not in self.landmarks
            ):
                raise ValueError(f"constraint references unknown landmark: {constraint.landmark}")
            if isinstance(constraint, DistanceConstraint) and (
                constraint.first_landmark not in self.landmarks
                or constraint.second_landmark not in self.landmarks
            ):
                raise ValueError("distance constraint references unknown landmark")
            if isinstance(
                constraint,
                (AngleConstraint, TangentContinuityConstraint, CurvatureContinuityConstraint),
            ) and (
                constraint.first_movement not in self.movements
                or constraint.second_movement not in self.movements
            ):
                raise ValueError("constraint references unknown movement")
            if isinstance(constraint, LoopClosureConstraint) and (
                constraint.start_landmark not in self.landmarks
                or constraint.end_landmark not in self.landmarks
            ):
                raise ValueError("loop-closure constraint references unknown landmark")
            if isinstance(constraint, ContactConstraint) and (
                constraint.contact_id not in contact_ids
            ):
                raise ValueError(f"constraint references unknown contact: {constraint.contact_id}")
            if isinstance(constraint, OrderingConstraint):
                if (
                    constraint.before_movement not in movement_order
                    or constraint.after_movement not in movement_order
                ):
                    raise ValueError("ordering constraint references unknown movement")
                if (
                    movement_order[constraint.before_movement]
                    >= movement_order[constraint.after_movement]
                ):
                    raise ValueError("ordering constraint must follow the declared movement order")

        parameter_ids = [parameter.id for parameter in self.parameters]
        parameter_targets = [parameter.target for parameter in self.parameters]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("parameter ids must be unique")
        if len(parameter_targets) != len(set(parameter_targets)):
            raise ValueError("parameter targets must be unique")
        shared_groups: dict[str, tuple[object, ...]] = {}
        for parameter in self.parameters:
            if parameter.target.startswith(("movement.", "occurrence.")):
                target_movement = parameter.target.split(".", 2)[1]
                if target_movement not in self.movements:
                    raise ValueError(
                        f"parameter target references unknown movement: {target_movement}"
                    )
            if parameter.shared_group is not None:
                definition = (
                    parameter.category,
                    parameter.initial,
                    parameter.lower,
                    parameter.upper,
                    parameter.prior_mean,
                    parameter.prior_sigma,
                )
                previous = shared_groups.setdefault(parameter.shared_group, definition)
                if previous != definition:
                    raise ValueError(
                        f"parameters in shared_group {parameter.shared_group!r} "
                        "must share bounds and priors"
                    )
        return self
