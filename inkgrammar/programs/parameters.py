from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Literal

from .models import DirectedGlyphProgram, ParameterCategory, ProgramParameter


@dataclass(frozen=True)
class ProgramParameterBinding:
    parameter_id: str
    category: ParameterCategory
    target: str
    root: Literal["alignment", "style", "movement", "occurrence"]
    owner_id: str | None
    path: str
    initial: float
    lower: float
    upper: float
    prior_mean: float | None
    prior_sigma: float | None
    shared_group: str | None


def _binding(parameter: ProgramParameter) -> ProgramParameterBinding:
    root, remainder = parameter.target.split(".", 1)
    owner_id = None
    path = remainder
    if root in {"movement", "occurrence"}:
        owner_id, path = remainder.split(".", 1)
    return ProgramParameterBinding(
        parameter_id=parameter.id,
        category=parameter.category,
        target=parameter.target,
        root=root,
        owner_id=owner_id,
        path=path,
        initial=parameter.initial,
        lower=parameter.lower,
        upper=parameter.upper,
        prior_mean=parameter.prior_mean,
        prior_sigma=parameter.prior_sigma,
        shared_group=parameter.shared_group,
    )


def bind_program_parameters(
    program: DirectedGlyphProgram,
    *,
    category: ParameterCategory | None = None,
) -> tuple[ProgramParameterBinding, ...]:
    """Return validated optimizer bindings in declared parameter order."""
    selected = (
        parameter
        for parameter in program.parameters
        if category is None or parameter.category == category
    )
    return tuple(_binding(parameter) for parameter in selected)


def resolve_program_parameter_values(
    program: DirectedGlyphProgram,
    overrides: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Resolve bounded values and make declared shared groups one scalar."""
    definitions = {parameter.id: parameter for parameter in program.parameters}
    values = {parameter.id: parameter.initial for parameter in program.parameters}
    overrides = overrides or {}
    unknown = set(overrides) - set(definitions)
    if unknown:
        raise ValueError(f"unknown program parameter ids: {sorted(unknown)}")
    for parameter_id, raw_value in overrides.items():
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"parameter value must be a finite number: {parameter_id}")
        value = float(raw_value)
        parameter = definitions[parameter_id]
        if not isfinite(value):
            raise ValueError(f"parameter value must be a finite number: {parameter_id}")
        if not parameter.lower <= value <= parameter.upper:
            raise ValueError(f"parameter value is outside bounds: {parameter_id}")
        values[parameter_id] = value

    groups: dict[str, list[ProgramParameter]] = {}
    for parameter in program.parameters:
        if parameter.shared_group is not None:
            groups.setdefault(parameter.shared_group, []).append(parameter)
    for group_id, parameters in groups.items():
        explicitly_set = [
            values[parameter.id] for parameter in parameters if parameter.id in overrides
        ]
        if explicitly_set and any(value != explicitly_set[0] for value in explicitly_set[1:]):
            raise ValueError(f"shared parameter group has conflicting overrides: {group_id}")
        shared_value = explicitly_set[0] if explicitly_set else values[parameters[0].id]
        for parameter in parameters:
            values[parameter.id] = shared_value
    return values
