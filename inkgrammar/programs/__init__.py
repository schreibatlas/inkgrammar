"""InkGrammar-specific directed glyph programs and compilation."""

from .compiler import (
    CompiledGlyphProgram,
    CompiledProgramMovement,
    compile_glyph_program,
)
from .diagnostics import render_program_diagnostic
from .fitting import (
    DirectedFitConfig,
    DirectedGlyphFitResult,
    ProgramFitStageResult,
    fit_known_glyph,
)
from .models import (
    AngleConstraint,
    ContactConstraint,
    CurvatureContinuityConstraint,
    DirectedGlyphProgram,
    DistanceConstraint,
    LoopClosureConstraint,
    OrderingConstraint,
    PositionConstraint,
    ProgramContact,
    ProgramLandmark,
    ProgramMovement,
    ProgramParameter,
    ProgramProvenance,
    ProgramStroke,
    TangentContinuityConstraint,
)
from .objective import (
    ProgramFitEnergy,
    ProgramObjective,
    ProgramObjectiveWeights,
    evaluate_program_objective,
    render_program_coverage,
)
from .observation import (
    GlyphObservation,
    ObservationFrame,
    TopologyDiagnostics,
    glyph_units_to_source_px,
    prepare_glyph_observation,
    source_px_to_glyph_units,
    standard_frame,
)
from .parameters import (
    ProgramParameterBinding,
    bind_program_parameters,
    resolve_program_parameter_values,
)

__all__ = [
    "AngleConstraint",
    "CompiledGlyphProgram",
    "CompiledProgramMovement",
    "ContactConstraint",
    "CurvatureContinuityConstraint",
    "DirectedFitConfig",
    "DirectedGlyphFitResult",
    "DirectedGlyphProgram",
    "DistanceConstraint",
    "GlyphObservation",
    "LoopClosureConstraint",
    "ObservationFrame",
    "OrderingConstraint",
    "PositionConstraint",
    "ProgramContact",
    "ProgramFitEnergy",
    "ProgramFitStageResult",
    "ProgramLandmark",
    "ProgramMovement",
    "ProgramObjective",
    "ProgramObjectiveWeights",
    "ProgramParameter",
    "ProgramParameterBinding",
    "ProgramProvenance",
    "ProgramStroke",
    "TangentContinuityConstraint",
    "TopologyDiagnostics",
    "bind_program_parameters",
    "compile_glyph_program",
    "evaluate_program_objective",
    "fit_known_glyph",
    "glyph_units_to_source_px",
    "prepare_glyph_observation",
    "render_program_coverage",
    "render_program_diagnostic",
    "resolve_program_parameter_values",
    "source_px_to_glyph_units",
    "standard_frame",
]
