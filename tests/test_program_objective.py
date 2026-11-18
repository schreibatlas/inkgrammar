from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image, ImageFilter
from planarcurve import Affine2D
from strokeoutline import RasterFrame, StrokeInput, render_strokes

from inkgrammar.dataset.transforms import standard_frame
from inkgrammar.fitting.models import PrimitiveTemplate
from inkgrammar.programs import write_program_fit_diagnostic
from inkgrammar.programs.compiler import compile_glyph_program
from inkgrammar.programs.fitting import DirectedFitConfig, fit_known_glyph
from inkgrammar.programs.models import DirectedGlyphProgram
from inkgrammar.programs.objective import (
    ProgramObjective,
    ProgramObjectiveWeights,
    evaluate_program_objective,
    render_program_coverage,
)
from inkgrammar.programs.observation import prepare_glyph_observation

ROOT = Path(__file__).resolve().parents[1]
PROGRAM_PATH = ROOT / "data" / "programs" / "synthetic" / "lowercase-a-v1.yaml"
PRIMITIVES_PATH = ROOT / "data" / "programs" / "synthetic" / "primitives.yaml"


def load_inputs() -> tuple[DirectedGlyphProgram, dict[str, PrimitiveTemplate]]:
    program = DirectedGlyphProgram.model_validate(yaml.safe_load(PROGRAM_PATH.read_text()))
    primitive_data = yaml.safe_load(PRIMITIVES_PATH.read_text())["primitives"]
    primitives = {
        primitive_id: PrimitiveTemplate.model_validate(value)
        for primitive_id, value in primitive_data.items()
    }
    return program, primitives


def synthetic_observation(
    program: DirectedGlyphProgram,
    primitives: dict[str, PrimitiveTemplate],
    parameter_values: dict[str, float],
    *,
    canvas_size: tuple[int, int] = (180, 170),
    origin: tuple[float, float] = (28.0, 142.0),
    x_height: float = 90.0,
    blur_radius: float = 0.0,
    noise_sigma: float = 0.0,
    seed: int = 0,
):
    width, height = canvas_size
    frame = standard_frame(origin, x_height)
    dummy_mask = np.zeros((height, width), dtype=bool)
    dummy_mask[0, 0] = True
    dummy = prepare_glyph_observation(
        Image.new("L", canvas_size, 255),
        mask=dummy_mask,
        frame=frame,
    )
    compiled = compile_glyph_program(program, primitives, parameter_values=parameter_values)
    coverage = render_program_coverage(compiled, dummy, samples_per_movement=64, supersampling=3)
    grayscale = np.clip(255.0 - coverage * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(grayscale)
    if blur_radius:
        image = image.filter(ImageFilter.GaussianBlur(blur_radius))
    if noise_sigma:
        noise = np.random.default_rng(seed).normal(0.0, noise_sigma, size=(height, width))
        noisy = np.clip(np.asarray(image, dtype=float) + noise, 0.0, 255.0).astype(np.uint8)
        image = Image.fromarray(noisy)
    if blur_radius or noise_sigma:
        return prepare_glyph_observation(image, frame=frame, mask_mode="grayscale")
    mask = coverage >= 0.18
    ink_image = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8))
    return prepare_glyph_observation(ink_image, mask=mask, frame=frame)


def test_program_objective_reports_decomposed_terms_and_reuses_observation_data() -> None:
    program, primitives = load_inputs()
    true_values = {
        "alignment-scale": 1.08,
        "alignment-rotation": 3.0,
        "alignment-translation-x": 0.04,
        "alignment-translation-y": -0.025,
        "entry-length": 0.3,
        "bowl-length": 0.9,
        "stem-angle": 4.0,
        "slant": 7.0,
        "stroke-width": 0.15,
    }
    observation = synthetic_observation(program, primitives, true_values)
    objective = ProgramObjective(
        program=program,
        primitive_templates=primitives,
        observation=observation,
        weights=ProgramObjectiveWeights(),
        samples_per_movement=40,
        supersampling=3,
    )

    true_energy, compiled = objective.evaluate(true_values)
    initial_energy = evaluate_program_objective(observation, program, primitives)
    rendered = render_program_coverage(compiled, observation)

    variable_width_compiled = replace(
        compiled,
        movements=tuple(
            replace(movement, width_profile=[(0.0, 0.08), (1.0, 0.2)])
            for movement in compiled.movements
        ),
    )
    variable_width_coverage = render_program_coverage(
        variable_width_compiled,
        observation,
        samples_per_movement=40,
    )
    assert variable_width_coverage.shape == observation.mask.shape
    assert np.isfinite(variable_width_coverage).all()

    assert rendered.shape == observation.mask.shape
    assert true_energy.total < initial_energy.total
    assert true_energy.ink >= 0.0
    assert true_energy.distance >= 0.0
    assert true_energy.centerline == 0.0
    assert true_energy.topology == 0.0
    assert true_energy.constraints >= 0.0
    assert true_energy.prior >= 0.0
    assert true_energy.width == 0.0
    assert true_energy.complexity > 0.0
    assert true_energy.complexity_breakdown.primitive_instances == len(program.movements)
    assert true_energy.complexity_breakdown.gesture_instances is None
    assert true_energy.complexity_breakdown.free_parameters == len(program.parameters)
    assert true_energy.complexity_breakdown.pen_lifts == len(program.strokes) - 1
    assert true_energy.complexity_breakdown.exception_overrides == 0
    weights = ProgramObjectiveWeights()
    assert weights.complexity == 0.001
    assert true_energy.total == pytest.approx(
        weights.ink * true_energy.ink
        + weights.distance * true_energy.distance
        + weights.centerline * true_energy.centerline
        + weights.topology * true_energy.topology
        + weights.constraints * true_energy.constraints
        + weights.prior * true_energy.prior
        + weights.width * true_energy.width
        + weights.complexity * true_energy.complexity
    )


def test_program_objective_uses_optional_skeleton_evidence() -> None:
    program, primitives = load_inputs()
    values = {
        "alignment-scale": 1.0,
        "alignment-rotation": 0.0,
        "alignment-translation-x": 0.0,
        "alignment-translation-y": 0.0,
        "entry-length": 0.28,
        "bowl-length": 0.85,
        "stem-angle": 0.0,
        "slant": 0.0,
        "stroke-width": 0.12,
    }
    observation = synthetic_observation(program, primitives, values)
    analyzed = prepare_glyph_observation(
        observation.image,
        mask=observation.mask,
        frame=observation.frame,
        analyze_skeleton=True,
    )

    energy = evaluate_program_objective(analyzed, program, primitives)

    assert energy.centerline > 0.0
    assert energy.topology >= 0.0
    assert energy.width > 0.0


def test_synthetic_lowercase_a_fit_is_bounded_topology_preserving_and_direction_cautious() -> None:
    program, primitives = load_inputs()
    true_values = {
        "alignment-scale": 1.1,
        "alignment-rotation": 4.0,
        "alignment-translation-x": 0.04,
        "alignment-translation-y": -0.03,
        "entry-length": 0.31,
        "bowl-length": 0.92,
        "stem-angle": 5.0,
        "slant": 8.0,
        "stroke-width": 0.15,
    }
    observation = synthetic_observation(program, primitives, true_values)
    initial_values = {
        "alignment-scale": 0.92,
        "alignment-rotation": -3.0,
        "alignment-translation-x": -0.04,
        "alignment-translation-y": 0.03,
        "entry-length": 0.2,
        "bowl-length": 0.68,
        "stem-angle": -6.0,
        "slant": -5.0,
        "stroke-width": 0.08,
    }
    config = DirectedFitConfig(
        max_iterations_per_stage=10,
        seeds=(7, 19),
        seed_perturbation=0.04,
        samples_per_movement=40,
        supersampling=3,
    )

    result = fit_known_glyph(
        observation,
        program,
        primitives,
        initial_values=initial_values,
        config=config,
    )

    assert result.success
    assert result.topology_preserved
    assert result.energy.total < result.initial_energy
    assert result.parameters["entry-length"] == result.parameters["exit-length"]
    assert result.route_ambiguous
    assert result.direction_ambiguous
    assert "static raster" in result.direction_uncertainty_note
    assert result.provenance.trajectory == "synthetic"
    assert result.stages
    assert result.stages[0].stage == "alignment"
    assert result.stages[-1].stage == "joint"
    for parameter in program.parameters:
        assert parameter.lower <= result.parameters[parameter.id] <= parameter.upper
    assert abs(result.parameters["stroke-width"] - true_values["stroke-width"]) < 0.03
    assert abs(result.parameters["alignment-scale"] - true_values["alignment-scale"]) < 0.25
    assert abs(result.parameters["entry-length"] - true_values["entry-length"]) < 0.15
    assert abs(result.parameters["bowl-length"] - true_values["bowl-length"]) < 0.25
    assert all(
        later.objective <= earlier.objective
        for earlier, later in zip(result.stages, result.stages[1:], strict=False)
    )


@pytest.mark.parametrize(
    ("blur_radius", "noise_sigma", "canvas_size", "origin"),
    [
        (0.0, 0.0, (180, 170), (28.0, 142.0)),
        (0.7, 3.0, (220, 190), (43.0, 157.0)),
    ],
)
def test_synthetic_observation_handles_slant_width_noise_and_crop_padding(
    blur_radius: float,
    noise_sigma: float,
    canvas_size: tuple[int, int],
    origin: tuple[float, float],
) -> None:
    program, primitives = load_inputs()
    values = {
        "alignment-scale": 1.06,
        "alignment-rotation": 2.0,
        "alignment-translation-x": 0.02,
        "alignment-translation-y": -0.01,
        "entry-length": 0.29,
        "bowl-length": 0.88,
        "stem-angle": 3.0,
        "slant": 6.0,
        "stroke-width": 0.14,
    }

    observation = synthetic_observation(
        program,
        primitives,
        values,
        canvas_size=canvas_size,
        origin=origin,
        blur_radius=blur_radius,
        noise_sigma=noise_sigma,
        seed=41,
    )

    assert observation.frame.origin_source_px == origin
    assert observation.mask.any()
    assert observation.distance_field.shape == observation.mask.shape


def test_synthetic_fit_improves_on_blurred_noisy_padded_observation() -> None:
    program, primitives = load_inputs()
    true_values = {
        "alignment-scale": 1.06,
        "alignment-rotation": 2.0,
        "alignment-translation-x": 0.02,
        "alignment-translation-y": -0.01,
        "entry-length": 0.29,
        "bowl-length": 0.88,
        "stem-angle": 3.0,
        "slant": 6.0,
        "stroke-width": 0.14,
    }
    observation = synthetic_observation(
        program,
        primitives,
        true_values,
        canvas_size=(220, 190),
        origin=(43.0, 157.0),
        blur_radius=0.7,
        noise_sigma=3.0,
        seed=41,
    )
    initial_values = {
        "alignment-scale": 0.94,
        "alignment-rotation": -1.0,
        "alignment-translation-x": -0.02,
        "alignment-translation-y": 0.02,
        "entry-length": 0.25,
        "bowl-length": 0.74,
        "stem-angle": -3.0,
        "slant": -3.0,
        "stroke-width": 0.1,
    }

    result = fit_known_glyph(
        observation,
        program,
        primitives,
        initial_values=initial_values,
        config=DirectedFitConfig(
            max_iterations_per_stage=8,
            seeds=(11,),
            samples_per_movement=40,
            supersampling=3,
        ),
    )

    assert result.success
    assert result.energy.total < result.initial_energy
    assert result.topology_preserved
    assert result.parameters["entry-length"] == result.parameters["exit-length"]
    assert abs(result.parameters["alignment-scale"] - true_values["alignment-scale"]) < 0.3
    assert abs(result.parameters["entry-length"] - true_values["entry-length"]) < 0.2
    assert abs(result.parameters["bowl-length"] - true_values["bowl-length"]) < 0.25
    assert abs(result.parameters["stroke-width"] - true_values["stroke-width"]) < 0.03



def test_program_render_parity_matches_final_strokeoutline_output() -> None:
    program, primitives = load_inputs()
    values = {
        "alignment-scale": 1.0,
        "alignment-rotation": 0.0,
        "alignment-translation-x": 0.0,
        "alignment-translation-y": 0.0,
        "entry-length": 0.28,
        "bowl-length": 0.85,
        "stem-angle": 0.0,
        "slant": 0.0,
        "stroke-width": 0.12,
    }
    observation = synthetic_observation(program, primitives, values)
    objective = ProgramObjective(
        program=program,
        primitive_templates=primitives,
        observation=observation,
        weights=ProgramObjectiveWeights(),
    )
    _, compiled = objective.evaluate(values)
    coverage = render_program_coverage(compiled, observation)

    width, height = observation.image.size
    frame = observation.frame
    glyph_to_pixel = Affine2D.from_basis(
        frame.origin_source_px,
        np.asarray(frame.x_axis_source_px) * frame.x_height_px,
        np.asarray(frame.y_axis_source_px) * frame.x_height_px,
    )
    raster_frame = RasterFrame(
        width,
        height,
        glyph_to_pixel.compose(compiled.frame_transform),
    )
    strokes = tuple(
        StrokeInput(
            movement.movement_id,
            movement.cubic_path,
            movement.width_profile if movement.width_profile is not None else 0.12,
        )
        for movement in compiled.movements
    )
    final_coverage = render_strokes(
        strokes,
        raster_frame=raster_frame,
        supersampling=3,
    ).combined_coverage

    assert final_coverage is not None
    np.testing.assert_array_equal(coverage, final_coverage)


def test_program_fit_diagnostic_writes_all_loss_and_provenance_panels(tmp_path: Path) -> None:
    program, primitives = load_inputs()
    values = {
        "alignment-scale": 1.0,
        "alignment-rotation": 0.0,
        "alignment-translation-x": 0.0,
        "alignment-translation-y": 0.0,
        "entry-length": 0.28,
        "bowl-length": 0.85,
        "stem-angle": 0.0,
        "slant": 0.0,
        "stroke-width": 0.12,
    }
    observation = synthetic_observation(program, primitives, values)
    energy, compiled = ProgramObjective(
        program=program,
        primitive_templates=primitives,
        observation=observation,
        weights=ProgramObjectiveWeights(),
    ).evaluate(values)
    output = tmp_path / "letter-a.png"
    report = write_program_fit_diagnostic(
        compiled,
        program,
        observation,
        energy,
        output,
        segmentation_alternatives=(observation.mask, ~observation.mask),
        stroke_hypotheses=(np.asarray([[20.0, 30.0], [45.0, 50.0]]),),
        gesture_boundaries={"body": (0.1, 0.8)},
        gesture_names={"body": "underturn"},
        geometry_family_ids={"body": "curve-family-017"},
        route_ambiguous=True,
        direction_ambiguous=True,
    )

    assert output.is_file()
    assert Image.open(output).size == (672, 870)
    assert len(report["panels"]) == 15
    assert report["segmentation_alternative_count"] == 2
    assert report["stroke_hypothesis_count"] == 1
    assert report["gesture_names"] == {"body": "underturn"}
    assert report["geometry_family_ids"] == {"body": "curve-family-017"}
    assert report["ambiguity"] == {"route": True, "direction": True}
    assert set(report["loss_breakdown"]) >= {
        "total",
        "ink",
        "distance",
        "centerline",
        "topology",
        "constraints",
        "prior",
        "width",
        "complexity",
    }
    assert set(report["complexity"]) == {
        "primitive_instances",
        "gesture_instances",
        "free_parameters",
        "pen_lifts",
        "exception_overrides",
    }
    assert report["provenance"] == program.provenance.model_dump(mode="json")
    assert json.loads(output.with_suffix(".json").read_text()) == report
