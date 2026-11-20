from __future__ import annotations

from string import ascii_lowercase

from inkgrammar.gestures import (
    GestureGlyphProgram,
    build_synthetic_lowercase_grammar,
    compile_gesture_program,
    expand_gesture_program,
)
from inkgrammar.programs.compiler import compile_glyph_program
from inkgrammar.programs.models import LoopClosureConstraint


def test_synthetic_programs_cover_all_lowercase_and_compile_with_legacy_parity() -> None:
    grammar = build_synthetic_lowercase_grammar()
    default_programs = [program for program in grammar.programs if program.variant == "default"]

    assert len(default_programs) == 26
    assert {program.character for program in default_programs} == set(ascii_lowercase)
    assert len({program.id for program in grammar.programs}) == len(grammar.programs)
    assert all(program.provenance.construction == "synthetic" for program in grammar.programs)

    for program in grammar.programs:
        round_trip = GestureGlyphProgram.model_validate(program.model_dump(mode="json"))
        assert round_trip.model_dump(mode="json") == program.model_dump(mode="json")

        directed = expand_gesture_program(program, grammar.templates)
        compiled_direct = compile_glyph_program(directed, grammar.primitive_templates)
        compiled_gesture = compile_gesture_program(
            program,
            grammar.templates,
            grammar.primitive_templates,
        )
        assert [item.movement_id for item in compiled_gesture.movements] == [
            item.movement_id for item in compiled_direct.movements
        ]
        assert [item.direction for item in compiled_gesture.movements] == [
            item.direction for item in compiled_direct.movements
        ]
        assert len(compiled_gesture.stroke_paths) == len(program.strokes)


def test_structural_cases_preserve_loops_crossings_detached_marks_and_ambiguity() -> None:
    grammar = build_synthetic_lowercase_grammar()
    programs = {(program.character, program.variant): program for program in grammar.programs}

    assert grammar.structural_cases["oval-plus-exit"] == ("a",)
    assert grammar.structural_cases["stem-plus-underturn"] == ("n", "r")
    assert grammar.structural_cases["double-hump"] == ("m", "w")
    assert grammar.structural_cases["ascender-and-descender-loops"] == ("b", "g", "p", "q", "y")
    assert grammar.structural_cases["crossing"] == ("x",)
    assert grammar.structural_cases["detached-dot-and-main-stroke"] == ("i", "j")
    assert grammar.structural_cases["retracing"] == ("s",)

    a = expand_gesture_program(programs[("a", "default")], grammar.templates)
    a_program = programs[("a", "default")]
    a = expand_gesture_program(a_program, grammar.templates)
    a_roles = [grammar.templates[item.gesture].role for item in a_program.gestures.values()]
    assert a_roles == ["oval", "exit"]
    assert any(isinstance(item, LoopClosureConstraint) for item in a.constraints)

    double_hump = programs[("m", "default")].gestures
    assert len(grammar.templates[next(iter(double_hump.values())).gesture].primitives) == 2

    crossing = programs[("x", "default")]
    crossing_alternative = programs[("x", "reverse-order")]
    assert crossing.contacts[0].mode == "must-cross-without-turn"
    crossing_primitives = [
        item.primitive
        for item in expand_gesture_program(crossing, grammar.templates).movements.values()
    ]
    assert crossing_primitives == ["curve-family-009", "curve-family-010"]
    reverse_crossing = expand_gesture_program(crossing_alternative, grammar.templates)
    reverse_crossing_primitives = [item.primitive for item in reverse_crossing.movements.values()]
    assert reverse_crossing_primitives == ["curve-family-010", "curve-family-009"]

    dot_default = programs[("i", "default")]
    dot_alternative = programs[("i", "dot-first")]
    assert len(dot_default.strokes) == len(dot_alternative.strokes) == 2
    assert grammar.templates["dot"].primitives[0].attach_to_previous is False
    dot_default_primitives = [
        item.primitive
        for item in expand_gesture_program(dot_default, grammar.templates).movements.values()
    ]
    assert dot_default_primitives == ["curve-family-002", "curve-family-012"]
    dot_alternative_primitives = [
        item.primitive
        for item in expand_gesture_program(dot_alternative, grammar.templates).movements.values()
    ]
    assert dot_alternative_primitives == ["curve-family-012", "curve-family-002"]
    assert dot_alternative.provenance.notes is not None

    retrace = programs[("s", "default")]
    retrace_alternative = programs[("s", "reverse-first")]
    retrace_directions = [
        item.direction
        for item in expand_gesture_program(retrace, grammar.templates).movements.values()
    ]
    assert retrace_directions == ["forward", "reverse"]
    reverse_retrace = expand_gesture_program(retrace_alternative, grammar.templates)
    reverse_retrace_directions = [item.direction for item in reverse_retrace.movements.values()]
    assert reverse_retrace_directions == ["reverse", "forward"]
    assert retrace_alternative.provenance.notes is not None
    assert grammar.structural_cases["ambiguous-traversal-or-order"] == ("i", "x", "s")
