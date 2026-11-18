# Gesture grammar workflows

The grammar workflow is an additive, versioned path for reusable geometry families,
semantic gesture templates, and composed lowercase glyph programs. It keeps the
existing `DirectedGlyphProgram` compiler and `StrokeOutline` renderer as the execution
path. It does not replace existing fitting projects or infer reviewed stroke intent
from a raster image.

## Project file

Pass a YAML project file with `format: inkgrammar-gesture-project` and
`format_version: 1`. All configured input files must be project-relative and resolve
beneath the project directory. Absolute input paths and symlinks that escape that
directory are rejected. `output_dir` is required and may be relative to the project
or absolute.

```yaml
format: inkgrammar-gesture-project
format_version: 1
dataset: synthetic-lowercase
style: sample-style
geometry_learning:
  allowed_change_counts: [0, 1, 2]
  held_out_fraction: 0.2
validation:
  letters: abcdefghijklmnopqrstuvwxyz
files:
  geometry_occurrences: geometry.yaml
  gesture_templates: gesture-templates.yaml
  gesture_occurrences: gesture-occurrences.yaml
  gesture_programs: gesture-programs.yaml
  primitive_templates: primitive-templates.yaml
  # movement_annotations: reviewed-movements.yaml
output_dir: runs/synthetic-lowercase
```

Each input document is YAML with its own `format` and `format_version: 1`:

- `inkgrammar-geometry-occurrences`: a `families` mapping from family ID to directed
  occurrence records. Each record has an ID, source observation ID, ordered points,
  split (`training` or `held-out`), and optional evidence IDs.
- `inkgrammar-gesture-templates`: a `templates` list of `GestureTemplate` records.
- `inkgrammar-gesture-occurrences`: an `occurrences` list of `GestureOccurrence`
  records, including `OccurrenceEvidence` and curves keyed by primitive slot ID.
- `inkgrammar-gesture-programs`: a `programs` list of versioned
  `GestureGlyphProgram` records.
- `inkgrammar-primitive-templates`: a `primitives` mapping from geometry family ID
  to an existing `PrimitiveTemplate`. These seeded templates remain the compiler's
  geometry input during this migration stage.
- `inkgrammar-movement-annotations`: optional reviewed movement annotations.

## Commands

Run commands through the installed CLI or `python -m inkgrammar`:

```text
inkgrammar grammar validate PROJECT.yaml
inkgrammar grammar materialize PROJECT.yaml
inkgrammar grammar learn-primitives PROJECT.yaml
inkgrammar grammar fit-gestures PROJECT.yaml
inkgrammar grammar fit-glyphs PROJECT.yaml
inkgrammar grammar control-sheet PROJECT.yaml
inkgrammar grammar inspect-letter PROJECT.yaml --letter a
```

`fit-gestures` consumes the geometry-family artifact produced by `learn-primitives`.
`fit-glyphs` expands each semantic gesture program into the existing directed program
and compiles it with the configured primitive templates. `control-sheet` renders the
configured programs through `StrokeOutline`. `inspect-letter` writes a deterministic
report containing the matching program variants, template roles, geometry-family IDs,
and provenance.

Artifacts are written below the configured `output_dir`, separated by command:

- `materialize/` contains project-relative input snapshots and a SHA-256 manifest. Its
  `project.yaml` points to the snapshots and can be loaded independently.
- `learn-primitives/geometry-families.json` contains fitted families and separate
  training and held-out metrics.
- `fit-gestures/gesture-dictionary.json` contains bounded ranked matches, evidence,
  ambiguity, reuse, and fit metrics.
- `fit-glyphs/glyph-programs.json` contains expanded directed programs and compile
  summaries.
- `validate/project.json`, `control-sheet/`, and `inspect-letter/` contain the
  corresponding validation, visual, and focused inspection artifacts.

Repeated runs over unchanged project inputs produce the same JSON and image artifacts.
The materialization manifest records paths relative to the project and snapshot, not
machine-specific absolute input paths.

## Ownership and evidence

Geometry-family fitting uses the PlanarCurve fitting APIs through the InkGrammar
adapter. Raster segmentation, skeleton path ranking, and stroke outlining remain owned
by RasterRegions, SkeletonPaths, and StrokeOutline. Input ownership is bounded to the
project directory. Static-image candidates remain inferred evidence; the CLI does not
upgrade them to reviewed movement labels or verified handwriting.

The checked-in project data does not provide an annotated real-image alphabet. A
control sheet is a deterministic rendering of configured programs, not a real-image
accuracy result. Real-image training and held-out claims require accessible source data
and reviewed splits.

## Fitting-generation migration gates

The grammar workflow introduces no `family_fit_v5` stack and does not remove or silently
redirect existing fitting generations. The older objective-v3 path remains frozen, and
the current family-fit dispatch remains available while the additive grammar path is
proven. CLI artifacts intentionally report grammar concepts and fit metrics, not
optimizer generation names.

Do not deprecate or delete an existing generation until all applicable gates are
reviewed and recorded:

1. Synthetic lowercase `a` through `z` and the structural cases have deterministic
   program, compiler, and ambiguity regressions.
2. Representative training and held-out family and gesture metrics are compared against
   the retained path, with runtime budgets recorded separately.
3. Directed movement order, direction, landmarks, parameter bounds, pen lifts, and
   exception overrides have explicit migration parity evidence.
4. StrokeOutline coverage and reviewed control-sheet outputs have rendering parity
   checks.
5. Existing project inputs and provenance can be migrated without losing annotations,
   evidence IDs, held-out splits, or ambiguity.
6. Any remaining behavior differences are documented and explicitly reviewed before
   old entry points are retired.
