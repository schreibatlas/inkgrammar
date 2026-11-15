from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "inkgrammar"


def _tree(relative_path: str) -> ast.Module:
    source = PACKAGE / relative_path
    return ast.parse(source.read_text(encoding="utf-8"), filename=str(source))


def _module_definitions(relative_path: str) -> set[str]:
    return {
        node.name
        for node in _tree(relative_path).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _imports_from(relative_path: str, module: str) -> set[str]:
    return {
        alias.name
        for node in ast.walk(_tree(relative_path))
        if isinstance(node, ast.ImportFrom) and node.module == module
        for alias in node.names
    }


def _imports_inkgrammar(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".", maxsplit=1)[0] == "inkgrammar" for alias in node.names):
                return True
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".", maxsplit=1)[0] == "inkgrammar"
        ):
            return True
        return False


def test_compatibility_facade_modules_stay_deleted() -> None:
    removed_modules = (
        "geometry.py",
        "comparison/mask.py",
        "fitting/skeleton_graph.py",
        "fitting/trajectory_parameterization.py",
        "skeleton_walk.py",
    )
    assert all(not (PACKAGE / relative_path).exists() for relative_path in removed_modules)


def test_generic_curve_algorithms_remain_in_planarcurve() -> None:
    assert "fit_similarity_transform" not in _module_definitions("comparison/path.py")
    assert "fit_similarity_transform" in _imports_from(
        "comparison/path.py", "planarcurve.transforms"
    )

    assert "covering_walk" in _imports_from("comparison/path.py", "skeletonpaths")
    assert "analyze_mask" in _imports_from("comparison/skeleton.py", "skeletonpaths")
    assert "OutlineConfig" not in _module_definitions("outline.py")

    spline_forwarders = {
        "bspline_from_geometry",
        "control_roughness_matrix",
        "spline_design_matrix",
        "spline_to_cubic_segments",
        "validate_bspline_geometry",
    }
    assert not spline_forwarders.intersection(_module_definitions("fitting/spline.py"))
    assert "sample_curve_by_arclength" in _imports_from("fitting/spline.py", "planarcurve.spline")
    assert "to_planar_bspline" in _module_definitions("fitting/planarcurve_adapters.py")
    assert "bspline_geometry_to_cubic_segments" in _module_definitions(
        "fitting/planarcurve_adapters.py"
    )


def test_external_geometry_libraries_do_not_depend_on_inkgrammar() -> None:
    for package_name in ("planarcurve", "rasterregions", "skeletonpaths", "strokeoutline"):
        spec = importlib.util.find_spec(package_name)
        assert spec is not None and spec.origin is not None
        package_root = Path(spec.origin).parent
        offenders = [path for path in package_root.rglob("*.py") if _imports_inkgrammar(path)]
        assert not offenders, f"{package_name} imports InkGrammar: {offenders}"
