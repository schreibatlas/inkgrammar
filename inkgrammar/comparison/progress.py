from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_STAGE_INDEX = {
    "input_validation": 1,
    "reference_hash": 2,
    "reference_cache": 3,
    "reference_mask_rows": 4,
    "reference_segmentation": 5,
    "reference_descriptors": 6,
    "sample_segmentation": 7,
    "coarse_ranking": 8,
    "reference_extraction": 9,
    "path_refinement": 10,
    "pair_report": 11,
    "batch_summary": 12,
    "completion": 13,
    "failure": 13,
}


class ProgressReporter:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        enabled: bool,
        verbose: bool,
        expected_samples: int,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.output_dir / "progress.jsonl"
        self.run_path = self.output_dir / "run.json"
        self.enabled = enabled
        self.verbose = verbose
        self.started_clock = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.completed_samples = 0
        self.expected_samples = expected_samples
        self.current_stage = "starting"
        self.status = "running"
        self.failure_details: dict[str, str] | None = None
        self.artifact_paths: dict[str, str] = {
            "output_dir": str(self.output_dir),
            "progress_jsonl": str(self.progress_path),
            "run_json": str(self.run_path),
        }
        self._write_run()

    def _elapsed(self) -> float:
        return round(time.monotonic() - self.started_clock, 3)

    def _write_run(self) -> None:
        payload = {
            "status": self.status,
            "current_stage": self.current_stage,
            "started_at": self.started_at,
            "elapsed_seconds": self._elapsed(),
            "completed_sample_count": self.completed_samples,
            "expected_sample_count": self.expected_samples,
            "failure_details": self.failure_details,
            "artifact_paths": self.artifact_paths,
        }
        temporary = self.run_path.with_name(f".{self.run_path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.run_path)

    def stage(
        self,
        name: str,
        *,
        completed: int = 1,
        total: int = 1,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.current_stage = name
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": name,
            "stage_index": _STAGE_INDEX[name],
            "completed": completed,
            "total": total,
            "elapsed_seconds": self._elapsed(),
            "details": details or {},
        }
        with self.progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
        if self.enabled:
            label = (details or {}).get("message", name.replace("_", " ").title())
            line = f"[compare-letter] {event['stage_index']:02d}/13 {label}"
            if total > 1:
                line += f" ({completed}/{total})"
            print(line, file=sys.stderr, flush=True)
        self._write_run()

    def sample_completed(self) -> None:
        self.completed_samples += 1
        self._write_run()

    def complete(self, artifacts: dict[str, str]) -> None:
        self.status = "completed"
        self.artifact_paths = artifacts
        self.stage("completion", details={"message": "Done"})

    def fail(self, error: BaseException) -> None:
        self.status = "failed"
        self.failure_details = {"type": type(error).__name__, "message": str(error)}
        details: dict[str, Any] = {"message": f"Failed: {error}"}
        diagnostic_path = getattr(error, "diagnostic_path", None)
        if diagnostic_path is not None:
            self.artifact_paths["diagnostic"] = str(diagnostic_path)
            details["diagnostic_path"] = str(diagnostic_path)
        self.stage("failure", details=details)
