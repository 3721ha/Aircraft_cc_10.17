"""Pure preflight checks for curriculum weight initialization and selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_curriculum_start(root: Path, output: Path, k: int, iteration: int,
                             iterations: int, initial: str | None = None,
                             mode: str | None = None,
                             expected_sha256: str | None = None) -> dict:
    """Resolve exactly one input; never silently ignore an explicit choice."""
    if k < 1 or iteration < 0 or iterations < 1:
        raise ValueError("curr requires K >= 1, iter0 >= 0 and n_iters >= 1")
    root, output = root.resolve(), output.resolve()
    mode = mode or ("resume" if iteration else "fresh")
    if mode not in {"fresh", "best", "resume"}:
        raise ValueError("MUMT_CURR_START_MODE must be fresh, best or resume")
    if iteration > 0 and (initial or mode != "resume"):
        raise ValueError(
            "Resume reads latest weights only. Remove MUMT_CURR_INIT and use "
            "mode=resume, or start a NEW output at iter0=0 from best.")
    if iteration == 0 and mode == "resume":
        raise ValueError("Resume requires iter0 > 0")
    key = f"curr_k{k}"
    log_path = output / "pilot_log.json"
    if iteration:
        if not log_path.is_file():
            raise FileNotFoundError(log_path)
        log = json.loads(log_path.read_text(encoding="utf-8"))
        rows = log.get(key, [])
        if (not isinstance(rows, list) or not rows
                or not isinstance(rows[-1], dict)
                or rows[-1].get("iter") != iteration - 1):
            raise ValueError("Resume iter0 does not follow the last logged iteration")
        path = output / f"{key}.pt"
        logged_sha256 = rows[-1].get("checkpoint", {}).get("sha256")
        if not logged_sha256:
            raise ValueError("Resume requires a logged latest checkpoint SHA256")
        if expected_sha256 and expected_sha256.lower() != logged_sha256.lower():
            raise ValueError("Resume expected SHA256 differs from the logged checkpoint")
        expected_sha256 = logged_sha256
    else:
        if log_path.exists() or any(output.glob("curr_k*.pt")):
            raise FileExistsError(
                "Fresh/refine output already contains a run; use a NEW output directory")
        if initial:
            path = Path(initial)
            if not path.is_absolute():
                path = root / path
        else:
            if mode == "best":
                raise ValueError("Best refinement requires MUMT_CURR_INIT")
            path = next((p for p in (output / "pilot_bc.pt",
                                    output / "dagger_latest.pt",
                                    output / "pilot_final.pt") if p.is_file()),
                        output / "pilot_bc.pt")
        if mode == "best":
            if path.name != f"{key}_best.pt":
                raise ValueError("Best refinement requires the matching curr_kK_best.pt")
            if not expected_sha256:
                raise ValueError("Best refinement requires MUMT_CURR_EXPECTED_SHA256")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Chosen curriculum checkpoint does not exist: {path}")
    digest = file_sha256(path)
    if expected_sha256 and digest != expected_sha256.lower():
        raise ValueError(f"Chosen checkpoint SHA256 mismatch: {path}")
    return {"mode": mode, "path": str(path), "sha256": digest,
            "iter_start": iteration, "weights_only": True}


def validate_best_record(log: dict, key: str, path: Path, k: int,
                         seeds: list[int], task_hash: str) -> tuple:
    record = log.get(key)
    if not isinstance(record, dict) or not path.is_file():
        raise ValueError("Resume requires both the best checkpoint and its log record")
    if file_sha256(path) != record.get("checkpoint_sha256"):
        raise ValueError("Best checkpoint SHA256 does not match its log record")
    validation = record.get("validation", {})
    if (validation.get("k") != k or validation.get("seeds") != seeds
            or validation.get("task_config_sha256") != task_hash):
        raise ValueError("Resume best record uses different K, validation seeds or task config")
    rl = validation["rl"]
    return (float(rl["success_rate"]), float(rl["mean_return"]),
            -float(rl["violations_per_step"]))
