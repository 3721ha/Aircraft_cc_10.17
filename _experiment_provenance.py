"""Capture enough local provenance to identify and reproduce an experiment."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from functools import wraps
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import zipfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: str | Path, value) -> None:
    """Write JSON without leaving a truncated manifest after interruption."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp",
        dir=str(destination.parent), text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=1)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def artifact_record(path: str | Path, role: str) -> dict:
    artifact = Path(path).resolve()
    return {
        "role": role,
        "path": str(artifact),
        "exists": artifact.is_file(),
        "size_bytes": artifact.stat().st_size if artifact.is_file() else None,
        "sha256": _sha256(artifact) if artifact.is_file() else None,
    }


def capture_source_snapshot(
        repo_root: str | Path,
        output_dir: str | Path,
        entry_scripts: list[str | Path]) -> dict:
    """Freeze run sources before importing the simulator modules."""
    root = Path(repo_root).resolve()
    provenance_dir = Path(output_dir).resolve() / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    paths = {Path(path).resolve() for path in entry_scripts}
    paths.update((root / "Aircraft_mumt" / "aircraft_sim").rglob("*.py"))
    paths.add(Path(__file__).resolve())
    for directory in (root, root / "Aircraft_mumt"):
        for pattern in ("requirements*.txt", "pyproject.toml", "setup.cfg", "setup.py"):
            paths.update(directory.glob(pattern))

    source_bytes = []
    for path in sorted(paths, key=lambda item: str(item).lower()):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = (path.relative_to(root).as_posix() if path.is_relative_to(root)
                    else f"external/{path.name}")
        content = path.read_bytes()
        source_bytes.append((relative, content))
    files = [{"path": relative,
              "sha256": hashlib.sha256(content).hexdigest(),
              "size_bytes": len(content)}
             for relative, content in source_bytes]
    snapshot_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                   + f"_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    bundle_path = provenance_dir / f"{snapshot_id}_source.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for relative, content in source_bytes:
            bundle.writestr(relative, content)
    with zipfile.ZipFile(bundle_path, "r") as bundle:
        archived_files = []
        for item in files:
            content = bundle.read(item["path"])
            archived_files.append({
                "path": item["path"],
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            })
    if archived_files != files:
        raise RuntimeError(f"source snapshot verification failed: {bundle_path}")
    return {
        "files": files,
        "manifest_sha256": canonical_sha256(files),
        "snapshot_path": (bundle_path.relative_to(root).as_posix()
                          if bundle_path.is_relative_to(root) else str(bundle_path)),
        "snapshot_sha256": _sha256(bundle_path),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def changed_source_files(source_snapshot: dict) -> list[str]:
    """Return source paths that no longer match the frozen startup snapshot."""
    root = Path(__file__).resolve().parent
    changed = []
    for item in source_snapshot.get("files", []):
        relative = item["path"]
        path = (root / relative if not relative.startswith("external/")
                else None)
        if path is None or not path.is_file():
            changed.append(relative)
            continue
        if _sha256(path) != item["sha256"]:
            changed.append(relative)
    return changed


def relocate_source_snapshot(source_snapshot: dict,
                             output_dir: str | Path) -> dict:
    """Move a startup snapshot into its run directory after preflight passes."""
    root = Path(__file__).resolve().parent
    source_path = Path(source_snapshot["snapshot_path"])
    if not source_path.is_absolute():
        source_path = root / source_path
    destination_dir = Path(output_dir).resolve() / "provenance"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source_path.name
    if source_path.resolve() != destination.resolve():
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copy2(source_path, destination)
        if _sha256(destination) != source_snapshot["snapshot_sha256"]:
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"source snapshot changed while moving: {source_path}")
        source_path.unlink()
    source_snapshot["snapshot_path"] = (
        destination.relative_to(root).as_posix()
        if destination.is_relative_to(root) else str(destination))
    return source_snapshot


_ACTIVE_RUN = None


def register_active_run(run_record: dict, persist_callback) -> None:
    """Register the current invocation so its outer wrapper can finalize it."""
    global _ACTIVE_RUN
    run_record["status"] = "running"
    run_record.setdefault("running_at_utc", datetime.now(timezone.utc).isoformat())
    _ACTIVE_RUN = (run_record, persist_callback, time.monotonic())


def _finish_active_run(error=None) -> None:
    global _ACTIVE_RUN
    if _ACTIVE_RUN is None:
        return
    run_record, persist_callback, started_monotonic = _ACTIVE_RUN
    now = datetime.now(timezone.utc).isoformat()
    changed = changed_source_files(run_record.get("effective_source", {}))
    run_record["source_integrity"] = {
        "verified_at_utc": now,
        "unchanged": not changed,
        "changed_files": changed,
    }
    if changed:
        run_record["status"] = "failed"
        run_record["failed_at_utc"] = now
        run_record["failure"] = {
            "type": "SourceDriftDuringRun",
            "message": "Source files changed while this run was active.",
        }
    elif error is not None and run_record.get("status") != "completed":
        run_record["status"] = "failed"
        run_record["failed_at_utc"] = now
        run_record["failure"] = {
            "type": type(error).__name__,
            "message": str(error)[:2000],
            "traceback": "".join(traceback.format_exception(
                type(error), error, error.__traceback__))[-8000:],
        }
    else:
        run_record["status"] = "completed"
        run_record["completed_at_utc"] = now
    run_record["duration_seconds"] = round(
        time.monotonic() - started_monotonic, 3)
    persist_callback(run_record)
    _ACTIVE_RUN = None
    if changed and error is None:
        raise RuntimeError(
            "Source files changed while the run was active; results are not "
            "valid. See the run provenance record for changed files.")


def track_run_lifecycle(function):
    """Finalize a registered run on return or exception, preserving early exits."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            result = function(*args, **kwargs)
        except BaseException as error:
            try:
                _finish_active_run(error)
            except Exception as persist_error:
                error.add_note(f"Could not persist run failure status: {persist_error}")
            raise
        _finish_active_run()
        return result
    return wrapped


def capture_run_provenance(
        repo_root: str | Path,
        output_dir: str | Path,
        entry_scripts: list[str | Path],
        phase: str,
        task_config,
        algorithm_config=None,
        runtime_parameters: dict | None = None,
        inputs: list[dict] | None = None,
        source_snapshot: dict | None = None) -> dict:
    """Write a source bundle and return a self-contained run manifest."""
    root = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    provenance_dir = output / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)

    source_snapshot = source_snapshot or capture_source_snapshot(
        root, output, entry_scripts)
    changed = changed_source_files(source_snapshot)
    if changed:
        raise RuntimeError(
            "Source changed after the startup snapshot and before training: "
            + ", ".join(changed))
    run_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
              + f"_{os.getpid()}_{uuid.uuid4().hex[:8]}")

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args], check=True, capture_output=True,
                text=True, encoding="utf-8", errors="replace")
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = git("status", "--short", "--untracked-files=all")
    config = _json_value(task_config)
    algorithm = _json_value(algorithm_config)
    return {
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "argv": list(sys.argv),
        "working_directory": os.getcwd(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "python_path": list(sys.path),
        "platform": platform.platform(),
        "execution_environment": {
            key: os.environ[key]
            for key in sorted(os.environ)
            if key.startswith("MUMT_") or key in {
                "PYTHONHASHSEED", "PYTHONPATH", "CUDA_VISIBLE_DEVICES",
                "CUDA_LAUNCH_BLOCKING", "CUBLAS_WORKSPACE_CONFIG",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            }
        },
        "git": {
            "head": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(status),
            "status_short": status.splitlines() if status else [],
        },
        "runtime_parameters": _json_value(runtime_parameters or {}),
        "task_config": config,
        "task_config_sha256": canonical_sha256(config),
        "algorithm_config": algorithm,
        "algorithm_config_sha256": canonical_sha256(algorithm),
        "dependencies": sorted(
            ({"name": dist.metadata.get("Name", ""), "version": dist.version}
             for dist in importlib.metadata.distributions()),
            key=lambda item: item["name"].lower()),
        "effective_source": {
            **source_snapshot,
        },
        "inputs": list(inputs or []),
        "outputs": [],
    }


def add_input(record: dict, path: str | Path, role: str) -> None:
    record.setdefault("inputs", []).append(artifact_record(path, role))


def add_output(record: dict, path: str | Path, role: str) -> None:
    outputs = record.setdefault("outputs", [])
    outputs[:] = [item for item in outputs if item.get("role") != role]
    outputs.append(artifact_record(path, role))
