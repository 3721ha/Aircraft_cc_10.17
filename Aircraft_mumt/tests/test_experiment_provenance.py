import hashlib
from pathlib import Path
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _experiment_provenance import (  # noqa: E402
    capture_source_snapshot,
    changed_source_files,
    relocate_source_snapshot,
    register_active_run,
    track_run_lifecycle,
)


def test_source_snapshot_archive_matches_manifest():
    with tempfile.TemporaryDirectory(dir=ROOT / "runs") as temp_dir:
        startup_dir = Path(temp_dir) / "startup"
        snapshot = capture_source_snapshot(
            ROOT, startup_dir, [ROOT / "_p3_pilot.py"])
        startup_archive = Path(snapshot["snapshot_path"])
        if not startup_archive.is_absolute():
            startup_archive = ROOT / startup_archive
        snapshot = relocate_source_snapshot(
            snapshot, Path(temp_dir) / "run")
        archive_path = Path(snapshot["snapshot_path"])
        if not archive_path.is_absolute():
            archive_path = ROOT / archive_path
        assert archive_path.is_file()
        assert not startup_archive.exists()
        assert changed_source_files(snapshot) == []

        with zipfile.ZipFile(archive_path) as archive:
            for item in snapshot["files"]:
                content = archive.read(item["path"])
                assert hashlib.sha256(content).hexdigest() == item["sha256"]
                assert len(content) == item["size_bytes"]


def test_source_drift_is_detected():
    snapshot = {"files": [{
        "path": "_experiment_provenance.py",
        "sha256": "not-the-current-hash",
    }]}
    assert changed_source_files(snapshot) == ["_experiment_provenance.py"]


def test_run_lifecycle_records_completed_and_failed_statuses():
    completed = {"effective_source": {"files": []}}
    completed_persisted = []
    register_active_run(completed, completed_persisted.append)

    @track_run_lifecycle
    def succeed():
        return "ok"

    assert succeed() == "ok"
    assert completed["status"] == "completed"
    assert completed["completed_at_utc"]
    assert completed["source_integrity"]["unchanged"]
    assert completed_persisted == [completed]

    failed = {"effective_source": {"files": []}}
    failed_persisted = []
    register_active_run(failed, failed_persisted.append)

    @track_run_lifecycle
    def fail():
        raise ValueError("expected smoke failure")

    try:
        fail()
    except ValueError as error:
        assert str(error) == "expected smoke failure"
    else:
        raise AssertionError("the wrapped failure must be re-raised")
    assert failed["status"] == "failed"
    assert failed["failure"]["type"] == "ValueError"
    assert failed["failed_at_utc"]
    assert failed_persisted == [failed]


def test_source_drift_fails_the_run():
    run_record = {"effective_source": {"files": [{
        "path": "_experiment_provenance.py",
        "sha256": "not-the-current-hash",
    }]}}
    persisted = []
    register_active_run(run_record, persisted.append)

    @track_run_lifecycle
    def finish_after_source_drift():
        return "computed"

    try:
        finish_after_source_drift()
    except RuntimeError as error:
        assert "Source files changed" in str(error)
    else:
        raise AssertionError("source drift must invalidate the run")
    assert run_record["status"] == "failed"
    assert run_record["failure"]["type"] == "SourceDriftDuringRun"
    assert run_record["source_integrity"]["changed_files"] == [
        "_experiment_provenance.py"]
    assert persisted == [run_record]
