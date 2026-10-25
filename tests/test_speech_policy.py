"""Thread-selection boundaries and benchmark-evidence validation."""

import json
from pathlib import Path
from typing import Any

import pytest

from hoast.speech_policy import qualify_threads, select_threads


def test_inclusive_twenty_percent_rule() -> None:
    """Accept equality and reject a slow case even when others improve."""
    assert select_threads({"en": 1.2, "zh": 0.8}, {"en": 1, "zh": 1}) == 1
    assert select_threads({"en": 1.200001, "zh": 0.8}, {"en": 1, "zh": 1}) == 2
    assert select_threads({"short": 0.4}, {"short": 1}) == 1


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_latency(value: float) -> None:
    """Reject missing or nonfinite/nonpositive evidence rather than defaulting.

    Args:
        value:
            Invalid warm-median latency in seconds.

    """
    with pytest.raises(ValueError):
        select_threads({"en": value}, {"en": 1})
    with pytest.raises(ValueError):
        select_threads({}, {})
    with pytest.raises(ValueError):
        select_threads({"en": 1}, {"zh": 1})


def test_matched_trial_provenance(tmp_path: Path) -> None:
    """Use warm medians only and reject profile/service/model contamination.

    Args:
        tmp_path:
            Empty temporary directory for tiny embedded benchmark JSON fixtures.

    """
    one, two = tmp_path / "one.json", tmp_path / "two.json"
    records = []
    for threads, seconds, path in ((1, 1.2, one), (2, 1.0, two)):
        record = {
            "model": "melo",
            "backend": "cpu",
            "threads": threads,
            "status": "completed",
            "exit_code": 0,
            "supervisor_outcome": "completed",
            "process_reaped": True,
            "python_version": "3.14.7 native fixture",
            "hoast_unchanged_and_stopped": True,
            "profile_enabled": False,
            "runs": [
                {
                    "case": "en",
                    "text": "Hello.",
                    "warm": False,
                    "seconds": 999,
                    "repetition": 0,
                },
                {
                    "case": "en",
                    "text": "Hello.",
                    "warm": True,
                    "seconds": seconds,
                    "repetition": 1,
                },
                {
                    "case": "en",
                    "text": "Hello.",
                    "warm": True,
                    "seconds": seconds,
                    "repetition": 2,
                },
            ],
        }
        path.write_text(json.dumps(record))
        records.append(record)
    policy = qualify_threads(one, two)
    assert policy.threads == 1 and policy.ratios == {"en": 1.2}
    assert len(policy.result_hashes[0]) == 64
    assert policy.provenance_notes
    for field, value in (
        ("profile_enabled", True),
        ("hoast_unchanged_and_stopped", False),
        ("backend", "GPU"),
    ):
        invalid = {**records[0], field: value}
        one.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            qualify_threads(one, two)


def _record(threads: int) -> dict[str, Any]:
    """Create a complete schema-two embedded trial with tiny synthetic identity.

    Args:
        threads:
            One or two execution threads; every warm call takes one second.

    """
    metadata = {"artifact": {"bytes": 3, "mtime_ns": 1, "ctime_ns": 1}}
    return {
        "schema_version": 2,
        "model": "melo",
        "backend": "cpu",
        "threads": threads,
        "status": "completed",
        "exit_code": 0,
        "supervisor_outcome": "completed",
        "process_reaped": True,
        "process_group_cleaned": True,
        "descendants_reaped": True,
        "hoast_unchanged_and_stopped": True,
        "profile_enabled": False,
        "python_version": "3.14.7 native fixture",
        "cache": "fixture-cache",
        "effective_config": {"threads": threads, "seed": 1234},
        "artifact_identity": {"artifact": {"bytes": 3, "sha256": "a" * 64}},
        "artifact_metadata_before": metadata,
        "artifact_metadata_after": metadata,
        "artifacts_unchanged": True,
        "case_inventory": ["en", "zh"],
        "runs": [
            {
                "case": name,
                "text": name,
                "warm": repetition > 0,
                "seconds": 1.0,
                "repetition": repetition,
            }
            for name in ("en", "zh")
            for repetition in range(3)
        ],
    }


def _paths(
    tmp_path: Path, one: dict[str, Any], two: dict[str, Any]
) -> tuple[Path, Path]:
    """Write isolated embedded evidence without real artifacts or model execution.

    Args:
        tmp_path:
            Empty pytest scratch directory.

        one:
            One-thread fixture record.

        two:
            Two-thread fixture record.

    """
    first, second = tmp_path / "one.json", tmp_path / "two.json"
    first.write_text(json.dumps(one))
    second.write_text(json.dumps(two))
    return first, second


def test_complete_provenance_and_no_overwrite(tmp_path: Path) -> None:
    """Strict evidence passes and source result bytes remain untouched.

    Args:
        tmp_path:
            Isolated fixture result directory.

    """
    paths = _paths(tmp_path, _record(1), _record(2))
    before = [path.read_bytes() for path in paths]
    selected = qualify_threads(
        *paths, expected_cases=["en", "zh"], require_complete_provenance=True
    )
    assert selected.threads == 1 and not selected.provenance_notes
    assert [path.read_bytes() for path in paths] == before


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), True])
def test_every_warm_sample_is_validated(tmp_path: Path, seconds: float) -> None:
    """Reject individual invalid calls even when other samples produce a valid median.

    Args:
        tmp_path:
            Isolated fixture result directory.

        seconds:
            Invalid warm latency inserted alongside valid calls.

    """
    one, two = _record(1), _record(2)
    one["runs"][1]["seconds"] = seconds
    with pytest.raises(ValueError):
        qualify_threads(*_paths(tmp_path, one, two))


@pytest.mark.parametrize(
    "field,value",
    [
        ("supervisor_outcome", "timeout"),
        ("process_reaped", False),
        ("process_group_cleaned", False),
        ("descendants_reaped", False),
        ("artifacts_unchanged", False),
        ("python_version", "3.12.0"),
    ],
)
def test_bad_supervisor_evidence(tmp_path: Path, field: str, value: object) -> None:
    """Reject explicit failed cleanup, mutation or non-native evidence.

    Args:
        tmp_path:
            Isolated result directory.

        field:
            Supervisor/provenance field to invalidate.

        value:
            Invalid replacement value, applied to both matched records.

    """
    one, two = _record(1), _record(2)
    one[field] = two[field] = value
    with pytest.raises(ValueError):
        qualify_threads(*_paths(tmp_path, one, two))


def test_distinct_repetitions_and_expected_coverage(tmp_path: Path) -> None:
    """Reject duplicate warm repetitions and absent expected representative cases.

    Args:
        tmp_path:
            Isolated result directory.

    """
    one, two = _record(1), _record(2)
    one["runs"][2]["repetition"] = 1
    with pytest.raises(ValueError, match="distinct"):
        qualify_threads(*_paths(tmp_path, one, two))
    paths = _paths(tmp_path, _record(1), _record(2))
    with pytest.raises(ValueError, match="expected"):
        qualify_threads(*paths, expected_cases=["en", "zh", "mixed"])
    one, two = _record(1), _record(2)
    one["runs"] = [run for run in one["runs"] if run["case"] != "zh"]
    with pytest.raises(ValueError):
        qualify_threads(*_paths(tmp_path, one, two))


def test_legacy_provenance_is_explicit_not_fabricated(tmp_path: Path) -> None:
    """Keep clean historical paired baselines reusable while strict mode rejects gaps.

    Args:
        tmp_path:
            Isolated historical fixture results.

    """
    records = [_record(1), _record(2)]
    for record in records:
        for field in (
            "schema_version",
            "artifact_identity",
            "artifact_metadata_before",
            "artifact_metadata_after",
            "artifacts_unchanged",
            "cache",
            "effective_config",
            "process_group_cleaned",
            "descendants_reaped",
            "case_inventory",
        ):
            record.pop(field)
    paths = _paths(tmp_path, *records)
    assert qualify_threads(*paths).provenance_notes
    with pytest.raises(ValueError, match="Incomplete"):
        qualify_threads(*paths, require_complete_provenance=True)


def test_changed_artifact_and_effective_config_fail(tmp_path: Path) -> None:
    """Require matched contents/settings rather than merely equal case labels.

    Args:
        tmp_path:
            Isolated fixture result directory.

    """
    one, two = _record(1), _record(2)
    two["artifact_identity"]["artifact"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="artifact_identity"):
        qualify_threads(*_paths(tmp_path, one, two))
    one, two = _record(1), _record(2)
    two["effective_config"]["seed"] = 0
    with pytest.raises(ValueError, match="configurations"):
        qualify_threads(*_paths(tmp_path, one, two))
