"""Inclusive thread selection with strict measurements and explicit provenance gaps."""

import hashlib
import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True, frozen=True)
class ThreadSelection:
    """Per-case latency evidence; selection is not a model-quality approval."""

    threads: int
    """One if every representative case meets the inclusive 1.2 ratio, else two."""

    ratios: dict[str, float]
    """Warm one-thread median divided by two-thread median, by exact case name."""

    medians_one: dict[str, float]
    """One-thread end-to-end warm medians in seconds."""

    medians_two: dict[str, float]
    """Two-thread end-to-end warm medians in seconds."""

    result_hashes: tuple[str, str]
    """SHA256 of the immutable source JSONs in one/two order."""

    provenance_notes: tuple[str, ...] = ()
    """Historical evidence gaps; empty only when complete new provenance is present."""


def select_threads(one: dict[str, float], two: dict[str, float]) -> int:
    """Choose one thread only when no measured case is more than 20% slower.

    Args:
        one:
            Positive finite one-thread latency medians in seconds, keyed by case.

        two:
            Corresponding two-thread medians with exactly the same nonempty keys.

    """
    if not one or one.keys() != two.keys():
        raise ValueError("Thread-policy cases must match and be nonempty")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (*one.values(), *two.values())
    ):
        raise ValueError("Thread-policy latencies must be positive and finite")
    return 1 if all(one[case] <= 1.2 * two[case] for case in one) else 2


def _inventory(record: dict[str, Any]) -> set[str]:
    """Infer expected cases from explicit inventory, or all calls in historical evidence.

    Args:
        record:
            Trial containing case_inventory when available and a nonempty runs list.

    """
    names = record.get("case_inventory")
    if names is None:
        names = [run["case"] for run in record["runs"]]
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Expected case inventory must contain nonempty names")
    if "case_inventory" in record and len(set(names)) != len(names):
        raise ValueError("Expected case inventory has duplicates")
    return set(names)


def _medians(
    record: dict[str, Any], threads: int, expected: set[str]
) -> dict[str, float]:
    """Extract finite warm samples with unique positive repetition IDs for every case.

    Args:
        record:
            Complete isolated benchmark with successful supervisor/root reaping evidence.

        threads:
            Required one/two execution budget.

        expected:
            Exact required representative case names; extra or missing cases fail.

    """
    if (
        record.get("status") != "completed"
        or record.get("exit_code") != 0
        or record.get("threads") != threads
        or record.get("hoast_unchanged_and_stopped") is not True
        or record.get("profile_enabled", False)
        or record.get("supervisor_outcome") != "completed"
        or record.get("process_reaped") is not True
    ):
        raise ValueError(
            "Thread policy requires completed, isolated, unprofiled, reaped trials"
        )
    for field in ("process_group_cleaned", "descendants_reaped", "artifacts_unchanged"):
        if field in record and record[field] is not True:
            raise ValueError(f"Unsuccessful trial evidence: {field}")
    version = record.get("python_version")
    if version is not None and not str(version).startswith("3.14."):
        raise ValueError("Thread policy requires native Python 3.14 evidence")
    groups: dict[str, list[float]] = {}
    repetitions: dict[str, set[int]] = {}
    for run in record["runs"]:
        name = run["case"]
        if name not in expected:
            raise ValueError(f"Unexpected trial case: {name}")
        if type(run["warm"]) is not bool:
            raise ValueError("Warm status must be boolean")
        repetition = run.get("repetition")
        if (
            type(repetition) is not int
            or repetition < 0
            or run["warm"] != (repetition > 0)
        ):
            raise ValueError("Warm status and repetition ID are inconsistent")
        seen = repetitions.setdefault(name, set())
        if repetition in seen:
            raise ValueError("Trial repetitions must be distinct within each case")
        seen.add(repetition)
        if run["warm"]:
            value = run["seconds"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("Every warm latency must be positive and finite")
            groups.setdefault(name, []).append(float(value))
    if groups.keys() != expected or any(len(values) < 2 for values in groups.values()):
        raise ValueError(
            "Thread policy requires at least two distinct warm calls per expected case"
        )
    return {name: statistics.median(values) for name, values in groups.items()}


def _provenance_notes(record: dict[str, Any], label: str) -> list[str]:
    """Describe historical gaps and validate supplied immutable artifact evidence.

    Args:
        record:
            Historical or schema-two trial; supplied bad provenance always fails.

        label:
            One/two-thread label attached to returned evidence-gap explanations.

    """
    notes = []
    for field in (
        "artifact_identity",
        "cache",
        "effective_config",
        "python_version",
        "process_group_cleaned",
        "descendants_reaped",
        "artifacts_unchanged",
        "artifact_metadata_before",
        "artifact_metadata_after",
        "case_inventory",
    ):
        if field not in record:
            notes.append(
                f"{label}: missing {field}; historical evidence is less complete"
            )
    identity = record.get("artifact_identity")
    if identity is not None:
        if not isinstance(identity, dict) or not identity:
            raise ValueError("Artifact identity must be nonempty")
        for metadata in identity.values():
            checksum, size = metadata.get("sha256"), metadata.get("bytes")
            if (
                not isinstance(checksum, str)
                or len(checksum) != 64
                or any(c not in "0123456789abcdef" for c in checksum)
                or type(size) is not int
                or size < 0
            ):
                raise ValueError("Invalid artifact hash/size identity")
    before, after = (
        record.get("artifact_metadata_before"),
        record.get("artifact_metadata_after"),
    )
    if before is not None and after is not None and before != after:
        raise ValueError("Artifact metadata changed within a trial")
    if identity is not None and before is not None and identity.keys() != before.keys():
        raise ValueError("Artifact hash and metadata inventories differ")
    return notes


def qualify_threads(
    one_path: Path,
    two_path: Path,
    *,
    expected_cases: Sequence[str] | None = None,
    require_complete_provenance: bool = False,
) -> ThreadSelection:
    """Qualify matched warm trials without rewriting historical result files.

    Missing historical artifact/group provenance is reported, not invented. Clean
    existing baselines remain usable by default. Strict mode requires complete
    recorded provenance; manually upgraded evidence must be written to a separate
    file and must not pretend a post-trial hash proves historical immutability.

    Args:
        one_path:
            Completed one-thread result JSON, read without modification.

        two_path:
            Matched completed two-thread result JSON, read without modification.

        expected_cases:
            Explicit exact representative cases; None infers each result's inventory,
            falling back to every case appearing in historical calls.

        require_complete_provenance:
            Reject provenance gaps when true, rather than accepting marked historical
            evidence. Measurement validity and supplied negative evidence always fail.

    """
    one_bytes, two_bytes = one_path.read_bytes(), two_path.read_bytes()
    one, two = json.loads(one_bytes), json.loads(two_bytes)
    for field in (
        "model",
        "backend",
        "gpu_precision",
        "model_dir",
        "inputs",
        "artifact_manifests",
        "stochastic_policy",
        "cache",
        "artifact_identity",
        "python_version",
        "native_package_versions",
    ):
        if one.get(field) != two.get(field):
            raise ValueError(f"Thread-policy trial mismatch: {field}")
    for record in (one, two):
        config = record.get("effective_config")
        record["_execution_config"] = (
            {key: value for key, value in config.items() if key != "threads"}
            if config is not None
            else None
        )
        texts: dict[str, str | None] = {}
        for run in record["runs"]:
            name, text = run["case"], run.get("text")
            if name in texts and texts[name] != text:
                raise ValueError("Thread-policy text changed within a trial")
            texts[name] = text
        record["_texts"] = texts
    if one["_execution_config"] != two["_execution_config"]:
        raise ValueError("Thread-policy effective configurations differ")
    if one["_texts"] != two["_texts"]:
        raise ValueError("Thread-policy trial texts differ")
    inventory_one, inventory_two = _inventory(one), _inventory(two)
    if inventory_one != inventory_two:
        raise ValueError("Thread-policy expected inventories differ")
    expected = inventory_one if expected_cases is None else set(expected_cases)
    if not expected or (
        expected_cases is not None and len(expected) != len(expected_cases)
    ):
        raise ValueError("Explicit expected cases must be nonempty and distinct")
    if inventory_one != expected:
        raise ValueError("Thread-policy inventory does not match expected cases")
    notes = _provenance_notes(one, "one") + _provenance_notes(two, "two")
    if (
        require_complete_provenance
        or one.get("schema_version", 1) >= 2
        or two.get("schema_version", 1) >= 2
    ) and notes:
        raise ValueError("Incomplete benchmark provenance: " + "; ".join(notes))
    one_medians, two_medians = _medians(one, 1, expected), _medians(two, 2, expected)
    return ThreadSelection(
        select_threads(one_medians, two_medians),
        {name: one_medians[name] / two_medians[name] for name in one_medians},
        one_medians,
        two_medians,
        (hashlib.sha256(one_bytes).hexdigest(), hashlib.sha256(two_bytes).hexdigest()),
        tuple(notes),
    )
