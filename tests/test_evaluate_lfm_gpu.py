"""Offline routing evaluator checks using only embedded synthetic fixtures."""

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from finetune.generate import digest
from finetune.tooling import tool_registry
from hoast.lfm import tool_declarations
from hoast.llm import GeneratedCallError
from tools.benchmark_lfm_gpu import sha256
from tools.evaluate_lfm_gpu import (
    POLICY,
    compare_baseline,
    evaluate_rows,
    invoke,
    load_replay,
    load_workload,
    main,
    safety_summary,
    source_hashes,
    workload_prefix,
)
from tools.evaluate_lfm_gpu import (
    test_selection_ids as choose_test_ids,
)


def test_python_runner_sources_and_removed_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fingerprint the Python runner and reject the removed backend at the CLI.

    Args:
        monkeypatch:
            Isolated command-line arguments.
    """
    sources = source_hashes()
    assert "hoast/lfm_gpu/runner.py" in sources
    assert not any("native_runner" in name or "/native/" in name for name in sources)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--corpus",
            "x",
            "--manifest",
            "x",
            "--split",
            "dev",
            "--backend",
            "native",
            "--output",
            "x",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def fixture(tmp_path: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    """Construct a small independently pinned validation and test corpus.

    Args:
        tmp_path:
            Isolated temporary fixture directory.
    """
    tools = tool_declarations(tool_registry())
    schema = digest(tools)
    rows = []
    for user, target in [
        ("turn on", [{"name": "set_light", "arguments": {"on": True}}]),
        ("help", []),
    ]:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": c} for c in target],
            },
        ]
        rows.append(
            {
                "id": digest([messages, schema])[:24],
                "messages": messages,
                "tools": tools,
                "family": "fixture",
                "command_language": "en",
                "surface_language": "en",
                "entity_languages": {},
                "asr_errors": [],
                "music_state": "off",
            }
        )
    validation_rows = deepcopy(rows)
    for row in validation_rows:
        row["messages"][1]["content"] += " validation"
        row["id"] = digest([row["messages"], schema])[:24]
    for name, items in (("validation", validation_rows), ("test", rows)):
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in items)
        )
    manifest = {
        "schema_sha256": schema,
        "source_manifest": {
            "schema_sha256": schema,
            "files": {
                f"{name}.jsonl": sha256(tmp_path / f"{name}.jsonl")
                for name in ("validation", "test")
            },
            "coverage": {name: {"rows": 2} for name in ("validation", "test")},
        },
        "fusion_dev": {"count": 1, "ids": [validation_rows[1]["id"]]},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    ids = choose_test_ids(rows)
    selection = {
        "version": 1,
        "algorithm": "lfm-final-test-v1",
        "benchmark_manifest_sha256": sha256(path),
        "test_jsonl_sha256": sha256(tmp_path / "test.jsonl"),
        "schema_sha256": schema,
        "ids": ids,
        "ids_sha256": digest(ids),
        "count": len(ids),
    }
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    return tmp_path, path, rows


def test_workload_validation_and_held_out(tmp_path: Path) -> None:
    """Select only named validation IDs, retain all test rows, and reject tampering.

    Args:
        tmp_path:
            Isolated temporary fixture directory.
    """
    corpus, manifest, rows = fixture(tmp_path)
    selected, identity = load_workload(corpus, manifest, "dev")
    assert (
        selected[0]["messages"][1]["content"]
        == rows[1]["messages"][1]["content"] + " validation"
    )
    assert identity["split"] == "dev"
    assert (
        len(load_workload(corpus, manifest, "test", corpus / "selection.json")[0]) == 2
    )
    with pytest.raises(ValueError, match="selection manifest"):
        load_workload(corpus, manifest, "test")
    selected_manifest = json.loads((corpus / "selection.json").read_text())
    selected_manifest["ids"].reverse()
    (corpus / "selection.json").write_text(json.dumps(selected_manifest))
    with pytest.raises(ValueError, match="deterministic algorithm"):
        load_workload(corpus, manifest, "test", corpus / "selection.json")
    with (corpus / "validation.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_workload(corpus, manifest, "dev")


def test_replay_scoring_and_comparison(tmp_path: Path) -> None:
    """Score ordered labels, abstention, malformed outputs and baseline alignment.

    Args:
        tmp_path:
            Isolated temporary fixture directory.
    """
    corpus, manifest, rows = fixture(tmp_path)
    workload, identity = load_workload(
        corpus, manifest, "test", corpus / "selection.json"
    )
    rows = workload
    outputs = [
        {
            "id": row["id"],
            "raw": "<|tool_call_start|>[set_light(on=True)]<|tool_call_end|><|im_end|>"
            if row["messages"][-1]["tool_calls"]
            else "Please clarify.<|im_end|>",
            "tokens": 8,
        }
        for row in rows
    ]
    path = tmp_path / "replay.jsonl"
    path.write_text("".join(json.dumps(item) + "\n" for item in outputs))
    predictions = iter(load_replay(path, workload))
    records: list[dict[str, Any]] = []
    results = evaluate_rows(
        workload,
        lambda row: ((x := next(predictions))["raw"], x["tokens"]),
        records.append,
    )
    assert all(result["correct"] for result in results)
    baseline = tmp_path / "baseline.jsonl"
    baseline.write_text(
        "".join(
            json.dumps(item) + "\n"
            for item in [
                {
                    "stage": "context",
                    "workload": identity,
                    "policy": POLICY,
                    "artifacts": None,
                },
                *records,
                {"stage": "complete"},
            ]
        )
    )
    comparison = compare_baseline(baseline, identity, results, None)
    assert comparison["raw_equal"] == 2 and not comparison["artifacts_verified"]
    assert comparison["slices"]["no_tool"]["n"] == 1
    assert comparison["current_safety"]["no_tool_cases"] == 1
    with pytest.raises(ValueError, match="workload"):
        compare_baseline(baseline, dict(identity, split="dev"), results, None)
    path.write_text("".join(json.dumps(item) + "\n" for item in reversed(outputs)))
    with pytest.raises(ValueError, match="misaligned"):
        load_replay(path, workload)


def test_selection_and_safety_count_invalid_attempts(tmp_path: Path) -> None:
    """Selection ignores input order; safety counts syntactically parsed invalid calls.

    Args:
        tmp_path:
            Isolated temporary fixture directory.
    """
    _, _, rows = fixture(tmp_path)
    assert choose_test_ids(rows) == choose_test_ids(list(reversed(rows)))
    no_tool = next(row for row in rows if not row["messages"][-1]["tool_calls"])
    raw = "<|tool_call_start|>[set_light(on=1)]<|tool_call_end|><|im_end|>"
    results = evaluate_rows([no_tool], lambda _: (raw, 8), lambda _: None)
    assert results[0]["error"] is not None
    assert safety_summary(results) == {
        "no_tool_cases": 1,
        "no_tool_unparseable": 0,
        "no_tool_parsed_call_attempts": 1,
        "no_tool_invalid_call_attempts": 1,
    }


def test_ordered_targets_are_not_set_comparison(tmp_path: Path) -> None:
    """Treat reordered valid tool calls as different from ordered targets.

    Args:
        tmp_path:
            Isolated temporary fixture directory.
    """
    _, _, rows = fixture(tmp_path)
    row = rows[0]
    calls = row["messages"][-1]["tool_calls"]
    calls.append(
        {"type": "function", "function": {"name": "get_weather", "arguments": {}}}
    )
    ordered = "<|tool_call_start|>[set_light(on=True), get_weather()]<|tool_call_end|><|im_end|>"
    reversed_calls = "<|tool_call_start|>[get_weather(), set_light(on=True)]<|tool_call_end|><|im_end|>"
    assert evaluate_rows([row], lambda _: (ordered, 12), lambda _: None)[0]["correct"]
    assert not evaluate_rows([row], lambda _: (reversed_calls, 12), lambda _: None)[0][
        "correct"
    ]


def test_prefix_uses_exact_templated_full_prompts(tmp_path: Path) -> None:
    """Cache only the token prefix actually shared by every rendered prompt.

    Args:
        tmp_path:
            Isolated temporary fixture directory.
    """
    _, _, rows = fixture(tmp_path)

    class Tokenizer:
        """Minimal tokenizer tracking its rendered message inputs."""

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Render system and user content for an in-memory fixture.

            Args:
                messages:
                    Ordered system and user messages.
                **kwargs:
                    Template options, including tools.
            """
            assert kwargs["tools"] == rows[0]["tools"]
            assert [message["role"] for message in messages] == ["system", "user"]
            return messages[0]["content"] + ":" + messages[1]["content"]

        def __call__(self, prompt: str, **kwargs: Any) -> Any:
            """Encode characters as integer IDs in one array.

            Args:
                prompt:
                    Synthetic rendered prompt.
                **kwargs:
                    Tokenizer options.
            """
            assert kwargs["add_special_tokens"] is False
            return SimpleNamespace(input_ids=np.array([[ord(char) for char in prompt]]))

    assert workload_prefix(rows, Tokenizer()) == tuple(map(ord, "system:"))


def test_invoke_failure_modes() -> None:
    """Preserve generated invalid text, mark truncation, and propagate runtime errors."""

    def bad(_: str) -> Any:
        """Raise a generated-call failure for a synthetic utterance."""
        raise GeneratedCallError("bad", "<|tool_call_start|>")

    def truncated(_: str) -> Any:
        """Raise the generation adapter's explicit truncation failure."""
        raise ValueError("Generation reached output limit")

    def failure(_: str) -> Any:
        """Raise a non-generation infrastructure failure."""
        raise RuntimeError("failed")

    assert invoke(bad, "x") == ("<|tool_call_start|>", 0)
    assert invoke(truncated, "x") == ("", 128)
    with pytest.raises(RuntimeError):
        invoke(failure, "x")
