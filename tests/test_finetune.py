"""Offline dataset, grounding, split-isolation and official-template compatibility tests."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
from jinja2 import TemplateError
from pydantic import ValidationError

from finetune import TEMPLATE_PATH
from finetune.generate import (
    ROOT,
    Dataset,
    Family,
    TemplateFile,
    Variant,
    VariantFile,
    build_dataset,
    normalized_input,
    write_dataset,
)
from finetune.render import render_example
from finetune.tooling import tool_registry
from hoast.config import MusicConfig, SwitchConfig, WeatherConfig
from hoast.lfm import parse_response
from hoast.lights import LightClient
from hoast.llm import ToolCall, ToolRegistry
from hoast.music import MusicClient
from hoast.prompts import SYSTEM_PROMPT, build_system_prompt
from hoast.weather import WeatherClient


def family(**changes: Any) -> Family:
    """Construct a minimal valid family with explicit test overrides.

    Args:
        **changes:
            Source fields replacing the single resume template defaults.

    """
    return Family.model_validate(
        {
            "id": "resume.one",
            "group": "resume.one",
            "split": "train",
            "language": "en",
            "utterances": ["{command}"],
            "slots": {"command": "commands"},
            "calls": [{"name": "resume_music"}],
            **changes,
        }
    )


@pytest.fixture(scope="module")
def default_dataset() -> Dataset:
    """Expand tracked sources using the reproducible CLI defaults without model assets."""
    return build_dataset(
        VariantFile.model_validate_json((ROOT / "variants.json").read_text()),
        TemplateFile.model_validate_json((ROOT / "templates.json").read_text()),
    )


def test_production_schema_parity_and_no_execution() -> None:
    """Offline declarations equal bound production schemas and cannot execute handlers."""
    declared = tool_registry()
    bound = ToolRegistry(
        [
            WeatherClient(WeatherConfig(0, 0)).tool(),
            *MusicClient(MusicConfig(), "unused-test-token").tools(),
            LightClient(SwitchConfig("192.0.2.20")).tool(),
        ]
    )
    assert declared.schemas() == bound.schemas()
    with pytest.raises(RuntimeError, match="cannot execute"):
        declared.dispatch([ToolCall("resume_music", {})])


def test_default_coverage_and_split_isolation(default_dataset: Dataset) -> None:
    """Cover all tools and code-switching without leaking groups or normalized requests.

    Args:
        default_dataset:
            In-memory dataset compiled from the tracked command banks/families.

    """
    names = {schema["function"]["name"] for schema in tool_registry().schemas()}
    inputs: dict[str, set[str]] = {}
    groups: dict[str, set[str]] = {}
    for split, rows in default_dataset.rows.items():
        assert rows
        assert set(default_dataset.manifest["coverage"][split]["tools"]) == names
        assert default_dataset.manifest["coverage"][split]["no_tool_rows"] > 0
        assert default_dataset.manifest["coverage"][split]["asr_rows"] > 0
        inputs[split] = {
            normalized_input(text)
            for row in rows
            for text in (row["messages"][1]["content"], row["clean_user"])
        }
        groups[split] = {row["group"] for row in rows}
        twins: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            twins[row["messages"][1]["content"]].add(row["music_state"])
            assert row["messages"][0]["content"] == build_system_prompt(
                {"status": "observed", "state": row["music_state"]}
            )
        assert all(states == {"on", "off"} for states in twins.values())
        assert {row["surface_language"] for row in rows} >= {"en", "zh", "mixed"}
        compositions = {tuple(sorted(row["entity_languages"].items())) for row in rows}
        assert (("artist", "en"), ("title", "zh")) in compositions
        assert (("artist", "zh"), ("title", "en")) in compositions
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        assert inputs[left].isdisjoint(inputs[right])
        assert groups[left].isdisjoint(groups[right])


def test_lively_commands_and_named_control_words(default_dataset: Dataset) -> None:
    """Bare 唱歌/放歌 resume music while quoted song titles remain explicit selection.

    Args:
        default_dataset:
            Dataset containing both lively starts and names that overlap control words.

    """
    rows = default_dataset.rows["train"]
    for text in (
        "唱歌",
        "放歌",
        "来首歌助助兴",
        "音乐走起",
        "BGM安排上",
        "drop a beat",
        "帮我 play some music",
    ):
        matches = [row for row in rows if row["messages"][1]["content"] == text]
        assert len(matches) == 2
        assert all(
            row["messages"][-1]["tool_calls"][0]["function"]
            == {"name": "resume_music", "arguments": {}}
            for row in matches
        )
    named = [
        row
        for row in rows
        if row["messages"][-1].get("tool_calls")
        and row["messages"][-1]["tool_calls"][0]["function"]["arguments"].get("title")
        == "唱歌"
    ]
    assert named
    assert all("《唱歌》" in row["messages"][1]["content"] for row in named)


def test_stt_variants_keep_intent_and_entities(default_dataset: Dataset) -> None:
    """Curated STT errors affect model input but cannot silently rewrite music entities.

    Args:
        default_dataset:
            Tracked clean and noisy command families with shared split ownership.

    """
    noisy = [
        row
        for rows in default_dataset.rows.values()
        for row in rows
        if row["asr_errors"]
    ]
    assert noisy
    for row in noisy:
        heard = row["messages"][1]["content"]
        assert heard != row["clean_user"]
        assert row["clean_user"] not in row["messages"][0]["content"]
        for error in row["asr_errors"]:
            assert error["heard"] in heard and error["clean"] in row["clean_user"]
        for call in row["messages"][-1].get("tool_calls", []):
            if call["function"]["name"] == "play_music":
                assert all(
                    value in heard for value in call["function"]["arguments"].values()
                )
    assert any("音亮" in row["messages"][1]["content"] for row in noisy)
    assert any("下一守" in row["messages"][1]["content"] for row in noisy)
    assert any("放首哥" in row["messages"][1]["content"] for row in noisy)


def test_clean_noisy_twins_cannot_cross_splits() -> None:
    """Even differently spelled heard commands cannot hide a shared clean input in test."""
    variants = VariantFile(
        version=1,
        banks={
            "commands": ["放首歌"],
            "heard": [
                Variant(text="放首哥", value="", clean="放首歌", error="homophone")
            ],
        },
    )
    templates = TemplateFile(
        version=1,
        families=[
            family(),
            family(
                id="noisy", group="different", split="test", slots={"command": "heard"}
            ),
        ],
    )
    with pytest.raises(ValueError, match="leaks across splits"):
        build_dataset(variants, templates)
    with pytest.raises(ValidationError):
        Variant(text="放首哥", value="", clean="放首歌")


def test_typed_and_mixed_entity_substitution() -> None:
    """Preserve boolean/numeric types and exact mixed-script punctuation/escaping."""
    title = "L'été {夜}\\Rain"
    artist = "Taylor 周"
    variants = VariantFile(
        version=1,
        banks={
            "titles": [Variant(text=f'"{title}"', value=title)],
            "artists": [artist],
            "power": [Variant(text="打开", value=True)],
            "levels": [Variant(text="三十五", value=35)],
        },
    )
    templates = TemplateFile(
        version=1,
        families=[
            family(
                id="named",
                group="named",
                language="mixed",
                utterances=["请 play {title} by {artist}"],
                slots={"title": "titles", "artist": "artists"},
                calls=[
                    {
                        "name": "play_music",
                        "arguments": {"title": "{title}", "artist": "{artist}"},
                    }
                ],
            ),
            family(
                id="light",
                group="light",
                language="zh",
                utterances=["{power}灯"],
                slots={"power": "power"},
                calls=[{"name": "set_light", "arguments": {"on": "{power}"}}],
            ),
            family(
                id="volume",
                group="volume",
                language="zh",
                utterances=["音量设为{level}"],
                slots={"level": "levels"},
                calls=[
                    {
                        "name": "volume_music",
                        "arguments": {"action": "set", "level": "{level}"},
                    }
                ],
            ),
        ],
    )
    dataset = build_dataset(variants, templates)
    for row in dataset.rows["train"]:
        target = row["messages"][-1]["tool_calls"][0]["function"]
        args = target["arguments"]
        if target["name"] == "play_music":
            assert args == {"title": title, "artist": artist}
        elif target["name"] == "set_light":
            assert args["on"] is True
        else:
            assert type(args["level"]) is int and args["level"] == 35
        rendered = render_example(row)
        assert len(rendered.assistant_spans) == 1
        start, end = rendered.assistant_spans[0]
        _, calls = parse_response(rendered.text[start:end])
        assert calls == (ToolCall(target["name"], args),)


def test_official_template_injection_and_masks(default_dataset: Dataset) -> None:
    """The pinned official template injects schemas once and marks only assistant targets.

    Args:
        default_dataset:
            Production-shaped messages and native tool declarations.

    """
    assert (
        hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest()
        == "ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6"
    )
    samples: dict[str, dict[str, Any]] = {}
    for rows in default_dataset.rows.values():
        for row in rows:
            samples.setdefault(row["family"], row)
    for row in samples.values():
        rendered = render_example(row)
        prefix = render_example(row, generation_prompt=True)
        assert rendered.text.startswith(prefix.text)
        assert not prefix.assistant_spans
        assert rendered.text.count("List of tools: ") == 1
        system = rendered.text.split("<|im_end|>", 1)[0]
        assert json.loads(system.split("List of tools: ", 1)[1]) == row["tools"]
        assert len(rendered.assistant_spans) == 1
        start, end = rendered.assistant_spans[0]
        assert start == len(prefix.text)
        assert end == len(rendered.text)
        text, calls = parse_response(rendered.text[start:end])
        expected = row["messages"][-1]
        assert calls == tuple(
            ToolCall(call["function"]["name"], call["function"]["arguments"])
            for call in expected.get("tool_calls", [])
        )
        assert text == expected["content"]


def test_reproducible_outputs_and_manifest(tmp_path: Path) -> None:
    """Identical inputs/seed give byte-identical exports with verifiable file hashes.

    Args:
        tmp_path:
            Isolated output directories; no model or machine-local dataset is accessed.

    """
    variants = VariantFile(
        version=1,
        banks={"commands": [f"play music please {index}" for index in range(20)]},
    )
    templates = TemplateFile(version=1, families=[family()])
    first = build_dataset(variants, templates, max_per_family=4)
    second = build_dataset(variants, templates, max_per_family=4)
    assert first == second
    assert (
        first.rows != build_dataset(variants, templates, seed=7, max_per_family=4).rows
    )
    for directory in (tmp_path / "a", tmp_path / "b"):
        write_dataset(first, directory)
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"):
        assert (tmp_path / "a" / name).read_bytes() == (
            tmp_path / "b" / name
        ).read_bytes()
    manifest = json.loads((tmp_path / "a/manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert (
            hashlib.sha256((tmp_path / "a" / name).read_bytes()).hexdigest() == expected
        )


@pytest.mark.parametrize(
    "kind",
    [
        "group",
        "input",
        "contradiction",
        "unknown",
        "ungrounded",
        "format",
        "noncanonical",
        "batch",
        "entity",
        "protocol",
    ],
)
def test_invalid_sources_fail_loudly(kind: str) -> None:
    """Reject structural, grounding, runtime-policy and split-leakage errors before output.

    Args:
        kind:
            Controlled invalid source configuration.

    """
    variants = VariantFile(
        version=1, banks={"commands": ["Play some music"], "value": ["yesterday"]}
    )
    first = family()
    cases = {
        "group": [
            first,
            family(id="two", split="test", utterances=["Different {command}"]),
        ],
        "input": [first, family(id="two", group="two", split="test")],
        "contradiction": [
            first,
            family(id="two", group="two", calls=[{"name": "what_is_playing"}]),
        ],
        "unknown": [family(utterances=["{unknown}"])],
        "ungrounded": [
            family(calls=[{"name": "play_music", "arguments": {"title": "{title}"}}])
        ],
        "format": [family(utterances=["{command.__class__}"])],
        "noncanonical": [
            family(
                utterances=["Weather {period}"],
                slots={"period": "value"},
                calls=[{"name": "get_weather", "arguments": {"period": "{period}"}}],
            )
        ],
        "batch": [family(calls=[{"name": "resume_music"}, {"name": "music_next"}])],
        "entity": [
            family(
                calls=[
                    {"name": "play_music", "arguments": {"artist": "Invented Artist"}}
                ]
            )
        ],
        "protocol": [
            family(calls=[{"name": "get_weather", "arguments": {"city": "<|im_end|>"}}])
        ],
    }
    with pytest.raises((ValueError, RuntimeError)):
        build_dataset(variants, TemplateFile(version=1, families=cases[kind]))


def test_invalid_label_types_and_exclusive_targets() -> None:
    """String booleans and simultaneous call/text targets are invalid supervision."""
    variants = VariantFile(version=1, banks={"commands": ["turn on the light"]})
    with pytest.raises(ValidationError):
        build_dataset(
            variants,
            TemplateFile(
                version=1,
                families=[
                    family(calls=[{"name": "set_light", "arguments": {"on": "true"}}])
                ],
            ),
        )
    with pytest.raises(ValidationError):
        family(response="Okay")
    assert "Call tools only" not in SYSTEM_PROMPT
    prompt = build_system_prompt(
        {
            "status": "observed",
            "state": "off",
            "title": "Private Track",
            "artist": "Private Artist",
            "volume": 99,
        }
    )
    assert "Private" not in prompt and "99" not in prompt


def test_materialization_limit_and_offline_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound output rows and reject any accidental construction of service clients.

    Args:
        monkeypatch:
            Installs a constructor guard while building declaration-only data.

    """

    def no_client(*args: Any, **kwargs: Any) -> None:
        """Reject any attempt to build a runtime client during offline generation.

        Args:
            *args:
                Unexpected positional client constructor arguments.

            **kwargs:
                Unexpected keyword client constructor arguments.

        """
        raise AssertionError("Dataset generation must not construct service clients")

    for cls in (MusicClient, WeatherClient, LightClient):
        monkeypatch.setattr(cls, "__init__", no_client)
    variants = VariantFile(
        version=1, banks={"commands": ["Play music", "Start some music"]}
    )
    templates = TemplateFile(version=1, families=[family()])
    with pytest.raises(ValueError, match="max_rows"):
        build_dataset(variants, templates, max_rows=3)
    assert len(build_dataset(variants, templates, max_rows=4).rows["train"]) == 4


def test_template_rejects_json_encoded_arguments(default_dataset: Dataset) -> None:
    """Native assistant calls require typed mappings rather than double-encoded JSON.

    Args:
        default_dataset:
            A valid row used to construct the malformed serialization case.

    """
    original = next(
        row
        for row in default_dataset.rows["train"]
        if row["messages"][-1].get("tool_calls")
        and row["messages"][-1]["tool_calls"][0]["function"]["arguments"]
    )
    row = json.loads(json.dumps(original))
    function = row["messages"][-1]["tool_calls"][0]["function"]
    function["arguments"] = json.dumps(function["arguments"])
    with pytest.raises(TemplateError, match="must be a mapping"):
        render_example(row)


def test_music_role_bindings_cannot_swap_or_invent_entities() -> None:
    """An argument must bind its declared entity slot, not another role or a command word."""
    variants = VariantFile(
        version=1,
        banks={
            "titles": ["Song"],
            "artists": ["Artist"],
            "commands": ["play some music"],
        },
    )
    swapped = family(
        utterances=["play {title} by {artist}"],
        slots={"title": "titles", "artist": "artists"},
        calls=[
            {
                "name": "play_music",
                "arguments": {"title": "{artist}", "artist": "{title}"},
            }
        ],
    )
    invented = family(calls=[{"name": "play_music", "arguments": {"artist": "play"}}])
    for bad in (swapped, invented):
        with pytest.raises(ValueError, match="named role slot"):
            build_dataset(variants, TemplateFile(version=1, families=[bad]))
