"""Offline canonical dataset regressions using embedded records."""

from copy import deepcopy
from typing import Any

import pytest

from finetune.generate import SPLITS, Dataset, canonical_dataset, dataset_coverage


def source_row(
    text: str,
    calls: list[dict[str, Any]],
    identifier: str,
    *,
    clean: str | None = None,
    state: str = "off",
) -> dict[str, Any]:
    """Construct an original, unnormalized routing example without external files.

    Args:
        text:
            Original user surface, including optional punctuation or curated noise.

        calls:
            Native function name/argument dictionaries; empty means abstention.

        identifier:
            Distinct fixture source identifier and family name.

        clean:
            Clean source when the original row contains a curated STT error.

        state:
            On/off music context for the routing example.

    """
    return {
        "id": identifier,
        "family": identifier,
        "group": identifier,
        "command_language": "en",
        "surface_language": "en",
        "entity_languages": {},
        "clean_user": text if clean is None else clean,
        "asr_errors": [],
        "music_state": state,
        "messages": [
            {"role": "system", "content": f"Music state: {state}."},
            {"role": "user", "content": text},
            {
                "role": "assistant",
                "content": "" if calls else "Okay.",
                **(
                    {
                        "tool_calls": [
                            {"type": "function", "function": call} for call in calls
                        ]
                    }
                    if calls
                    else {}
                ),
            },
        ],
        "tools": [],
    }


def dataset(items: dict[str, list[dict[str, Any]]]) -> Dataset:
    """Wrap fixture splits with source coverage and schema identity.

    Args:
        items:
            Explicit split ownership for embedded examples.

    """
    rows = {split: items.get(split, []) for split in SPLITS}
    return Dataset(
        rows,
        {
            "schema_sha256": "fixture",
            "coverage": dataset_coverage(rows),
            "families": {
                row["family"]: {"written_rows": 1}
                for values in rows.values()
                for row in values
            },
        },
    )


def test_canonical_entities_dedup_and_provenance() -> None:
    """Remove delimiters and normalize copied names while preserving every source ID."""
    calls = [
        {"name": "play_music", "arguments": {"title": "Don't Stop", "artist": "G.E.M."}}
    ]
    first = source_row('play "Don\'t Stop" by "G.E.M."!', calls, "first")
    second = source_row("play 《Don't Stop》 by G.E.M.!", calls, "second")
    original = deepcopy(first)
    result = canonical_dataset(dataset({"train": [first, second]}))
    assert first == original
    assert len(result.rows["train"]) == 1
    row = result.rows["train"][0]
    assert row["messages"][-2]["content"] == "play Don't Stop by G.E.M.!"
    assert row["messages"][-1]["tool_calls"][0]["function"]["arguments"] == {
        "title": "Don't Stop",
        "artist": "G.E.M.",
    }
    assert [item["id"] for item in row["normalization"]["sources"]] == [
        "first",
        "second",
    ]
    assert result.manifest["coverage"]["train"]["rows"] == 1
    assert result.manifest["source_coverage"]["train"]["rows"] == 2


def test_canonical_collisions_fail() -> None:
    """Catch cross-split leakage and labels made contradictory by losing quotes."""
    named = source_row(
        'play "Play"', [{"name": "play_music", "arguments": {"title": "Play"}}], "named"
    )
    same = source_row(
        "play 《Play》",
        [{"name": "play_music", "arguments": {"title": "Play"}}],
        "same",
    )
    generic = source_row(
        "play Play", [{"name": "resume_music", "arguments": {}}], "generic"
    )
    with pytest.raises(ValueError, match="collision"):
        canonical_dataset(dataset({"train": [named], "test": [same]}))
    with pytest.raises(ValueError, match="collision"):
        canonical_dataset(dataset({"train": [named, generic]}))


def test_typed_and_query_arguments_are_not_stripped() -> None:
    """Canonicalize user language without corrupting enums, queries, or scalar types."""
    calls = [
        {
            "name": "get_weather",
            "arguments": {"period": "next_week", "city": "Paris, France"},
        },
        {"name": "set_light", "arguments": {"on": False}},
    ]
    row = source_row(
        "Weather in Paris, France next week; turn the light off!", calls, "typed"
    )
    result = canonical_dataset(dataset({"train": [row]})).rows["train"][0]
    assert result["messages"][-1] == row["messages"][-1]
    assert (
        result["messages"][-2]["content"]
        == "Weather in Paris, France next week; turn the light off!"
    )


def test_punctuation_variants_share_ownership_not_literal_labels() -> None:
    """Allow distinct punctuated inputs with copied labels but reject cross-split reuse."""
    one = source_row(
        "play Don't Stop",
        [{"name": "play_music", "arguments": {"title": "Don't Stop"}}],
        "one",
    )
    two = source_row(
        "play Dont Stop",
        [{"name": "play_music", "arguments": {"title": "Dont Stop"}}],
        "two",
    )
    result = canonical_dataset(dataset({"train": [one, two]}))
    assert len(result.rows["train"]) == 2
    with pytest.raises(ValueError, match="collision"):
        canonical_dataset(dataset({"train": [one], "test": [two]}))
