"""Offline canonical dataset and speech-label audits using embedded records."""

from copy import deepcopy
from typing import Any

import pytest

from finetune.augment import (
    aligned_entity,
    assemble,
    build_jobs,
    speech_row,
    volume_signature,
)
from finetune.generate import SPLITS, Dataset, canonical_dataset, dataset_coverage
from hoast.input_text import canonicalize_text


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
            Clean speech source when the original row contains a curated STT error.

        state:
            On/off context used to verify shared-audio handling.

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


def speech_result(
    job: dict[str, str], transcript: str, audit: str | None = None
) -> dict[str, Any]:
    """Build a persisted-worker-shaped observation with explicit script audit text.

    Args:
        job:
            Split-owned synthesis job produced by build_jobs.

        transcript:
            Recognizer output, canonicalized as the real worker does.

        audit:
            Optional same-length simplified-script representation for identity checks.

    """
    text = canonicalize_text(transcript)
    return {
        "job": job,
        "status": "ok" if text else "empty",
        "raw_transcript": transcript,
        "transcript": text,
        "audit_transcript": text if audit is None else audit,
        "audit_source": canonicalize_text(job["text"]),
    }


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


def test_synthesis_uses_clean_text_and_shares_state_twins() -> None:
    """Avoid speaking a curated ASR typo and avoid synthesizing music-state twins twice."""
    calls = [{"name": "volume_music", "arguments": {"action": "louder"}}]
    rows = canonical_dataset(
        dataset(
            {
                "train": [
                    source_row("把音亮调大！", calls, "noisy", clean="把音量调大！"),
                    source_row("把音量调大！", calls, "clean", state="on"),
                ]
            }
        )
    ).rows
    jobs, owners = build_jobs(rows)
    assert len(jobs) == 1 and jobs[0]["text"] == "把音量调大！"
    assert len(owners[jobs[0]["id"]]) == 2


def test_script_equivalent_names_are_copied_as_heard() -> None:
    """Accept verified script/spacing variants without teaching name reconstruction."""
    calls = [{"name": "play_music", "arguments": {"title": "夜曲", "artist": "邓紫棋"}}]
    rows = canonical_dataset(
        dataset({"train": [source_row("播放邓紫棋的《夜曲》", calls, "song")]})
    ).rows
    jobs, owners = build_jobs(rows)
    record = speech_result(jobs[0], "播放鄧紫棋的夜曲", "播放邓紫棋的夜曲")
    result = speech_row(owners[jobs[0]["id"]][0], record, {})
    assert result["speech"]["training_eligible"]
    assert (
        result["messages"][-1]["tool_calls"][0]["function"]["arguments"]["artist"]
        == "鄧紫棋"
    )
    wrong = speech_row(
        owners[jobs[0]["id"]][0], speech_result(jobs[0], "播放某某的夜曲"), {}
    )
    assert not wrong["speech"]["training_eligible"]
    assert "entity_changed/artist" in wrong["speech"]["review_reasons"]


def test_entity_alignment_does_not_borrow_artist_or_verb_text() -> None:
    """Keep a title's actual casing/position when the same letters occur elsewhere."""
    source = "play Play by Coldplay"
    heard = "Play PLAY by Cold play"
    assert aligned_entity("Play", source, source, heard, heard) == "PLAY"
    assert aligned_entity("Coldplay", source, source, heard, heard) == "Cold play"
    assert (
        aligned_entity("Play", "Play Play", "Play Play", "Play, Play", "Play, Play")
        is None
    )
    text = "play 'Til Tuesday by G.E.M."
    assert aligned_entity("'Til Tuesday", text, text, text, text) == "'Til Tuesday"
    assert aligned_entity("G.E.M.", text, text, text, text) == "G.E.M."


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


def test_negation_loss_and_empty_speech_require_review() -> None:
    """Never auto-label a lost negation or failed recognition as a clean training row."""
    rows = canonical_dataset(
        dataset({"train": [source_row("Don't start music.", [], "negative")]})
    ).rows
    jobs, owners = build_jobs(rows)
    for heard in ("Start music", ""):
        result = speech_row(owners[jobs[0]["id"]][0], speech_result(jobs[0], heard), {})
        assert not result["speech"]["training_eligible"]
        assert not result["messages"][-1].get("tool_calls")


def test_observed_cross_split_collision_blocks_training() -> None:
    """Flag a transcript collapsing a held-out command onto a training phrase."""
    calls = [{"name": "resume_music", "arguments": {}}]
    rows = canonical_dataset(
        dataset(
            {
                "train": [source_row("Resume the music.", calls, "training")],
                "test": [source_row("Start some music.", calls, "heldout")],
            }
        )
    ).rows
    jobs, owners = build_jobs(rows)
    records = {job["id"]: speech_result(job, "Resume the music.") for job in jobs}
    candidates, coverage = assemble(rows, jobs, owners, records, {})
    assert coverage["train"]["eligible_rows"] == 0
    assert "cross_split_collision" in candidates["test"][0]["speech"]["review_reasons"]


def test_integer_percentage_spellings_preserve_the_bound_value() -> None:
    """Approve integer word/digit variants without accepting a different heard value."""
    assistant = {
        "tool_calls": [
            {
                "function": {
                    "name": "volume_music",
                    "arguments": {"action": "set", "level": 35},
                }
            }
        ]
    }
    assert volume_signature(
        "set volume thirty five percent", assistant
    ) == volume_signature("set volume 35", assistant)
    assert volume_signature("把音量设为三十五", assistant) == volume_signature(
        "把音量设为35", assistant
    )
    assert volume_signature("set volume 50", assistant) != volume_signature(
        "set volume 35", assistant
    )
