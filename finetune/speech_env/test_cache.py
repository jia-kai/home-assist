"""Offline checks for verified raw-speech migration across normalization policies."""

import hashlib
import json
from typing import Any

import pytest

from finetune.speech_worker import inference_signature, migrate_record


class IdentityScript:
    """ASCII fixture for the script-only audit converter."""

    def convert(self, text: str) -> str:
        """Return unchanged fixture text.

        Args:
            text:
                Canonical ASCII test text.

        """
        return text


def test_inference_fingerprint_excludes_only_cache_code() -> None:
    """Reject inference/import changes while allowing a separate migration helper."""
    source = """
from backend_a import Model
class SpeechWorker:
    def recognize(self, waves):
        return Model(beam_size=1).generate(waves)
def main():
    for offset in range(10):
        worker.recognize(waves)
"""
    original = inference_signature(source)
    assert (
        inference_signature(source + "\ndef migrate_cache():\n    return None\n")
        == original
    )
    assert inference_signature(source.replace("beam_size=1", "beam_size=2")) != original
    assert inference_signature(source.replace("backend_a", "backend_b")) != original


@pytest.mark.parametrize("suppressed", [False, True])
def test_migration_reprocesses_raw_text_and_preserves_provenance(
    suppressed: bool,
) -> None:
    """Use acoustic no-speech evidence rather than a previous normalization outcome.

    Args:
        suppressed:
            Whether recorded acoustic confidence requires retaining an empty transcript.

    """
    prior = {
        "device": "cuda",
        "batch_size": 16,
        "worker_sha256": "old",
        "normalizer_sha256": "old",
    }
    current = {
        **prior,
        "worker_sha256": "new",
        "normalizer_sha256": "new",
        "inference_sha256": "same",
    }
    job = {"id": "fixture", "text": "Don't stop, please.", "split": "train"}
    prior_hash = hashlib.sha256(json.dumps(prior, sort_keys=True).encode()).hexdigest()
    record: dict[str, Any] = {
        "job": job,
        "settings_sha256": prior_hash,
        "status": "empty",
        "raw_transcript": "Don't stop, please.",
        "transcript": "",
        "avg_logprob": -2.0 if suppressed else -0.1,
        "no_speech_probability": 0.9 if suppressed else 0.1,
        "audio_sha256": "same-audio",
    }
    migrated = migrate_record(record, job, current, prior, IdentityScript())
    assert migrated["transcript"] == ("" if suppressed else "Don't stop, please.")
    assert migrated["audio_sha256"] == "same-audio"
    assert migrated["inference_provenance"]["settings_sha256"] == prior_hash
    assert record["transcript"] == ""
    with pytest.raises(ValueError, match="incompatible"):
        migrate_record(
            record, job, {**current, "device": "cpu"}, prior, IdentityScript()
        )
    with pytest.raises(ValueError, match="provenance"):
        migrate_record(
            record, {**job, "split": "test"}, current, prior, IdentityScript()
        )
