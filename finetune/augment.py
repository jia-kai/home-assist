"""Build split-owned TTS-to-STT variants with explicit label and transcript audits."""

import argparse
import hashlib
import json
import logging
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from hoast.input_text import NORMALIZATION_VERSION, canonicalize_text, content_signature
from hoast.lfm import tool_declarations
from hoast.llm import ToolCall

from .generate import (
    SPLITS,
    Dataset,
    VariantFile,
    dataset_coverage,
    digest,
    surface_language,
    write_dataset,
)
from .tooling import tool_registry

logger = logging.getLogger(__name__)


def signature(text: str) -> str:
    """Compare word content without case, spacing, or punctuation differences.

    Args:
        text:
            Canonical user language; model input itself is not case-folded.

    """
    return content_signature(text)


def build_jobs(
    rows: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, Any]]]]:
    """Share audio between music-state twins and clean/curated-noise source variants.

    Args:
        rows:
            Canonical source rows indexed by their immutable source split.

    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for split, items in rows.items():
        for row in items:
            if row["normalization"]["split"] != split:
                raise ValueError("Source split provenance does not match its file")
            groups[(split, signature(row["clean_user"]))].append(row)
    jobs: list[dict[str, str]] = []
    owners: dict[str, list[dict[str, Any]]] = {}
    for (split, clean), items in sorted(groups.items()):
        options = [
            (source["id"], source["clean_user"])
            for row in items
            for source in row["normalization"]["sources"]
            if signature(canonicalize_text(source["clean_user"])) == clean
        ]
        if not options:
            raise ValueError("Missing original clean text for synthesis")
        text = min(options)[1]
        identifier = digest([split, text, "tts_stt_v1"])[:24]
        jobs.append({"id": identifier, "text": text, "split": split})
        owners[identifier] = items
    return jobs, owners


def _entity_span(value: str, text: str) -> re.Match[str] | None:
    """Locate an entity while allowing ASR capitalization and spacing differences.

    Args:
        value:
            Known source entity; punctuation has already been canonicalized.

        text:
            Transcript or synthesis text in an aligned script representation.

    """
    pattern = r"\s*".join(
        re.escape(character) for character in value if not character.isspace()
    )
    if not pattern:
        return None
    if value[0].isascii() and value[0].isalnum():
        pattern = r"(?<![A-Za-z0-9])" + pattern
    if value[-1].isascii() and value[-1].isalnum():
        pattern += r"(?=s?(?:[^A-Za-z0-9]|$))"
    matches = list(re.finditer(pattern, text))
    if not matches:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    return matches[0] if len(matches) == 1 else None


def aligned_entity(
    value: str, source: str, audit_source: str, heard: str, audit: str
) -> str | None:
    """Copy the heard span aligned to the labelled source entity, not another mention.

    Character alignment operates on short offline command strings. It prevents a
    title such as Play from being borrowed from Coldplay or from the request verb.
    Entire entity spans must survive in one matching block; changed names fail.

    Args:
        value:
            Canonical intended music entity from the validated source target.

        source:
            Canonical clean synthesis text, preserving source script and casing.

        audit_source:
            Same-length script-equivalent source used for character alignment.

        heard:
            Actual canonical transcript from which a returned label is copied.

        audit:
            Same-length script-equivalent transcript used for alignment.

    """
    if len(source) != len(audit_source) or len(heard) != len(audit):
        return None
    match = _entity_span(value, source)
    if match is None:
        return None
    begin = len(signature(audit_source[: match.start()]))
    end = len(signature(audit_source[: match.end()]))
    offsets = [
        index
        for index, character in enumerate(audit)
        if unicodedata.category(character)[0] in "LNM"
        for _ in character.casefold()
    ]
    for block in SequenceMatcher(
        None, signature(audit_source), signature(audit), autojunk=False
    ).get_matching_blocks():
        if block.a <= begin < end <= block.a + block.size:
            first = block.b + begin - block.a
            last = block.b + end - block.a - 1
            start = offsets[first]
            prefix = re.match(r"[^\w\s]+", value)
            if prefix and heard[:start].endswith(prefix[0]):
                start -= len(prefix[0])
            stop = offsets[last] + 1
            suffix = re.search(r"[^\w\s]+$", value)
            if suffix and heard[stop:].startswith(suffix[0]):
                stop += len(suffix[0])
            return heard[start:stop]
    return None


def volume_signature(text: str, assistant: dict[str, Any]) -> str:
    """Compare known integer volume values across digits, spoken words, and percent units.

    Only the already-labelled integer in [0, 100] is mapped. A different heard
    number is not corrected or guessed. This audit does not alter LLM input.

    Args:
        text:
            Script-normalized source or recognized command for label auditing.

        assistant:
            Known intended target, used to bind a volume amount rather than infer one.

    """
    for call in assistant.get("tool_calls", []):
        function = call["function"]
        if function["name"] != "volume_music":
            continue
        level = function["arguments"]["level"]
        if type(level) is not int or not 0 <= level <= 100:
            raise ValueError("Volume label must be an integer percentage in [0, 100]")
        small = (
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        )
        tens = (
            "",
            "",
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
        )
        english = (
            "one hundred"
            if level == 100
            else small[level]
            if level < 20
            else tens[level // 10] + (" " + small[level % 10] if level % 10 else "")
        )
        digits = "零一二三四五六七八九"
        chinese = (
            "一百"
            if level == 100
            else digits[level]
            if level < 10
            else (digits[level // 10] if level >= 20 else "")
            + "十"
            + (digits[level % 10] if level % 10 else "")
        )
        english_pattern = (
            r"(?<![A-Za-z])"
            + re.escape(english).replace(r"\ ", r"\s+")
            + r"(?![A-Za-z])"
        )
        text = re.sub(english_pattern, str(level), text, flags=re.IGNORECASE)
        text = text.replace(chinese, str(level))
        text = re.sub(
            r"\b(?:percentage\s+points?|percent)\b|个百分点|百分之",
            "",
            text,
            flags=re.IGNORECASE,
        )
    return signature(text)


def speech_row(
    source: dict[str, Any], record: dict[str, Any], confusions: dict[str, str]
) -> dict[str, Any]:
    """Create an intended-command example and conservatively assess training eligibility.

    Name changes are never silently restored. Script/case/space-equivalent entities
    are copied from the actual transcript when alignment proves their identity.
    Other transcript edits require review unless covered by existing curated STT
    confusions. Failed synthesis/recognition is retained for end-to-end accounting.

    Args:
        source:
            Canonical row whose known clean spoken command supplies the intended label.

        record:
            Persisted speech worker result, including raw/canonical transcript,
            aligned script-only audit text, model fingerprint, and audio provenance.

        confusions:
            Reviewed canonical heard-to-clean command substitutions from variant banks.

    """
    row = deepcopy(source)
    heard = record["transcript"]
    if heard != canonicalize_text(heard):
        raise ValueError("Worker transcript is not canonical")
    clean = canonicalize_text(record["job"]["text"])
    intended = deepcopy(source["messages"][-1])
    assistant = row["messages"][-1]
    reasons: list[str] = []
    if record["status"] != "ok":
        reasons.append("speech_unavailable")
    else:
        audit = record["audit_transcript"]
        audit_source = record["audit_source"]
        aligned = len(audit) == len(heard) and len(audit_source) == len(clean)
        if not aligned:
            reasons.append("script_alignment")
        for call in assistant.get("tool_calls", []):
            function = call["function"]
            if function["name"] != "play_music":
                continue
            for key in ("title", "artist"):
                if key not in function["arguments"]:
                    continue
                value = function["arguments"][key]
                copied = (
                    aligned_entity(value, clean, audit_source, heard, audit)
                    if aligned
                    else None
                )
                if copied is None:
                    reasons.append(f"entity_changed/{key}")
                else:
                    function["arguments"][key] = copied
        corrected = audit
        for noisy, original in sorted(
            confusions.items(), key=lambda pair: -len(pair[0])
        ):
            corrected = corrected.replace(noisy, original)
        if volume_signature(audit_source, intended) not in {
            volume_signature(audit, intended),
            volume_signature(corrected, intended),
        }:
            reasons.append("transcript_changed")
    row["messages"][-2]["content"] = heard
    row["clean_user"] = clean
    row["surface_language"] = surface_language(heard)
    row["asr_errors"] = (
        [] if heard == clean else [{"kind": "tts_stt", "clean": clean, "heard": heard}]
    )
    row["entity_languages"] = {
        key: surface_language(value)
        for call in assistant.get("tool_calls", [])
        if call["function"]["name"] == "play_music"
        for key, value in call["function"]["arguments"].items()
        if isinstance(value, str) and value
    }
    row["speech"] = {
        **record,
        "source_ids": [source["id"]],
        "intended_assistant": intended,
        "review_reasons": reasons,
        "training_eligible": not reasons,
    }
    row["id"] = digest([record["job"]["id"], source["music_state"], intended])[:24]
    tool_registry().validate(
        [
            ToolCall(call["function"]["name"], call["function"]["arguments"])
            for call in assistant.get("tool_calls", [])
        ]
    )
    return row


def assemble(
    rows: dict[str, list[dict[str, Any]]],
    jobs: list[dict[str, str]],
    owners: dict[str, list[dict[str, Any]]],
    records: dict[str, dict[str, Any]],
    confusions: dict[str, str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build unique speech observations and flag lexical collisions across source splits.

    Args:
        rows:
            Frozen canonical clean/curated-noise examples for all source splits.

        jobs:
            Completed synthesis jobs, each bound to exactly one source split.

        owners:
            All canonical parent rows represented by each unique spoken command.

        records:
            One actual speech result per job; missing records raise.

        confusions:
            Reviewed heard-to-clean command spellings used only for label auditing.

    """
    output: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    ownership: dict[str, set[str]] = defaultdict(set)
    for split, items in rows.items():
        for row in items:
            for text in (row["messages"][-2]["content"], row["clean_user"]):
                ownership[signature(text)].add(split)
    unique: dict[str, dict[str, Any]] = {}
    for job in jobs:
        record = records[job["id"]]
        if record["job"] != job:
            raise ValueError("Transcript record does not match its synthesis job")
        for source in owners[job["id"]]:
            row = speech_row(source, record, confusions)
            if row["id"] in unique:
                unique[row["id"]]["speech"]["source_ids"].append(source["id"])
                continue
            unique[row["id"]] = row
            output[job["split"]].append(row)
            if row["messages"][-2]["content"]:
                ownership[signature(row["messages"][-2]["content"])].add(job["split"])
    for split, items in output.items():
        for row in items:
            text = row["messages"][-2]["content"]
            if text and ownership[signature(text)] != {split}:
                row["speech"]["review_reasons"].append("cross_split_collision")
                row["speech"]["training_eligible"] = False
    coverage = {
        split: {
            "source_rows": len(rows[split]),
            "speech_rows": len(items),
            "eligible_rows": sum(row["speech"]["training_eligible"] for row in items),
            "review_reasons": dict(
                Counter(
                    reason
                    for row in items
                    for reason in row["speech"]["review_reasons"]
                )
            ),
            "command_languages": dict(
                Counter(row["command_language"] for row in items)
            ),
            "speech_outcomes": dict(Counter(row["speech"]["status"] for row in items)),
        }
        for split, items in output.items()
    }
    return output, coverage


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Publish one UTF-8 JSONL artifact atomically.

    Args:
        path:
            Destination under the already-created augmentation directory.

        rows:
            Serializable source-linked observations or reviewed training examples.

    """
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    """Run isolated speech inference and publish candidates, reviewed data, and coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("finetune/generated"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variants", type=Path, default=Path(__file__).with_name("variants.json")
    )
    parser.add_argument(
        "--speech-python",
        type=Path,
        default=Path("finetune/speech_env/.venv/bin/python"),
    )
    parser.add_argument(
        "--assets", type=Path, default=Path(".cache/hoast/finetune/speech-assets")
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--reuse-transcripts", action="store_true")
    parser.add_argument(
        "--renormalize-from",
        type=Path,
        help="Prior speech directory with compatible raw transcripts/audio",
    )
    parser.add_argument(
        "--prior-worker-source",
        type=Path,
        help="Verified source snapshot for a cache without an inference fingerprint",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.output / "augmentation.log"),
            logging.StreamHandler(),
        ],
    )
    try:
        if args.reuse_transcripts and args.renormalize_from is not None:
            raise ValueError("Choose reuse-transcripts or renormalize-from, not both")
        manifest = json.loads((args.data / "manifest.json").read_text())
        if manifest["normalization"]["version"] != NORMALIZATION_VERSION:
            raise ValueError("Generate canonical data before speech augmentation")
        if manifest["schema_sha256"] != digest(tool_declarations(tool_registry())):
            raise ValueError("Tool schema changed; regenerate canonical data")
        variants = VariantFile.model_validate_json(
            args.variants.read_text()
        ).model_dump()
        if manifest["variants_sha256"] != digest(variants):
            raise ValueError("Variant bank does not match the source dataset")
        normalizer_hash = hashlib.sha256(
            (
                Path(__file__).resolve().parent.parent / "hoast/input_text.py"
            ).read_bytes()
        ).hexdigest()
        if manifest["normalization"]["normalizer_sha256"] != normalizer_hash:
            raise ValueError("Source normalizer changed; regenerate canonical data")
        rows: dict[str, list[dict[str, Any]]] = {}
        for split in SPLITS:
            path = args.data / f"{split}.jsonl"
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != manifest["files"][path.name]
            ):
                raise ValueError("Source dataset checksum mismatch")
            rows[split] = [json.loads(line) for line in path.read_text().splitlines()]
        jobs, owners = build_jobs(rows)
        jobs_path = args.output / "jobs.jsonl"
        write_rows(jobs_path, jobs)
        speech = args.output / "speech"
        if not args.reuse_transcripts:
            if not args.speech_python.is_file():
                raise FileNotFoundError("Prepare the isolated speech environment first")
            with (args.output / "speech-process.log").open("w") as log:
                subprocess.run(
                    [
                        # Resolving the interpreter symlink would bypass its venv.
                        str(args.speech_python.absolute()),
                        "-m",
                        "finetune.speech_worker",
                        "--jobs",
                        str(jobs_path.resolve()),
                        "--output",
                        str(speech.resolve()),
                        "--assets",
                        str(args.assets.resolve()),
                        "--device",
                        args.device,
                        "--batch-size",
                        str(args.batch_size),
                        *(
                            ["--reuse-from", str(args.renormalize_from.resolve())]
                            if args.renormalize_from is not None
                            else []
                        ),
                        *(
                            [
                                "--prior-worker-source",
                                str(args.prior_worker_source.resolve()),
                            ]
                            if args.prior_worker_source is not None
                            else []
                        ),
                    ],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=Path(__file__).resolve().parent.parent,
                )
        settings = json.loads((speech / "settings.json").read_text())
        if (
            settings["device"] != args.device
            or settings["batch_size"] != args.batch_size
        ):
            raise ValueError(
                "Cached speech device/batch size differs from the requested settings"
            )
        worker_hash = hashlib.sha256(
            Path(__file__).with_name("speech_worker.py").read_bytes()
        ).hexdigest()
        if settings["worker_sha256"] != worker_hash:
            raise ValueError("Speech worker changed; use a new output directory")
        for name, expected in settings["speech_frontend_sha256"].items():
            actual = hashlib.sha256(
                (Path(__file__).resolve().parent.parent / "hoast" / name).read_bytes()
            ).hexdigest()
            if actual != expected:
                raise ValueError("Speech frontend changed; use a new output directory")
        if settings["normalizer_sha256"] != normalizer_hash:
            raise ValueError("Speech and source normalization policies differ")
        fingerprint = hashlib.sha256(
            json.dumps(settings, sort_keys=True).encode()
        ).hexdigest()
        records = {
            job["id"]: json.loads((speech / f"{job['id']}.json").read_text())
            for job in jobs
        }
        if any(record["settings_sha256"] != fingerprint for record in records.values()):
            raise ValueError("Speech results mix incompatible inference settings")
        for record in records.values():
            if record["status"] in ("ok", "empty", "invalid_input"):
                audio = speech / record["audio_file"]
                if (
                    hashlib.sha256(audio.read_bytes()).hexdigest()
                    != record["audio_sha256"]
                ):
                    raise ValueError("Roundtrip audio checksum mismatch")
        confusions = {
            canonicalize_text(value["text"]): canonicalize_text(value["clean"])
            for bank in variants["banks"].values()
            for value in bank
            if isinstance(value, dict) and value.get("clean") is not None
        }
        candidates, coverage = assemble(rows, jobs, owners, records, confusions)
        if any(
            not items or not any(row["speech"]["status"] == "ok" for row in items)
            for items in candidates.values()
        ):
            raise ValueError("Each source split needs successful speech observations")
        for split, items in candidates.items():
            write_rows(args.output / f"{split}-roundtrip.jsonl", items)
            if split != "train":
                write_rows(
                    args.output / f"{split}-roundtrip-independent.jsonl",
                    [
                        row
                        for row in items
                        if "cross_split_collision"
                        not in row["speech"]["review_reasons"]
                    ],
                )
        review = [
            row
            for items in candidates.values()
            for row in items
            if not row["speech"]["training_eligible"]
        ]
        write_rows(args.output / "review.jsonl", review)
        augmented = deepcopy(rows)
        seen = {digest(row["messages"]) for row in augmented["train"]}
        labels_by_input = {
            digest([row["messages"][:-1], row["tools"]]): digest(row["messages"][-1])
            for row in augmented["train"]
        }
        accepted: list[dict[str, Any]] = []
        for row in candidates["train"]:
            key = digest(row["messages"])
            if row["speech"]["training_eligible"] and key not in seen:
                input_key = digest([row["messages"][:-1], row["tools"]])
                target = digest(row["messages"][-1])
                if labels_by_input.setdefault(input_key, target) != target:
                    raise ValueError(
                        "Speech augmentation creates contradictory labels for identical model input"
                    )
                seen.add(key)
                accepted.append(row)
        augmented["train"].extend(accepted)
        if not accepted:
            raise ValueError(
                "Speech augmentation produced no eligible new training examples; inspect the review file"
            )
        augmented_manifest = deepcopy(manifest)
        augmented_manifest["speech_augmentation"] = {
            "source_manifest_sha256": hashlib.sha256(
                (args.data / "manifest.json").read_bytes()
            ).hexdigest(),
            "speech_settings": settings,
            "coverage": coverage,
            "new_training_rows": len(accepted),
            "unique_audio_jobs": len(jobs),
            "reused_audio_jobs": sum(
                bool(record.get("reused_inference")) for record in records.values()
            ),
            "label_policy": "only_script_case_space_equivalence_or_reviewed_command_confusions_with_grounded_music_entities",
            "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        augmented_manifest["canonical_source_coverage"] = augmented_manifest["coverage"]
        augmented_manifest["coverage"] = dataset_coverage(augmented)
        for split, items in augmented.items():
            for family, count in Counter(row["family"] for row in items).items():
                augmented_manifest["families"][family]["written_rows"] = count
        write_dataset(Dataset(augmented, augmented_manifest), args.output / "data")
        (args.output / "augmentation.json").write_text(
            json.dumps(
                augmented_manifest["speech_augmentation"], ensure_ascii=False, indent=2
            )
        )
        logger.info(
            "augmentation status=ok unique_audio=%d new_training_rows=%d review_rows=%d coverage=%s",
            len(jobs),
            len(accepted),
            len(review),
            coverage,
        )
    except Exception:
        logger.exception("augmentation status=failed")
        raise


if __name__ == "__main__":
    main()
