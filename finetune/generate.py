"""Build offline conversational tool-routing JSONL from typed variant banks and templates."""

import argparse
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from tempfile import TemporaryDirectory
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hoast.agent import validate_action_batch
from hoast.lfm import tool_declarations
from hoast.llm import ToolCall
from hoast.logging import configure_logging, get_logger
from hoast.prompts import build_system_prompt

from . import TEMPLATE_PATH
from .tooling import tool_registry

logger = get_logger(__name__)
ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")
type Scalar = str | int | bool


class StrictModel(BaseModel):
    """Reject unknown configuration fields and implicit scalar coercion."""

    model_config: ConfigDict = ConfigDict(extra="forbid", strict=True, frozen=True)
    """Strict source-file contract, with immutable model attributes."""


class Variant(StrictModel):
    """A spoken surface form and its exact typed argument value."""

    text: str
    """Text inserted into a command, including any deliberate quoting or affixes."""

    value: Scalar
    """Argument value; strings retain spelling and Unicode, booleans/integers retain types."""

    clean: str | None = None
    """Intended clean surface for a curated STT error; metadata only, never model input."""

    error: (
        Literal["homophone", "near_homophone", "segmentation", "latin_asr"] | None
    ) = None
    """Explicit STT-confusion category, required exactly when clean is supplied."""

    @model_validator(mode="after")
    def paired_asr_metadata(self) -> Self:
        """Require a distinct nonempty clean form and error category together."""
        if (self.clean is None) != (self.error is None):
            raise ValueError("STT variants require both clean and error")
        if self.clean is not None and (
            not self.clean.strip() or self.clean == self.text
        ):
            raise ValueError(
                "STT clean text must be nonempty and differ from heard text"
            )
        return self


class VariantFile(StrictModel):
    """Named banks of reusable lexical/entity choices."""

    version: Literal[1]
    """Source format version."""

    banks: dict[str, list[Variant | Scalar]]
    """Variant objects, or scalar shorthand used as both surface text and value."""


class CallTemplate(StrictModel):
    """One supervised tool call with optional typed placeholders."""

    name: str
    """Fixed production tool name; tool names are never expanded from user text."""

    arguments: dict[str, Scalar] = Field(default_factory=dict)
    """Argument literals or named placeholders; a whole placeholder retains its value type."""


class Family(StrictModel):
    """A split-owned family of related command surface templates."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    """Unique template-family identifier used for deterministic sampling."""

    group: str = Field(min_length=1)
    """Semantic/surface parent shared by related translations; must belong to one split."""

    split: Literal["train", "validation", "test"]
    """Explicit split assignment applied before expanding variants or music states."""

    language: Literal["en", "zh", "mixed"]
    """Command-frame language; inserted entity names can change the full surface language."""

    utterances: list[str] = Field(min_length=1)
    """Related format strings using only simple named fields, without conversions/specifiers."""

    slots: dict[str, str] = Field(default_factory=dict)
    """Template field names mapped to variant-bank names."""

    calls: list[CallTemplate] = Field(default_factory=list)
    """Ordered tool targets; empty only for a no-tool response example."""

    response: str | None = None
    """Brief assistant text for no-tool examples; routing abstention is the supervised signal."""

    @model_validator(mode="after")
    def exclusive_target(self) -> Self:
        """Require exactly one of tool calls or a nonempty no-tool response."""
        if (self.calls and self.response is not None) or (
            not self.calls and not (self.response and self.response.strip())
        ):
            raise ValueError(
                "A family requires calls or a nonempty response, exclusively"
            )
        return self


class TemplateFile(StrictModel):
    """Versioned collection of manually split command families."""

    version: Literal[1]
    """Source format version."""

    families: list[Family] = Field(min_length=1)
    """Families expanded independently, with global duplicate/leakage checks."""


@dataclass(slots=True)
class Dataset:
    """Validated records and a reproducibility/coverage manifest."""

    rows: dict[str, list[dict[str, Any]]]
    """Self-contained conversational rows keyed by split; tools are read-only shared metadata."""

    manifest: dict[str, Any]
    """Source/prompt/schema hashes, generation parameters, and actual row coverage."""


def digest(value: object) -> str:
    """Hash a canonical JSON value for stable record identities and provenance.

    Args:
        value:
            JSON-serializable data; NaN and infinity are rejected.

    """
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def fields(template: str) -> set[str]:
    """Find simple placeholders and reject Python attribute/index/format expressions.

    Args:
        template:
            A source template, with double braces allowed for literal braces.

    """
    result: set[str] = set()
    for _, name, spec, conversion in Formatter().parse(template):
        if name is None:
            continue
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or spec or conversion:
            raise ValueError(
                f"Only simple named placeholders are allowed: {template!r}"
            )
        result.add(name)
    return result


def normalized_input(text: str) -> str:
    """Normalize case, width, whitespace and terminal punctuation for leakage checks.

    Args:
        text:
            Generated user command; interior entity punctuation is preserved.

    """
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split()).rstrip(
        ".!?。！？"
    )


def surface_language(text: str) -> str:
    """Describe visible Latin/Han script composition without translating entity labels.

    Args:
        text:
            Utterance or entity value whose script composition is reported.

    """
    han = re.search(r"[\u3400-\u9fff]", text) is not None
    latin = re.search(r"[A-Za-z]", text) is not None
    return "mixed" if han and latin else "zh" if han else "en" if latin else "other"


def expand_value(value: Scalar, chosen: dict[str, Variant]) -> Scalar:
    """Expand a label while preserving types for whole-field substitutions.

    Args:
        value:
            Literal scalar or format string from a tool argument/response template.

        chosen:
            Selected variants keyed by simple placeholder names.

    """
    if not isinstance(value, str):
        return value
    names = fields(value)
    if names - chosen.keys():
        raise ValueError(f"Unknown argument placeholders: {names - chosen.keys()}")
    match = re.fullmatch(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", value)
    if match:
        return chosen[match[1]].value
    return value.format_map({name: variant.value for name, variant in chosen.items()})


def build_dataset(
    variants: VariantFile,
    templates: TemplateFile,
    *,
    seed: int = 42,
    max_per_family: int = 128,
    max_rows: int = 100_000,
) -> Dataset:
    """Expand and validate split-owned templates without model or service access.

    Cartesian indices are sampled before materializing rows; each selected command
    produces both on/off states. Related groups and normalized commands cannot cross
    splits. Contradictory normalized labels fail. Exact within-split duplicates are
    removed; punctuation/case variants remain in the same split. Actual post-dedup
    counts are reported, so max_per_family is an upper bound, not a guaranteed count.

    Args:
        variants:
            Strictly parsed lexical banks, with typed entity/value aliases.

        templates:
            Strictly parsed families and their explicit split ownership.

        seed:
            Integer seed for stable per-family sampling and per-split shuffling.

        max_per_family:
            Maximum sampled base commands per family before on/off expansion;
            zero expands the entire Cartesian space, capped at one million per family.

        max_rows:
            Positive total output-row limit across all splits, bounding materialized
            output while detecting accidental Cartesian expansion.

    """
    if type(seed) is not int or type(max_per_family) is not int or max_per_family < 0:
        raise ValueError(
            "seed and max_per_family must be integers; the limit cannot be negative"
        )
    if type(max_rows) is not int or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")
    banks: dict[str, list[Variant]] = {}
    for name, options in variants.banks.items():
        if not options:
            raise ValueError(f"Empty variant bank: {name}")
        banks[name] = [
            option
            if isinstance(option, Variant)
            else Variant(text=str(option), value=option)
            for option in options
        ]
    registry = tool_registry()
    schemas = tool_declarations(registry)
    schema_hash = digest(schemas)
    rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    group_splits: dict[str, str] = {}
    family_ids: set[str] = set()
    labels: dict[str, tuple[str, str]] = {}
    exact_inputs: set[str] = set()
    family_counts: dict[str, dict[str, int]] = {}
    duplicates = 0
    for family in templates.families:
        if family.id in family_ids:
            raise ValueError(f"Duplicate family ID: {family.id}")
        family_ids.add(family.id)
        owner = group_splits.setdefault(family.group, family.split)
        if owner != family.split:
            raise ValueError(f"Group leaks across splits: {family.group}")
        slot_names = sorted(family.slots)
        for name in slot_names:
            if not name.isidentifier() or family.slots[name] not in banks:
                raise ValueError(f"Unknown bank or invalid slot in {family.id}: {name}")
        target_fields = set().union(
            *(
                fields(value)
                for call in family.calls
                for value in call.arguments.values()
                if isinstance(value, str)
            )
        )
        if family.response is not None:
            target_fields |= fields(family.response)
        for utterance in family.utterances:
            names = fields(utterance)
            if names - family.slots.keys() or target_fields - names:
                raise ValueError(
                    f"Unknown or ungrounded placeholders in {family.id}: {utterance}"
                )
            if family.slots.keys() - names:
                raise ValueError(
                    f"Unused slots in {family.id}: {family.slots.keys() - names}"
                )
        axes = [banks[family.slots[name]] for name in slot_names]
        combinations = math.prod(len(axis) for axis in axes)
        total = len(family.utterances) * combinations
        if total > 1_000_000:
            raise ValueError(f"Family Cartesian space exceeds one million: {family.id}")
        limit = total if max_per_family == 0 else min(total, max_per_family)
        rng = random.Random(digest([seed, family.id]))
        selected = (
            range(total) if limit == total else sorted(rng.sample(range(total), limit))
        )
        counts = {
            "cartesian_commands": total,
            "sampled_commands": limit,
            "written_rows": 0,
        }
        family_counts[family.id] = counts
        for index in selected:
            template_index, remainder = divmod(index, combinations)
            chosen: dict[str, Variant] = {}
            for name, axis in reversed(list(zip(slot_names, axes, strict=True))):
                remainder, option_index = divmod(remainder, len(axis))
                chosen[name] = axis[option_index]
            text = family.utterances[template_index].format_map(
                {name: option.text for name, option in chosen.items()}
            )
            clean_text = family.utterances[template_index].format_map(
                {
                    name: option.clean if option.clean is not None else option.text
                    for name, option in chosen.items()
                }
            )
            asr_errors = [
                {
                    "slot": name,
                    "clean": option.clean,
                    "heard": option.text,
                    "kind": option.error,
                }
                for name, option in chosen.items()
                if option.error is not None
            ]
            if not text.strip():
                raise ValueError(f"Empty generated command in {family.id}")
            if "<|" in text or "<|" in clean_text:
                raise ValueError("Source commands cannot contain chat protocol tokens")
            calls = [
                ToolCall(
                    call.name,
                    {
                        key: expand_value(value, chosen)
                        for key, value in call.arguments.items()
                    },
                )
                for call in family.calls
            ]
            validated = registry.validate(calls)
            validate_action_batch(calls)
            for call, arguments, call_template in zip(
                calls, validated, family.calls, strict=True
            ):
                if any(
                    isinstance(value, str) and "<|" in value
                    for value in call.arguments.values()
                ):
                    raise ValueError("Tool labels cannot contain chat protocol tokens")
                if (
                    arguments.model_dump(mode="json", exclude_unset=True)
                    != call.arguments
                ):
                    raise ValueError(
                        f"Noncanonical target arguments in {family.id}: {call}"
                    )
                if call.name == "play_music" and not any(
                    isinstance(value, str) and bool(value.strip())
                    for key in ("title", "artist")
                    for value in [call.arguments.get(key)]
                ):
                    raise ValueError(
                        "Generic music starts must target resume_music, not empty play_music"
                    )
                if call.name == "play_music":
                    for key in ("title", "artist"):
                        value = call.arguments.get(key)
                        if value and (
                            key not in chosen
                            or call_template.arguments.get(key) != "{" + key + "}"
                        ):
                            raise ValueError(
                                f"Music entity target must bind its named role slot: {key}"
                            )
                        if value and (not isinstance(value, str) or value not in text):
                            raise ValueError(
                                f"Music entity must be copied exactly from the request: {key}"
                            )
            assistant: dict[str, Any] = {"role": "assistant", "content": ""}
            if calls:
                assistant["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                    for call in calls
                ]
            else:
                assert family.response is not None
                response = expand_value(family.response, chosen)
                if not isinstance(response, str) or not response.strip():
                    raise ValueError(f"Invalid no-tool response in {family.id}")
                if "<|" in response:
                    raise ValueError(
                        "Response text cannot contain chat protocol tokens"
                    )
                assistant["content"] = response
            label_hash = digest(assistant)
            for surface in (text, clean_text):
                previous = labels.setdefault(
                    normalized_input(surface), (family.split, label_hash)
                )
                if previous[0] != family.split:
                    raise ValueError(
                        f"Normalized clean/heard input leaks across splits in {family.id}: {surface!r}"
                    )
                if previous[1] != label_hash:
                    raise ValueError(
                        f"Contradictory targets in {family.id}: {surface!r}"
                    )
            if text in exact_inputs:
                duplicates += 1
                continue
            exact_inputs.add(text)
            entities = {
                key: surface_language(value)
                for call in calls
                if call.name == "play_music"
                for key, value in call.arguments.items()
                if isinstance(value, str) and value
            }
            for state in ("on", "off"):
                if sum(len(items) for items in rows.values()) >= max_rows:
                    raise ValueError(
                        "Dataset exceeds max_rows; reduce expansion or explicitly raise the limit"
                    )
                messages = [
                    {
                        "role": "system",
                        "content": build_system_prompt(
                            {"status": "observed", "state": state}
                        ),
                    },
                    {"role": "user", "content": text},
                    deepcopy(assistant),
                ]
                rows[family.split].append(
                    {
                        "id": digest([messages, schema_hash])[:24],
                        "family": family.id,
                        "group": family.group,
                        "command_language": family.language,
                        "surface_language": surface_language(text),
                        "entity_languages": entities,
                        "clean_user": clean_text,
                        "asr_errors": asr_errors,
                        "music_state": state,
                        "messages": messages,
                        "tools": schemas,
                    }
                )
                counts["written_rows"] += 1
    coverage: dict[str, Any] = {}
    for split, items in rows.items():
        random.Random(digest([seed, split])).shuffle(items)
        coverage[split] = {
            "rows": len(items),
            "tools": dict(
                sorted(
                    Counter(
                        call["function"]["name"]
                        for row in items
                        for call in row["messages"][-1].get("tool_calls", [])
                    ).items()
                )
            ),
            "no_tool_rows": sum(
                not row["messages"][-1].get("tool_calls") for row in items
            ),
            "asr_rows": sum(bool(row["asr_errors"]) for row in items),
            "asr_error_kinds": dict(
                sorted(
                    Counter(
                        error["kind"] for row in items for error in row["asr_errors"]
                    ).items()
                )
            ),
            "batch_shapes": dict(
                sorted(
                    Counter(
                        "+".join(
                            call["function"]["name"]
                            for call in row["messages"][-1].get("tool_calls", [])
                        )
                        or "no_tool"
                        for row in items
                    ).items()
                )
            ),
            "command_languages": dict(
                sorted(Counter(row["command_language"] for row in items).items())
            ),
            "surface_languages": dict(
                sorted(Counter(row["surface_language"] for row in items).items())
            ),
            "entity_compositions": dict(
                sorted(
                    Counter(
                        json.dumps(row["entity_languages"], sort_keys=True)
                        for row in items
                        if row["entity_languages"]
                    ).items()
                )
            ),
        }
    return Dataset(
        rows,
        {
            "version": 1,
            "format": "lfm_conversational_tool_routing",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "seed": seed,
            "max_per_family": max_per_family,
            "max_rows": max_rows,
            "music_states": ["on", "off"],
            "schema_sha256": schema_hash,
            "prompt_sha256": digest(
                [
                    build_system_prompt({"status": "observed", "state": state})
                    for state in ("on", "off")
                ]
            ),
            "variants_sha256": digest(variants.model_dump()),
            "templates_sha256": digest(templates.model_dump()),
            "chat_template_sha256": hashlib.sha256(
                TEMPLATE_PATH.read_bytes()
            ).hexdigest(),
            "duplicate_commands_removed": duplicates,
            "families": family_counts,
            "groups": group_splits,
            "coverage": coverage,
        },
    )


def write_dataset(dataset: Dataset, output: Path) -> None:
    """Stage complete JSONL files, replace generated outputs, and publish the manifest last.

    Args:
        dataset:
            Fully validated dataset; no output is published before validation succeeds.

        output:
            Output directory, created as needed. Only train/validation/test JSONL
            and manifest.json are replaced; unrelated files remain untouched.

    """
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".build-", dir=output) as temporary:
        staging = Path(temporary)
        hashes: dict[str, str] = {}
        for split, rows in dataset.rows.items():
            path = staging / f"{split}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(
                        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                    )
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {**dataset.manifest, "files": hashes}
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for split in SPLITS:
            (staging / f"{split}.jsonl").replace(output / f"{split}.jsonl")
        (staging / "manifest.json").replace(output / "manifest.json")


def main() -> None:
    """Build all configured splits and log coverage without loading any model or client."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=Path, default=ROOT / "variants.json")
    parser.add_argument("--templates", type=Path, default=ROOT / "templates.json")
    parser.add_argument("--output", type=Path, default=ROOT / "generated")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument(
        "--max-per-family",
        type=int,
        default=128,
        help="Sampled commands before on/off expansion; 0 expands all",
    )
    args = parser.parse_args()
    configure_logging(log_file=Path(".cache/hoast/diagnostics/finetune-generate.log"))
    try:
        variants = VariantFile.model_validate_json(
            args.variants.read_text(encoding="utf-8")
        )
        templates = TemplateFile.model_validate_json(
            args.templates.read_text(encoding="utf-8")
        )
        dataset = build_dataset(
            variants,
            templates,
            seed=args.seed,
            max_per_family=args.max_per_family,
            max_rows=args.max_rows,
        )
        write_dataset(dataset, args.output)
        for split, coverage in dataset.manifest["coverage"].items():
            logger.info(
                "dataset split=%s rows=%d asr_rows=%d tools=%s",
                split,
                coverage["rows"],
                coverage["asr_rows"],
                coverage["tools"],
            )
        logger.info("dataset status=ready output=%s", args.output)
    except Exception:
        logger.exception("dataset status=failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
