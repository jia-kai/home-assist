"""Evaluate routing without tool execution, one backend per process.

Example after coordinating exclusive GPU access::

    python -m tools.evaluate_lfm_gpu --corpus /tmp/ttc/1000/lfm-current-corpus \
        --manifest docs/lfm-routing-benchmark-manifest.json --split dev \
        --backend custom --checkpoint models/openvino-int8 --device 'UHD Graphics 630' \
        --gpu-coordinated --output custom.jsonl

Run ``openvino-gpu`` or ``openvino-cpu`` separately with the same checkpoint and
compare using ``--baseline custom.jsonl``. ``--backend replay --predictions FILE``
accepts JSONL records containing exactly ``id``, ``raw`` and integer ``tokens``
in selected order, for an external runner such as C++; replay does not establish
that the external runner used the same prompt, artifacts, or generation policy.
The dev selection is validation-only; ``--split test`` requires a separately
frozen test-selection manifest and is intended only after choices are frozen.
No network or tool dispatch occurs.
Output is exclusive-created, flushed JSONL, including exceptions on failure.

The custom backend accepts explicit workspace tuning flags; all settings,
device and driver are recorded. They make no performance or parity claim.
--prefix-cache snapshots the exact common token prefix of the selected workload;
the first matching request includes its construction. Compare only matched
checkpoint hashes and benchmark settings. Coordinate GPU use.
"""

import argparse
import hashlib
import json
import logging
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from finetune.evaluate import canonical, score, summarize
from finetune.generate import digest
from finetune.tooling import tool_registry
from hoast.input_text import canonical_user_messages
from hoast.lfm import LFM2, LFMConfig, parse_response, tool_declarations
from hoast.llm import GeneratedCallError
from tools.benchmark_lfm_gpu import choose_device, common_prefix, sha256

LOGGER = logging.getLogger(__name__)
POLICY = {
    "max_new_tokens": 128,
    "max_context_tokens": 4096,
    "repetition_penalty": 1.05,
    "sampling": False,
}
ARTIFACTS = (
    "openvino_model.xml",
    "openvino_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "config.json",
)
SOURCE_ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILES = (
    "tools/evaluate_lfm_gpu.py",
    "finetune/evaluate.py",
    "finetune/generate.py",
    "finetune/tooling.py",
    "hoast/input_text.py",
    "hoast/agent.py",
    "hoast/llm.py",
    "hoast/lfm.py",
    "hoast/prompts.py",
    "hoast/lights.py",
    "hoast/music.py",
    "hoast/weather.py",
    "hoast/lfm_gpu/generation.py",
    "hoast/lfm_gpu/runner.py",
    "hoast/lfm_gpu/weights.py",
    "hoast/lfm_gpu/lfm_gpu_kernels.cl",
    "hoast/lfm_gpu/lfm_gpu_ops.cl",
    "hoast/lfm_gpu/lfm_gpu_runner.cl",
    "tools/benchmark_lfm_gpu.py",
)


def source_hashes() -> dict[str, str]:
    """Fingerprint evaluator, parsing, rendering and GPU runtime sources."""
    return {name: sha256(SOURCE_ROOT / name) for name in SOURCE_FILES}


def test_selection_ids(rows: list[dict[str, Any]]) -> list[str]:
    """Select at most two test rows per label, state, language and ASR stratum.

    Within each stratum, rank SHA-256 of ``lfm-final-test-v1:<row-id>``;
    return selected IDs sorted lexicographically. Selection uses labels and
    source metadata only, never model predictions.

    Args:
        rows:
            Verified test rows with label and coverage metadata.
    """
    groups: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for row in rows:
        shape = (
            "+".join(
                call["function"]["name"]
                for call in row["messages"][-1].get("tool_calls", [])
            )
            or "no_tool"
        )
        key = (
            shape,
            row["music_state"],
            row["command_language"],
            "noisy" if row["asr_errors"] else "clean",
        )
        groups[key].append(row["id"])
    ids = []
    for members in groups.values():
        ids.extend(
            sorted(
                members,
                key=lambda item: hashlib.sha256(
                    ("lfm-final-test-v1:" + item).encode("utf-8")
                ).hexdigest(),
            )[:2]
        )
    return sorted(ids)


def load_workload(
    corpus: Path, manifest_path: Path, split: str, test_selection: Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify a manifest-pinned current-schema workload and its selection.

    Args:
        corpus:
            Directory containing the generated split files.
        manifest_path:
            Durable benchmark manifest containing source file hashes and dev IDs.
        split:
            Either dev (selected validation IDs) or test (selected test IDs).
        test_selection:
            Required frozen selection manifest for test; forbidden for dev.
    """
    if (split == "test") != (test_selection is not None):
        raise ValueError("test requires a selection manifest; dev must not use one")
    if split not in ("dev", "test"):
        raise ValueError("split must be dev or test")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest["source_manifest"]
    schema = digest(tool_declarations(tool_registry()))
    if schema != manifest["schema_sha256"] or schema != source["schema_sha256"]:
        raise ValueError("Benchmark schema does not match current tool declarations")
    name = "validation.jsonl" if split == "dev" else "test.jsonl"
    path = corpus / name
    if sha256(path) != source["files"][name]:
        raise ValueError(f"Corpus hash mismatch: {name}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    indexed = {row["id"]: row for row in rows}
    if (
        len(indexed) != len(rows)
        or len(rows)
        != source["coverage"]["validation" if split == "dev" else "test"]["rows"]
    ):
        raise ValueError("Duplicate IDs or unexpected row count")
    schemas = tool_declarations(tool_registry())
    for row in rows:
        if (
            row["tools"] != schemas
            or row["id"] != digest([row["messages"], schema])[:24]
        ):
            raise ValueError("Row schema or content identity mismatch")
    if split == "dev":
        ids = manifest["fusion_dev"]["ids"]
        if len(ids) != manifest["fusion_dev"]["count"] or len(set(ids)) != len(ids):
            raise ValueError("Invalid selected IDs")
        if any(row_id not in indexed for row_id in ids):
            raise ValueError("Selected ID absent from validation")
        rows = [indexed[row_id] for row_id in ids]
    selection_hash = None
    if split == "test":
        if test_selection is None:
            raise ValueError("Missing test selection manifest")
        selection = json.loads(test_selection.read_text(encoding="utf-8"))
        if (
            selection["version"] != 1
            or selection["algorithm"] != "lfm-final-test-v1"
            or selection["benchmark_manifest_sha256"] != sha256(manifest_path)
            or selection["test_jsonl_sha256"] != source["files"][name]
            or selection["schema_sha256"] != schema
        ):
            raise ValueError("Test selection provenance mismatch")
        ids = selection["ids"]
        if (
            ids != test_selection_ids(rows)
            or selection["count"] != len(ids)
            or selection["ids_sha256"] != digest(ids)
        ):
            raise ValueError("Test selection does not match deterministic algorithm")
        if set(ids) & set(manifest["fusion_dev"]["ids"]):
            raise ValueError("Selected validation and test IDs overlap")
        rows = [indexed[row_id] for row_id in ids]
        selection_hash = sha256(test_selection)
    if not rows:
        raise ValueError("Empty workload")
    identity = {
        "split": split,
        "dataset_sha256": source["files"][name],
        "schema_sha256": schema,
        "ids_sha256": digest([row["id"] for row in rows]),
        "manifest_sha256": sha256(manifest_path),
    }
    if selection_hash is not None:
        identity["selection_manifest_sha256"] = selection_hash
    return rows, identity


def load_replay(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Read exact ordered raw predictions from an external non-dispatching runner.

    Args:
        path:
            JSONL predictions, each with only id, raw and tokens.
        rows:
            Manifest-verified workload in required order.
    """
    values = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if len(values) != len(rows):
        raise ValueError("Replay count mismatch")
    for value, row in zip(values, rows, strict=True):
        if (
            set(value) != {"id", "raw", "tokens"}
            or value["id"] != row["id"]
            or not isinstance(value["raw"], str)
            or type(value["tokens"]) is not int
            or not 1 <= value["tokens"] <= 128
        ):
            raise ValueError("Invalid or misaligned replay prediction")
    return values


def evaluate_rows(
    rows: list[dict[str, Any]],
    generate: Callable[[dict[str, Any]], tuple[str, int]],
    emit: Callable[[dict[str, Any]], None],
) -> list[dict[str, Any]]:
    """Score ordered model outputs using production validation and emit each result.

    Args:
        rows:
            Verified labeled examples, in workload order.
        generate:
            Non-dispatching function returning raw protocol text and output count.
        emit:
            Flushed JSON-compatible record sink.
    """
    results = []
    for row in rows:
        started = time.perf_counter()
        raw, count = generate(row)
        result = score(row, raw, count)
        result["wall_seconds"] = time.perf_counter() - started
        results.append(result)
        emit({"stage": "prediction", **result})
    return results


def safety_summary(results: list[dict[str, Any]]) -> dict[str, int]:
    """Count parsed attempted calls and malformed responses on no-tool labels.

    Parsed calls are counted even when argument or action-batch validation fails.
    Unparseable responses are separate because their intended calls are unknown.
    This does not imply tool execution or measure real-world harm.

    Args:
        results:
            Ordered scored outputs including raw response and expected calls.
    """
    counts: Counter[str] = Counter()
    for result in results:
        if result["expected"]:
            continue
        counts["no_tool_cases"] += 1
        try:
            _, calls = parse_response(result["raw"])
        except ValueError:
            counts["no_tool_unparseable"] += 1
            continue
        if calls:
            counts["no_tool_parsed_call_attempts"] += 1
            if result["error"] is not None:
                counts["no_tool_invalid_call_attempts"] += 1
    return {
        name: counts[name]
        for name in (
            "no_tool_cases",
            "no_tool_unparseable",
            "no_tool_parsed_call_attempts",
            "no_tool_invalid_call_attempts",
        )
    }


def compare_baseline(
    path: Path,
    identity: dict[str, Any],
    results: list[dict[str, Any]],
    artifact_hashes: dict[str, str] | None,
    backend_config: dict[str, Any] | None = None,
    sources: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compare completed baseline records after strict workload and artifact alignment.

    Args:
        path:
            Completed evaluator JSONL result file.
        identity:
            Current workload fingerprint.
        results:
            Ordered current prediction records.
        artifact_hashes:
            Current model artifact hashes, or None for unverified replay.
        backend_config:
            Current backend metadata; compare workspaces and prefix when both have them.
        sources:
            Current source hashes; required to match for live held-out comparisons.
    """
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if (
        not records
        or records[0].get("stage") != "context"
        or records[-1].get("stage") != "complete"
    ):
        raise ValueError("Baseline is incomplete")
    context = records[0]
    if context["workload"] != identity or context["policy"] != POLICY:
        raise ValueError("Baseline workload or generation policy differs")
    if (
        artifact_hashes is not None
        and context["artifacts"] is not None
        and artifact_hashes != context["artifacts"]
    ):
        raise ValueError("Baseline checkpoint artifacts differ")
    if (
        identity["split"] == "test"
        and artifact_hashes is not None
        and context["artifacts"] is not None
        and (sources is None or context.get("source_sha256") != sources)
    ):
        raise ValueError("Held-out baseline source fingerprints differ")
    reference_config = context.get("backend_config", {})
    if (
        backend_config is not None
        and "workspace" in backend_config
        and "workspace" in reference_config
    ):
        for key in ("workspace", "prefix_sha256", "device", "driver"):
            if backend_config.get(key) != reference_config.get(key):
                raise ValueError(f"Baseline backend configuration differs: {key}")
    predictions = [r for r in records if r.get("stage") == "prediction"]
    if len(predictions) != len(results):
        raise ValueError("Baseline prediction count mismatch")
    raw_equal = calls_equal = correctness_equal = 0
    slices: dict[str, Counter[str]] = defaultdict(Counter)
    for baseline, current in zip(predictions, results, strict=True):
        if baseline["id"] != current["id"] or canonical(
            baseline["expected"]
        ) != canonical(current["expected"]):
            raise ValueError("Baseline prediction alignment mismatch")
        raw_equal += baseline["raw"] == current["raw"]
        calls_equal += (
            baseline["error"] is None
            and current["error"] is None
            and canonical(baseline["actual"]) == canonical(current["actual"])
        )
        correctness_equal += baseline["correct"] == current["correct"]
        labels = [
            "language/" + current["command_language"],
            "surface/" + current["surface_language"],
            "state/" + current["music_state"],
            "stt/" + ("noisy" if current["asr_errors"] else "clean"),
        ]
        labels += (
            [
                "tool/" + name
                for name in sorted({call["name"] for call in current["expected"]})
            ]
            if current["expected"]
            else ["no_tool"]
        )
        for label in labels:
            slices[label]["n"] += 1
            slices[label]["raw_equal"] += baseline["raw"] == current["raw"]
            slices[label]["baseline_correct"] += baseline["correct"]
            slices[label]["current_correct"] += current["correct"]
    return {
        "baseline_sha256": sha256(path),
        "count": len(results),
        "raw_equal": raw_equal,
        "validated_calls_equal": calls_equal,
        "correctness_equal": correctness_equal,
        "artifacts_verified": artifact_hashes is not None
        and context["artifacts"] is not None,
        "slices": {name: dict(counts) for name, counts in sorted(slices.items())},
        "baseline_safety": safety_summary(predictions),
        "current_safety": safety_summary(results),
    }


def workload_prefix(rows: list[dict[str, Any]], tokenizer: Any) -> tuple[int, ...]:
    """Derive an exact common prompt-token prefix for a verified workload.

    Args:
        rows:
            Ordered rows with system and user messages and tool declarations.
        tokenizer:
            Checkpoint tokenizer with the native template.
    """
    sequences = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            canonical_user_messages(row["messages"][:-1]),
            tools=row["tools"],
            tokenize=False,
            add_generation_prompt=True,
        )
        sequences.append(
            [
                int(token)
                for token in tokenizer(
                    prompt, add_special_tokens=False, return_tensors="np"
                ).input_ids[0]
            ]
        )
    return common_prefix(sequences)


def backend(
    args: argparse.Namespace, rows: list[dict[str, Any]]
) -> tuple[
    Callable[[dict[str, Any]], tuple[str, int]],
    Callable[[], None],
    dict[str, str],
    dict[str, Any],
]:
    """Load one non-dispatching live backend using the serving generation policy.

    Args:
        args:
            Explicit backend, checkpoint and device options.
        rows:
            Verified ordered workload used to derive an optional exact token prefix.
    """
    checkpoint = args.checkpoint.resolve()
    hashes = {name: sha256(checkpoint / name) for name in ARTIFACTS}
    tools = tool_registry()
    if args.backend.startswith("openvino"):
        cache = tempfile.TemporaryDirectory(prefix="lfm-eval-")
        model = LFM2(
            LFMConfig(
                checkpoint,
                cache_dir=Path(cache.name),
                device="GPU" if args.backend == "openvino-gpu" else "CPU",
            ),
            tools,
        )
        model.load()

        def generate(row: dict[str, Any]) -> tuple[str, int]:
            """Generate the row's user message with its exact saved system prompt.

            Args:
                row:
                    Verified corpus row.
            """
            model.config = replace(
                model.config, system_prompt=row["messages"][0]["content"]
            )
            return invoke(model.generate, row["messages"][1]["content"])

        def close() -> None:
            """Release the model and temporary compiled cache."""
            try:
                model.close()
            finally:
                cache.cleanup()

        return generate, close, hashes, {"device": model.config.device}
    # GPU and tokenizer packages are intentionally lazy for replay/offline tests.
    import pyopencl as cl
    from transformers import PreTrainedTokenizerFast

    from hoast.lfm_gpu.generation import GenerationConfig, GPUGeneration
    from hoast.lfm_gpu.runner import Runner, WorkspaceSpec

    devices = [
        device
        for platform in cl.get_platforms()
        for device in platform.get_devices(device_type=cl.device_type.GPU)
    ]
    device = choose_device(devices, args.device)
    metadata = json.loads((checkpoint / "tokenizer_config.json").read_text())
    tokenizer_class = metadata.pop("tokenizer_class")
    if tokenizer_class == "TokenizersBackend":
        if metadata.pop("backend") != "tokenizers":
            raise ValueError("Unexpected tokenizer backend")
        metadata["additional_special_tokens"] = metadata.pop("extra_special_tokens")
        metadata["extra_special_tokens"] = metadata.pop("model_specific_special_tokens")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(checkpoint / "tokenizer.json"),
            chat_template=(checkpoint / "chat_template.jinja").read_text(),
            **metadata,
        )
    elif tokenizer_class == "PreTrainedTokenizerFast":
        if "backend" in metadata and metadata["backend"] != "tokenizers":
            raise ValueError("Unexpected tokenizer backend")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            checkpoint, local_files_only=True
        )
    else:
        raise ValueError("Unexpected tokenizer class")
    spec = WorkspaceSpec(
        4096,
        args.chunk_size,
        prepared_prefill=args.prepared_prefill,
        prepared_kn=args.prepared_kn,
        vec4_matvec=args.vec4_matvec,
        tiled_attention=args.tiled_attention,
        prefix_cache=args.prefix_cache,
        persistent_kn=args.persistent_kn,
        persistent_kn_budget_bytes=args.persistent_kn_budget_bytes,
    )
    queue = cl.CommandQueue(cl.Context([device]))
    runner = Runner(queue, checkpoint, spec)
    prefix = workload_prefix(rows, tokenizer) if args.prefix_cache else None
    adapter = GPUGeneration(
        runner,
        tokenizer,
        tools,
        GenerationConfig(system_prompt=rows[0]["messages"][0]["content"]),
        prefix,
    )

    def generate_custom(row: dict[str, Any]) -> tuple[str, int]:
        """Generate one custom GPU sequence with the row's system prompt.

        Args:
            row:
                Verified corpus row.
        """
        adapter.config = replace(
            adapter.config, system_prompt=row["messages"][0]["content"]
        )
        return invoke(adapter.generate, row["messages"][1]["content"])

    def close_custom() -> None:
        """Finish queued work."""
        queue.finish()

    metadata = {
        "workspace": asdict(spec),
        "device": device.name,
        "driver": device.driver_version,
        "prefix_tokens": len(prefix) if prefix else 0,
        "prefix_sha256": digest(prefix) if prefix else None,
    }
    return generate_custom, close_custom, hashes, metadata


def invoke(generate: Callable[[str], Any], user: str) -> tuple[str, int]:
    """Preserve repairable output and mark truncation while propagating other failures.

    Args:
        generate:
            Serving-compatible generation method.
        user:
            Canonical corpus user utterance.
    """
    try:
        result = generate(user)
        return result.raw, result.output_tokens
    except GeneratedCallError as error:
        return error.raw, 0
    except ValueError as error:
        if str(error) == "Generation reached output limit":
            return "", 128
        raise


def main() -> None:
    """Evaluate a pinned workload, saving incremental results and failure tracebacks."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument(
        "--test-selection",
        type=Path,
        help="Frozen test selection manifest, required for test",
    )
    parser.add_argument(
        "--backend",
        choices=("custom", "openvino-gpu", "openvino-cpu", "replay"),
        required=True,
    )
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--prepared-prefill", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--prepared-kn", action="store_true")
    parser.add_argument("--vec4-matvec", action="store_true")
    parser.add_argument("--tiled-attention", action="store_true")
    parser.add_argument("--persistent-kn", action="store_true")
    parser.add_argument("--persistent-kn-budget-bytes", type=int, default=0)
    parser.add_argument(
        "--prefix-cache",
        action="store_true",
        help="Cache the exact shared workload token prefix",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--device", help="Unique OpenCL GPU name substring for custom backend"
    )
    parser.add_argument(
        "--gpu-coordinated",
        action="store_true",
        help="Confirm GPU use was coordinated externally",
    )
    parser.add_argument(
        "--predictions", type=Path, help="Raw JSONL predictions for replay"
    )
    parser.add_argument("--baseline", type=Path, help="Completed prior evaluator JSONL")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.split == "test") != (args.test_selection is not None):
        parser.error("--test-selection is required exactly for --split test")
    if args.backend in ("custom", "openvino-gpu") and not args.gpu_coordinated:
        parser.error("GPU runs require --gpu-coordinated after external coordination")
    if args.backend != "replay" and args.checkpoint is None:
        parser.error("live backends require --checkpoint")
    if args.backend == "custom" and not args.device:
        parser.error("custom backend requires --device")
    if (args.backend == "replay") != (args.predictions is not None):
        parser.error("--predictions is required only for replay")
    with args.output.open("x", encoding="utf-8") as stream:

        def emit(record: dict[str, Any]) -> None:
            """Write one durable JSONL record.

            Args:
                record:
                    Serializable progress or result record.
            """
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

        try:
            sources = source_hashes()
            rows, identity = load_workload(
                args.corpus, args.manifest, args.split, args.test_selection
            )
            artifacts = None
            backend_config: dict[str, Any] = {}
            close: Callable[[], None] = lambda: None
            if args.backend == "replay":
                predictions = iter(load_replay(args.predictions, rows))

                def replay_generate(row: dict[str, Any]) -> tuple[str, int]:
                    """Return the next aligned externally generated raw prediction.

                    Args:
                        row:
                            The verified workload row expected at this position.
                    """
                    value = next(predictions)
                    if value["id"] != row["id"]:
                        raise ValueError("Replay mismatch")
                    return value["raw"], value["tokens"]

                generate = replay_generate
            else:
                generate, close, artifacts, backend_config = backend(args, rows)
            try:
                emit(
                    {
                        "stage": "context",
                        "workload": identity,
                        "backend": args.backend,
                        "checkpoint": str(args.checkpoint.resolve())
                        if args.checkpoint
                        else None,
                        "artifacts": artifacts,
                        "source_sha256": sources,
                        "policy": POLICY,
                        "backend_config": backend_config,
                        "replay_sha256": sha256(args.predictions)
                        if args.predictions
                        else None,
                    }
                )
                results = evaluate_rows(rows, generate, emit)
                if source_hashes() != sources:
                    raise RuntimeError(
                        "Evaluator or runner source changed during evaluation"
                    )
                if (
                    artifacts is not None
                    and args.checkpoint is not None
                    and {name: sha256(args.checkpoint / name) for name in ARTIFACTS}
                    != artifacts
                ):
                    raise RuntimeError("Checkpoint changed during evaluation")
                comparison = (
                    compare_baseline(
                        args.baseline,
                        identity,
                        results,
                        artifacts,
                        backend_config,
                        sources,
                    )
                    if args.baseline
                    else None
                )
                emit(
                    {
                        "stage": "complete",
                        "metrics": summarize(results),
                        "safety": safety_summary(results),
                        "comparison": comparison,
                    }
                )
            finally:
                close()
        except Exception:
            emit({"stage": "error", "traceback": traceback.format_exc()})
            LOGGER.exception("Evaluation failed; partial output retained")
            raise


if __name__ == "__main__":
    main()
