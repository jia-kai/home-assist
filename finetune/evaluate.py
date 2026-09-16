"""Evaluate native tool routing without executing tools; retain every prediction."""

import argparse
import hashlib
import json
import logging
import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerFast

from hoast.agent import validate_action_batch
from hoast.input_text import canonical_user_messages
from hoast.lfm import parse_response

from .tooling import tool_registry

logger = logging.getLogger(__name__)
REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"


def canonical(value: Any) -> str:
    """Serialize structured calls with order and JSON scalar types preserved.

    Args:
        value:
            JSON-compatible ordered calls or metadata.

    """
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def score(row: dict[str, Any], raw: str, count: int) -> dict[str, Any]:
    """Check production syntax/schemas/batch constraints and exact target arguments.

    Args:
        row:
            Held-out record including expected assistant calls and slice metadata.

        raw:
            Generated completion with native special tokens retained.

        count:
            Generated token count including EOS; 128 means output-limit failure.

    """
    expected = [call["function"] for call in row["messages"][-1].get("tool_calls", [])]
    actual: list[dict[str, Any]] = []
    error: str | None = None
    try:
        if count >= 128:
            raise ValueError("Generation reached output limit")
        text, calls = parse_response(raw)
        validated = tool_registry().validate(calls)
        validate_action_batch(calls)
        if not calls and not text:
            raise ValueError("Empty response is not a valid abstention")
        actual = [
            {
                "name": call.name,
                "arguments": (
                    {**call.arguments, "level": arguments.model_dump()["level"]}
                    if call.name == "volume_music"
                    else call.arguments
                ),
            }
            for call, arguments in zip(calls, validated, strict=True)
        ]
    except (ValueError, RuntimeError) as exc:
        error = str(exc)
        logger.debug("invalid id=%s reason=%s raw=%r", row["id"], error, raw)
    return {
        "id": row["id"],
        "user": row["messages"][-2]["content"],
        "expected": expected,
        "actual": actual,
        "raw": raw,
        "error": error,
        "correct": error is None and canonical(expected) == canonical(actual),
        "tokens": count,
        **(
            {
                "speech_status": row["speech"]["status"],
                "speech_training_eligible": row["speech"]["training_eligible"],
                "speech_review_reasons": row["speech"]["review_reasons"],
            }
            if "speech" in row
            else {}
        ),
        **{
            key: row[key]
            for key in (
                "family",
                "command_language",
                "surface_language",
                "entity_languages",
                "asr_errors",
                "music_state",
            )
        },
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute exact-match accuracy and slice counts, including false tool activation.

    Args:
        results:
            One scored prediction per held-out row.

    """
    groups: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        labels = [
            "overall",
            "language/" + result["command_language"],
            "surface/" + result["surface_language"],
            "entities/" + canonical(result["entity_languages"]),
            "stt/" + ("noisy" if result["asr_errors"] else "clean"),
            "state/" + result["music_state"],
        ]
        labels += [
            "tool/" + name for name in sorted({c["name"] for c in result["expected"]})
        ]
        if not result["expected"]:
            labels.append("no_tool")
        if "speech_status" in result:
            labels.append("speech/" + result["speech_status"])
            labels.append(
                "speech_audit/"
                + (
                    "eligible"
                    if result["speech_training_eligible"]
                    else "review_required"
                )
            )
        for label in labels:
            groups[label].append(result["correct"])
    return {
        "slices": {
            key: {
                "n": len(values),
                "correct": sum(values),
                "accuracy": sum(values) / len(values),
            }
            for key, values in sorted(groups.items())
        },
        "invalid": sum(result["error"] is not None for result in results),
        "false_tool_activations": sum(
            not result["expected"] and bool(result["actual"]) for result in results
        ),
        "speech_input_failures": sum(
            result.get("speech_status") in ("empty", "tts_rejected", "invalid_input")
            for result in results
        ),
    }


def backend(
    args: argparse.Namespace, tokenizer: Any
) -> tuple[Callable[[list[str]], list[tuple[str, int]]], dict[str, Any]]:
    """Load one inference backend and return a prompt-batch generator plus metadata.

    Args:
        args:
            CLI options identifying checkpoint, device backend, and CPU threads.

        tokenizer:
            Checkpoint's native fast tokenizer used for all rendering and decoding.

    """
    if args.backend == "openvino":
        # Deployment dependencies are optional in the isolated CUDA environment.
        import openvino as ov
        import openvino_genai

        graph = ov.Core().read_model(str(Path(args.model) / "openvino_model.xml"))
        constants: Counter[str] = Counter()
        for op in graph.get_ops():
            if op.get_type_name() == "Constant":
                constants[op.get_output_element_type(0).get_type_name()] += math.prod(
                    op.get_output_shape(0)
                )
        assert constants["i8"] + constants["u8"] > 100_000_000, (
            "Missing INT8 compressed weights"
        )
        pipeline = openvino_genai.LLMPipeline(
            args.model,
            "CPU",
            INFERENCE_NUM_THREADS=args.threads,
            PERFORMANCE_HINT="LATENCY",
        )

        def generate(prompts: list[str]) -> list[tuple[str, int]]:
            """Generate serial OpenVINO completions, retaining protocol tokens.

            Args:
                prompts:
                    Fully rendered native prompts without extra special tokens.

            """
            results: list[tuple[str, int]] = []
            for prompt in prompts:
                inputs = tokenizer(
                    prompt, add_special_tokens=False, return_tensors="np"
                )
                output = pipeline.generate(
                    ov.Tensor(inputs.input_ids),
                    do_sample=False,
                    max_new_tokens=128,
                    repetition_penalty=1.0,
                )
                ids = list(output.tokens[0])
                results.append(
                    (tokenizer.decode(ids, skip_special_tokens=False), len(ids))
                )
            return results

        return generate, {
            "constant_elements": dict(constants),
            "device": "CPU",
            "threads": args.threads,
        }

    # CUDA dependencies are optional for the OpenVINO-only evaluation path.
    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(args.threads)
    model: Any = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=REVISION if not Path(args.model).exists() else None,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = model.to("cuda").eval()
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    def generate(prompts: list[str]) -> list[tuple[str, int]]:
        """Generate batched CUDA completions and strip only post-EOS padding.

        Args:
            prompts:
                Native prompts padded on the left for causal generation.

        """
        inputs = tokenizer(
            prompts, add_special_tokens=False, padding=True, return_tensors="pt"
        ).to("cuda")
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=128,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        results: list[tuple[str, int]] = []
        for sequence in output[:, inputs.input_ids.shape[1] :].tolist():
            if tokenizer.eos_token_id in sequence:
                sequence = sequence[: sequence.index(tokenizer.eos_token_id) + 1]
            results.append(
                (tokenizer.decode(sequence, skip_special_tokens=False), len(sequence))
            )
        return results

    return generate, {"device": torch.cuda.get_device_name(), "dtype": "bfloat16"}


def main() -> None:
    """Write exhaustive predictions, slice metrics, and warmed end-to-end throughput."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=["cuda", "openvino"], default="cuda")
    parser.add_argument(
        "--data", type=Path, default=Path("finetune/generated/test.jsonl")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.output / "evaluation.log"),
            logging.StreamHandler(),
        ],
    )
    try:
        if not 0 <= args.shard_index < args.num_shards:
            raise ValueError("shard-index must be within a positive num-shards")
        rows = [json.loads(line) for line in args.data.read_text().splitlines()]
        rows = rows[args.shard_index :: args.num_shards]
        if not rows:
            raise ValueError("Evaluation shard is empty")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            args.model,
            revision=REVISION if not Path(args.model).exists() else "main",
            extra_special_tokens={},
        )
        prompts: list[str | None] = []
        for row in rows:
            if row.get("speech", {}).get("status") in (
                "empty",
                "tts_rejected",
                "invalid_input",
            ):
                prompts.append(None)
            else:
                prompts.append(
                    tokenizer.apply_chat_template(
                        canonical_user_messages(row["messages"][:-1]),
                        tools=row["tools"],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
        generate, metadata = backend(args, tokenizer)
        warmup = next((prompt for prompt in prompts if prompt is not None), None)
        if warmup is not None:
            generate([warmup])
        results: list[dict[str, Any]] = []
        start = time.perf_counter()
        started_unix = time.time()
        with (args.output / "predictions.jsonl").open("w") as output:
            for offset in range(0, len(rows), args.batch_size):
                batch_prompts = prompts[offset : offset + args.batch_size]
                valid_prompts = [
                    prompt for prompt in batch_prompts if prompt is not None
                ]
                predictions = iter(generate(valid_prompts) if valid_prompts else [])
                for row, prompt in zip(
                    rows[offset : offset + args.batch_size], batch_prompts, strict=True
                ):
                    raw, count = ("", 0) if prompt is None else next(predictions)
                    result = score(row, raw, count)
                    if prompt is None:
                        result["error"] = (
                            "Speech input unavailable; inference not attempted"
                        )
                    results.append(result)
                    output.write(canonical(result) + "\n")
                output.flush()
                logger.info("evaluated=%d/%d", len(results), len(rows))
        elapsed = time.perf_counter() - start
        metrics = {
            **summarize(results),
            **metadata,
            "model": args.model,
            "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "seconds": elapsed,
            "examples_per_second": len(rows) / elapsed,
            "output_tokens_per_second": sum(r["tokens"] for r in results) / elapsed,
            "batch_size": args.batch_size,
            "max_new_tokens": 128,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "generation_started_unix": started_unix,
            "generation_finished_unix": time.time(),
        }
        (args.output / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False)
        )
        logger.info("evaluation status=ok accuracy=%s", metrics["slices"]["overall"])
    except Exception:
        logger.exception("evaluation status=failed")
        raise


if __name__ == "__main__":
    main()
