"""Single- or multi-GPU completion-only LoRA SFT using Liquid's TRL recipe."""

import argparse
import hashlib
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, set_seed
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from hoast.input_text import canonicalize_text

from . import TEMPLATE_PATH

logger = logging.getLogger(__name__)
MODEL = "LiquidAI/LFM2.5-350M"
REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"


def training_data(path: Path, tokenizer: Any) -> Dataset:
    """Render JSONL without columnar coercion and audit completion token boundaries.

    Args:
        path:
            Frozen train or validation JSONL, never the held-out test file.

        tokenizer:
            Native checkpoint tokenizer with its unmodified chat template.

    """
    examples: list[dict[str, Any]] = []
    lengths: list[int] = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        for message in row["messages"]:
            if message["role"] == "user" and message["content"] != canonicalize_text(
                message["content"]
            ):
                raise ValueError(
                    "Training user text must be canonical; regenerate the dataset"
                )
        prompt = tokenizer.apply_chat_template(
            row["messages"][:-1],
            tools=row["tools"],
            tokenize=False,
            add_generation_prompt=True,
        )
        text = tokenizer.apply_chat_template(
            row["messages"],
            tools=row["tools"],
            tokenize=False,
            add_generation_prompt=False,
        )
        assert text.startswith(prompt)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        assert ids[: len(prefix)] == prefix and len(ids) > len(prefix)
        assert ids.count(tokenizer.bos_token_id) == 1
        assert len(ids) <= 4096, "Refuse to truncate supervised examples"
        lengths.append(len(ids))
        examples.append(
            {
                "prompt": prompt,
                "completion": text[len(prompt) :],
                "input_ids": ids,
                "completion_mask": [0] * len(prefix) + [1] * (len(ids) - len(prefix)),
            }
        )
    logger.info("dataset=%s rows=%d max_tokens=%d", path, len(examples), max(lengths))
    return Dataset.from_list(examples)


def audit_masks(dataset: Any, collator: Callable[..., Any], tokenizer: Any) -> None:
    """Check real TRL labels against rendered prompt boundaries and padded targets.

    Args:
        dataset:
            Tokenized SFTTrainer dataset retaining prompt/completion strings.

        collator:
            Trainer's actual completion-only collator, returning padded tensors.

        tokenizer:
            Native tokenizer with the configured EOS/padding token.

    """
    records = [dataset[index] for index in range(min(64, len(dataset)))]
    batch = collator(records)
    assert len({len(record["input_ids"]) for record in records}) > 1
    for index, record in enumerate(records):
        prefix = tokenizer(record["prompt"], add_special_tokens=False)["input_ids"]
        ids = record["input_ids"]
        assert ids[: len(prefix)] == prefix
        assert (
            ids
            == tokenizer(
                record["prompt"] + record["completion"], add_special_tokens=False
            )["input_ids"]
        )
        assert tokenizer.eos_token_id in ids[len(prefix) :]
        expected = [-100] * len(prefix) + ids[len(prefix) :]
        expected += [-100] * (batch["labels"].shape[1] - len(ids))
        assert batch["labels"][index].tolist() == expected
        assert ids.count(tokenizer.bos_token_id) == 1
    logger.info(
        "mask_audit status=ok examples=%d prompt=masked padding=masked eos=supervised",
        len(records),
    )


def main() -> None:
    """Train on rank-local GPUs, select by validation loss, and save merged HF weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("finetune/generated"))
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.output / f"rank-{rank}.log"),
            logging.StreamHandler(),
        ],
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size < 1:
            raise ValueError("WORLD_SIZE must be positive")
        torch.cuda.set_device(rank)
        torch.set_num_threads(4)
        set_seed(42)
        revision = REVISION
        logger.info(
            "rank=%d world_size=%d device=%s revision=%s",
            rank,
            world_size,
            torch.cuda.get_device_name(rank),
            revision,
        )
        # Upstream names the generic fast backend using Transformers 5 terminology.
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            MODEL,
            revision=revision,
            extra_special_tokens={},
        )
        assert tokenizer.chat_template == TEMPLATE_PATH.read_text()
        tokenizer.pad_token = tokenizer.eos_token
        manifest = json.loads((args.data / "manifest.json").read_text())
        for name in ("train.jsonl", "validation.jsonl"):
            assert (
                hashlib.sha256((args.data / name).read_bytes()).hexdigest()
                == manifest["files"][name]
            )
        train = training_data(args.data / "train.jsonl", tokenizer)
        validation = training_data(args.data / "validation.jsonl", tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            revision=revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model.config.use_cache = False
        assert model.config.rope_theta == model.config.rope_parameters["rope_theta"]
        targets = ["q_proj", "k_proj", "v_proj"]
        module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
        assert set(targets) <= module_names
        config = SFTConfig(
            output_dir=str(args.output / "checkpoints"),
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            bf16=True,
            max_length=4096,
            completion_only_loss=True,
            dataset_kwargs={"skip_prepare_dataset": True},
            packing=False,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            logging_steps=10,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            seed=42,
            data_seed=42,
            report_to="none",
            ddp_find_unused_parameters=False,
            max_steps=args.max_steps,
            dataloader_num_workers=0,
        )
        trainer = SFTTrainer(
            model=model,
            args=config,
            train_dataset=train,
            eval_dataset=validation,
            processing_class=tokenizer,
            peft_config=LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=targets,
                task_type="CAUSAL_LM",
            ),
        )
        assert trainer.train_dataset is not None
        audit_masks(trainer.train_dataset, trainer.data_collator, tokenizer)
        batch = trainer.data_collator([trainer.train_dataset[0]])
        mask = batch["labels"][0] != -100
        assert mask.any() and not mask[0]
        logger.info(
            "mask supervised_tokens=%d total_tokens=%d", mask.sum(), mask.numel()
        )
        result = trainer.train()
        trainer.save_model(str(args.output / "adapter"))
        trainer.accelerator.wait_for_everyone()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(args.output / "adapter")
            provenance = {
                "model": MODEL,
                "revision": revision,
                "world_size": world_size,
                "effective_batch_size": (
                    config.per_device_train_batch_size
                    * config.gradient_accumulation_steps
                    * world_size
                ),
                "seed": 42,
                "manifest_sha256": hashlib.sha256(
                    (args.data / "manifest.json").read_bytes()
                ).hexdigest(),
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_validation_loss": trainer.state.best_metric,
                "metrics": result.metrics,
                "history": trainer.state.log_history,
            }
            (args.output / "training.json").write_text(json.dumps(provenance, indent=2))
            # Reload from disk on CPU, proving the adapter is independently usable.
            base = AutoModelForCausalLM.from_pretrained(
                MODEL, revision=revision, dtype=torch.float32
            )
            adapted: Any = PeftModel.from_pretrained(
                base, args.output / "adapter"
            ).eval()
            probe = tokenizer("Hello", return_tensors="pt")
            with torch.inference_mode():
                before = adapted(**probe).logits
                merged = adapted.merge_and_unload().eval()
                after = merged(**probe).logits
            torch.testing.assert_close(before, after, atol=2e-4, rtol=2e-4)
            merged.config.use_cache = True
            merged.save_pretrained(args.output / "merged", safe_serialization=True)
            tokenizer.save_pretrained(args.output / "merged")
            logger.info(
                "merge status=ok max_logit_delta=%g", (before - after).abs().max()
            )
        trainer.accelerator.wait_for_everyone()
    except Exception:
        logger.exception("training status=failed rank=%d", rank)
        raise


if __name__ == "__main__":
    main()
