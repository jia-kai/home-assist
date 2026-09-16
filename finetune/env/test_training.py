"""Offline integration regression for TRL special tokens and completion-only labels."""

import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import Lfm2Config, Lfm2ForCausalLM, PreTrainedTokenizerFast
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from finetune.train import audit_masks, training_data


def test_prepared_native_tokens_survive_trl(tmp_path: Path) -> None:
    """Reject duplicate BOS/EOS and prompt/padding loss using a tiny offline model.

    Args:
        tmp_path:
            Isolated pytest directory for embedded dataset and trainer artifacts.

    """
    vocabulary = {
        "[UNK]": 0,
        "<bos>": 1,
        "<eos>": 2,
        "user": 3,
        "assistant": 4,
        "hello": 5,
        "Okay": 6,
        "please": 7,
        "clarify": 8,
    }
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 1)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<eos>",
    )
    tokenizer.chat_template = "{{ bos_token }} {% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% if m['role'] == 'assistant' %}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    path = tmp_path / "train.jsonl"
    rows = [
        {
            "tools": [],
            "messages": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": "Okay"},
            ],
        }
        for content in ("hello", "hello please clarify")
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    data = training_data(path, tokenizer)
    assert tokenizer(data[0]["prompt"]).input_ids[:2] == [1, 1]
    config = Lfm2Config(
        vocab_size=len(vocabulary),
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        layer_types=["conv", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    model: Any = Lfm2ForCausalLM(config)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=data,
        args=SFTConfig(
            output_dir=str(tmp_path / "trainer"),
            use_cpu=True,
            bf16=False,
            completion_only_loss=True,
            dataset_kwargs={"skip_prepare_dataset": True},
            report_to="none",
        ),
    )
    audit_masks(trainer.train_dataset, trainer.data_collator, tokenizer)
