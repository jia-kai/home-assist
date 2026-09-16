"""Preview dataset rows through the official LFM template without loading a tokenizer/model."""

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

from transformers.utils.chat_template_utils import render_jinja_template

from hoast.logging import configure_logging, get_logger

from . import TEMPLATE_PATH

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class RenderedExample:
    """Native chat text and assistant generation regions before tokenization."""

    text: str
    """Rendered conversation, including official role/tool tokens and exactly one tool list."""

    assistant_spans: tuple[tuple[int, int], ...]
    """Half-open character offsets of assistant supervision spans, not token indices."""


def render_example(
    row: dict[str, Any], *, generation_prompt: bool = False
) -> RenderedExample:
    """Render structured messages/tools using Transformers' official Jinja machinery.

    Args:
        row:
            Dataset record with messages and native LFM tool declarations. Argument
            values in assistant tool calls must be mappings, not JSON strings.

        generation_prompt:
            Drop the supervised final assistant turn and add an empty assistant
            prefix, as production generation does.

    """
    messages = row["messages"][:-1] if generation_prompt else row["messages"]
    # Transformers' installed annotation says str, but this API returns text/spans lists.
    result: Any = render_jinja_template(
        conversations=[messages],
        tools=row["tools"],
        chat_template=TEMPLATE_PATH.read_text(encoding="utf-8"),
        return_assistant_tokens_mask=True,
        add_generation_prompt=generation_prompt,
        bos_token="<|startoftext|>",
    )
    text = result[0][0]
    assert isinstance(text, str)
    spans = tuple((start, end) for start, end in result[1][0])
    assert all(
        type(start) is int and type(end) is int and 0 <= start <= end <= len(text)
        for start, end in spans
    )
    return RenderedExample(text, spans)


def main() -> None:
    """Print one dataset record's native training text for template inspection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    configure_logging()
    try:
        if args.index < 0:
            raise ValueError("index cannot be negative")
        with args.dataset.open(encoding="utf-8") as source:
            line = next(islice(source, args.index, None), None)
        if line is None:
            raise ValueError("Dataset does not contain that record index")
        result = render_example(json.loads(line))
        sys.stdout.write(result.text)
        logger.info(
            "render status=ok assistant_character_spans=%s", result.assistant_spans
        )
    except Exception:
        logger.exception("render status=failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
