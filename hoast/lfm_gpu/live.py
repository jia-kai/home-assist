"""Lifecycle-managed strict-prefix OpenCL backend for live LFM sessions."""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Self

from ..lfm import LFM2, tool_declarations
from ..llm import Generation, _clear_exception_frames, _reclaim_after_generation
from ..memory import reclaim_memory
from ..prompts import build_system_prompt
from .generation import GenerationConfig, GPUGeneration


def stable_prefix(tokenizer: Any, system_prompt: str, tools: Any) -> tuple[int, ...]:
    """Derive a token prefix covering the complete rendered system and tool definitions.

    Args:
        tokenizer:
            Native tokenizer and chat template used for inference.

        system_prompt:
            Complete system instruction for one fixed playback state.

        tools:
            Registry whose actual declarations are rendered by the template.
    """
    sequences: list[list[int]] = []
    rendered_system: str | None = None
    for user in ("alpha", "zeta", "!", "中"):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
            tools=tool_declarations(tools),
            tokenize=False,
            add_generation_prompt=True,
        )
        marker = "<|im_end|>\n"
        end = prompt.find(marker)
        if end < 0 or "<|im_start|>system\n" not in prompt[:end]:
            raise ValueError("Unexpected LFM system template")
        system = prompt[: end + len(marker)]
        if rendered_system is not None and rendered_system != system:
            raise ValueError("System template depends on user history")
        rendered_system = system
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="np").input_ids
        if ids.ndim != 2 or ids.shape[0] != 1 or not ids.shape[1]:
            raise ValueError(
                "Tokenizer must return one nonempty integer token sequence"
            )
        sequences.append([int(token) for token in ids[0]])
    length = 0
    for values in zip(*sequences):
        if len(set(values)) != 1:
            break
        length += 1
    if not length:
        raise ValueError("Cannot derive a nonempty stable LFM prefix")
    prefix = tuple(sequences[0][:length])
    decoded = tokenizer.decode(
        list(prefix), skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if rendered_system is None or not decoded.startswith(rendered_system):
        raise ValueError("Stable prefix does not cover complete system and tools")
    return prefix


class LiveLFM2(LFM2):
    """Own one OpenCL runner with two explicitly initialized strict prefixes."""

    _live_active: bool = False
    """Whether generation is active, including synchronous callbacks."""

    _base_prompt: str | None = None
    """Base instruction used to construct both snapshots, or None when unloaded."""

    def load(self) -> Self:
        """Load the native tokenizer and exactly one capable OpenCL GPU; never fall back."""
        with self._lock:
            if self._live_active:
                raise RuntimeError("Reentrant live model operation is not allowed")
            if (self._model is None) != (self._base_prompt is None) or (
                (self._model is None) != (self._tokenizer is None)
            ):
                raise RuntimeError("Inconsistent live model lifecycle state")
            if self._model is not None:
                return self
            # Optional inference dependencies are loaded only when the model is opened.
            import pyopencl as cl
            from transformers import PreTrainedTokenizerFast

            from .runner import Runner, WorkspaceSpec

            if self.config.device != "GPU":
                raise ValueError("Live OpenCL LFM requires GPU device")
            path = self.config.model_path.resolve()
            if not (path / "openvino_model.xml").is_file():
                raise FileNotFoundError(f"Prepare LFM first: {path}")
            metadata = json.loads((path / "tokenizer_config.json").read_text())
            tokenizer_class = metadata.pop("tokenizer_class")
            if tokenizer_class == "TokenizersBackend":
                if metadata.pop("backend") != "tokenizers":
                    raise ValueError("Unexpected LFM tokenizer backend metadata")
                metadata["additional_special_tokens"] = metadata.pop(
                    "extra_special_tokens"
                )
                metadata["extra_special_tokens"] = metadata.pop(
                    "model_specific_special_tokens"
                )
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=str(path / "tokenizer.json"),
                    chat_template=(path / "chat_template.jinja").read_text(),
                    **metadata,
                )
            elif tokenizer_class == "PreTrainedTokenizerFast":
                if "backend" in metadata and metadata["backend"] != "tokenizers":
                    raise ValueError("Unexpected LFM tokenizer backend metadata")
                tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    path, local_files_only=True
                )
            else:
                raise ValueError("Unexpected LFM tokenizer backend metadata")
            devices = [
                device
                for platform in cl.get_platforms()
                for device in platform.get_devices(device_type=cl.device_type.GPU)
            ]
            if len(devices) != 1:
                raise RuntimeError(
                    f"Expected exactly one OpenCL GPU, found {len(devices)}"
                )
            base_prompt = self.config.system_prompt
            prompts = tuple(
                build_system_prompt({"status": "observed", "state": state}, base_prompt)
                for state in ("on", "off")
            )
            prefixes = tuple(
                stable_prefix(
                    tokenizer,
                    prompt,
                    self.tools,
                )
                for prompt in prompts
            )
            spec = WorkspaceSpec(
                self.config.max_context_tokens,
                min(128, self.config.max_context_tokens),
                prepared_prefill=True,
                prefix_cache=True,
                prefix_slots=2,
            )
            runner = Runner(cl.CommandQueue(cl.Context([devices[0]])), path, spec)
            adapter = GPUGeneration(
                runner,
                tokenizer,
                self.tools,
                replace(self._generation_config(), system_prompt=prompts[1]),
                prefixes[0],
                strict_prefix_cache=True,
                second_prefix=prefixes[1],
            )
            adapter.initialize_prefix()
            self._model, self._tokenizer, self._base_prompt = adapter, tokenizer, base_prompt
            return self

    def close(self) -> None:
        """Release the model, rejecting calls from an active generation callback."""
        with self._lock:
            if self._live_active:
                raise RuntimeError("Reentrant live model operation is not allowed")
            super().close()
            self._base_prompt = None

    def _generation_config(self) -> GenerationConfig:
        """Translate settings, mapping the loaded base prompt to the off snapshot."""
        system_prompt = self.config.system_prompt
        if self._base_prompt is not None and system_prompt == self._base_prompt:
            system_prompt = build_system_prompt(
                {"status": "observed", "state": "off"}, self._base_prompt
            )
        return GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            max_context_tokens=self.config.max_context_tokens,
            max_prompt_chars=self.config.max_prompt_chars,
            max_output_chars=self.config.max_output_chars,
            system_prompt=system_prompt,
            repetition_penalty=self.config.repetition_penalty,
        )

    def generate_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        on_text: Callable[[str], None] | None = None,
    ) -> Generation:
        """Generate from untrusted history, forwarding raw prevalidation text.

        Fragments precede final size and protocol validation; they are not safe
        tool calls. Session filters protocol text before exposing it to consumers.

        Args:
            messages:
                Caller-owned native history without internal repair metadata.

            on_text:
                Optional synchronous raw-text callback; exceptions abort inference.
        """
        return self._run_live(messages, on_text)

    def _generate_repair(
        self,
        messages: Sequence[Mapping[str, Any]],
        feedback: str,
        on_text: Callable[[str], None] | None = None,
    ) -> Generation:
        """Generate with Session-owned feedback, keeping caller history untrusted.

        Args:
            messages:
                Base conversation history without internal metadata.

            feedback:
                Transient diagnostic created by Session.

            on_text:
                Optional synchronous raw-text callback.
        """
        return self._run_live(messages, on_text, feedback)

    def _run_live(
        self,
        messages: Sequence[Mapping[str, Any]],
        on_text: Callable[[str], None] | None,
        feedback: str | None = None,
    ) -> Generation:
        """Serialize inference and cleanup, rejecting callback reentrancy.

        Args:
            messages:
                Caller-owned base conversation history.

            on_text:
                Optional synchronous raw-text callback.

            feedback:
                Session-created diagnostic, or None for a public request.
        """
        with self._lock:
            if self._live_active:
                raise RuntimeError("Reentrant live model operation is not allowed")
            if self._model is None or self._tokenizer is None:
                raise RuntimeError("Load LFM2 before generating")
            if self._base_prompt is None:
                raise RuntimeError("Inconsistent live model lifecycle state")
            self._live_active = True
            failure: BaseException | None = None
            try:
                if self.config.reclaim_after_call:
                    reclaim_memory("lfm.before")
                adapter: GPUGeneration = self._model
                adapter.config = self._generation_config()
                if feedback is not None:
                    return adapter._generate_repair(messages, feedback, on_text)
                return adapter.generate_messages(messages, on_text)
            except BaseException as error:
                failure = error
                if self.config.reclaim_after_call:
                    _clear_exception_frames(error)
                raise
            finally:
                try:
                    if self.config.reclaim_after_call:
                        _reclaim_after_generation("lfm.after", failure, reclaim_memory)
                finally:
                    self._live_active = False
