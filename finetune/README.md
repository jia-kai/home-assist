# Tool-routing fine-tuning data

This directory builds **supervised routing examples**, without loading a model,
reading service credentials, or executing tools. Separate training and evaluation
commands below load real checkpoints without executing tools.
Generation uses the same prompt builder, tool declarations, strict argument
validation, and batch limits as the application.

## Build and inspect

From the repository root:

```sh
uv run python -m finetune.generate
# Sample a larger Cartesian subset per template family:
uv run python -m finetune.generate --seed 42 --max-per-family 256
# Expand every combination, subject to the explicit total-row bound:
uv run python -m finetune.generate --max-per-family 0 --max-rows 100000
# Inspect the native LFM rendering of one record:
uv run python -m finetune.render finetune/generated/train.jsonl --index 0
```

Outputs in the gitignored `finetune/generated/` directory:

- `train.jsonl`, `validation.jsonl`, `test.jsonl`: self-contained examples.
- `manifest.json`: generation settings, source/prompt/schema/template hashes,
  output-file hashes, family/group ownership, and actual coverage counts.

The default seed is 42 and the per-family limit is 128 **base commands**, before
expanding each into music-on and music-off contexts. A limit of zero means complete
expansion. Sampling selects Cartesian indices without materializing the whole
combination space. Each family is capped at one million possible combinations;
the default total materialized-row limit is 100,000. Generated files are staged
only after validation, then replaced, with the manifest published last.

The checked default basic-punctuation build contains 7,510 training rows, 890 validation
rows, and 894 test rows. Of those, 1,360 contain curated STT confusions. Counts are determined
by source files and sampling settings; consult the generated manifest after edits.
Logs and tracebacks go to `.cache/hoast/diagnostics/finetune-generate.log`.

## Source files and labels

| File                       | Purpose                                                     |
| -------------------------- | ----------------------------------------------------------- |
| `variants.json`            | Lexical variants, typed aliases, names, and STT confusions    |
| `templates.json`           | Command families, slot bindings, split ownership, and labels |
| `generate.py`              | Cartesian sampling, validation, deduplication, and export    |
| `tooling.py`               | All eight production declarations with execution disabled  |
| `lfm_chat_template.jinja`  | Unmodified upstream reference template                       |
| `render.py`                | Offline template preview and assistant-span inspection      |

`hoast/prompts.py` owns the concise routing instructions and the shared
`build_system_prompt` function. System music context contains **only on/off state**
in this dataset. There are no background song names, artists, player IDs, or volume
readings. Names in the user request and its target arguments are explicit entities.
The runtime additionally supports unavailable/unconfigured status; these initial
examples use the full eight-tool registry and observed on/off state.

Production schemas come from `declare_weather_tool`, `declare_music_tools`, and
`declare_light_tool`. Runtime clients bind those declarations to real handlers;
dataset generation uses their non-executable handlers and never constructs clients.

### Canonical targets

- Generic playback requests → `resume_music()` in either on/off state.
- Explicit song title and/or artist → `play_music(...)`, copying supplied entities.
- Pause/stop → `pause_music()`; next/switch/skip song → `music_next()`.
- Current-song/artist questions → `what_is_playing()`.
- Volume, lights, and weather → their typed tool arguments.
- Unsupported, ambiguous, hypothetical, or explicitly negated actions → no tool call.

Lively generic starts include **唱歌**, **放歌**, **来首歌助助兴**, **音乐走起**,
**BGM安排上**, “drop a beat,” and “give this room a soundtrack.” Quoted titles such
as `《唱歌》` remain named-song requests. Generic examples use `resume_music` as one
canonical label, although the runtime also accepts empty `play_music` arguments.

The application renders real tool results itself. It also replaces no-tool model
prose with its standard clarification; synthetic acknowledgements in no-tool rows
supervise abstention, not a promise that the exact sentence will be spoken.
No tool results or successful playback outcomes are fabricated in these records.

Volume values are integers in `[0, 100]`. `set` requires a value; `louder` and
`quieter` add/subtract the supplied percentage points, defaulting to five when
omitted. Relative results clamp to `[0, 100]`, and an explicit relative zero is a
no-op. Supervised targets include the effective integer amount even for a default
request. Signed/fractional values and values outside the supported range are not
valid volume arguments.

### Basic punctuation in user text

`hoast/input_text.py` supplies the same idempotent normalization for STT output,
external LLM user turns, and generated data. It normalizes Unicode compatibility
forms and whitespace while preserving case, contractions, possessives, and dotted
initials. ASCII/Chinese commas, stops, question/exclamation marks, semicolons,
colons, and apostrophes are retained. Curly word apostrophes normalize to `'`;
Chinese sentence punctuation such as `，` retains its width. Title brackets,
decorative double quotes, and other symbols are removed. For example:

```text
播放《晴天》，谢谢！         → 播放 晴天，谢谢！
play "Don't Stop" by G.E.M. → play Don't Stop by G.E.M.
set volume 35%             → set volume 35
```

Unsupported signed/fractional digit literals are rejected before punctuation
removal rather than rewritten into different values. Numeric tool schemas enforce
the allowed integer range. System/tool schemas, assistant targets, tool results,
and internal repair feedback keep their structured syntax.

The JSON source banks retain original spellings and punctuation for provenance
and natural TTS prosody. Published model-input user strings use normalization
version 2 (`basic_punctuation`).
Copied music entity labels are canonicalized consistently; the model is not taught
to invent missing title brackets or double quotes. Location query labels and typed
numeric/boolean arguments retain their tool semantics. Each canonical record keeps
its original source IDs, text, and labels under `normalization.sources` for auditing,
not as model input. Equivalent rows are merged within a split. Formatting-independent
content checks prevent cross-split leakage; contradictory targets are checked
against exact model input so valid punctuation variants can keep their copied labels.

### Typed slots

A scalar bank entry is shorthand for identical surface text and value. Use an
explicit object when pronunciation/formatting differs from the argument:

```json
{"text": "三十五", "value": 35}
```

A whole argument placeholder, such as `"level": "{level}"`, retains that integer.
Booleans work the same way. A title can have surface `《晴天》` and value `晴天`.
Only simple named placeholders are supported: no attribute access, indexing,
format specifications, or conversions. Double braces represent literal braces.
All declared slots must appear in every utterance template in that family; all
target placeholders must be grounded in the utterance.

Target arguments are checked against the production Pydantic models and must
already be canonical. For example, an unsupported weather period cannot silently
normalize to `now`. Application limits also apply: at most four calls, one music
mutation, and one light action. A playback query is not a mutation.

## Chinese, English, and code-switching

The corpus includes Chinese command frames with English names, English frames with
Chinese names, mixed title/artist languages, and fully code-switched commands:

```text
播放Taylor Swift的"Love Story"
play 《晴天》 by Coldplay
请play 《夜曲》，歌手是 Adele
来点 music
```

Nonempty title/artist targets must use their named role slots (`"title": "{title}"`
and `"artist": "{artist}"`), preventing swapped bindings and literal labels borrowed
from command words. Those strings must also occur **exactly** in the user input. The builder
does not translate, transliterate, infer an omitted artist, or restore names from
background context. Names containing control words, quotes, braces, and backslashes
are covered by tests. Artist/title combinations represent user search constraints;
they are not claims that every cross-product pairing exists in a music catalog.
Template authors still need to review the linguistic roles in each sentence;
structural validation does not establish arbitrary natural-language correctness.

`command_language` identifies the template frame. `surface_language` and
`entity_languages` report Han/Latin script composition of the actual rendered
request and arguments. These are coverage metadata, not model inputs or a language
identification model.

## STT errors and Chinese near-homophones

Curated variants carry a reviewed clean form and confusion category:

```json
{
  "text": "把音亮调大点",
  "value": "louder",
  "clean": "把音量调大点",
  "error": "homophone"
}
```

Examples include 音亮/音量, 下一守/下一首, 放首哥/放首歌, 暂亭/暂停,
播方/播放, and English `loader`/`louder` and `quiter`/`quieter`. Curated errors
affect command slots; they are not random character replacements across the whole
request. A corrupted command word therefore cannot corrupt a quoted title or artist.
The corpus also contains uncertain-name clarification examples.

Each noisy row stores `clean_user` and `asr_errors` for audit/evaluation.
**Do not feed this metadata to the model.** Only the heard/transcribed text is in
`messages`. Clean and noisy versions are checked together for split leakage and
contradictory labels. Proper names continue to be copied from the heard request;
arbitrary catalog-name correction is not taught.

These substitutions are synthetic robustness probes, not estimates of Whisper's
error distribution. Add reviewed errors from actual transcripts as they become
available, while retaining the clean/noisy provenance and split ownership.

## Splits, reproducibility, and coverage

Each family has an explicit `split` and a parent `group`. Related translations,
wrappers, on/off twins, and STT variants belong to the same split. The generator
rejects normalized clean/heard input collisions across splits, conflicting targets,
duplicate family IDs, malformed placeholders, and invented music entities.

Canonical duplicate model-input/target rows within a split are merged with their
source provenance; casing variants can remain in that split. Sampling uses stable per-family seeds, never Python's
process-dependent `hash()`. Each split is deterministically shuffled. Changing the
seed or row limits is a new dataset version, recorded in the manifest.

Family-level splitting prevents the most direct template/augmentation leakage;
it does not prove semantic independence. Entity vocabularies are shared, so these
splits test routing and composition rather than unseen-entity generalization.
Keep a separately reviewed real-command evaluation set before judging a fine-tune.
Report per-tool, language, entity-composition, STT-error, and no-tool performance,
not just total accuracy: combinatorial entity rows are intentionally more numerous.

## Official LFM schema injection audit

Reviewed against Liquid AI's [tool-use documentation](https://docs.liquid.ai/lfm/key-concepts/tool-use)
and the [LFM2.5-350M native template](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/chat_template.jinja)
on September 15, 2026.

Runtime `hoast/lfm.py`:

1. Builds strict schemas with `ToolRegistry.schemas()`.
2. Converts them with `tool_declarations()` to LFM's native
   `{name, description, parameters}` objects. Parameter types, enums, required
   fields and constraints remain; schema bookkeeping is omitted.
3. Calls the checkpoint tokenizer's `apply_chat_template(..., tools=..., tokenize=False,
   add_generation_prompt=True)`.
4. The **official template** appends one `List of tools: [...]` block inside the
   system message. The caller does not append another tool list.
5. Runtime tokenizes that rendered text with `add_special_tokens=False`, avoiding
   a duplicate beginning-of-sequence token.

The installed OpenVINO export template and current native Liquid AI template were
byte-identical. The reference copy here is pinned by SHA-256:

```text
ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6
```

Pinned matching export:
[`OpenVINO/LFM2.5-350M-int8-ov` revision `b6a4a9c42aa2dc4bacefa45717befc45acc3a1bf`](https://huggingface.co/OpenVINO/LFM2.5-350M-int8-ov/blob/b6a4a9c42aa2dc4bacefa45717befc45acc3a1bf/chat_template.jinja).
The upstream template is unmodified; its source model's terms are linked in the
[Liquid AI repository](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE).

Each JSONL row has `messages` and a separate native `tools` list. Assistant
`tool_calls[].function.arguments` is a **JSON object**, not a JSON-encoded string.
The official template renders Python-style calls between `<|tool_call_start|>` and
`<|tool_call_end|>`, and has `{% generation %}` assistant supervision markers.
Tests verify single schema injection, role boundaries, assistant character spans,
typed booleans/numbers, escaping, and round-tripping targets through our LFM parser.

When preparing training text, keep only `messages` and `tools` as model inputs:

```python
text = tokenizer.apply_chat_template(
    row["messages"],
    tools=row["tools"],
    tokenize=False,
    add_generation_prompt=False,
)
```

Use the tokenizer associated with the native checkpoint being fine-tuned and verify
its template against this reference. Configure assistant-only/completion loss in
the selected trainer and inspect token masks there; the preview reports character
spans, not token IDs. Do not prepend `clean_user`, labels, or manifest metadata.

Preserve argument dictionaries when loading data. Columnar loaders can add null
fields to heterogeneous nested dictionaries, including empty argument maps. One
way to avoid that is to read each JSONL line with `json.loads`, render it with the
native tokenizer first, and store plain prompt/completion strings for the trainer:

```python
prompt = tokenizer.apply_chat_template(
    row["messages"][:-1], tools=row["tools"],
    tokenize=False, add_generation_prompt=True,
)
assert text.startswith(prompt)
training_row = {"prompt": prompt, "completion": text[len(prompt):]}
```

Tokenize already-rendered strings without adding another set of special tokens.

## Checks

```sh
uv run pytest tests/test_finetune.py -q
uv run pyright finetune hoast/prompts.py
uv run ruff check finetune hoast/prompts.py
```

Tests require no downloaded model, external dataset, service credentials, or audio
devices. Template validation establishes format compatibility, not routing accuracy
of the untuned model. Training and real-model evaluation are separate steps.

## TTS-to-STT augmentation

The augmentation workflow synthesizes **clean source commands**, transcribes the
audio, and canonicalizes the recognized user text. It does not speak the curated
misspellings from the hand-written STT variants. Audio is shared by music-on/off
twins and equivalent clean/noisy source records, while source IDs and split
ownership are retained.

On a CUDA-capable Linux training host, prepare the isolated speech environment
and run from the repository root:

```sh
uv sync --locked
uv sync --project finetune/speech_env --locked --python 3.12
uv run python -m finetune.generate
AUGMENT_DIR=.cache/hoast/finetune/speech-data
uv run python -m finetune.augment --data finetune/generated --output "$AUGMENT_DIR" --device cuda --batch-size 16
```

`--device cuda` runs both native Kokoro synthesis and Whisper-small inference on
CUDA. **Whisper processes independent utterances in real batches**. Kokoro's
duration expansion supports one utterance per forward pass, so its synthesis is
GPU-accelerated but sequential within each batch. Use `--device cpu` explicitly
for the CPU alternative; there is no silent device fallback. `--batch-size` controls
the ASR batch and can be reduced for memory limits. Low-confidence retries can use
smaller batches.

The worker downloads pinned English Kokoro v1.0 (`af_heart`), Mandarin Kokoro v1.1
(`zf_001`), and Whisper-small assets. Han-containing text uses the Mandarin model
with English phoneme insertions. Silent title marks are removed for synthesis,
initialisms are spoken as letters, and English possessives on Mandarin names retain
their spoken suffix. Returned audio is mono float32 at 24 kHz, then resampled to
16 kHz for ASR. Language detection and decoder prompts are independent per
utterance, without source-text hints or shared transcript history.

The GPU augmentation backend uses native Torch TTS and FP16 CTranslate2 ASR;
the edge recognizer uses CPU INT8 and its own streaming/VAD context. These are
related synthetic robustness probes, not identical executions or estimates of
real microphone noise. Model revisions, voices, precision, sample rates, decoding
settings, frontend hashes, and actual batch sizes are recorded with the outputs.

### Label audit and outputs

Every speech observation retains its intended target, original source IDs,
synthesis text, raw transcript, canonical transcript, and audio checksum. The raw
transcript is never fed to the LLM. Case, spacing, and verified Chinese script
equivalents can be accepted while copying music labels **as actually heard**.
Known integer-volume word/digit spellings are equivalent for audit purposes;
different numeric values are never repaired into the expected value. Existing
reviewed command-confusion substitutions can also establish a label match.

Other wording changes, changed names, missing speech, or split collisions require
review and are excluded from automatic training augmentation. This includes
negation or intent changes that would otherwise turn into mislabeled commands.
Detailed review reasons are retained instead of silently correcting names or
dropping unsuccessful cases.

| Output                       | Contents                                                          |
| ---------------------------- | ----------------------------------------------------------------- |
| `data/train.jsonl`            | Canonical training rows plus eligible, nonduplicate speech variants |
| `data/validation.jsonl`       | Canonical validation split                                        |
| `data/test.jsonl`             | Canonical test split                                              |
| `data/manifest.json`          | Source hashes, coverage, and augmentation settings                 |
| `train-roundtrip.jsonl`       | All training speech candidates, including review-required cases    |
| `validation-roundtrip.jsonl`  | Held-out validation speech observations                            |
| `test-roundtrip.jsonl`        | All source-held-out intended-command speech observations            |
| `test-roundtrip-independent.jsonl` | Speech observations without cross-split input collisions       |
| `review.jsonl`                | Observations requiring label/transcript review                     |
| `speech/`                    | WAV audio, per-job results, settings, and diagnostics               |
| `augmentation.json`           | Eligibility, failures, source coverage, and added-row counts        |

The roundtrip held-out files measure the full intended-command path: if STT loses
a name, the original intent remains the end-to-end target and the row is flagged.
Such a row is not a clean transcript-grounded extraction example. The evaluator
reports audited eligible/review-required slices and retains failed speech inputs
as failures without attempting LLM inference. State twins and speech variants are
related observations, not independent real-speaker samples.

Speech jobs are hash-addressed and resumable under matching settings. Use a new
output directory when model, frontend, normalization, or decoding settings change.
After a completed speech run, `--reuse-transcripts` rebuilds the audited dataset
from the retained results without resynthesizing; source/schema/normalizer checks
still apply. `--variants` selects the matching source variant bank when using a
frozen dataset outside the default directory. Logs, audio, and detailed results
belong in ignored output storage.

When only text normalization changes, `--renormalize-from` can reuse a previous
`speech/` directory into a **different output directory**. Raw transcripts are
normalized again, including script-audit text, while original inference provenance
and audio hashes are retained. A verified `worker-source.py` snapshot proves that
the model execution and retry implementation match; caches without that snapshot
require `--prior-worker-source` pointing to their exact recorded worker source.
Model/voice/decoder settings, device, batch size, source text, and split must match.
Missing jobs are synthesized/transcribed normally. No new inference is claimed
for reused recordings. All resulting labels and split checks are rerun.

```sh
PREVIOUS_SPEECH=.cache/hoast/finetune/prior-run/speech
AUGMENT_DIR=.cache/hoast/finetune/updated-data
uv run python -m finetune.augment --data finetune/generated --output "$AUGMENT_DIR" --renormalize-from "$PREVIOUS_SPEECH" --device cuda --batch-size 16
```

Use the `*-roundtrip-independent.jsonl` files for text-independent held-out scores.
The complete roundtrip files retain source-held-out observations and any collision
flags for end-to-end accounting; they must not be presented as independent textual
holdouts if overlaps are reported.

To train on the resulting data, pass `--data "$AUGMENT_DIR/data"` to
`finetune.train`. To evaluate speech inputs:

```sh
MODEL_DIR=.cache/hoast/finetune/run/openvino-int8
uv run python -m finetune.evaluate --backend openvino --model "$MODEL_DIR" --data "$AUGMENT_DIR/test-roundtrip.jsonl" --batch-size 1 --threads 2 --output "$AUGMENT_DIR/eval-int8"
```

## LoRA training and INT8 export

See [RESULTS.md](RESULTS.md) for held-out routing accuracy and correct/incorrect
examples. Paths in the commands below are example output locations you can change.

Official references:

- [Liquid fine-tuning overview](https://docs.liquid.ai/lfm/fine-tuning)
- [Liquid's TRL recipe](https://docs.liquid.ai/lfm/fine-tuning/trl)
- [Optimum Intel OpenVINO export](https://huggingface.co/docs/optimum-intel/openvino/export)

The application/export environment uses CPU Torch. CUDA training has a separate,
locked uv project in `finetune/env`, using Python 3.14, Torch 2.9.1+cu128,
Transformers 4.57.6, TRL 0.26.2, and PEFT 0.18.1. Run from the repository root:

```sh
uv sync --locked
uv sync --project finetune/env --locked
uv run python -m finetune.generate
RUN_DIR=.cache/hoast/finetune/run
GPU_WORKERS=1
mkdir -p "$RUN_DIR"
OMP_NUM_THREADS=4 finetune/env/.venv/bin/torchrun --standalone --nproc_per_node="$GPU_WORKERS" -m finetune.train --output "$RUN_DIR" > "$RUN_DIR/train.log" 2>&1
OMP_NUM_THREADS=4 .venv/bin/python -m finetune.export --model "$RUN_DIR/merged" --output "$RUN_DIR/openvino-int8" > "$RUN_DIR/export.log" 2>&1
```

Set `GPU_WORKERS` to the number of CUDA GPUs you want to use, with one process per
GPU. Training requires BF16-capable CUDA hardware and sufficient device memory.
Detailed metrics/logs can contain local paths and device information; keep them
in an ignored output directory such as the example above.

`train.py` pins `LiquidAI/LFM2.5-350M` revision
`9e6c6ccf47cd318696e137d381a7ded8fe4df09f`. It uses the native tokenizer JSON
and verifies byte-identical chat-template rendering. Upstream's generic tokenizer
backend is named `TokenizersBackend` in Transformers 5; loading it explicitly as
`PreTrainedTokenizerFast` with an empty extra-special-token mapping allows the
OpenVINO-compatible Transformers 4 stack to use the same vocabulary and template.
The upstream RoPE value is checked against the loaded configuration.

Settings follow Liquid's first-pass LoRA recipe: rank 16, alpha 32, dropout 0.05,
q/k/v attention projections, learning rate 2e-4, and three epochs. The recipe's
`o_proj` name does not exist in this architecture, so it contributes no adapter;
the actual adapter has 491,520 parameters. Each rank uses BF16, four examples per
device and four accumulation steps. Effective batch size is `16 × GPU_WORKERS`;
changing the worker count changes the optimization batch size.
The seed is 42; warmup is 5%, followed by cosine decay. Checkpoint selection uses
validation completion loss only. There is no test-based checkpoint selection.

JSONL is rendered and tokenized with `add_special_tokens=False` before conversion
to a columnar dataset, preserving heterogeneous argument dictionaries. Explicit
completion masks are passed to TRL with `skip_prepare_dataset=True`: TRL's default
string preparation would insert another BOS and append another EOS after the
native template's trailing newline. Actual collated prompt, completion/EOS, and
padding labels are audited on 64 unequal-length records. Only assistant completions
contribute to the loss. Packing is disabled; overlength examples raise rather than
silently truncate. The maximum sequence length is 4096. Add `--max-steps 2` and a
**different output directory**
for a short smoke run. Each rank retains a diagnostic log. `training.json`
records revision, manifest hash, validation history, and the selected checkpoint.
Adapters are reloaded from disk, merged in FP32, and checked for logit equivalence
before saving a standalone Hugging Face checkpoint.

The export produces **INT8 weight compression**, not full INT8 activation
quantization, with a stateful graph and tokenizer files. `export.py` uses Optimum's
official export API followed by NNCF 8-bit asymmetric weight compression. Its
version-scoped patch for Optimum Intel 1.27.0 computes short-convolution history by
concatenation: the upstream traced advanced-index cache update fails during GenAI decoding with
a `ScatterNDUpdate` shape mismatch. Tests compare patched outputs and cache states
with native prefill and multiple decoding steps, including short prompts.

### Held-out evaluation

```sh
RUN_DIR=.cache/hoast/finetune/run
finetune/env/.venv/bin/python -m finetune.evaluate --model LiquidAI/LFM2.5-350M --output "$RUN_DIR/eval-base"
finetune/env/.venv/bin/python -m finetune.evaluate --model "$RUN_DIR/merged" --output "$RUN_DIR/eval-tuned"
.venv/bin/python -m finetune.evaluate --backend openvino --batch-size 1 --threads 2 --model "$RUN_DIR/openvino-int8" --output "$RUN_DIR/eval-int8"
```

The default dataset is `finetune/generated/test.jsonl`. Use `--data` to explicitly
evaluate validation instead. Greedy generation is capped at 128 tokens with a
repetition penalty of 1.0. Native special tokens are retained for production
parsing; schemas and batch limits are checked without executing handlers.
Accuracy requires exact ordered tool names and typed arguments. For no-tool rows,
any valid nonempty abstention text is accepted. Malformed/empty/length-limited
outputs fail. `predictions.jsonl` preserves every user utterance, target, output,
and reason; complete messages/tools are in the corresponding frozen JSONL dataset.
`metrics.json` includes slice counts, invalid outputs, false tool activations, and
warmed end-to-end throughput. CUDA batches and serial OpenVINO CPU measurements
have different hardware and batching; their throughput is not a quantization
speedup comparison. OpenVINO evaluation uses the actual GenAI pipeline and checks
that the graph contains at least 100 million INT8 weight elements.
Omitted relative-volume amounts are scored using the same five-point default as
runtime dispatch. Absolute settings still require an explicit value.

The generator can increase samples with `--max-per-family`, but this changes all
generated splits. Preserve frozen evaluation files before expanding training;
never tune from test errors. These synthetic, shared-entity splits do not establish
unseen-entity, real-STT, or unavailable-service generalization.

## Deploying to an edge device

### Portable model versus device cache

The exported OpenVINO IR (`.xml` graph and `.bin` weights) is not tied to the
training host or its CUDA GPUs. Deploy it on an OS/architecture and inference
device supported by the installed OpenVINO and OpenVINO GenAI versions. INT8 here
describes stored weights; execution precision depends on the target plugin.

Copy the model, tokenizer, and configuration files together. The export directory
contains `openvino_model.xml`, `openvino_model.bin`, tokenizer/detokenizer XML/BIN
files, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, and model
configuration files. Transfer any other tokenizer/configuration files generated
alongside them too. Only the export is needed for inference; `adapter/` and
`merged/` are training and re-export artifacts.

Compiled caches are device/runtime-specific. Let the target create its own cache;
do not copy a source machine's compiled blobs or Python virtual environment.
The export-time convolution patch is encoded into the IR and does not need to run
on the inference device. Runtime memory also includes activations and context
state, so model file size is not a total RAM requirement.

### Transfer and install

These shell examples assume a Linux target with SSH access. Replace the SSH host
placeholder and choose paths appropriate for your deployment. From the source:

```sh
RUN_DIR=.cache/hoast/finetune/run
EDGE_HOST=your-edge-host
ssh "$EDGE_HOST" 'mkdir -p "$HOME/models/lfm-int8"'
rsync -a --exclude='*.log' "$RUN_DIR/openvino-int8/" "$EDGE_HOST:models/lfm-int8/"
```

On the target, obtain the project checkout and run the following from its root:

```sh
uv sync --locked
export MODEL_DIR="$HOME/models/lfm-int8"
export CACHE_DIR=.cache/hoast
```

Use the runtime versions in the root lockfile; the separate CUDA training
environment is unnecessary on the target. Hardware-accelerated execution also
requires the appropriate device driver. OpenVINO's `GPU` device targets supported
Intel GPUs; it is not the CUDA backend. Start with `CPU`, then select `GPU` only
when the intended device is available through OpenVINO.

### Evaluate without executing actions

Generate the reference evaluation data on the target, or transfer a frozen
evaluation set alongside its manifest when comparing against a particular run:

```sh
uv run python -m finetune.generate
uv run python -m finetune.evaluate --backend openvino --model "$MODEL_DIR" --data finetune/generated/test.jsonl --batch-size 1 --threads 2 --output "$CACHE_DIR/finetune/edge-evaluation"
```

This runs the entire test set on the target CPU, saves predictions and metrics,
and never executes tool handlers. Compare dataset hashes before comparing runs.
Use target measurements for throughput and memory planning; benchmark results from
a different device or batching setup are not target-device latency estimates.

### Select the checkpoint in the assistant

The application accepts upstream `TokenizersBackend` metadata and the
`PreTrainedTokenizerFast` metadata saved with merged fine-tuned checkpoints.
Keep the exported tokenizer JSON and chat template intact.

The assistant selects the LFM artifact through `<cache>/lfm/model.json`.
`--model lfm` selects the model family, not a checkpoint directory. Register the
target-local absolute path with this snippet, which replaces the selection for
the chosen cache root:

```sh
uv run python - <<'PY'
import json
import os
from pathlib import Path

model = Path(os.environ["MODEL_DIR"]).resolve(strict=True)
required = (
    "openvino_model.xml", "openvino_model.bin", "tokenizer.json",
    "tokenizer_config.json", "chat_template.jinja",
)
for name in required:
    if not (model / name).is_file():
        raise FileNotFoundError(model / name)
manifest = Path(os.environ["CACHE_DIR"]) / "lfm" / "model.json"
manifest.parent.mkdir(parents=True, exist_ok=True)
manifest.write_text(
    json.dumps({"model_id": "LiquidAI/LFM2.5-350M", "path": str(model)}, indent=2) + "\n",
    encoding="utf-8",
)
PY
```

Configure the assistant's services as described in the [project setup](../README.md#setup),
then start text mode. Use the same tool schemas as training: the reference
evaluation includes all eight tools, while the assistant registers tools for
its configured services.

```sh
uv run python -m hoast --config config.toml --model lfm --cache "$CACHE_DIR" --device CPU --threads 2 --max-new-tokens 128 --text
```

For a supported Intel GPU, use `--device GPU --threads 1`. Execution is explicit;
a device-loading error does not trigger CPU fallback. Unlike the evaluator, the
assistant can execute configured tools. See [RESULTS.md](RESULTS.md) for known
routing limitations. Voice mode additionally requires speech models and satellite
configuration from the main README; omit `--text` after preparing those resources.

Model preparation commands can replace the selection manifest with the upstream
model. Register the fine-tuned export after completing model preparation. To
switch checkpoints, stop the assistant, change the manifest to the new local
export directory, and restart. Preserve the previous manifest if you need to
restore its selection. The target creates its compiled cache under
`<cache>/lfm/compiled/` when loading the model.

## Training and deployment checks

```sh
.venv/bin/pytest tests/test_finetune.py tests/test_finetune_evaluate.py tests/test_finetune_export.py tests/test_finetune_deploy.py tests/test_session.py -q
finetune/env/.venv/bin/python -m pytest finetune/env/test_training.py -q
finetune/speech_env/.venv/bin/python -m pytest finetune/speech_env/test_cache.py -q
.venv/bin/pyright --project finetune/env/pyrightconfig.json
.venv/bin/pyright --project finetune/speech_env/pyrightconfig.json
.venv/bin/pyright
.venv/bin/ruff check finetune hoast/lfm.py tests/test_finetune*.py
```
