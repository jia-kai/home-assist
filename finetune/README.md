# Tool-routing fine-tuning data

This directory builds **supervised routing examples**, without loading a model,
reading service credentials, or executing tools. It does not train or deploy a model.
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

The checked default build contains 6,730 training rows, 810 validation rows, and
810 test rows. Of those, 976 contain curated STT confusions. Counts are determined
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

Exact duplicate commands within a split are removed; casing and punctuation variants
can remain in that same split. Sampling uses stable per-family seeds, never Python's
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
