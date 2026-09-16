# Local LLM preparation, runtime, and measured results

## Models and setup

### Selected bounded-CPU configuration

The interactive CLI uses `--threads auto`: **GPU = one CPU core, CPU = two**.
Explicit `--threads 1|2` overrides this. LFM GPU uses FP16 computation,
`GPU_QUEUE_THROTTLE=LOW`, one stream and one compilation worker with the published
INT8 weights. `LFMConfig.from_cache(threads=None)` resolves the same defaults;
direct `LFMConfig` construction defaults to one thread. Library callers can use
`hoast.runtime.configure_cpu_budget` to enforce the aggregate process budget.

Nine current five-tool-registry cases, each repeated three times, passed 27/27:

| Device/settings | CPU cores | Warm median |
| --------------- | --------- | ----------- |
| GPU FP16 / LOW  | 1         | 0.6207 s    |
| GPU FP16 / LOW  | 2         | 0.6077 s    |
| CPU FP32        | 2         | 1.0367 s    |
| CPU FP32        | 1         | 1.9528 s    |

One GPU host core is selected because its median is within about 2% of two.
History follow-ups, independent sessions and post-cancellation generation passed.
Compact prompt/schema formatting was rejected after reducing accuracy to 18/27;
the original template and tool schema remain the reference. FP16 KV caching and
prefix caching did not improve this workload. Raw IR profiling is not the timing
of GenAI's rewritten execution graph. Reproducible local measurements are retained
in `.cache/hoast/diagnostics/lfm-tuning/`. Four-thread comparisons later in this
document are historical experiments outside the selected two-core budget.

**LFM2.5 on the iGPU is the CLI default.** Preparation also defaults to LFM and
GPU; `LFMConfig` and its `from_cache()` factory default to GPU. Select CPU
explicitly with `--device CPU` or `device="CPU"`. FunctionGemma
supports raw sessions on OpenVINO CPU/GPU (and PyTorch CPU), but observed weather
routing was poor. The backend smoke checks and historical synthetic scores below
do not establish equal agent accuracy. Deterministic answer rendering
grounds spoken facts in tool results; it does not correct a wrongly routed city
or period.

Install the locked Python 3.14 environment with `uv sync --locked` from the project
root. OpenVINO GenAI serves both model families on explicitly selected `CPU` or
`GPU`; device errors propagate without automatic CPU fallback. Both models have
been verified on the local i5-8500T CPU and Intel UHD 630 GPU. The comparative
FunctionGemma-versus-LFM measurements below are CPU-only. A separate matched
LFM CPU/iGPU routing measurement is summarized below.

### FunctionGemma

Accept the terms for [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it)
and authenticate with a read token from that account:

```sh
uv run hf auth login
uv run python -m tools.prepare_llm --model functiongemma download --revision 39eccb091651513a5dfb56892d3714c1b5b8276c
uv run python -m tools.prepare_llm --model functiongemma export --precision int8 --threads 2 --device CPU
USE_TORCH=0 uv run python -m tools.prepare_llm --model functiongemma compile --device GPU
```

Without `--revision`, download resolves `main` to an immutable commit and records
it in `.cache/hoast/model.json`. Authentication uses the standard Hugging Face
credential store. Export creates stateful OpenVINO IR with KV caching and prepares
both SDPA and paged attention (PA) for the chosen device. INT8 is symmetric;
experimental INT4 is symmetric, group size 128, ratio 1.0, with embedding/final
layers at the exporter's INT8 backup precision. FP32 is also available. No
calibration dataset is required. Historical measurements used INT8/PA/four CPU threads;
INT4 substantially reduced exact-call accuracy.

### LFM2.5

```sh
uv run python -m tools.prepare_llm --model lfm download
USE_TORCH=0 uv run python -m tools.prepare_llm --model lfm compile --device CPU
USE_TORCH=0 uv run python -m tools.prepare_llm --model lfm compile --device GPU
```

Download uses anonymous access to the public
`OpenVINO/LFM2.5-350M-int8-ov` conversion linked from LiquidAI's model card, pinned
to `b6a4a9c42aa2dc4bacefa45717befc45acc3a1bf`. `--revision` can select another
export commit. Provenance is recorded in `.cache/hoast/lfm/model.json`.
The publisher identifies `LiquidAI/LFM2.5-350M`, NNCF `INT8_ASYM` weights and group
size -1, but does not identify the upstream weight commit. Pinning the export
does not establish that missing source revision.

LFM uses published INT8 IR: `export` is FunctionGemma-only. Its native hybrid
pipeline chooses its own attention implementation. The export requires OpenVINO
>=2026.3.0; the measured environment uses 2026.3.1. The runtime adapts Transformers
5 tokenizer metadata to the pinned Transformers 4 generic fast-tokenizer API,
without changing serialized tokenizer/template files. Integration checks verified
token IDs against the OpenVINO tokenizer for English, Chinese, French, and Korean.
The preparation CLI compiles models without running a benchmark or tool handler.

### Cache and diagnostics

All paths default to `.cache/hoast/` relative to the working directory. Put
`--cache-dir /absolute/cache` before the preparation subcommand and pass the same
root to runtime `from_cache` methods. Downloaded HF snapshots, exported XML/BIN
IR, selection/provenance manifests, device compilation caches, and preparation
diagnostics are retained. Export identities include model provenance, dependency
versions and compression settings; existing completed exports reuse IR. A failed
new export/compile does not publish an export completion manifest.

`compile` operates on existing IR without downloading or exporting. Model files
are portable IR, not standalone executables; changing hardware, runtime or compile
settings can require recompilation. CPU and GPU compilation identities are
separate. Keep model ownership alive across requests to amortize startup.

On OpenVINO GenAI 2026.3.1, FunctionGemma PA rewrites its CPU blob on load and
spends about 1.4 seconds compiling. SDPA with `CACHE_MODE=OPTIMIZE_SPEED` loaded
its reusable compiled pipeline in about 0.38 seconds, excluding Python/tokenizer
startup, but had slower warm requests. Model download and IR are reused by both.

Preparation logs command arguments, Python version, and complete exception
tracebacks/chains/notes under `diagnostics/`. Export subprocess stdout/stderr and
conversion settings are saved together. Existing LFM `prepare-*.log` diagnostics
are also retained. Historical benchmark/probe programs, fixtures, raw reports and
local benchmark results have been removed; the summaries below retain their
methodology and qualifications and are not commands for rerunning those suites.

After preparation, model loading can run with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. `USE_TORCH=0` set before importing Transformers reduces
OpenVINO-only startup/memory; do not set it for FunctionGemma export or PyTorch
inference. The CPU PyTorch backend uses FP32; its thread settings are process-wide.

## Runtime and session contracts

```python
from hoast.llm import FunctionGemma, LLMConfig, ToolRegistry
from hoast.session import Session

tools = ToolRegistry([])  # Supply typed tools for a tool-using application.
with FunctionGemma(LLMConfig.from_cache(device="CPU"), tools) as model:
    session = Session(model)
    text = "".join(session.stream("Hello!"))
    pending = session.pending_calls
```

For LFM, use `LFM2(LFMConfig.from_cache(device="GPU"), tools)` from `hoast.lfm`.
FunctionGemma also supports PyTorch FP32 on CPU. `LLMConfig.from_cache` resolves
OpenVINO exports only and has no `backend` argument. To use the downloaded
checkpoint, construct the configuration directly:

```python
import json
from pathlib import Path

from hoast.llm import LLMConfig

cache = Path(".cache/hoast")
manifest = json.loads((cache / "model.json").read_text())
torch_config = LLMConfig(Path(manifest["path"]), backend="torch", cache_dir=cache)
```

Runtime settings belong to model/runner configuration; the system TOML requires
`[weather]` latitude and longitude and accepts optional `[music]` settings.
See the [README setup guide](../README.md#setup) for configuration, credentials,
and the direct CLI. Interactive Home Assistant usage:

```sh
uv run python -m hoast --config config.toml
uv run python -m hoast --config config.toml --model lfm --device GPU
uv run python -m hoast --config config.toml --model functiongemma --device CPU
```

- Model owners support idempotent `load()`/`close()` and context management,
  serialize inference/lifecycle operations, and retain no per-user history.
- `generate` and `generate_messages` return intercepted calls without executing
  handlers. `Generation` carries text, native raw protocol, calls, token counts,
  wall latency and backend first-token latency. Caller-supplied histories use
  native user/assistant/tool messages; configured system instructions are inserted
  by the model owner.
- `Session` owns history independently of model lifetime. Exhaust `stream(text)`
  before inspecting `pending_calls`; call `invoke_tools()` explicitly, then
  `stream()` for model continuation, or `complete(text)` to commit a nonempty,
  externally grounded assistant answer without inference. `complete` requires
  successful tool results and no pending calls or failed dispatch. It enables a
  new user turn and preserves native tool history. `reset()` clears conversation
  state while preserving the loaded model.
- `Session.request_tools(user_text: str, calls: Sequence[ToolCall]) -> None`
  records application-selected calls without inference or handler execution.
  The request and call sequence must be nonempty; the whole copied batch validates
  before the original user text and native assistant calls commit. Prior history
  is retained. Pending calls, pending continuation, or failed dispatch require
  finishing or resetting first. Dispatch with `invoke_tools`, then continue with
  `stream()` or `complete(text)` as for generated calls. `history` and
  `pending_calls` return isolated snapshots.
- `Session.invoke_tools(result_transform: Callable[[ToolCall, JsonValue], JsonValue]
  | None = None) -> tuple[JsonValue, ...]` dispatches the validated batch once and
  returns full JSON result snapshots in call order. By default it also stores full
  results in native history. A transform receives isolated call/result copies and
  projects only the model-visible result; its return must be JSON-compatible.
  History wraps every projected or full value under `result`, with an additional
  `name` for LFM. Full results are returned even with a transform. Handler,
  serialization, and projection failures require reset; handlers are never retried.
- Streams emit natural-language fragments, withholding possible native tool
  protocol until parsing/validation succeeds. Close abandoned iterators to cancel
  and join their workers. Failed/abandoned generation does not commit history;
  text already emitted cannot be retracted. A handler failure requires session
  reset because preceding side effects may already have occurred.
- `Session.stream(..., max_repair_attempts=2)` permits up to two extra generations
  for `GeneratedCallError`: generated syntax, unknown tools, or invalid arguments.
  The default is zero (fail-fast streaming). Repair-enabled streams buffer prose
  until validation succeeds, while retaining cancellation callbacks. Each repair
  includes the first 400 characters each of the validation error and rejected
  output in a transient user message; full diagnostics remain in logs. Only the
  original request and validated reply enter committed history.
  Exhaustion preserves prior history and any pending continuation. Inference,
  context/output-limit, and handler failures are not repaired.
- Define handlers with `Tool`, `ToolArguments` (Pydantic), and `ToolRegistry`.
  Strict validation rejects implicit type coercion and extra fields. Nested
  argument models, lists, scalar types, and concrete string enums are supported;
  recursive schemas, arbitrary mappings, unions/optional-null schemas and
  template-reserved names are rejected at registration.
- All calls validate as a batch before any handler executes, and are validated
  again at dispatch. Handlers run in order; exceptions propagate without retry or
  rollback. Results must be JSON-compatible. Sessions snapshot results and wrap
  them for native continuation so scalar/empty results retain their meaning.
- FunctionGemma uses its required developer activation sentence and `<escape>`
  string syntax. LFM uses a native system prompt and
  `<|tool_call_start|>[function(argument=value)]<|tool_call_end|>`. Its restricted
  AST parser never evaluates expressions. Both reject duplicate keys, malformed
  or incomplete calls and invalid arguments before dispatch.
- LFM prompt declarations omit `title` and outer `additionalProperties` while
  retaining meaningful parameter constraints; runtime validation remains strict.
  Full schema bookkeeping caused malformed copied fields in an integration pilot.
- Defaults reserve up to 128 generated tokens within a 4096-token context budget.
  Over-budget prompts and length-exhausted outputs raise rather than truncate
  silently. LFM's default repetition penalty is 1.05; comparison runs use 1.0.
  Use `dataclasses.replace` for budgets, system prompts and other model settings;
  retain FunctionGemma's developer activation sentence.

### Local agent and tool contract

The interactive CLI defaults to `--model lfm --device GPU --threads auto`, with
`--cache .cache/hoast` and a required `--config` path. It sets a route-only system
prompt and a 384-token output budget. `/reset` clears history; EOF exits. There is
no positional `interactive` subcommand. The TOML requires `[weather]` containing
finite numeric WGS84 `latitude` (−90 through 90) and `longitude` (−180 through 180).
Optional `[music]` enables the Music Assistant integration; credentials are read
from the environment or the dotenv file selected by `--env-file` (default `.env`).
Booleans as coordinates, missing required fields, and unknown keys are rejected.

`hoast.agent.LocalAgent` accepts weather with an optional complete core music
registry, optional music next/query extensions, and optional light control.
`WeatherAgent` is a compatibility subclass with the same behavior and registry
validation. Standalone `play` and `stop` route deterministically to `resume_music`
and `pause_music` via `Session.request_tools`, preserving native history without
model inference. External user turns retain basic sentence punctuation and
apostrophes while losing title brackets and decorative double quotes. Shortcut
matching ignores case and terminal sentence punctuation. Longer requests use the model. Without configured
music tools, these shortcuts yield “Music isn't configured.” Standalone “louder”
and “quieter” also bypass inference for five-point volume changes; “quiter” is
accepted as a spelling alias for “quieter”. An explicit integer percentage amount
overrides the relative default, for example “quieter 20 percent”.

Other turns perform one model routing pass with up to two repair passes
for generated syntax or argument errors, then dispatches at most four
calls, including at most one music action, in order, without retrying execution. Batches exceeding
either limit fail before dispatch. Model routing prose is withheld. After all
results render successfully through the `Session.invoke_tools` result transform,
the agent calls `Session.complete` with their combined answer, then yields one
deterministic sentence-group chunk per result, separated by spaces. These are
application-rendered chunks, whereas raw `Session.stream` delivers real incremental
word-boundary fragments from the backend. Tool-free model text is discarded along
with history. `LocalAgent.stream`, not the LLM, supplies the fallback: “What would
you like to do with the music?” when music is enabled, otherwise “Please rephrase
your request.” No fallback executes an action. Exhausted call
repairs produce “I couldn't understand that request. Please rephrase it.” and
preserve prior history; the CLI remains ready for another request. Diagnostic
feedback is logged, not spoken. Other errors propagate
and reset history; closing an abandoned answer stream also resets history, even
after tool dispatch and answer commitment. Already executed tools are not undone.

The shared system prompt in `hoast/prompts.py` requires grounded arguments and
clarification for unsupported or ambiguous actions. Volume values are integer
percentages in `[0, 100]`: `volume_music(action='set', level=35)` sets an absolute
level; `volume_music(action='louder', level=20)` adds twenty percentage points;
`volume_music(action='quieter')` uses the five-point relative default.

Tool descriptions are compact action guidance. The weather description is “Get
weather, outside temperature, and rain forecast in Celsius.” Provider internals
stay out of the model-facing description. The agent's result transform stores
only grounded spoken answer strings under the native `result` field, rather than
detailed weather/music objects. Full outcomes and diagnostic reasons remain in
logs. `Session.complete` also stores the combined grounded assistant answer.
This compact model feed applies to LocalAgent; raw sessions retain full results
unless the caller supplies a transform.

For an externally grounded answer in a raw session, exhaust the routing stream,
inspect `pending_calls`, call `results = session.invoke_tools()`, render those
results in the application, then call `session.complete(text)`. For the weather
registry, the rendering step can be
`text = " ".join(render_weather(result) for result in results)` using
`hoast.agent.render_weather`. `complete` commits text; it does not yield it or
verify the application's grounding.

Music results use `hoast.agent.render_music`. It distinguishes acknowledged
requests (“Pause requested.” / “Resume requested.”) from immediately observed
states (“Music paused.” / “Music resumed.”), and describes new playback as a
mix based on a bounded seed label and at most the first artist. Refusals use short
actionable wording; detailed reasons and candidate lists are logged rather than
spoken. Pause/resume use native controls on the retained source. Idle retained
external AudioSource sessions, including Spotify Connect, resume using
`players/cmd/play` after a read-only queue-identity check. Idle queue sources and
missing or unsupported sources refuse fallback. Blank `play_music()` arguments
also resume existing playback without
searching or creating a queue. A supplied title or artist replaces the active
queue with a recommendation mix, whose seed need not play first. Version-only
duplicates with identical title and artist sets use the first available ranked
match. Title-only requests also choose the first available exact-title match
across artists, using Music Assistant's search order. An explicit artist remains
a strict filter; different matching collaborator sets require clarification.
State checks cannot prevent
server fallback during a dispatch race. See the
[Music Assistant guide](../README.md#music) for
setup, player selection, command examples, and result semantics.

Native LFM/GPU checks covered explicit pause/resume, blank play/continue, new
title/artist requests and retained-history controls. Valid-but-wrong routing still
occurred for "Play music" after previous playback (pause instead of resume), and
a combined forecast-plus-pause request omitted pause. Syntax/argument repair
does not detect these semantic mistakes. This multiword “play music” limitation
still applies; standalone “play” uses the deterministic shortcut. Use the direct CLI for deterministic
control testing; it bypasses model routing and invokes the same handlers.

### Music volume contract

`volume_music` requires `action` to be `louder`, `quieter`, or `set`. Every supplied
`level` must be an integer in `[0, 100]`; strings, booleans, fractions, and
out-of-range values fail strict validation. Set requires an explicit level.
Relative actions add/subtract the supplied percentage points, defaulting to five
when omitted. Results clamp to `[0, 100]`; an explicit relative zero is a no-op.
An unchanged target sends no mutation. Relative amounts are not multiplicative
ratios: raising a current 40 by 20 requests 60.

Volume control allows idle players and sends no playback, queue, or source-change
commands. Effective routing resolves sync leaders and active groups. Group targets
read `group_volume` and write `players/cmd/group_volume`; individual targets read
`volume_level` and write `players/cmd/volume_set`. Missing group readings are not
replaced with member readings. Advertised `volume_set` support is required;
missing current volume blocks relative changes but permits absolute settings.

- Compact player fields are `player_id`, `name`, `available`, `enabled`, `state`,
  `source`, `synced_to`, `active_group`, and group-aware `volume_level` (percentage
  or null). `players` lists each target without resolving members; `status`
  resolves the effective target and adds `queue`.
- Volume results include effective `player_id`, target `level` (possibly null for
  an unavailable relative reading), and `status`. `volume_unchanged` has
  `confirmation: "observed"`. `volume_set` also includes `observed_level`, the
  immediate readback or null; confirmation is `observed` only on a matching reading,
  otherwise `requested`. Spoken answers distinguish “Volume set to 35 percent.”,
  “Requested volume 35 percent.”, and “Volume is already 35 percent.”
- Unsupported control, missing relative readings, or routing/volume drift before
  dispatch return `cannot_volume` with `reason` and no mutation. Routing drift
  after dispatch returns `cannot_volume`, `confirmation: "requested"`, and
  `observed_player_id`. Snapshots are non-atomic; commands are never retried.
  Missing unique selection returns `player_required` with choices and a reason.
  Pre-dispatch refusals say “I can't adjust that volume. Please use the music app.”
  Post-dispatch uncertainty says “Volume change requested, but not confirmed.
  Please check the music app.”

The direct CLI accepts `volume louder`, `volume quieter`, `volume raise 20`,
`volume lower 10`, `volume set 0`, and the absolute shorthand `volume 35`.
`stop` aliases `pause`. Invalid or missing required values fail during parsing
with exit 2 before client setup or any mutation.
`cannot_volume` also exits 2 with the full JSON result; accepted or unchanged volume
exits 0. Global options precede the subcommand, as shown in the README.

### Weather tool and answer contract

| Field                                  | Exact semantics                                                        |
| -------------------------------------- | ---------------------------------------------------------------------- |
| `period` argument                      | Optional: `now` (default), `today`, `tomorrow`, or `next_week`.            |
| `city` argument                        | Optional string, at most 200 characters; empty/whitespace selects home.  |
| Destination and dates                  | Named cities are geocoded; dates use the destination timezone.           |
| `next_week`                             | Next Monday through Sunday, even when the current day is Monday.        |
| `temperature_c`, daily extrema          | Celsius; missing provider measurements remain JSON null.                |
| Current `precipitation_mm`              | Total in the `interval_seconds` preceding `time`, not ongoing rain.      |
| Daily `rain_mm`                         | Rain total in millimeters, excluding snow.                              |
| Daily probability                      | Maximum precipitation probability in percent, including snow.           |
| Weekly temperature summary             | Minimum daily low and maximum daily high across the seven days.         |
| Weekly probability summary             | Maximum daily probability, not probability of any event during a week.  |
| Weekly `rainy_dates`                    | Dates with `rain_mm > 0`; empty means no forecast rain.                  |
| Missing weekly inputs                  | Each summary metric is null if any relevant daily input is null.        |
| City resolution failure                | `ambiguous_city` returns choices; `city_not_found` requests city/country. |

The daily and summary probability key is `precipitation_probability_max_pct`.
Successful results contain `city`, `timezone`, and `period`, plus current readings
or chronological `days`; weekly results also contain `summary`. City-resolution
results instead contain `status`, `period`, `city_query`, and `choices`. Missing
keys, malformed data, and provider/network errors raise. Period `now`, including
omitted periods, returns today's `days` plus `current_temperature_c` (possibly
null) and current observation fields; explicit `today` omits those observations.
Period strings are case-insensitive, trimmed, and normalize spaces/hyphens to
underscores. Unknown strings, including `current`, select `now`; non-strings fail
validation.
The city value `Home` (case-insensitive, trimmed) selects configured home coordinates
without geocoding, just like an empty city. Qualified names such as `Home, Kansas`
are geocoded as city queries.
After region/country filtering, candidates rank by case-insensitive exact city
name, national-capital status (`PPLC`), descending population, then great-circle
proximity to configured home. Missing population ranks below known counts;
complete ties retain provider order. Optional feature codes and population remain
null when absent. No matches return `city_not_found`. Candidate metadata is logged
at debug level; selected city, ranking tuple, criteria, and distance in kilometers
are logged at info level. Routing instructions preserve all user-specified state,
province, region, and country qualifiers in the single `city` argument.

The renderer rounds temperatures and percentages to integers (nearest, ties to
even) and produces compact text, without a model summarization pass:

- Default: “It's light rain, 12 degrees outside now, 1 to 12 degrees, 30% chance of
  light rain.”
- Explicit today: “For today, it's clear, 10 to 20 degrees, no rain.”
- Named city: “In Paris, France, it's clear, 15 degrees outside now, 10 to 20
  degrees, no rain.”
- City and period: “For tomorrow, in Paris, France, it's heavy rain, 10 to 18
  degrees, 80% chance of heavy rain.”

Weekly text uses “For next week”, the most frequent daily condition (earliest on
ties), weekly temperature extrema, and the maximum daily precipitation probability.
That percentage is neither rain-only nor a weekly occurrence probability. Rain
intensity is heavy if any rainy day's condition says heavy, light if all rainy
days say light or drizzle, and otherwise unspecified. Zero rain across selected
days produces “no rain”; missing measurements produce unavailable clauses.
Explicit `now` and `current` requests use the default current-plus-today wording.
Weather retrieval uses the network; model inference is local.

## Historical measurement scope

Measurements were made September 11–13, 2026 on an Intel i5-8500T (six cores,
AVX2, 16 GB RAM) and UHD 630 using OpenVINO/GenAI 2026.3.1 and Transformers 4.57.6.
Model revisions are the pinned revisions above. CPU inference uses an FP32 hint;
the CPU lacks native BF16 and AVX-512/VNNI. GPU uses an FP16 hint. These are
small synthetic diagnostic fixtures, not a broad accuracy or agent-completion
benchmark. No real tool/device handler, live search, or backup provider was run.

Warm latency includes prompt formatting, tokenization, generation, decoding and
validation, excluding model loading, queueing and real tools/network access.
Tokens per wall second includes prefill/application overhead, not decode-only
throughput. P95 uses nearest rank. First-token numbers come from a Transformers
streamer or OpenVINO generation metrics and have different timing boundaries.
RSS includes imports, runtime caches and allocations, not just model weights.

### FunctionGemma CPU tuning

The sweep covered 1/2/4/6 threads across PyTorch FP32 and OpenVINO FP32/INT8/INT4,
with 3/5-thread and attention/cache follow-ups. Four threads offered the best
observed balance; five were similar and six slower. One unmeasured warm-up preceded
three or five passes through six cases: three city temperatures, two light actions
and a no-call greeting. The greeting checks call absence, not prose quality.

| Configuration           | Threads | Median  | P95     | First token | Tokens/s | Exact calls |
| ----------------------- | ------- | ------- | ------- | ----------- | -------- | ----------- |
| PyTorch FP32            | 4       | 1084 ms | 2299 ms | 216 ms      | 17.8     | 18/18       |
| OpenVINO FP32 PA        | 4       | 729 ms  | 1667 ms | 59 ms       | 25.7     | 18/18       |
| OpenVINO INT8 PA        | 4       | 303 ms  | 664 ms  | 46 ms       | 61.9     | 30/30       |
| OpenVINO INT8 SDPA      | 4       | 493 ms  | 886 ms  | 222 ms      | 40.2     | 30/30       |
| OpenVINO INT4 PA        | 4       | 376 ms  | 650 ms  | 46 ms       | 66.7     | 3/18        |

INT8 PA had about 3.6× lower median latency than tuned PyTorch. INT4 throughput
reflects different, incorrect outputs. INT8 weights occupy 256.6 MiB versus
1022.7 MiB FP32; final INT8 peak process RSS with `USE_TORCH=0` was 998 MiB versus
2069 MiB PyTorch. FP32 backend outputs and token counts matched in all 18 paired
checks, including termination tokens. Fresh processes used existing filesystem
caches; these are not cold-disk startup measurements.

### FunctionGemma CPU versus UHD 630

| Metric                            | CPU, four threads | UHD 630 GPU  |
| --------------------------------- | ----------------- | ------------ |
| Stored weights / inference hint   | INT8 / FP32       | INT8 / FP16  |
| Exact calls                       | 30/30             | 30/30        |
| Median / P95 request              | 303 / 652 ms      | 432 / 985 ms |
| First token                       | 45.8 ms           | 43.5 ms      |
| Tokens / request wall second      | 62.5              | 43.2         |
| Tokenizer/pipeline load           | 2.79 s            | 1.96 s       |
| Steady-state process CPU time     | 42.19 s           | 17.31 s      |
| Steady-state wall interval        | 10.79 s           | 15.55 s      |
| Average busy CPU-core equivalents | 3.91              | 1.11         |
| Whole-machine busy CPU            | 77.8%             | 39.5%        |

The matched sequential five-pass runs used PA, one stream, one warm-up and the
same cached artifact. All 30 raw outputs and input/output token counts matched.
GPU latency was 1.42× CPU, but process CPU time fell about 59%. GPU still used
about one host core on average, with per-request interval averages up to 1.97
cores. Process CPU time sums all threads; the measured interval also includes
diagnostic writes. Whole-machine `/proc/stat` utilization includes unrelated work
and cannot all be attributed to this model. Loading is outside the interval;
power/energy and instantaneous utilization peaks were not measured. An initial
unprofiled run measured 544 ms GPU versus 301 ms CPU, showing cache/system-load
sensitivity while still favoring CPU latency.

UHD 630 has 24 EUs, 350–1100 MHz limits, PCI `8086:3e92` and `i915`. Theoretical
FP32/FP16 peaks at 1.10 GHz are 422.4/844.8 GFLOP/s; sustained throughput was not
inferred from those peaks. System OpenCL 3.0 driver `24.35.30872.36` and ICD loader
2.3.5 expose FP32/FP16 and export/import, but no OpenVINO GPU INT8 capability or
packed integer dot-product extension. Byte operations/storage do not imply native
INT8 matrix acceleration.

The 269,040,441-byte source weight file contains 127 int8 constants with
268,042,240 elements. This establishes compressed source storage, not eight-bit
GPU residency. A separate source-IR compilation reports GPU.0, FP16 fully connected
operations, and mixed node types (428 FP16, 92 FP32, 74 int8, 15 uint8 plus index
types). That audit is not the timed GenAI PA graph or a per-kernel arithmetic
profile. Explicit GPU selection used no AUTO/HETERO fallback; host tokenization
and orchestration still run on CPU.

### LFM2.5 CPU versus iGPU routing latency

A September 13 matched run used the compact weather-only registry and routing
prompt, the prepared INT8 export, native attention, repetition penalty 1.05 and
384-token output budget. Six short queries received one warm-up each, followed by
five repetitions (30 measured requests per device), in separate sequential
processes. CPU used four threads. No tool execution or network time was included;
the current multi-capability registry was not the measured workload.

| Metric                         | CPU, four threads | UHD 630 iGPU |
| ------------------------------ | ----------------- | ------------ |
| Median / P95 request            | 517 / 642 ms      | 612 / 703 ms |
| Median first token              | 253 ms            | 193 ms       |
| Mean busy CPU-core equivalents  | 3.97              | 1.61         |
| Mean CPU time per request       | 2.11 s            | 0.98 s       |
| Output tokens per wall second   | 32.3              | 28.2         |
| Correct calls                   | 30/30             | 30/30        |

The iGPU's median request was 18% slower, while process CPU time per request was
54% lower. CPU time sums user/system time across all process threads, so it can
exceed elapsed time. Throughput includes prompt processing and application work;
it is not decode-only throughput. First token precedes call validation and is not
time to a spoken tool result. GPU is the deployment default to leave CPU capacity
for other Home Assistant work; these small sequential runs are not concurrent
speech-service or long-conversation benchmarks. Temporary scripts and raw results
remain outside the repository under `/tmp/opencode`.

### FunctionGemma versus LFM2.5 on CPU

Each deployable INT8 configuration used four threads, greedy one-beam decoding,
repetition penalty 1.0 and native templates/parsers. FunctionGemma uses symmetric
INT8/PA; LFM uses asymmetric INT8/native hybrid attention. Semantic tasks and
budgets match, but tokenizer lengths and rendered prompts differ. This is not an
isolated architecture/quantization experiment or evidence about GGUF/llama.cpp,
GPU speed, other precisions or fine-tuning. Each model ran in a separate sequential
process: 30 speed generations after warm-up and 174 single-pass accuracy cases.

| Six-case speed workload        | FunctionGemma | LFM2.5       |
| ------------------------------ | ------------- | ------------ |
| Exact checks                   | 30/30         | 30/30        |
| Median / P95 request           | 313 / 697 ms  | 817 / 916 ms |
| Median first token             | 46.9 ms       | 562.9 ms     |
| Generated tokens / wall second | 60.2          | 19.6         |
| Mean prompt / output tokens    | 189.5 / 22.2  | 197 / 16.3   |
| Mean busy CPU-core equivalents | 3.92          | 3.96         |
| Model/tokenizer load           | 2.80 s        | 1.18 s       |
| Weight file                    | 256.6 MiB     | 339.0 MiB    |
| RSS after load                 | 622.8 MiB     | 561.7 MiB    |
| RSS after speed suite          | 997.8 MiB     | 959.8 MiB    |
| Whole-process peak RSS         | 1150.2 MiB    | 1237.1 MiB   |
| RSS after owner close          | 360.7 MiB     | 323.9 MiB    |

LFM median latency was 2.61× FunctionGemma, mostly before the first token. Both
used approximately 1 GiB warmed up. RSS includes retained reports and libraries;
these are not independent per-suite peaks. No swapping was observed.

| Suite median latency            | FunctionGemma | LFM2.5  |
| ------------------------------- | ------------- | ------- |
| Multilingual next-tool          | 305 ms        | 976 ms  |
| Translated tool results         | 1120 ms       | 2303 ms |
| Direct multiple choice          | 583 ms        | 284 ms  |
| Answer-recording tool questions | 901 ms        | 670 ms  |
| Plain-excerpt summaries         | 925 ms        | 1242 ms |
| Native-result summaries         | 1407 ms       | 1681 ms |

Direct LFM answers are much terser; many FunctionGemma attempts decline or fail.
These timings include failures, not equal amounts of useful prose.

### Multilingual routing and tool-result fidelity

Ten requests per language used the same seven English tools and required exact
ordered next calls and arguments. Explicit alias normalization did not change
scores. FP32 FunctionGemma had identical per-language counts to INT8.

| Language | FunctionGemma | LFM2.5 |
| -------- | ------------- | ------ |
| English  | 7/10          | 6/10   |
| Chinese  | 5/10          | 5/10   |
| French   | 6/10          | 5/10   |
| Korean   | 4/10          | 6/10   |
| Mixed    | 8/10          | 6/10   |
| Total    | 30/50         | 28/50  |

FunctionGemma reversed a French off request, invented artists in Chinese/Korean
light requests, and extracted a Korean phrase rather than a canonical city.
All 50 INT8 responses parsed, demonstrating that syntax does not establish correct
actions. The English baseline also failed; these are not solely language effects.
Mixed prompts include familiar English action words, and ten cases do not show
that mixed language is generally easier.

LFM often read status before acting. Of 11 targeted follow-ups with synthetic
status results, 5 issued the correct next action. Other replies claimed completion
without an action, asked for confirmation, or reversed a boolean. This is not a
replacement 50-case completion score or equal-turn comparison. Some web-search
versus no-call labels are application-policy choices.

Translated tool-result lexical coverage was 89/112 FunctionGemma and 99/112 LFM
across 32 responses. FunctionGemma FP32 reached 90/112 but retained failures.
**Term matching is not semantic accuracy or language compliance**: copying,
negation and wrong-entity binding can match, while valid paraphrases may not.
LFM more consistently produces Chinese/French/Korean prose, but both mishandle
paused versus playing or unknown versus off. FunctionGemma frequently quotes
foreign reports under English introductions. LFM sometimes says both playing and
paused, or unknown and no unknown state; four English weather responses requested
another web search despite instructions. Native-response trials change both user
language and requested output language, not just one independent variable.
Trials were single-pass without warm-up; script-dependent token lengths and lazy
runtime work prevent treating language timing as controlled throughput rankings.

### Knowledge and synthetic search summaries

The knowledge fixture had 36 questions (six each: shell, Python, math, geography,
music, everyday facts) plus four supplied-evidence controls. Choice positions
were seeded/shuffled/balanced. No code snippets or live search were executed.
Strict scores require the requested answer format; conservative recovery accepts
an unambiguous matching letter/text, excluding echoed options and contradictions.

| Generated-answer metric       | FunctionGemma | LFM2.5 |
| ----------------------------- | ------------- | ------ |
| Strict direct answer          | 0/40          | 21/40  |
| Recovered direct answer       | 2/40          | 22/40  |
| Recovered knowledge-only      | 2/36          | 19/36  |
| Recovered evidence controls   | 0/4           | 3/4    |
| Correct answer-recording tool | 0/40          | 4/40   |

FunctionGemma FP32 also recovered 2/40; a second INT8 shuffle recovered 0/40.
This measures failure to answer through these interfaces, not absence of latent
knowledge. The answer-recording tool differs from natural application tools.
LFM knowledge recovery by domain was 4/6, 1/6, 1/6, 5/6, 3/6, 5/6 in the order
above, remaining weak on tested code evaluation and computation.

FunctionGemma FP32 teacher-forced plain-text answer ranking reached 19/36 knowledge
and 2/4 controls: shell 4/6, Python 2/6, math 1/6, geography/music/everyday 4/6 each.
It selected mean conditional log probability per answer token, excluding EOS,
without a chat template/choice letter. Median/P95 time for four candidate forward
passes was 280/341 ms, not usable answer-generation latency. Uniform guessing is
25%; candidate wording/length biases remain despite normalization, and six items
per domain do not justify broad domain conclusions.

Six synthetic search cases covered weather, a concert, outage timing, conflicting
venue listings, unspecified Wi-Fi, and a misleading page-footer instruction.
The target was at most 70 words using supplied evidence only.

| Required-term coverage        | FunctionGemma INT8 | LFM2.5 INT8 |
| ----------------------------- | ------------------ | ----------- |
| Plain excerpts                | 2/24               | 19/24       |
| Native tool results           | 21/24              | 20/24       |

FunctionGemma declined all six plain requests; the two terms occur in refusals.
Native INT8 summaries met all content requirements in 3/6 manually reviewed
cases; 5/6 met the length limit. It confused an 11:40 update with the 11:15 outage
resolution, omitted known specs, and quoted a misleading footer. FP32 native
coverage was 18/24, also 5/6 within length; increased precision retained errors.
Standalone INT8 plain/native median generation was 1010/1412 ms versus FP32
1827/3367 ms. Those exploratory runs had no repeated trials or warm-up, and first
requests can include lazy initialization. Token statistics exclude exceptions
without generation metrics rather than treating them as zero-token successes.

LFM preserved weather and concert facts but likewise confused outage timing,
misdated a venue update in the plain format, asserted no Wi-Fi where it was merely
unspecified, and repeated a false 99-credit footer price alongside the true 8.
Neither model is established as a dependable unattended search summarizer.
These short synthetic snippets do not test full-page or long-context compression.

### Backup-routing experiment

An eleven-tool music/weather/backup experiment evaluated context-bound
`ask_backup_llm()` and explicit `ask_backup_llm(question=...)`. No backup endpoint
or real device operation was invoked; GPT-6 was a proposed provider, not a tested
backup. Policy assigned reasoning, code, translation, synthesis and general
knowledge to backup, routine music/weather to local tools, and greetings or
missing-argument clarification locally. These are policy labels, not calibrated
intrinsic difficulty judgments.

52 distinct cases (36 English plus 16 multilingual variants) ran with canonical
and seeded shuffled tool order: 104 trials/model/interface, 416 initial generations
and 10 bounded continuations in total. Backup was required in 40/104 trials.
A mixed request received one synthetic local result/continuation only when its
initial local calls exactly matched the reference and omitted backup. This bound
can reject meaningful paraphrases or longer plans; misses do not prove unrestricted
agents could never complete the task.

| Interface     | Model         | Valid backup / required | Recall | Precision | False backup |
| ------------- | ------------- | ----------------------- | ------ | --------- | ------------ |
| Context-bound | FunctionGemma | 3/40                    | 7.5%   | 75%       | 1/64         |
| Context-bound | LFM2.5        | 5/40                    | 12.5%  | 100%      | 0/64         |
| Question      | FunctionGemma | 4/40                    | 10%    | 100%      | 0/64         |
| Question      | LFM2.5        | 6/40                    | 15%    | 100%      | 0/64         |

“Valid” requires the whole batch to pass schema and one-backup-call limits; an
unrelated malformed call can block a shaped backup proposal. Context-bound
FunctionGemma had 4/5 parseable-attempt precision and 4/40 recall, but only 3/4
executable precision and 3/40 recall. Invalid negative plans were not counted as
valid no-backup decisions. High precision from four to six calls does not establish
reliability: missed escalation dominated.

Context-bound median/P95 attempt latency was 529/1172 ms FunctionGemma and
2681/4780 ms LFM; explicit-question latency was 635/1189 ms and 1663/3200 ms.
Times include bounded second generations but exclude actual provider/tool work.
Canonical order ran first in the same process, so order timing is confounded by
cache/warm-up and is not a causal ordering-speed result.

The evaluating assistant manually reviewed 234 distinct proposal/context groups,
covering 272 parsed calls and 20 identifiable malformed fragments, plus no-call
and failed responses. Tool selection and argument fidelity were graded separately;
faithful city translations/paraphrases received semantic credit, while missing,
invented or invalid arguments did not. No external judge or subagent assigned
those grades.

| Argument / complete-plan metric        | FunctionGemma | LFM2.5 |
| -------------------------------------- | ------------- | ------ |
| Correct local arguments, context-bound | 9/30          | 25/47  |
| Correct local arguments, question      | 11/29         | 25/46  |
| Full semantic plans, context-bound     | 35/104        | 48/104 |
| Full semantic plans, question          | 33/104        | 46/104 |
| Literal plans, context-bound           | 35/104        | 40/104 |
| Literal plans, question                | 32/104        | 36/104 |

Argument denominators are proposed parameterized calls, not requests; missing
calls do not enter those denominators. Empty objects on wrong zero-argument tools
earn no tool-selection credit. Failures included invented cities/artists, wrong
forecast periods, volume 140%, and converting 1:15 to 165 instead of 75 seconds.
Explicit backup questions were fully correct/partial/incorrect in 1/2/1 of four
FunctionGemma calls and 3/3/0 of six LFM calls. Rewrites dropped named subjects,
target languages, explanation requests or code-correction requirements.

The experiment does not support these untuned models as sole escalation gates or
their generated questions as the sole handoff context. Any future backup gateway
should retain the original request and verified tool results, enforce bounded
calls, and evaluate actual provider quality/latency separately. No backup gateway
is part of the current local agent. Its Music Assistant tools are separate from
this historical synthetic experiment.
