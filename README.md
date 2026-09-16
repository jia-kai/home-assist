# Home Assistant

A local home assistant with tool-based actions and grounded answers. It runs
LFM2.5-350M through OpenVINO on Intel GPU by default, with CPU execution available.

## Setup

This is a source-tree development project. The bootstrap installs
[uv](https://docs.astral.sh/uv/) into user-local storage if needed, installs Python
3.14 and the locked dependencies into `.venv`, and prepares every model used by the
default assistant: LFM2.5, Whisper-small, and English/Chinese Kokoro.

```sh
./prepare.sh
# Also prepare the optional gated FunctionGemma model, after obtaining HF access:
./prepare.sh --with-functiongemma
```

The script can be invoked from another directory and always prepares this checkout.
It reuses verified downloads, prepares LFM's GPU/CPU caches, provisions Chinese's
Python 3.12 environment, and runs the compact kernel/bilingual speech check. Model
steps run serially with at most two CPU cores and a 4 GiB memory cap. Detailed logs
are saved under `.cache/hoast/diagnostics/prepare/`.

Prerequisites are Linux, Bash, a C++17 compiler, `systemd-run`/`systemctl` with a
running user manager, `taskset`, and a working Intel OpenCL driver for UHD 630.
The resource guard requires at least 7 GiB available memory before each model step.
Network access is needed for missing dependencies/models; bootstrapping uv also
needs curl or wget. System packages and configuration files are not modified.
If uv was newly installed, add its directory (normally `$HOME/.local/bin`) to your
shell's PATH before using the `uv` commands below. The bootstrap adjusts PATH for
its own run without editing shell startup files.

Other commands below run from the project root. Artifacts and compilation caches
live in `.cache/hoast/`. Alternative models and backend details are covered in
[docs/llm.md](docs/llm.md).

Copy the example configuration, then set your home coordinates in decimal degrees:

```sh
cp config.example.toml config.toml
```

`config.toml` is gitignored. Configure the optional `[music]` table as described
below, or remove it to disable music integration.

The required `[weather]` table accepts finite numeric WGS84 `latitude` (−90 to 90)
and `longitude` (−180 to 180). Replace the example coordinates with your own.
Unknown configuration keys are rejected.

## Run

```sh
uv run python -m hoast --config config.toml
# Use CPU instead of the default Intel GPU:
uv run python -m hoast --config config.toml --device CPU
# Explicit keyboard/text mode:
uv run python -m hoast --config config.toml --text
```

Voice mode connects to `[satellite].host` in `config.toml`, receives command audio,
runs STT and the tool-based agent, then plays its bounded reply through system
audio. Startup warms English and Chinese TTS, feeds both generated clips to STT,
then warms LLM inference without dispatching tools. Listening starts after warm-up.
The satellite stays busy until playback drains. A gap of **more than 30 seconds**
between the previous recognized capture's end and the next capture's first audio
starts a fresh LLM session. Capture ends adaptively after **600 ms of detected
silence following speech**. `satellite.capture_seconds` is the hard maximum
(six seconds by default), not a mandatory wait. Initial silence waits for speech
or the maximum; short pauses are tolerated. Silero uses 32 ms frames, so endpoint
notification has frame/packet granularity while retained trailing silence is 600 ms.

To hear exactly what the controller captured, run:

```sh
uv run python -m hoast --config config.toml --debug-audio
```

Debug mode plays a high start tone, the captured command, and a lower end tone
through system audio before STT. It deliberately adds the replay duration to each
turn; the satellite stays busy until all processing/playback finishes.
Replay and synthesized replies share the TTS playback API and queue. Linux playback
uses an available PipeWire/PulseAudio adapter to mix with the satellite's audio.

The system prompt gets only music **on/off** state (or explicit unavailability)
before every request. It does not include background title, artist, volume, source,
or player metadata. Explicit user requests and tool answers retain their own context.
With `[switch]` configured, the agent can turn its light relay on or off. Observed
music starts/resumes with available track metadata produce
“Now playing {title} by {artist}”; the mix seed is not used as the current track.
TTS input is limited to 400 characters at sentence/word boundaries after consuming
the complete agent response; the text console retains the full grounded answer.

In text mode, use `/reset` to clear history and EOF to exit. Conversational output goes to
stdout; diagnostics go to stderr and `.cache/hoast/diagnostics/agent.log`
(or the selected `--cache` root).

`--help` describes options and displays defaults. The LFM CPU budget defaults to
`--threads auto`: one core for GPU execution, two for CPU. Explicit `1` or `2`
overrides it. GPU inference uses FP16 computation with INT8-stored weights.

## Speech

Local speech uses Kokoro with a CPU graph and zero-copy INT8-weight GPU decoder
convolution for TTS, and faster-whisper small INT8 for STT. The TTS runtime targets
Intel UHD 630; both engines default to two inference threads. Initialize and test from the
project root:

```sh
uv run python -m tools.prepare_stt
uv run python -m tools.prepare_tts
uv run python -m hoast.tts "The weather is sunny." --output /tmp/speech.wav
uv run python -m hoast.stt /tmp/speech.wav
# Compact kernel correctness and in-memory TTS-to-STT check:
uv run python -m tools.guarded_run --threads 2 --memory-gib 4 --timeout 240 -- .venv/bin/python -m tools.check_speech
# Prepare automatic Mandarin and mixed English/Chinese speech:
uv run python -m tools.guarded_run --threads 2 --memory-gib 4 --timeout 900 -- .venv/bin/python -m tools.prepare_tts --chinese
uv run python -m hoast.tts "Hello world. 你好，欢迎回家。" --output /tmp/chinese.wav
uv run python -m hoast.stt /tmp/chinese.wav --language zh
```

Han-containing utterances use the official Chinese Kokoro v1.1 bilingual voice.
The Chinese model and phonemizer initialize on first use and are cached by the
`TTS` instance; English-only requests do not initialize them. Speech CLIs cap the
whole process to two CPU cores. Both language models share the same GPU context
and compiled kernel cache. Add `--chinese` to the compact check to cover bilingual
speech after its preparation.

See [docs/speech.md](docs/speech.md) for voice/language options, reusable Python
engines and local artifact paths. The linked study documents preserve the tuning
journey, rejected approaches and measured results.

For local-microphone wake-word development and the reSpeaker ESPHome interface,
see [Network microphone development](docs/voice-satellite.md). The simulator uses
OHF Linux Voice Assistant with microWakeWord; `python -m hoast.voice` receives
native wake/audio events and emits transcripts for the agent.

## Weather

Open-Meteo supplies observations, forecasts, and city geocoding.
Weather requests default to home; named cities are resolved
through geocoding. “Next week” means next Monday through Sunday in the destination
timezone. Model inference is local; weather retrieval needs network access.
The city value `Home` also selects configured home coordinates, ignoring case
and surrounding whitespace.
When multiple cities match, region/country qualifiers filter candidates first.
Ranking then prefers exact names, national capitals, larger populations, and
finally great-circle proximity to home. Missing population ranks below known
counts; ties across all criteria use provider order. Include a region or country
to disambiguate a city. Candidate metadata is logged at debug level; selection,
ranking criteria, and distance are logged at info level.

Spoken responses are brief. Unknown or ambiguous locations prompt a request for
the exact location. Candidate lists stay in diagnostics, long labels are
shortened, and music confirmations mention at most the first artist.

With `now` or no specified period, answers combine today's forecast with the
current temperature. Explicit `today` omits the current temperature. Answers
identify named locations and requested forecast periods. Temperatures are
Celsius, rounded to integers; rain chances are rounded percentages.
Unknown period strings (including `current`) select `now`. Recognized periods
ignore case and surrounding whitespace; `next week` and `next-week` select `next_week`.

## Music

### Setup

Music tools target Music Assistant 2.10.3's authenticated HTTP API.
In the Music Assistant UI, open **Settings → Profile** and create a long-lived
access token.
Put it in the project-root `.env` (already gitignored), or export the variable in
your shell. You can copy `.env.example` to `.env` as a starting point:

```dotenv
MUSIC_ASSISTANT_TOKEN=YOUR_ACCESS_TOKEN
```

Credentials stay outside TOML. The process environment takes precedence over the
selected dotenv file, including an explicitly empty value (which is an error).
Dotenv values are not interpolated and do not modify the process environment.

The direct CLI needs **no LLM, model download, compilation, or inference**. After
`uv sync --locked`, discover players and inspect their state:

```sh
uv run python -m hoast.music_cli players
uv run python -m hoast.music_cli --player YOUR_PLAYER_ID status
```

Without `--config`, it uses `http://localhost:8095`, an empty player ID, and
`MUSIC_ASSISTANT_TOKEN` from the environment or working-directory `.env`.
It does not implicitly load `config.toml`. An empty player ID selects a player
only when exactly one enabled, available nonprotocol player exists; otherwise
the JSON result asks for a player and includes choices. `players` lists choices
without selecting one.

Configure the optional `[music]` table in `config.toml`. Replace `YOUR_PLAYER_ID`
with the intended player's ID from the `players` command:

```toml
[music]
server_url = "http://localhost:8095"
player_id = "YOUR_PLAYER_ID"
token_env = "MUSIC_ASSISTANT_TOKEN"
```

All three fields are optional strings. Defaults are `http://localhost:8095`, an
empty player ID, and `MUSIC_ASSISTANT_TOKEN`. Change the server URL if your server
runs elsewhere; it must be an HTTP(S) origin without credentials, query, fragment,
or API path. The interactive runner accepts `--env-file` to select a token file.

### Direct CLI

**Global options must precede the subcommand.** `--player` and `--server` override
TOML/defaults; `--env-file` selects a dotenv file (default `.env`). Explicit
`--config` reads the system TOML and therefore requires `[weather]`; a weather-only
file uses music defaults for this CLI.

```sh
uv run python -m hoast.music_cli --config config.toml --player YOUR_PLAYER_ID status
uv run python -m hoast.music_cli --player YOUR_PLAYER_ID pause
uv run python -m hoast.music_cli --player YOUR_PLAYER_ID resume
uv run python -m hoast.music_cli --player YOUR_PLAYER_ID volume louder
uv run python -m hoast.music_cli --player YOUR_PLAYER_ID volume quieter
uv run python -m hoast.music_cli --player YOUR_PLAYER_ID volume YOUR_VOLUME_LEVEL
uv run python -m hoast.music_cli --player YOUR_PLAYER_ID play --title "YOUR_SONG_TITLE" --artist "YOUR_ARTIST_NAME"
```

Replace the `YOUR_…` placeholders before running these commands. Volume levels
must be integers from 1 to 100.

### Playback behavior

The CLI and agent share the same music handlers. `stop` is an alias for `pause`
and preserves the retained stream.
`play` with both title and artist omitted, empty, or whitespace-only delegates to
native resume, with the same state-dependent success or refusal as `resume` and
no search or new stream. A nonblank title and/or recording artist requests a new
mix; `--author` aliases `--artist`. Title/artist matching is exact after Unicode,
case, and whitespace normalization. Provider duplicates and version-only variants
with the same normalized title and full artist set collapse to the first available
ranked result; there is no version selector. Title-only requests choose the first
available exact-title match in Music Assistant's search order, even when covers
by other artists also match. Specify `--artist` to constrain the performer; that
constraint is never relaxed. Different collaborator sets matching an explicit artist can still
require clarification. Artist-only homonyms can collapse because names alone do
not establish artist identity. Missing or remaining ambiguous matches do not
start playback; candidates stay in diagnostic results and logs.

- **New play replaces the resolved active queue** with an Endless Mix seeded by
  the match. “Infinite” playback means provider-backed recommendations and dynamic
  refill, not a guarantee of unlimited music. The seed may not play first.
- **Play/resume and pause use MA's player controls**: `players/cmd/play` and
  `players/cmd/pause`. MA handles idle-queue restoration, source capabilities,
  protocol fallbacks, and group/sync routing. The selected player ID is sent to
  MA; effective group state is read for observations. Play can restore an idle
  queue or source even when local source/capability metadata is absent. Pause can
  become a stop when MA cannot pause the output; observed stops are reported as
  stopped. Empty queues and unsupported sources can still be rejected by MA.
- **What is playing?** uses the read-only `what_is_playing()` tool. Its observed
  result renders “Now playing {title} by {artist}” when both labels are available;
  paused/idle playback and missing metadata are described without inventing names.
  Play/resume automatically call this same lookup, reuse fresh post-command
  observations, and include its result as `playback` for text rendering. Explicit
  new mixes also read back the player after the queue operation. Requested-only
  starts retain acknowledgement wording until playback is observed.
- **Next song / switch song** uses the `music_next` tool, sending
  `players/cmd/next` once. MA chooses the appropriate external source, queue, or
  native next action. A changed observed media identity confirms advancement;
  delayed/unchanged/missing metadata yields “Next song requested.” Next is never
  retried. The direct CLI equivalent is `python -m hoast.music_cli next`.
  The read-only query is `python -m hoast.music_cli now-playing`.
- Explicit play/pause intents use directional commands, so asking to play an
  already-playing player does not pause it. The frontend button uses a toggle
  (and a Stop action for stop-only sources/radio); voice pause delegates the
  server's pause behavior. State checks are not atomic with dispatch.
  `confirmation: "requested"` means acknowledged, while `"observed"`
  means the expected state was seen in an immediate snapshot; neither proves
  audible playback or future recommendation supply.

Verified against official MA 2.10.3
[player commands](https://github.com/music-assistant/server/blob/3e21f8293fbcd8710dc2d3c8afc2c94418187cf9/music_assistant/controllers/players/controller.py)
and the current official frontend's
[Play button](https://github.com/music-assistant/frontend/blob/f838bc60c42c0b04077179d658e1d70130b53193/src/layouts/default/PlayerOSD/PlayerControlBtn/PlayBtn.vue)
and [Next button](https://github.com/music-assistant/frontend/blob/f838bc60c42c0b04077179d658e1d70130b53193/src/layouts/default/PlayerOSD/PlayerControlBtn/NextBtn.vue).

### Volume

Volume works while idle or playing and changes only volume, without playback,
queue, or source changes. `louder` and `quieter` adjust five percentage points,
clamped to 1–100; quieter at an existing zero is a no-op. An absolute value must
be an integer from 1–100. Invalid CLI arguments are rejected before client setup
or any mutation. Group/sync targets use the effective group's reading and volume
command. `players` and `status` expose group-aware `volume_level` (null when
unavailable); `status` resolves the effective target, while `players` lists each
target's own reading. Unsupported volume control or a missing relative reading
returns `cannot_volume`; an absolute setting can proceed without a current reading.
Volume results distinguish `level` (target) from `observed_level` (readback).
`volume_set` is `observed` only when immediate readback matches, otherwise
`requested`; `volume_unchanged` is an observed no-op. Routing or volume drift
before dispatch refuses mutation; routing drift afterward reports `cannot_volume`
with `confirmation: "requested"`, without retrying.

### Output and diagnostics

Stdout contains one compact JSON result, without a tool-call wrapper. Diagnostics
go to stderr; exceptions retain tracebacks, chains, and notes in
`.cache/hoast/diagnostics/music-cli.log`, relative to the working directory, with
the resolved token redacted. Commands are never retried.

| Exit | Meaning                                                                  |
| ---- | ------------------------------------------------------------------------ |
| 0    | Read succeeded, command accepted, or already in the requested state.       |
| 1    | Configuration, credential, tool validation, transport, or execution error. |
| 2    | CLI syntax error, or an actionable result below.                           |

Selection/search/volume refusals include `player_required`, `not_found`, `ambiguous`,
and `cannot_volume`. They retain their result on stdout even with exit code 2.
MA player-command errors, including empty-queue and unsupported-source failures,
are execution errors with exit code 1. An unconfirmed accepted command has
`confirmation: "requested"`; it is not retried.

## Local agent and model API

The shared concise routing prompt is in `hoast/prompts.py`. The offline
[fine-tuning toolkit](finetune/README.md) builds reproducible English, Chinese,
code-switched, and STT-confusion examples using production schemas and the official
LFM template. Generate the dataset with `uv run python -m finetune.generate`.

`hoast.agent.LocalAgent` accepts weather plus optional complete core music tools
(`pause_music`, `resume_music`, `play_music`, `volume_music`). The `music_next`
and `what_is_playing` extensions require that music group; `set_light` is independently optional.
Production music registration includes all six music tools. Each turn allows
at most four calls, including at most one music action and one light action.
Standalone “play” and “stop” deterministically resume and
pause through `Session.request_tools`, bypassing model inference while retaining
history. Matching ignores case, surrounding whitespace, and terminal `. ! ?`
punctuation. Longer requests, including “play music”, still use model routing and
can misroute. Without music configured, shortcuts say “Music isn't configured.”
Standalone “louder” and “quieter” also bypass inference and adjust by five points;
“quiter” is accepted as a spelling alias for “quieter”.
“Next”, “next song”, “next track”, “switch song”, “skip song”, and “skip track”
directly dispatch `music_next` with empty arguments.
“What is playing”, “what's playing”, and “what song is playing” directly dispatch
`what_is_playing`; the query is read-only and does not count as a music mutation.

Tool schemas live in `declare_music_tools()` (`hoast/music.py`),
`declare_weather_tool()` (`hoast/weather.py`), and `declare_light_tool()`
(`hoast/lights.py`). Runtime clients bind these declarations to their handlers;
`finetune/tooling.py` aggregates the full non-executable catalog for dataset work.
`hoast/llm_cli.py:main` assembles enabled tools from configuration and
passes them into `ToolRegistry` (`hoast/llm.py`), whose `schemas()` supplies the
LLM-facing tool list and whose dispatch validates and invokes handlers.
`LocalAgent.__post_init__` validates renderer-compatible registry composition;
its allowed-name sets do not register tools or control MA playback semantics.

Ambiguous control requests default to music. A model response without a tool call
prompts a clarification without changing playback. The agent stores grounded
spoken summaries in model history; full tool results remain available in
diagnostics and through the raw session API.

Invalid generated calls receive up to two repair attempts before any tool runs.
If repair fails, the agent asks you to rephrase and preserves prior history.
Other errors and closed, abandoned answer streams reset history. Tool execution
is never retried.

See [docs/llm.md](docs/llm.md) for model preparation, raw `Session` API usage,
agent contracts, and speed/quality measurements. Small models can misroute
requests or hallucinate facts; the documented measurements cover narrow
diagnostic workloads.

## Development

```sh
uv run pytest
uv run pyright hoast tools tests
uv run ruff check hoast tools tests
uv run ruff format --check hoast tools tests
```

Tests use embedded fixtures and temporary directories, without downloading models.
Configure application logging once with `hoast.logging.configure_logging`;
preparation commands retain full failures under `.cache/hoast/diagnostics/`.

## License

The project code is licensed under the [MIT License](LICENSE). Downloaded models
and third-party dependencies are covered by their respective licenses.
