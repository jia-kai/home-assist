# Home Assistant

A local home assistant with tool-based actions and grounded answers. It runs
LFM2.5-350M through OpenVINO on Intel GPU by default, with CPU execution available.

## Setup

Run all commands from the project root. This is a source-tree development project;
uv manages its environment without building or installing the project as a package.
[uv](https://docs.astral.sh/uv/) installs Python 3.14 and the locked dependencies
into `.venv`, including CPU-only PyTorch for model export.

```sh
uv sync --locked
# Public LFM INT8 export; no Hugging Face login required.
uv run python -m tools.prepare_llm --model lfm download
USE_TORCH=0 uv run python -m tools.prepare_llm --model lfm compile
```

Artifacts and compilation caches live in `.cache/hoast/`, relative to the working
directory. GPU execution requires a working Intel OpenCL driver. Alternative
models and backend details are covered in [docs/llm.md](docs/llm.md).

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
```

Use `/reset` to clear history and EOF to exit. Conversational output goes to
stdout; diagnostics go to stderr and `.cache/hoast/diagnostics/agent.log`
(or the selected `--cache` root).

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
- **Pause/resume control only the existing native stream**, including a supported
  Spotify Connect source. They issue no search, queue replacement, queue-resume,
  or source-selection requests. An idle retained external AudioSource, such as
  Spotify Connect, resumes with native `players/cmd/play` after a read-only check
  that its source URI is not an MA queue. Idle players without such a source still
  refuse fallback; resume those in the source app or request new music. A conversational
  “play music” or “continue music” is intended to route to resume; a new title or
  artist requests a new mix. Player and source must support native pause/unpause.
- Group/sync routing is resolved by the client. State checks are not atomic with
  Music Assistant's command handling: the server can still fall back if state changes
  during dispatch. Detected source drift is reported without retries or restoration.
  `confirmation: "requested"` means acknowledged, while `"observed"`
  means the expected state was seen in an immediate snapshot; neither proves
  audible playback or future recommendation supply.

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

Actionable JSON statuses are `player_required`, `cannot_resume`, `not_playing`,
`not_found`, `ambiguous`, and `cannot_volume`. These retain their full result on
stdout, even when the exit code is 2. A refusal reporting
`confirmation: "requested"` can follow a command that was sent before source drift
was detected.

## Local agent and model API

`hoast.agent.LocalAgent` accepts weather alone or all five tools: `get_weather`
and the four music tools above. Each turn allows at most four calls, including at
most one music action. Standalone “play” and “stop” deterministically resume and
pause through `Session.request_tools`, bypassing model inference while retaining
history. Matching ignores case, surrounding whitespace, and terminal `. ! ?`
punctuation. Longer requests, including “play music”, still use model routing and
can misroute. Without music configured, shortcuts say “Music isn't configured.”
Standalone “louder” and “quieter” also bypass inference and adjust by five points;
“quiter” is accepted as a spelling alias for “quieter”.

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
