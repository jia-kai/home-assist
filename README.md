# Hoast

A local home assistant for weather, music and light control, with English and
Chinese voice support. Language and speech models run on your machine; weather
requests and music services still need network access.

Hoast uses LFM2.5 for language, SenseVoiceSmall for recognition, and MeloTTS for
speech. The language model uses an Intel GPU by default, with a CPU option.
Speech runs on the CPU.

## What you can ask

- “What's the weather today?” or “Will it rain in Seattle tomorrow?”
- “Play some music,” “stop music,” or “next song.”
- “Play Blue in Green by Miles Davis.”
- “Set music volume to 30 percent.”
- “Turn on the light” or “关灯。”

Music requires Music Assistant; light control requires a compatible Kasa relay.
These integrations are optional. Weather uses Celsius and defaults to your home
location. Small local models can misunderstand requests, so check the result of
important actions.

## Setup

Run commands from the project root. The current bootstrap targets Linux with:

- Bash, Git, `taskset`, and a running systemd user manager.
- An Intel GPU with a working OpenCL driver. Preparation compiles GPU models even
  if you intend to use CPU inference afterward.
- At least **7 GiB of available RAM** during model preparation, plus disk space
  for dependencies and models.
- Internet access for the initial downloads, and curl or wget if uv is missing.

### 1. Prepare the models

```sh
./prepare.sh
```

This installs uv if needed, creates a Python 3.14 environment in `.venv`, and
prepares the language, speech and voice-activity models. It does not configure or
start services.

**Prepare once.** If your models are already prepared, skip this step. Existing
model outputs are not overwritten; use fresh output paths for a new export.
For offline setup, other models and custom paths, see the
[speech preparation guide](docs/speech2-native.md) and
[language model guide](docs/llm.md).

### 2. Configure your home

For a new installation:

```sh
cp config.example.toml config.toml
```

Edit `config.toml`:

- Set `[weather].latitude` and `longitude` to your home coordinates.
- Remove `[music]` unless you have configured Music Assistant below.
- Remove `[switch]` unless you have configured your Kasa relay below.
- For voice mode, set `[satellite].host` to your voice satellite's address.
  Text mode does not require a satellite.

Keep existing configuration when upgrading. `config.toml` and local secret files
are gitignored; never put access tokens or API keys in the TOML file.

### 3. Try text mode

```sh
.venv/bin/python -m hoast --config config.toml --text
```

Use `/reset` to clear the conversation and Ctrl+D to exit. To run the language
model on CPU instead:

```sh
.venv/bin/python -m hoast --config config.toml --text --device CPU
```

## Voice mode

Voice mode receives audio from an ESPHome voice satellite and plays replies
through the computer's audio output. It does not use the computer's microphone
automatically. See the [reSpeaker guide](respeaker/README.md) for supported
hardware, firmware and installation.

Update the satellite table in `config.toml`:

```toml
[satellite]
host = "192.168.1.30"
language = "auto"
capture_seconds = 6.0
# For an encrypted API, also set:
# key_env = "ESPHOME_API_KEY"
```

If encryption is enabled, put `ESPHOME_API_KEY` in `.env` or the process
environment. Use the key configured on the device.

```sh
.venv/bin/python -m hoast --config config.toml
```

Wait for the initialization announcement, then say the satellite's wake word
and your request. Capture ends after about half a second of silence, up to the
configured maximum. Each voice command starts a fresh conversation; include the
full request rather than relying on an earlier command's context.

To replay captured audio while troubleshooting:

```sh
.venv/bin/python -m hoast --config config.toml --debug-audio
```

Replay adds time to each turn. For local-microphone development and advanced
speech options, see [voice satellite development](docs/voice-satellite.md) and
the [speech guide](docs/speech.md).

## Music

1. Start Music Assistant and configure your music providers and player. The
   assistant currently requires the selected player to use **AirPlay 2 PTP**.
2. In Music Assistant, open **Settings → Profile** and create a long-lived access
   token. Store it in the project-root `.env` or export it in your shell:

   ```dotenv
   MUSIC_ASSISTANT_TOKEN=YOUR_ACCESS_TOKEN
   ```

3. Set the music configuration, changing the server URL if needed:

   ```toml
   [music]
   server_url = "http://localhost:8095"
   player_id = "YOUR_PLAYER_ID"
   token_env = "MUSIC_ASSISTANT_TOKEN"
   ```

4. List players, then replace `YOUR_PLAYER_ID` with the desired player's ID:

   ```sh
   .venv/bin/python -m hoast.music_cli --config config.toml players
   ```

Music Assistant must be available when Hoast starts. Remove `[music]` to disable
this integration.

“Play” resumes existing playback. A request with a song title or artist starts
a new mix; specifying the artist helps avoid choosing a different recording.
“Stop” uses Music Assistant's pause control, which can stop sources that cannot
be paused.

You can check the connection without loading any models:

```sh
.venv/bin/python -m hoast.music_cli --config config.toml status
```

## Light control

Set your compatible Kasa relay's IP address in `config.toml`:

```toml
[switch]
ip = "192.168.1.40"
```

This enables on/off control of the connected light. Remove `[switch]` to disable
it. See the [switch guide](docs/switch.md) for device setup and direct controls.

## Docker

The included Compose stack runs Hoast and Music Assistant. Prepare the models
and configure voice mode first. Review the Compose CPU affinity, memory limit,
UID/GID and audio paths for your machine; the supplied settings target UID 1000
with PipeWire/PulseAudio and Intel graphics.

Create `.env` and `music.env` from their example files if they do not already
exist. Keep any existing secrets. Set your tokens/API key in `.env`, and set
`AIRPLAY_IP` in `music.env` to the fixed address of your AirPlay receiver.

```sh
docker compose up -d --build
```

The optional AEC reference bridge is disabled in the Compose file. See the
[reSpeaker guide](respeaker/README.md) before enabling it.

Do not run native Hoast and its container at the same time. If a systemd service
manages your deployment, use that service to stop/start Hoast rather than
launching a competing Compose instance.

## Troubleshooting

- **Model preparation fails:** check `.cache/hoast/diagnostics/prepare/` and
  `./prepare.sh --help`. If outputs already exist, use them directly or choose
  fresh output paths; do not delete working models just to rerun setup.
- **No voice input or reply:** check the satellite address, API key and computer's
  audio output. Use `--debug-audio` to hear what was captured.
- **Music is unavailable:** check that Music Assistant is running, the token and
  player ID are correct, and the player uses AirPlay 2 PTP.
- **The container restarts:** inspect its logs and check for memory-limit kills.
  The supplied memory limit may need adjustment for your workload. Input/output
  bounds and memory observations do not guarantee OOM-free operation.

For the current container:

```sh
docker logs --timestamps --tail 100 hoast
```

For installations managed by the user `hoast.service`, the journal retains
output across container recreation:

```sh
journalctl --user -u hoast.service --since '1 day ago'
```

Logs can contain spoken requests and local device details; review them before
sharing. Memory observations are recorded at voice stages and every minute
without forced garbage collection. `python -m hoast` disables transparent huge
pages for its process before loading inference libraries; host settings and other
services are unchanged. See the [memory diagnostics guide](docs/memory-diagnostics.md)
for policy verification, periodic mapping observations and preserving OOM evidence.
Advanced budgets and optional reclamation
are described in the [language model guide](docs/llm.md).

## Further reading

- [Speech setup and runtime options](docs/speech.md)
- [Language model setup and advanced configuration](docs/llm.md)
- [Voice satellite development](docs/voice-satellite.md)
- [Fine-tuning your language model](finetune/README.md)

## License

Project code is licensed under the [MIT License](LICENSE). Downloaded models and
third-party dependencies have their own licenses; review them before redistribution.
