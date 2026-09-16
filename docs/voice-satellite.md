# Network microphone development

## Selected stack

Use **ESPHome + microWakeWord** on the XIAO ESP32S3, with the XVF3800 doing
microphone-array processing. For development, use the Open Home Foundation's
**Linux Voice Assistant (LVA)** with this machine's microphone. Hoast connects
directly to LVA using `aioesphomeapi`; the ESPHome device is the intended compatible
target pending firmware build and hardware testing. Home Assistant Core is not
needed for the direct connection.

```text
Local mic → LVA / microWakeWord ── ESPHome native API ──→ hoast.voice → STT
                                                              ↓ transcript
                                                        hoast agent CLI

Later: XVF3800 → XIAO / microWakeWord ── same native API ──→ hoast.voice
```

The **satellite is the TCP server**, normally on port 6053. The host connects and
subscribes; the satellite then sends wake requests and command audio. The receiver
negotiates `API_AUDIO`, carrying mono 16 kHz signed little-endian 16-bit PCM over
the native TCP connection. It does not use a separate UDP audio listener.

### Official sources checked September 15, 2026

- [Seeed's XVF3800 Home Assistant guide](https://wiki.seeedstudio.com/respeaker_xvf3800_xiao_home_assistant/)
  recommends the [formatBCE ESPHome integration](https://github.com/formatBCE/Respeaker-XVF3800-ESPHome-integration).
  That integration is community-maintained and explicitly linked by the vendor.
- [ESPHome microWakeWord](https://esphome.io/components/micro_wake_word/) and
  [voice assistant](https://esphome.io/components/voice_assistant/) document local
  wake detection and native voice-pipeline control.
- [OHF Linux Voice Assistant](https://github.com/OHF-Voice/linux-voice-assistant)
  implements a Linux satellite with microWakeWord and the ESPHome native API.
- [OHF pymicro-wakeword](https://github.com/OHF-Voice/pymicro-wakeword) provides
  streaming microWakeWord inference and feature extraction on Linux.
- [ESPHome aioesphomeapi](https://github.com/esphome/aioesphomeapi) supplies the host
  protocol client used here.

This choice reuses existing satellite software and the hardware firmware's
protocol. A Wyoming satellite or an openWakeWord detector would target a different
transport or inference engine.

## Run with a local microphone

### Interactive wake-only test

After preparing the satellite, run this single command to hear **“I'm here”**
whenever **“Okay Nabu”** is detected:

```sh
uv run python -m tools.wake_test
```

It starts its own satellite and native API client, replaces the wake chime with
the spoken reply, and rearms after playback. No STT or agent process is needed.
Stop an existing satellite on port 6053 before starting this test. Ctrl-C closes
the client and stops its satellite. Use `--input-device 'YOUR MICROPHONE NAME'`
to select a microphone. The reply is synthesized once using the prepared Kokoro
model and cached in `.cache/hoast/wake-test/im-here.wav`; output uses the satellite's
default playback device. Detailed diagnostics are in
`.cache/hoast/diagnostics/wake-test.log`.

### Full transcript flow

For the complete agent and system-audio response loop, configure:

```toml
[satellite]
host = "127.0.0.1"
port = 6053
capture_seconds = 6.0
language = "auto"
# key_env = "ESPHOME_API_KEY"  # Set this secret in .env for an encrypted ESP device.
```

Start the satellite independently, then run `uv run python -m hoast --config config.toml`.
Startup synthesizes English and Chinese warm-up clips without playing them, feeds
both through STT, then warms LLM inference without dispatching any tools. It fails
with logged diagnostics if required prepared speech models are missing. Native
subscription starts only after warm-up completes.

The combined agent uses native CPU activations instead of the optional fused
Snake extension: real startup testing found a CPU-plugin crash with that extension
and the LLM loaded together. GPU convolution offload remains enabled. Standalone
TTS commands retain their own activation configuration.

For each recognized command, the controller refreshes music status in the system
prompt, runs the grounded agent and its configured weather/music/light tools,
prints the answer, and speaks at most 400 characters through system audio.
Complete sentences/words are preferred when truncating. The native run ends only
after playback drains, so the satellite cannot trigger another turn during TTS.
This is half-duplex; barge-in is not enabled. Streaming Silero VAD finishes
capture after 600 ms of detected silence following speech, rather than waiting
for the configured maximum duration. Initial silence does not trigger the
endpoint, and speech resumption resets the trailing-silence counter.

Background system music context contains only on/off state, with unavailable or
unconfigured status kept explicit. Titles, artists, volume and source identifiers
are excluded from this context; `what_is_playing` supplies metadata for explicit
queries and verified playback responses. Routing diagnostics distinguish the
LLM-selected tool batch from implicit lookups inside playback handlers. Raw model
output is retained at debug level in the agent diagnostic log.

Use `--debug-audio` with `python -m hoast` or the standalone `python -m hoast.voice`
to replay each captured command before transcription. Playback is a 120 ms high
start tone, an 80 ms gap, the original command audio, another gap, and a lower
end tone. It drains before STT starts and does not replay warm-up clips. The
satellite remains busy throughout. Cancellation stops additional submissions
between blocks, drains queued output, and suppresses STT for the cancelled command.

Replay uses the application's existing `TTS.play_samples` and shared playback
queue. The complete cue/command waveform is resampled from 16 to 24 kHz once,
avoiding seams between queued blocks; the original captured PCM still goes to STT.
The standalone STT CLI owns one playback-only TTS instance for replay without
loading synthesis models. Linux output prefers PipeWire/PulseAudio adapters to
avoid direct ALSA `dmix` contention with the satellite. `wait_playback` closes a
drained queue; shared ownership does not keep the physical device open forever.

Conversation reuse is based on capture timestamps: start a new session if the
next first PCM arrives more than 30 seconds after the previous nonempty command
capture ended. Exactly 30 seconds reuses the session. Empty STT does not extend it.
The existing `[switch].ip` enables the verified `set_light(on=true/false)` tool.
Music confirmations use actual observed track labels and mention only the first
structured artist. Missing metadata or requested-only playback retains a cautious
confirmation rather than claiming “Now playing”.
`what_is_playing()` is the shared read-only lookup used for these confirmations;
play/resume call it implicitly and render its `playback` result when confirmed.
“Next song” and “switch song” dispatch `music_next`, using MA's player-level Next
command once. MA handles idle-queue restoration for Play and source/group routing
for all transport controls, matching its player controls rather than restricting
the agent to an already-paused native stream.

Disconnect cancels pending inference-to-tool dispatch and suppresses pending TTS.
Already dispatched effects persist; audio already playing is drained before
shutdown or reconnect. Inference runs on one worker while the native network loop
continues processing keepalives and cancellation.

Prerequisites: Linux with a working PulseAudio or PipeWire PulseAudio server,
microphone input, `libmpv`, Git, and uv. On Debian/Ubuntu the upstream documented
packages include `libmpv-dev`, `pulseaudio`/`pipewire`, and `alsa-utils`. The helper
does not install system packages or change the default microphone.

Run from the project root:

```sh
uv sync --locked
uv run python -m tools.voice_satellite --prepare
uv run python -m tools.voice_satellite
```

The helper installs LVA in `.cache/hoast/voice-satellite/.venv` using Python 3.13,
separate from hoast's Python 3.14 environment. It pins the LVA source revision
`7c6fbaa4ee3c9a2cdd25803ed40b32e108a99a4a`, `pymicro-wakeword==2.4.1`, and
`pymicro-features==2.0.2`. Other satellite dependencies follow that upstream
revision's constraints. The host's `aioesphomeapi==45.3.1` is locked in `uv.lock`.

The satellite binds to **127.0.0.1:6053**, uses one input channel, and runs
**Okay Nabu** detection. Its optional peripheral WebSocket server is disabled.
The model SHA-256 is checked at setup and launch:

```text
d89128c4d16a72de429119fb2254ce46649553c2a24f5dd840175c80d7b9d094
```

This matches the `.tflite` from the official
[Okay Nabu 20241226.3 release](https://github.com/OHF-Voice/micro-wake-word/releases/tag/okay_nabu_20241226.3).
The manifest uses cutoff **0.85**, feature step **10 ms**, and sliding window **5**.
LVA persists sensitivity and model selections in its checkout's `preferences.json`;
deliberate changes there can override the defaults.

If you need to select a microphone:

```sh
uv run python -m tools.voice_satellite --list-input-devices
# The device names are written to .cache/hoast/diagnostics/voice-satellite.log.
uv run python -m tools.voice_satellite --input-device 'YOUR MICROPHONE NAME'
```

In a second terminal, start transcription:

```sh
uv run python -m hoast.voice
# Automatic language detection, with a longer command window:
uv run python -m hoast.voice --language auto --seconds 10
```

Say **“Okay Nabu”**, wait for the chime, then speak your command. The receiver
records until **600 ms of detected trailing silence**, then transcribes. A native
audio-end message can also end recording. `--seconds` accepts a hard maximum of
0.1–30 seconds (default six); both elapsed time and buffered audio are bounded.
Silence is counted in received PCM, not gaps in network delivery. Silero scores
32 ms frames: 19 silent frames establish the threshold at 608 ms, and the retained
buffer is trimmed to 600 ms after the last non-silent frame. Packetization adds
notification latency. The listener processes at most four frames before yielding
to network control events; no audio history is rescanned or downloaded.

Each nonempty, uncancelled transcript is written as one line to stdout. To test
the existing agent's text integration:

```sh
uv run python -m hoast.voice | uv run python -m hoast --config config.toml --text
```

This diagnostic text pipe writes the answer to the terminal. It does not coordinate agent
completion, TTS playback, or barge-in with the satellite; wait for the answer
before starting another command. It is a development bridge for the STT/agent
boundary. Use default `python -m hoast` voice mode for coordinated local TTS completion.

Ctrl-C stops either foreground process. Diagnostics and complete tracebacks are
retained in `.cache/hoast/diagnostics/voice.log` and `voice-satellite.log`.
LVA may report an mDNS multicast warning on unrelated VPN interfaces even while
the explicit loopback connection works; discovery is not needed by hoast.

## Application integration contract

`hoast.voice.listen` takes an asynchronous PCM handler and an optional nonblocking
transcript sink. Its optional awaited `on_turn` hook receives `VoiceTurn` with the
transcript, first-PCM/capture-end monotonic timestamps, and a thread-safe cancellation
event. Native completion waits for this hook, including any system TTS playback.
The PCM handler receives bounded mono 16 kHz int16 PCM bytes and
returns text. `VoiceReceiver` owns one capture at a time, rejects overlapping
starts and server-side wake requests, and sends native run/STT/end/error events.
`listen` supplies the streaming endpoint; direct receiver test transports can omit
it. Recurrent state and partial frame buffers reset on every accepted wake.

Capture cancellation or disconnection discards incomplete audio. If STT is already
running, shutdown waits for it, then suppresses its transcript. Failed or empty
transcription returns an error event and ends the run so the satellite can rearm.
Transport failures reconnect after two seconds. Side effects performed inside a
custom PCM handler cannot be undone; keep that handler focused on recognition.

## Transition to the real device

Use the vendor-linked configuration at pinned revision
`7b96e8b00a60174441d09c5c0b26378cccd0b1f2`:

- [Entry-point YAML](https://github.com/formatBCE/Respeaker-XVF3800-ESPHome-integration/blob/7b96e8b00a60174441d09c5c0b26378cccd0b1f2/config/respeaker-xvf-satellite-example.yaml)
- [Hardware package](https://github.com/formatBCE/Respeaker-XVF3800-ESPHome-integration/blob/7b96e8b00a60174441d09c5c0b26378cccd0b1f2/packages/hardware.yaml)
- [Voice package](https://github.com/formatBCE/Respeaker-XVF3800-ESPHome-integration/blob/7b96e8b00a60174441d09c5c0b26378cccd0b1f2/packages/voice-assistant.yaml)

The referenced configuration requires ESPHome **2026.6.0 or newer**; current
official documentation reports **2026.8.2**. Before building, pin ESPHome and the
configuration's `respeaker_ref` and patched `i2s_audio_ref` together and validate
the resulting YAML. No ESP image has been compiled or flashed as part of the
Linux setup.

This hardware configuration uses XMOS-master **48 kHz, 32-bit stereo I2S** with
XIAO in secondary mode, selecting channel 1 for wake-word input and channel 0
for ASR. That is the internal hardware format, distinct from network PCM.
The package references XMOS image
`application_xvf3800_inthost-lr48-sqr-i2c-v1.0.7-release.bin`.
Follow Seeed's paired XMOS/ESP flashing instructions. Do not combine these settings
with the separate reSpeaker-owned 16 kHz YAML or with USB-mode DSP firmware.
Confirm the DSP reports version **1.0.7** and the expected image/channel routing
before audio testing; bundling an image in the package does not verify flashing.
Also verify the wake detector starts when the native voice client connects.

Select only the same Okay Nabu release for initial comparison. The vendor-linked
voice package also enables a VAD gate (cutoff 0.05), which LVA's selected wake path
does not reproduce. To compare matching detection decisions initially, disable
that extra firmware VAD gate, keep the 0.85/5 wake settings, and verify both
implementations with the same recorded input. That comparison still requires a
defined PCM injection/replay method on the ESP; acoustic playback into its array
does not deliver identical input. Retain the hardware DSP for normal use and tune
with real recordings when the device arrives.

Configure Wi-Fi and the native API encryption key using the firmware's secrets.
Then export that key as `ESPHOME_API_KEY` and connect:

```sh
uv run python -m hoast.voice --host YOUR_DEVICE_ADDRESS
```

Model weights, wake-word family, and transport are shared; Linux uses a different
TensorFlow Lite runtime from the ESP. The simulator cannot establish bit-identical
inference or reproduce beamforming, noise suppression, echo cancellation, wireless
loss, or far-field accuracy. Real-device acceptance remains: repeat wake→capture→STT
cycles, check the first spoken word, verify rearming/reconnect, and test playback
echo before enabling a complete spoken-response loop.

## Validation performed

`tests/test_endpoint.py` checks silence duration, hysteresis, initial silence,
resumed speech, fragmented packets, and renewed speech after an endpoint in the
same packet. Receiver tests check early STT, rearming, and detector failures.
`tests/test_voice_debug.py` mocks cue/PCM ordering, exact sample preservation,
cancellation, cleanup, and replay drain before STT. The real bundled VAD matched
batch scores across 200 frames and detected an endpoint on synthesized speech.
On this machine, frame inference averaged 0.139 ms (p95 0.182 ms); diagnostics:
`/tmp/opencode/adaptive-vad-benchmark-final.log`.

The complete pipeline is covered by mocks in `tests/test_application.py`: real
receiver callbacks, worker orchestration, agent sessions, and tool registry run
against mock speech/LLM engines, Kasa I/O, and music observations. Tests verify
bilingual warm-up order without tool effects, configured entry-point routing,
two full command cycles, playback drain before rearming, timeout-boundary session
reuse, fresh music prompt context, grounded announcements, bounded speech, empty
STT, inference failure, and disconnect during inference/playback. All fixtures are
in-memory or use temporary files; no hardware, downloads, or local model paths
are needed.

```sh
uv run pytest tests/test_application.py tests/test_voice.py tests/test_lights.py tests/test_satellite_config.py tests/test_music.py -q
```

- Linux satellite launched and opened this machine's analog microphone.
- Combined real-model startup passed: English/Chinese TTS, STT recognition of
  both generated clips, LLM warm-up with no dispatch, and native endpoint probing.
  Detailed output is retained in `/tmp/opencode/full-voice-compatible-startup.log`.
- Hoast loaded the prepared Whisper model and completed a native API subscription
  to the running LVA server; LVA initialized Okay Nabu at cutoff 0.85.
- The pinned upstream API server was exercised with deterministic PCM: four
  complete wake-request/audio/transcript/end cycles across two connections.
  A deterministic test handler supplied transcript text; this checks transport
  and lifecycle, not acoustic wake detection or Whisper recognition accuracy.
- Hardware-independent tests cover byte bounds, two consecutive captures,
  overlapping starts, empty timeout, abort, malformed PCM, recognizer failure,
  and disconnect during both capture and recognition.

A spoken positive wake-word trial and XVF3800 hardware validation remain manual.
