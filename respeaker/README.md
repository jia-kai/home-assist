# reSpeaker XVF3800 voice satellite

The XIAO ESP32-S3 detects **Okay Nabu** locally and serves the encrypted ESPHome
voice API to Hoast. It also sends host-timestamped audio over I²S to the XVF3800
as its AEC reference. The recommended DSP is the **official 1.0.9 48 kHz
I²S-master image**, with output routing chosen to preserve effective local AEC.

## Files

- `official.yaml`: recommended ESP32 build and pinned official XMOS image.
- `minimal.yaml`: paired 1.0.7 DSP fallback on the same lean ESP32 stack.
- `official-test5.yaml`, `official-1.0.8.yaml`: comparison profiles.
- `core/`: always-running host reference bridge and wire protocol.
- `firmware/`, `components/`: ESP32 C++ audio code and ESPHome microphone platform.
- `tools/`, `tests/`: setup commands, local diagnostics and offline tests.

Only the host bridge runs Python continuously; the ESP32 runs compiled firmware.
No SSID, password, API key, device address, player ID, or recording is tracked.
The root `config.toml` supplies non-secret player/device settings; the root `.env`
supplies credentials. `respeaker/secrets.yaml` is an ignored, generated ESPHome
build artifact, not a separately maintained settings file.

## Install

Use this repository's `uv` environment, ESPHome 2026.8.2, the board's USB serial
port, and an NTP server on the IoT AP host. `arduino-cli` is not required.

1. Set the explicit MA player ID and ESP endpoint in root `config.toml`. Keep
   Wi-Fi, ESPHome API and OTA secrets in root `.env`:

   ```toml
   [music]
   player_id = "PLAYER_ID"

   [satellite]
   host = "DEVICE_ADDRESS"
   key_env = "ESPHOME_API_KEY"
   ```

   ```dotenv
   RESPEAKER_WIFI_PASSWORD="AP_PASSWORD"
   ESPHOME_API_KEY="BASE64_NOISE_KEY"
   ESPHOME_OTA_PASSWORD="OTA_PASSWORD"
   ```

   Generate ESPHome's ignored YAML from those root files. The helper discovers
   the active AP SSID and the AP's address for NTP from the route to the device:

   ```sh
   uv run python -m respeaker.tools.prepare
   ```

2. Compile and upload the recommended profile over the XIAO USB port:

   ```sh
   uvx --python 3.13 --from 'esphome==2026.8.2' esphome config respeaker/official.yaml
   uvx --python 3.13 --from 'esphome==2026.8.2' esphome compile respeaker/official.yaml
   uvx --python 3.13 --from 'esphome==2026.8.2' esphome upload respeaker/official.yaml \
     --device ESP_SERIAL_PORT --upload_speed 115200
   uv run python -m respeaker.tools.status --device-host DEVICE_ADDRESS
   ```

   The ESP32 image embeds the official XMOS image and installs it over I²C if
   the reported version differs. Wait for `version=1.0.9` and
   `wake_route=ready`. Ready means the ESP read back both output routes.

3. Root Docker Compose runs Hoast directly against `[satellite].host` and reads
   `ESPHOME_API_KEY` from root `.env`. It also starts the MA reference bridge,
   which reads `[satellite].host` and `[music].player_id` from the same config
   and binds `/data/aec-reference/aec-reference-{player_id}.sock` in the shared
   data mount. On a first installation with a loopback satellite setting, the
   optional helper makes a private backup before changing the ignored root
   `config.toml`:

   ```sh
   uv run python -m respeaker.tools.activate --device-host DEVICE_ADDRESS
   docker compose up -d --build
   ```

   Skip the helper if `[satellite]` is already configured. Hoast does not start
   the Linux satellite; verify its native voice connection to the ESP after
   speech-model warm-up. If a standalone bridge container is already running,
   stop and remove it before starting the root stack; see the root README for
   the one-time migration command.

   The bridge can also run directly:

   ```sh
   uv run python -m respeaker.core.bridge \
     --socket-directory "$HOME/music-data/aec-reference"
   ```

The bridge's socket follows the configured AirPlay player. If the ESP's IP
changes, update only `[satellite].host`; Hoast and the bridge will use it after
their next start. `capture_seconds` is Hoast's maximum post-wake command
recording duration, not the AEC reference cache. `key_env` names a variable in
root `.env`, not the key itself.

### Reserve the device address on a NetworkManager-shared AP

For a stable address, reserve the ESP's Wi-Fi MAC and chosen IP in the AP's
dnsmasq configuration rather than setting `wifi.manual_ip` inside ESPHome.
Check that the chosen IP is available on the AP subnet. Add this line to a
root-owned file in `/etc/NetworkManager/dnsmasq-shared.d/`:

```text
dhcp-host=DEVICE_WIFI_MAC,RESERVED_IP
```

When ready to interrupt the AP briefly, reconnect its NetworkManager shared
connection so dnsmasq loads the reservation, then renew the ESP's lease by
restarting the device. Set `[satellite].host` in root `config.toml` to the
reserved IP if it changed, and start Hoast and the bridge afterward. Do not
place device-specific MACs or IPs in tracked configuration.

## Audio and synchronization

```text
MA AirPlay HAEC v2 Unix socket → host bridge → RTP/L16 UDP 5070 → ESP buffer
                                            RTCP SR UDP 5071 ↗      ↓
host chronyd → ESP SNTP clock ──────────────────────────────────→ timed I²S → XMOS AEC

XVF3800 processed output → 48-to-16 kHz decimator → Hoast voice API TCP 6053
XVF3800 wake output      → 48-to-16 kHz decimator → local Okay Nabu detector
XVF3800 I²S outputs      → optional diagnostic TCP 5072
```

The MA patch exports PCM with host presentation timestamps. A small bounded
queue hands HAEC v2 records to a background Unix sequence-packet writer; socket
backpressure cannot stall MA's Python audio loop. The bridge downmixes and
resamples to 48 kHz L16, then sends **one RTP packet per 10 ms** about
**three seconds before** its original presentation timestamp. If MA supplies
less lead, the bridge sends at real-time speed. The queue reports overflow and
expired audio rather than silently discarding it. RTCP sender reports map RTP
sample time to host time. The ESP synchronizes to host NTP and retains up to
**4.2 seconds of future reference** in PSRAM. It feeds the XMOS I²S far-end
input when each sample is due; the host pacing does not delay music playback.
Wake events and 16 kHz command PCM continue over the existing encrypted
ESPHome API. The LED ring is commanded on a wake event. No Home Assistant Core
or device speaker is needed for this host voice pipeline.

The official 1.0.9 image defaults both capture slots to ASR beam 0. This
profile instead selects **channel 0 = processed user-selected beam (8/0)** and
**channel 1 = ASR auto-select beam (7/3)** via I²C readback. Its LED-ring command
is 19. The on-board gains and system delay match the paired 1.0.7 image; the
official AEC high-pass default remains off. Matching the output routing was
necessary for comparable local AEC results; changing high-pass settings was
not required.

## Local tests

This experiment **does not use Music Assistant**. Each command opens device
capture first, plays the same deterministic PCM on the host speaker, and sends
a timestamped copy to I²S. WAVs stay under ignored `respeaker/captures/`:

```sh
uv run python -m respeaker.tools.local_trial --device-host DEVICE_ADDRESS --mode raw
uv run python -m respeaker.tools.local_trial --device-host DEVICE_ADDRESS --mode processed
uv run python -m respeaker.tools.local_trial --device-host DEVICE_ADDRESS --mode reference
uv run python -m respeaker.tools.local_trial --device-host DEVICE_ADDRESS --mode delay
uv run python -m respeaker.tools.analyze --reference respeaker/captures/stimulus.wav \
  --raw respeaker/captures/raw.wav --processed respeaker/captures/processed.wav \
  --farend respeaker/captures/reference.wav --delay respeaker/captures/delay.wav
```

`raw` temporarily routes pre-gain mic 0 to channel 1; `reference` selects the
far-end signal the DSP receives; `delay` records far-end and raw mic together
on one clock. Debug routing pauses wake detection and restores it on disconnect.
The capture tool first calls the encrypted ESPHome `arm_debug_audio` action with
a fresh one-time credential. Only then does TCP port 5072 open, for at most ten
seconds while waiting for a connection. The TCP client sends that 64-character
credential followed by `P`, `R`, `F`, or `D`; the device accepts one authenticated
session, closes the port afterward, and bounds each session to 65 seconds. PCM
is unencrypted on this development-only connection. After an eight-byte `HDBG`
header, it streams 48 kHz stereo signed 32-bit little-endian PCM. Flash the
updated ESP image before using the capture or local-trial tools.
The local trial rejects speaker output underruns. Normalized echo correlations
are useful for comparison, but separate captures and external speaker latency
do not give a calibrated dB attenuation measurement. `--reference-delay-ms`
changes *only* the local test reference, not MA playback.

For wake testing without STT, run
`uv run python -m respeaker.tools.wake_probe --device-host DEVICE_ADDRESS`
instead of the full Hoast voice client, then speak Okay Nabu near the array.
The LED ring and native wake/capture event still require a human-spoken trial.
Remote AirPlay speaker delay and long-run acoustic performance are not measured
by the local experiment.

## Other DSP images

The paired vendor **1.0.7** image in `minimal.yaml` is the tested fallback.
`official-1.0.8.yaml` and `official-test5.yaml` retain the routing used in their
comparison trials; their local AEC results should not be interpreted as a
firmware-only difference from the recommended, routing-matched profile.
All profile URLs and MD5 checks are pinned in the YAML. The generic official
`respeaker_xvf3800_i2s_dfu_firmware_v1.0.7.bin` is a **16 kHz I²S-slave** image
and cannot be substituted for a 48 kHz XMOS-master image.

The official 48 kHz `test5` image and paired image both report **version 1.0.7**.
If switching between them without first changing DSP version, the automatic
version check will skip the transfer. The `official-test5.yaml` profile exposes
an explicit encrypted action:

```sh
uv run python -m respeaker.tools.flash_xmos --device-host DEVICE_ADDRESS
```

Its same-version I²C DFU can take several minutes while the ESP is running.
Confirm transfer completion in device logs; an I²C version reading alone
cannot attest which 1.0.7 binary is installed. Reboot the ESP after a manual
XMOS update so its configured wake route and I²S clock state are restored.
