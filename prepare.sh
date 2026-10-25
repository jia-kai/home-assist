#!/usr/bin/env bash
# Bootstrap the locked Python 3.14 environment and explicit native speech roots.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: prepare.sh [options]
Installs uv/Python 3.14 if needed; prepares LFM, Silero VAD and SenseVoice/Melo EN+ZH.
Does not start services, edit configuration, or prepare a speech Python bridge.

  --with-functiongemma          Also prepare the gated optional model (HF access required)
  --offline                    Disable downloads (uv/HF, LFM selection and speech inputs required)
  --threads 1|2                Native preparation CPU budget (default: 2)
  --stt-backend cpu|GPU|ort     Explicit application backend (default: cpu)
  --stt-precision FP32|FP16|UINT8
                               Weight storage (default: FP32); ort requires UINT8
  --tts-precision FP32|FP16-decoder
                               Melo storage (default: FP16-decoder); other stages stay FP32
  --stt-output DIR             New STT artifact root (default: models/stt2 for FP32 OV)
  --tts-output DIR             New EN/ZH root (default: models/tts2; FP32: tts2-fp32)
  --speech-input-cache DIR     Pinned input root (default: models/speech2-inputs)
  --stt-source DIR             Explicit pinned official SenseVoice source
  --melo-source DIR            Explicit Melo git checkout at the audited commit
  --assets-manifest FILE       Explicit pinned EN/ZH Melo/BERT local inventory
  --g2p-resource-dir DIR       Static checkpoint20.npz/homographs.en data, never imported
  --nltk-resource-dir DIR      Unpacked pinned CMU dictionary/reference tagger data
  --community-source DIR       Explicit sherpa 2024-07-17 source (requires UINT8)
  --license-dir DIR            Explicit verified STT legal texts/provenance (local-only)
  -h, --help                   Show this help without provisioning anything

Existing artifact targets are never replaced, even if valid; select fresh outputs.
FP16 STT/GPU and community UINT8 are opt-in candidates, not readiness guarantees.
The bootstrap validates/fetches pinned inputs, then exports; it does not benchmark
or run speech inference. See README.md and docs/speech2-native.md for qualification.
EOF
}
with_functiongemma=false
offline=false
threads=2
stt_backend=cpu
stt_precision=FP32
tts_precision=FP16-decoder
stt_output=''
tts_output=''
speech_input_cache=models/speech2-inputs
source_args=()
community_source=false
while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --with-functiongemma) with_functiongemma=true; shift ;;
        --offline) offline=true; shift ;;
        --threads|--stt-backend|--stt-precision|--tts-precision|--stt-output|--tts-output|--speech-input-cache|--stt-source|--melo-source|--assets-manifest|--g2p-resource-dir|--nltk-resource-dir|--community-source|--license-dir)
            if (( $# < 2 )) || [[ -z $2 || $2 == --* ]]; then
                printf 'Missing value for %s. Run prepare.sh --help.\n' "$1" >&2
                exit 2
            fi
            case "$1" in
                --threads) threads=$2 ;;
                --stt-backend) stt_backend=$2 ;;
                --stt-precision) stt_precision=$2 ;;
                --tts-precision) tts_precision=$2 ;;
                --stt-output) stt_output=$2 ;;
                --tts-output) tts_output=$2 ;;
                --speech-input-cache) speech_input_cache=$2 ;;
                --community-source) source_args+=("$1" "$2"); community_source=true ;;
                *) source_args+=("$1" "$2") ;;
            esac
            shift 2 ;;
        *) printf 'Unknown argument: %s. Run prepare.sh --help.\n' "$1" >&2; exit 2 ;;
    esac
done
if [[ $threads != 1 && $threads != 2 ]] ||
   [[ $stt_backend != cpu && $stt_backend != GPU && $stt_backend != ort ]] ||
   [[ $stt_precision != FP32 && $stt_precision != FP16 && $stt_precision != UINT8 ]] ||
   [[ $tts_precision != FP32 && $tts_precision != FP16-decoder ]]; then
    printf 'Invalid threads/backend/precision; see prepare.sh --help.\n' >&2
    exit 2
fi
if [[ $stt_backend == ort && $stt_precision != UINT8 ]]; then
    printf '--stt-backend ort requires --stt-precision UINT8; there is no backend fallback.\n' >&2
    exit 2
fi
if "$community_source" && [[ $stt_precision != UINT8 ]]; then
    printf '--community-source requires --stt-precision UINT8.\n' >&2
    exit 2
fi
if [[ -z $stt_output ]]; then
    if [[ $stt_precision == FP32 ]]; then
        stt_output=models/stt2
    else
        stt_output="models/stt2-${stt_precision,,}-${stt_backend,,}"
    fi
fi
if [[ -z $tts_output ]]; then
    if [[ $tts_precision == FP32 ]]; then
        tts_output=models/tts2-fp32
    else
        tts_output=models/tts2
    fi
fi

root=$(dirname -- "${BASH_SOURCE[0]}")
cd -P -- "$root"
root=$PWD
log_dir="$root/.cache/hoast/diagnostics/prepare/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p -- "$log_dir"

# Preserve the failing stage's status and point to its durable diagnostics.
on_error() {
    local status=$?
    printf 'Preparation failed; diagnostics: %s\n' "$log_dir" >&2
    exit "$status"
}
trap on_error ERR
exec > >(tee -a "$log_dir/bootstrap.log") 2> >(tee -a "$log_dir/bootstrap.log" >&2)
printf 'Preparation diagnostics: %s\n' "$log_dir"
printf 'Policy: offline=%s threads=%s STT=%s/%s TTS=%s\n' "$offline" "$threads" "$stt_backend" "$stt_precision" "$tts_precision"

# Refuse prior artifacts before environment installation or any model work.
for target in "$stt_output" "$tts_output/EN" "$tts_output/ZH"; do
    if [[ -e $target || -L $target ]]; then
        printf 'Refusing to replace existing artifacts: %s. Select fresh output directories.\n' "$target" >&2
        exit 1
    fi
done
if "$offline"; then
    for selection in models/lfm/model.json; do
        if [[ ! -f $selection ]]; then
            printf 'Offline bootstrap requires the prepared LFM selection: %s. Run online setup first.\n' "$selection" >&2
            exit 1
        fi
    done
    if "$with_functiongemma" && [[ ! -f models/functiongemma/model.json ]]; then
        printf 'Offline FunctionGemma requires models/functiongemma/model.json from prior gated download.\n' >&2
        exit 1
    fi
fi
for tool in systemd-run systemctl taskset git tee; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'Missing prerequisite: %s. See README.md for setup requirements.\n' "$tool" >&2
        exit 1
    fi
done
if ! systemctl --user show-environment >/dev/null 2>"$log_dir/systemd.log"; then
    printf 'A running systemd user manager is required; see %s/systemd.log.\n' "$log_dir" >&2
    exit 1
fi
if "$offline"; then
    export UV_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
if ! command -v uv >/dev/null 2>&1; then
    if "$offline"; then
        printf 'Offline bootstrap requires uv already on PATH.\n' >&2
        exit 1
    fi
    install_dir="${UV_INSTALL_DIR:-${HOME:?HOME must be set}/.local/bin}"
    installer="$log_dir/uv-install.sh"
    printf 'Installing uv...\n'
    if command -v curl >/dev/null 2>&1; then
        curl --fail --silent --show-error --location https://astral.sh/uv/install.sh --output "$installer" 2>"$log_dir/uv-download.log"
    elif command -v wget >/dev/null 2>&1; then
        wget --no-verbose https://astral.sh/uv/install.sh -O "$installer" 2>"$log_dir/uv-download.log"
    else
        printf 'Install curl or wget to bootstrap uv.\n' >&2
        exit 1
    fi
    UV_INSTALL_DIR="$install_dir" UV_NO_MODIFY_PATH=1 sh "$installer" >"$log_dir/uv-install.log" 2>&1
    export PATH="$install_dir:$PATH"
    printf 'uv installed in %s; add this directory to PATH for future shells.\n' "$install_dir"
fi
uv_bin=$(command -v uv)
export UV_PROJECT_ENVIRONMENT="$root/.venv"
python_version=$(< "$root/.python-version")
if [[ $python_version != 3.14 && $python_version != 3.14.* ]]; then
    printf 'Native speech setup requires .python-version to select Python 3.14.\n' >&2
    exit 1
fi
printf 'Preparing Python 3.14 environment...\n'
"$uv_bin" python install "$python_version" >"$log_dir/python.log" 2>&1
"$uv_bin" sync --locked --python "$python_version" >"$log_dir/dependencies.log" 2>&1
python="$root/.venv/bin/python"
"$python" -c 'import sys; assert sys.version_info[:2] == (3, 14), sys.version' >"$log_dir/python-version.log" 2>&1

# Run one model step at a time with a hard memory cap and durable diagnostics.
# Args:
#   name:
#       Short diagnostic filename and progress label.
#
#   threads:
#       One or two allowed CPU cores and native preparation threads.
#
#   command:
#       Executable and remaining arguments, forwarded without evaluation.
#
run_model() {
    local name=$1 budget=$2
    shift 2
    printf 'Preparing %s...\n' "$name"
    "$python" -m tools.guarded_run --threads "$budget" --memory-gib 4 --timeout 1800 --log-file "$log_dir/$name.log" -- "$@"
}

if "$offline"; then
    run_model vad 1 "$python" -m tools.prepare_vad --offline
else
    run_model vad 1 "$python" -m tools.prepare_vad
fi

speech_inputs_manifest=models/speech2-inputs/inputs.json
mkdir -p -- "$(dirname -- "$speech_inputs_manifest")"
fetch_args=(--root "$speech_input_cache" --output "$speech_inputs_manifest" "${source_args[@]}")
if "$offline"; then fetch_args+=(--offline); else fetch_args+=(--download); fi
if [[ $stt_precision == UINT8 ]]; then fetch_args+=(--with-community); fi
run_model speech-inputs "$threads" "$python" -m tools.fetch_speech2 "${fetch_args[@]}"
# A file, rather than process substitution, preserves Python failure status.
"$python" -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    inputs = json.load(stream)
for key in ("stt_source", "melo_source", "assets_manifest", "g2p_resource_dir", "nltk_resource_dir", "community_source", "license_dir"):
    value = inputs[key] or ""
    sys.stdout.buffer.write(value.encode() + b"\0")
' "$speech_inputs_manifest" >"$log_dir/input-paths.bin"
mapfile -d '' -t inputs <"$log_dir/input-paths.bin"

if ! "$offline"; then
    run_model lfm-download 1 "$python" -m tools.prepare_llm --model lfm download
fi
run_model lfm-gpu 1 env USE_TORCH=0 "$python" -m tools.prepare_llm --model lfm compile --device GPU --threads 1
run_model lfm-cpu 2 env USE_TORCH=0 "$python" -m tools.prepare_llm --model lfm compile --device CPU --threads 2
if [[ $stt_precision == UINT8 ]]; then
    import_backend=openvino
    if [[ $stt_backend == ort ]]; then import_backend=onnxruntime; fi
    run_model stt2-community "$threads" "$python" -m tools.prepare_stt2_community --community "${inputs[5]}" --official "${inputs[0]}" --license-dir "${inputs[6]}" --output "$stt_output" --threads "$threads" --backend "$import_backend"
else
    run_model stt2 "$threads" "$python" -m tools.prepare_stt2 --source "${inputs[0]}" --license-dir "${inputs[6]}" --output "$stt_output" --threads "$threads" --precision "$stt_precision"
fi
tts_args=(--precision FP32)
if [[ $tts_precision == FP16-decoder ]]; then tts_args=(--precision FP16 --fp16-stages decoder); fi
for language in EN ZH; do
    run_model "tts2-$language" "$threads" "$python" -m tools.prepare_tts2 --source "${inputs[1]}" --assets-manifest "${inputs[2]}" --g2p-resource-dir "${inputs[3]}" --nltk-resource-dir "${inputs[4]}" --output "$tts_output" --language "$language" --threads "$threads" "${tts_args[@]}" --verify-original
done
if "$with_functiongemma"; then
    if ! "$offline"; then
        run_model functiongemma-download 1 "$python" -m tools.prepare_llm --model functiongemma download
    fi
    run_model functiongemma-export 2 "$python" -m tools.prepare_llm --model functiongemma export --precision int8 --device GPU --threads 2
fi
printf 'Preparation completed; independent speech qualification/runtime startup is still required.\n'
printf 'Run from this checkout (after setting config.toml):\n'
printf '  .venv/bin/python -m hoast --config config.toml --stt-backend %q --stt-model-dir %q --tts-model-dir %q --threads 2\n' "$stt_backend" "$stt_output" "$tts_output"
printf 'Optional offline bilingual smoke check (real inference; run separately):\n'
printf '  .venv/bin/python -m tools.check_speech2 --stt-model-dir %q --tts-model-dir %q --stt-backend %q --tts-backend cpu --threads 2 --chinese\n' "$stt_output" "$tts_output" "$stt_backend"
printf 'Diagnostics: %s\n' "$log_dir"
