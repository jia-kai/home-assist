#!/usr/bin/env bash
# Bootstrap the locked environments and prepare every default assistant model.
set -Eeuo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    printf 'Usage: %s [--with-functiongemma]\n' "${0##*/}"
    printf 'Installs uv if needed; prepares LFM, Whisper, English/Chinese Kokoro, and checks speech.\n'
    printf 'The optional FunctionGemma model requires prior Hugging Face access.\n'
    exit 0
fi
with_functiongemma=false
if [[ ${1:-} == --with-functiongemma && $# == 1 ]]; then
    with_functiongemma=true
elif (( $# != 0 )); then
    printf 'Unknown arguments. Run %s --help.\n' "${0##*/}" >&2
    exit 2
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

for tool in c++ systemd-run systemctl taskset; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'Missing prerequisite: %s. See README.md for setup requirements.\n' "$tool" >&2
        exit 1
    fi
done
if ! systemctl --user show-environment >/dev/null 2>"$log_dir/systemd.log"; then
    printf 'A running systemd user manager is required; see %s/systemd.log.\n' "$log_dir" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
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
printf 'Preparing Python environments; diagnostics: %s\n' "$log_dir"
"$uv_bin" python install "$python_version" >"$log_dir/python.log" 2>&1
"$uv_bin" sync --locked --python "$python_version" >"$log_dir/dependencies.log" 2>&1
python="$root/.venv/bin/python"

# Run one model step at a time with a hard memory cap and durable diagnostics.
# Args:
#   name:
#       Short diagnostic filename and progress label.
#
#   threads:
#       One or two allowed CPU cores and inference threads.
#
#   command:
#       Executable and remaining arguments, forwarded without evaluation.
#
run_model() {
    local name=$1 threads=$2
    shift 2
    printf 'Preparing %s...\n' "$name"
    "$python" -m tools.guarded_run --threads "$threads" --memory-gib 4 --timeout 1800 --log-file "$log_dir/$name.log" -- "$@"
}

run_model gpu-check 1 "$python" -c 'from hoast.tts_gpu import SharedGPU; gpu = SharedGPU(); gpu.close()'
run_model lfm-download 1 "$python" -m tools.prepare_llm --model lfm download
run_model lfm-gpu 1 env USE_TORCH=0 "$python" -m tools.prepare_llm --model lfm compile --device GPU --threads 1
run_model lfm-cpu 2 env USE_TORCH=0 "$python" -m tools.prepare_llm --model lfm compile --device CPU --threads 2
run_model stt 2 "$python" -m tools.prepare_stt
run_model tts 2 "$python" -m tools.prepare_tts --chinese
if "$with_functiongemma"; then
    run_model functiongemma-download 1 "$python" -m tools.prepare_llm --model functiongemma download
    run_model functiongemma-export 2 "$python" -m tools.prepare_llm --model functiongemma export --precision int8 --device GPU --threads 2
fi
run_model speech-check 2 "$python" -m tools.check_speech --chinese
printf 'Models are ready. Diagnostics: %s\n' "$log_dir"
