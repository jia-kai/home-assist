"""Build and numerically validate the optional AVX2 Kokoro CPU activation kernel."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import openvino as ov
from openvino.utils.node_factory import NodeFactory

from hoast.logging import configure_logging, get_logger
from hoast.tts_kernels import DEFAULT_KERNEL, KERNEL_SOURCE

logger = get_logger(__name__)


def validate_library(library: Path) -> None:
    """Check vector/tail paths and scalar/channel broadcasting against native ops.

    Args:
        library:
            Newly built extension library.

    """
    core = ov.Core()
    core.add_extension(library)
    factory = NodeFactory()
    factory.add_extension(library)
    rng = np.random.default_rng(20260914)
    for shape, scalar in (((1, 8, 32), True), ((2, 17, 19), False)):
        coefficient_shape = (1, 1, 1) if scalar else (1, shape[1], 1)
        x = ov.opset13.parameter(shape, ov.Type.f32)
        a = ov.opset13.parameter(coefficient_shape, ov.Type.f32)
        s = ov.opset13.parameter(coefficient_shape, ov.Type.f32)
        custom = factory.create("HoastSnake", [x, a, s], {})
        reference = ov.opset13.add(
            x,
            ov.opset13.multiply(
                s,
                ov.opset13.power(
                    ov.opset13.sin(ov.opset13.multiply(a, x)), np.float32(2)
                ),
            ),
        )
        # The native wrapper's Model constructor has an upstream self-type stub issue.
        model = ov.Model([custom, reference], [x, a, s])  # pyright: ignore[reportGeneralTypeIssues]
        compiled = core.compile_model(
            model,
            "CPU",
            {
                "INFERENCE_NUM_THREADS": 2,
                "NUM_STREAMS": "1",
                "INFERENCE_PRECISION_HINT": "f32",
            },
        )
        values = [
            rng.uniform(-20, 20, shape).astype(np.float32),
            rng.uniform(-2, 2, coefficient_shape).astype(np.float32),
            rng.uniform(-2, 2, coefficient_shape).astype(np.float32),
        ]
        output = compiled(values)
        np.testing.assert_allclose(
            output[compiled.output(0)], output[compiled.output(1)], rtol=1e-5, atol=2e-5
        )


def build(output: Path = DEFAULT_KERNEL) -> Path:
    """Compile an ABI-fingerprinted kernel and publish it only after validation.

    Args:
        output:
            Library destination; its directory is created if needed.

    """
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("The tuned speech kernel targets Linux x86_64 with AVX2")
    root = Path(ov.__file__).parent
    libraries = root / "libs"
    openvino_libraries = list(libraries.glob("libopenvino.so.*"))
    tbb_libraries = list(libraries.glob("libtbb.so.*"))
    if len(openvino_libraries) != 1 or len(tbb_libraries) != 1:
        raise RuntimeError("Cannot identify the installed OpenVINO/TBB libraries")
    source_hash = hashlib.sha256(KERNEL_SOURCE.read_bytes()).hexdigest()
    manifest_path = output.with_suffix(".json")
    if output.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with output.open("rb") as stream:
            library_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if (
            manifest["source_sha256"] == source_hash
            and manifest["openvino"] == ov.__version__
            and manifest["library_sha256"] == library_hash
        ):
            logger.info("speech.kernels status=cached path=%s", output)
            return output
    compiler = shutil.which("c++")
    if compiler is None:
        raise FileNotFoundError("A C++ compiler is required to build the speech kernel")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.so")
    command = [
        compiler,
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-O3",
        "-mavx2",
        "-fno-math-errno",
        "-ffp-contract=off",
        "-I" + str(root / "include"),
        str(KERNEL_SOURCE),
        str(openvino_libraries[0]),
        str(tbb_libraries[0]),
        "-lmvec",
        "-lm",
        "-Wl,-rpath," + str(libraries),
        "-o",
        str(temporary),
    ]
    with output.with_suffix(".build.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    validate_library(temporary)
    temporary.replace(output)
    with output.open("rb") as stream:
        library_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "openvino": ov.__version__,
        "source_sha256": source_hash,
        "library_sha256": library_hash,
        "isa": "AVX2",
        "compiler": compiler,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("speech.kernels status=ready path=%s", output)
    return output


def main() -> None:
    """Build the local optional accelerator and retain complete build diagnostics."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_KERNEL, help="CPU extension library path"
    )
    args = parser.parse_args()
    configure_logging(
        log_file=Path(".cache/hoast/diagnostics/build-speech-kernels.log")
    )
    try:
        build(args.output)
    except Exception:
        logger.exception("CPU speech kernel preparation failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
