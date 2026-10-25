"""Convert offline pinned Melo/BERT checkpoints to native Python 3.14 OpenVINO IR.

Run with the existing project environment, e.g. `python -m tools.prepare_tts2
--language EN --threads 1 --precision FP16`. No download or alternate interpreter is used. Output
is atomically published as OUTPUT/LANGUAGE; existing artifacts are never replaced.

The preparation-only source loader executes only verified neural helper modules
(commons, transforms, attentions, modules, models) under a private package. It
rewrites their `melo` imports to that private package and removes exactly the
training-only `import melo.monotonic_align as monotonic_align` AST node. The
SynthesizerTrn constructor and strict original checkpoint load are unchanged;
training methods are never invoked. Optional --verify-original invokes only
neural inference methods with deterministic injected noise. Melo API, utils, cleaner, text,
Numba and alignment are neither loaded nor required. See graphs.py for MIT
attribution of graph adaptations; the pinned source license is copied to output.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

from hoast.logging import configure_logging
from hoast.runtime import configure_cpu_budget
from tools.prepare_tts2_frontend import populate_frontend_assets

logger = logging.getLogger(__name__)
SOURCE_REVISION = "209145371cff8fc3bd60d7be902ea69cbdb7965a"
DEFAULT_SOURCE = Path("models/speech2-inputs/MeloTTS")
DEFAULT_ASSETS = Path("models/speech2-inputs/inputs.melo-assets.json")
DEFAULT_OUTPUT = Path("models/tts2")
DEFAULT_G2P_RESOURCE_DIR = Path("models/speech2-inputs/g2p_en")
DEFAULT_NLTK_RESOURCE_DIR = Path("models/speech2-inputs/nltk")
SOURCE_HASHES = {
    "melo/commons.py": "046943d265635c3d9c99195562e703564a26300981a7db5b87d4cd3fd180ae0d",
    "melo/transforms.py": "e6798989a0d75cbec45e7e24e3697e1c4886b3664c6a80a452645ef1d2a93f9c",
    "melo/attentions.py": "5261b73e4945651e9f7133375f4f83c1532a30f8227f6249c7006d696a737353",
    "melo/modules.py": "de80baee7dc8d89d621c541b899a84193df4bb3ef95f5e6d076cc37b6bb10bf6",
    "melo/models.py": "f96eef52845224b5797e54008013a5b6a5fafad2035713dc5181a5db8bf8bef4",
    "LICENSE": "88a50e5a02bbc2a5c2f084dc19da751aa97b1690f5fda76cd8005c8634d1ca70",
}
REPOSITORIES = {
    "EN": ("myshell-ai/MeloTTS-English", "bert-base-uncased"),
    "ZH": ("myshell-ai/MeloTTS-Chinese", "bert-base-multilingual-uncased"),
}
ASSET_REVISIONS = {
    "myshell-ai/MeloTTS-English": "bb4fb7346d566d277ba8c8c7dbfdf6786139b8ef",
    "myshell-ai/MeloTTS-Chinese": "af5d207a364ea4208c6f589c89f57f88414bdd16",
    "bert-base-uncased": "86b5e0934494bd15c9632b12f734a8a67f723594",
    "bert-base-multilingual-uncased": "7cbf9a625e29989f6b9c6c2fa68234c304f7e38f",
}
PRECISIONS = ("FP32", "FP16", "INT8")
DEFAULT_FP16_STAGES = ("bert", "decoder")
WAVEFORM_SNR_FLOORS = {"FP32": 60.0, "FP16": 40.0, "INT8": 30.0}
QUIET_WAVEFORM_RMS = 1e-4
PCM16_LSB = 1 / 32768
PRECISION_TOLERANCES = {
    "FP32": (3e-4, 3e-4),
    "FP16": (3e-3, 3e-3),
    "INT8": (2e-2, 2e-2),
}
# S=WordPiece count, P=phone count, F=frame count. All batch dimensions are one.
STAGE_CONTRACTS: dict[str, dict[str, Any]] = {
    "bert": {
        "inputs": {
            "input_ids": {"dtype": "int64", "shape": [1, "S"]},
            "attention_mask": {"dtype": "int64", "shape": [1, "S"]},
            "token_type_ids": {"dtype": "int64", "shape": [1, "S"]},
        },
        "outputs": {"token_features": {"dtype": "float32", "shape": [1, "S", 768]}},
    },
    "encoder": {
        "inputs": {
            "phones": {"dtype": "int64", "shape": [1, "P"]},
            "tones": {"dtype": "int64", "shape": [1, "P"]},
            "languages": {"dtype": "int64", "shape": [1, "P"]},
            "phone_features": {"dtype": "float32", "shape": [1, 768, "P"]},
            "speaker_id": {"dtype": "int64", "shape": [1]},
            "duration_noise": {"dtype": "float32", "shape": [1, 2, "P"]},
        },
        "outputs": {
            "mean": {"dtype": "float32", "shape": [1, 192, "P"]},
            "logscale": {"dtype": "float32", "shape": [1, 192, "P"]},
            "log_dp": {"dtype": "float32", "shape": [1, 1, "P"]},
            "log_sdp": {"dtype": "float32", "shape": [1, 1, "P"]},
            "speaker_condition": {"dtype": "float32", "shape": [1, 256, 1]},
        },
    },
    "flow": {
        "inputs": {
            "latent": {"dtype": "float32", "shape": [1, 192, "F"]},
            "speaker_condition": {"dtype": "float32", "shape": [1, 256, 1]},
        },
        "outputs": {"z": {"dtype": "float32", "shape": [1, 192, "F"]}},
    },
    "decoder": {
        "inputs": {
            "latent": {"dtype": "float32", "shape": [1, 192, "F"]},
            "speaker_condition": {"dtype": "float32", "shape": [1, 256, 1]},
        },
        "outputs": {"audio": {"dtype": "float32", "shape": [1, 1, "512*F"]}},
    },
}


def sha256(path: Path) -> str:
    """Hash a file using bounded-memory reads, following checkpoint symlinks.

    Args:
        path:
            Existing source/checkpoint/artifact file.

    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(source: Path) -> None:
    """Require the exact pinned git revision and byte-identical helper/license files.

    Args:
        source:
            Offline MeloTTS git checkout root; no upstream packages are imported.

    """
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"Melo source revision mismatch: {revision}")
    for relative, expected in SOURCE_HASHES.items():
        actual = sha256(source / relative)
        if actual != expected:
            raise ValueError(f"Pinned Melo source hash mismatch: {relative}: {actual}")


def inventory_files(directory: Path) -> dict[str, dict[str, str | int]]:
    """Recursively hash staged files under relative POSIX names before publication.

    Call before writing the root manifest so it does not inventory itself.

    Args:
        directory:
            Unpublished staging root, including nested frontend data and notices.

    """
    return {
        path.relative_to(directory).as_posix(): {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


class NeuralImports(ast.NodeTransformer):
    """Rewrite verified helper imports and exclude the sole training alignment import."""

    package: str
    """Private package name used instead of the installed melo namespace."""

    removed_alignment: int
    """Count of removed alignment import nodes; exactly one is required in models."""

    def __init__(self, package: str) -> None:
        """Configure private import rewriting without importing upstream package code.

        Args:
            package:
                Unique private package registered in sys.modules.

        """
        self.package = package
        self.removed_alignment = 0

    def visit_Import(self, node: ast.Import) -> ast.AST | None:
        """Remove the exact pinned alignment import, rejecting other Melo imports.

        Args:
            node:
                Import statement from a hash-verified neural helper AST.

        """
        if len(node.names) == 1:
            alias = node.names[0]
            if (
                alias.name == "melo.monotonic_align"
                and alias.asname == "monotonic_align"
            ):
                self.removed_alignment += 1
                return None
        if any(
            alias.name == "melo" or alias.name.startswith("melo.")
            for alias in node.names
        ):
            raise ValueError("Unexpected upstream Melo import")
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        """Redirect allowed neural imports to the private preparation package.

        Args:
            node:
                From-import statement from a hash-verified neural helper AST.

        """
        if node.module == "melo":
            if any(
                alias.name not in {"commons", "modules", "attentions"}
                for alias in node.names
            ):
                raise ValueError("Unexpected upstream Melo from-import")
            node.module = self.package
        elif node.module == "melo.commons":
            node.module = self.package + ".commons"
        elif node.module is not None and node.module.startswith("melo."):
            raise ValueError("Unexpected upstream Melo module import")
        return node


def load_neural_source(source: Path) -> ModuleType:
    """Load only pin-verified neural helpers and return their models module.

    The upstream __init__ is never executed. Import rewriting and alignment-node
    exclusion are the only AST modifications. Private modules are removed on a
    failure; successful preparation retains them for Torch source inspection.

    Args:
        source:
            MeloTTS checkout root matching SOURCE_REVISION and SOURCE_HASHES.

    """
    verify_source(source)
    package_name = "_hoast_melo_prepare_" + SOURCE_REVISION[:12]
    if package_name in sys.modules:
        raise RuntimeError("Pinned neural source is already loaded in this process")
    package = ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    loaded = [package_name]
    try:
        for name in ("commons", "transforms", "attentions", "modules", "models"):
            path = source / "melo" / f"{name}.py"
            tree = ast.parse(path.read_text(), filename=str(path))
            transformer = NeuralImports(package_name)
            tree = transformer.visit(tree)
            if transformer.removed_alignment != (1 if name == "models" else 0):
                raise ValueError("Pinned training-alignment import invariant failed")
            ast.fix_missing_locations(tree)
            qualified = package_name + "." + name
            module = ModuleType(qualified)
            module.__file__ = str(path)
            module.__package__ = package_name
            sys.modules[qualified] = module
            setattr(package, name, module)
            loaded.append(qualified)
            # Execution is restricted to the byte-verified pinned neural files.
            exec(compile(tree, str(path), "exec"), module.__dict__)  # noqa: S102
        return sys.modules[package_name + ".models"]
    except Exception:
        for qualified in loaded:
            sys.modules.pop(qualified, None)
        raise


def resolve_assets(
    manifest: dict[str, Any], repository: str
) -> tuple[Path, dict[str, Any]]:
    """Validate every listed local asset hash/size, never requesting a download.

    Args:
        manifest:
            Parsed cached speech-bench asset inventory.

        repository:
            Exact inventory repository key.

    """
    entry = manifest[repository]
    snapshot = Path(entry["snapshot"])
    if entry["revision"] != ASSET_REVISIONS[repository] or not entry["files"]:
        raise ValueError(f"Missing required asset provenance for {repository}")
    for item in entry["files"]:
        name = item["name"]
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Invalid asset-relative path: {name}")
        path = snapshot / name
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Offline asset integrity mismatch: {repository}/{name}")
    return snapshot, entry


def validate_config(config: dict[str, Any]) -> None:
    """Fail rather than publish artifacts contradicting the fixed stage contract.

    Args:
        config:
            Official Melo configuration, including model/data/symbol dimensions.

    """
    data = config["data"]
    model = config["model"]
    expected = {
        "inter_channels": 192,
        "hidden_channels": 192,
        "gin_channels": 256,
        "n_layers_trans_flow": 3,
        "resblock": "1",
        "upsample_rates": [8, 8, 2, 2, 2],
    }
    for key, value in expected.items():
        if model[key] != value:
            raise ValueError(f"Unsupported Melo configuration: {key}")
    if data["sampling_rate"] != 44100 or data["hop_length"] != 512:
        raise ValueError("Expected native 44100Hz / hop512 Melo assets")
    if (
        data["n_speakers"] <= 0
        or not config["num_languages"]
        or not config["num_tones"]
    ):
        raise ValueError("Explicit speaker/language/tone vocabulary required")
    if model.get("n_flow_layer", 4) != 4 or not model.get("use_transformer_flow", True):
        raise ValueError("Expected four transformer flow couplings")
    if model.get("flow_share_parameter", False):
        raise ValueError("Shared transformer flow is outside this pinned contract")


def example_inputs(stage: str, length: int, config: dict[str, Any]) -> tuple[Any, ...]:
    """Construct deterministic, bounded-length valid Torch inputs for qualification.

    Args:
        stage:
            One of bert, encoder, flow, decoder.

        length:
            Dynamic token/phone/frame count, at least one.

        config:
            Official Melo config used to bound generated discrete IDs.

    """
    # Torch is a heavy preparation-only dependency, imported after CPU budgeting.
    import torch

    generator = torch.Generator().manual_seed(20261005 + length)
    if stage == "bert":
        ids = torch.arange(length, dtype=torch.int64).unsqueeze(0) + 100
        return ids, torch.ones_like(ids), torch.zeros_like(ids)
    if stage == "encoder":
        speakers = config["data"]["spk2id"]
        chinese = "ZH" in speakers
        pool = (
            ("n", "i", "h", "ao", "sh", "i", "j", "ie", "_")
            if chinese
            else (
                "hh",
                "ah",
                "l",
                "ow",
                "w",
                "er",
                "d",
                "_",
            )
        )
        phones = torch.tensor(
            [
                [
                    config["symbols"].index(pool[index % len(pool)])
                    for index in range(length)
                ]
            ],
            dtype=torch.int64,
        )
        tones = torch.full_like(phones, 3 if chinese else 8)
        languages = torch.full_like(phones, 0 if chinese else 2)
        return (
            phones,
            tones,
            languages,
            torch.randn(1, 768, length, generator=generator),
            torch.tensor([next(iter(speakers.values()))], dtype=torch.int64),
            torch.randn(1, 2, length, generator=generator),
        )
    if stage in {"flow", "decoder"}:
        return (
            torch.randn(1, 192, length, generator=generator),
            torch.randn(1, 256, 1, generator=generator),
        )
    raise ValueError(f"Unknown stage: {stage}")


def array_error_metrics(actual: Any, expected: Any) -> dict[str, Any]:
    """Report finite shape-compatible errors without enforcing numerical budgets.

    Args:
        actual:
            Candidate numeric array, usually stage output or waveform.

        expected:
            Independent FP32 reference array of the same shape.

    """
    # NumPy is an optional preparation-only numerical dependency.
    import numpy as np

    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.shape != expected_array.shape:
        raise ValueError("Candidate and reference geometry differ")
    if not np.isfinite(actual_array).all() or not np.isfinite(expected_array).all():
        raise ValueError("Candidate or reference has nonfinite values")
    error = actual_array.astype(np.float64) - expected_array.astype(np.float64)
    mse = float(np.mean(error * error))
    power = float(np.mean(expected_array.astype(np.float64) ** 2))
    snr = None if mse == 0 else float(10 * np.log10(max(power, 1e-30) / mse))
    return {
        "max_abs_error": float(np.max(np.abs(error))),
        "rmse": mse**0.5,
        "signal_power": power,
        "signal_rms": power**0.5,
        "snr_db": snr,
        "exact": mse == 0,
    }


def compare_arrays(
    actual: Any, expected: Any, atol: float, rtol: float
) -> dict[str, Any]:
    """Enforce finite shape-compatible elementwise parity and report error metrics.

    Args:
        actual:
            Candidate numeric array, usually stage output or waveform.

        expected:
            Independent FP32 reference array of the same shape.

        atol:
            Explicit absolute elementwise error budget.

        rtol:
            Explicit relative elementwise error budget.

    """
    # NumPy is an optional preparation-only numerical dependency.
    import numpy as np

    metrics = array_error_metrics(actual, expected)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    return metrics


def waveform_snr_gate(
    signal_rms: float, max_abs_error: float, snr_db: float | None, precision: str
) -> dict[str, Any]:
    """Enforce SNR, allowing sub-PCM16-LSB errors on essentially silent references.

    This gate does not replace geometry, finite-value, exact-duration or allclose
    validation. Quiet means RMS strictly below 1e-4 in normalized audio units;
    its maximum absolute error must be at most one PCM16 LSB (1/32768).

    Args:
        signal_rms:
            Nonnegative reference waveform RMS in normalized full-scale units.

        max_abs_error:
            Nonnegative maximum candidate/reference error in full-scale units.

        snr_db:
            Reference/error power ratio in dB; None denotes exactly zero error.

        precision:
            Root storage policy selecting 60/40/30 dB for FP32/FP16/INT8.

    """
    if (
        not math.isfinite(signal_rms)
        or not math.isfinite(max_abs_error)
        or signal_rms < 0
        or max_abs_error < 0
        or (snr_db is not None and not math.isfinite(snr_db))
        or (snr_db is None and max_abs_error != 0)
    ):
        raise ValueError("Invalid waveform gate metrics")
    floor = WAVEFORM_SNR_FLOORS[precision]
    quiet = signal_rms < QUIET_WAVEFORM_RMS and max_abs_error <= PCM16_LSB
    if snr_db is not None and snr_db < floor and not quiet:
        raise ValueError(f"Candidate waveform SNR below {floor}dB")
    return {
        "snr_floor_db": floor,
        "quiet_exception_used": snr_db is not None and snr_db < floor and quiet,
        "quiet_rms_strict_upper_bound": QUIET_WAVEFORM_RMS,
        "quiet_max_abs_error_inclusive_bound": PCM16_LSB,
    }


def save_pipeline_evidence(
    directory: Path,
    durable_directory: Path,
    length: int,
    arrays: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    """Persist NPZ arrays and JSON metrics before gates, surviving staging deletion.

    Args:
        directory:
            Unpublished staging root receiving the artifact copy.

        durable_directory:
            Unique diagnostics run directory outside unpublished staging.

        length:
            Synthetic fixture phone count used in the evidence filename.

        arrays:
            Reference/candidate waveform and duration arrays, plus sample rate.
            Candidate audio may be absent until duration validation succeeds.

        metadata:
            Metrics, input seeds, gate status and diagnostic reasons available so far.

    """
    # NumPy is an optional preparation-only numerical dependency.
    import numpy as np

    evidence = f"qualification-{length}.npz"
    np.savez_compressed(durable_directory / evidence, **arrays)
    metadata_path = durable_directory / f"qualification-{length}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(durable_directory / evidence, directory / evidence)
    shutil.copyfile(metadata_path, directory / metadata_path.name)
    return evidence


def capture_original_reference(
    synthesizer: Any, bert: Any, models: ModuleType, config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Capture unadapted pinned inference, full BERT and original spline references.

    Standard normals are injected into upstream torch.randn/randn_like calls,
    not into adapted equations. Original SynthesizerTrn.infer supplies its dense
    alignment and waveform reference. This function runs before any graph
    adaptation/weight-normalization folding/BERT truncation.

    Args:
        synthesizer:
            Strictly loaded original FP32 model with original relative attention.

        bert:
            Locally loaded FP32 twelve-layer BertModel without pooling.

        models:
            Verified private upstream models module; sibling transforms is original.

        config:
            Official model configuration used to construct embedded synthetic inputs.

    """
    # Torch is an optional heavy preparation/parity dependency.
    import torch

    package_name = models.__package__
    if package_name is None:
        raise ValueError("Original neural models module has no private package")
    transform = sys.modules[package_name + ".transforms"]
    references: list[dict[str, Any]] = []
    with torch.inference_mode():
        for length in (4, 5, 9):
            token_inputs = example_inputs("bert", length, config)
            token_features = (
                bert(
                    input_ids=token_inputs[0],
                    attention_mask=token_inputs[1],
                    token_type_ids=token_inputs[2],
                    output_hidden_states=True,
                    return_dict=True,
                )
                .hidden_states[-3]
                .clone()
            )
            phone_inputs = list(example_inputs("encoder", length, config))
            phone_inputs[3] = token_features.transpose(1, 2).contiguous()
            phones, tones, languages, features, speaker_id, duration_noise = (
                phone_inputs
            )
            lengths = torch.tensor([length], dtype=torch.int64)
            zero_bert = torch.zeros(1, 1024, length)
            speaker = synthesizer.emb_g(speaker_id).unsqueeze(-1)
            hidden, mean, logscale, mask = synthesizer.enc_p(
                phones,
                lengths,
                tones,
                languages,
                zero_bert,
                features,
                g=None if synthesizer.use_vc else speaker,
            )
            log_dp = synthesizer.dp(hidden, mask, g=speaker)
            with patch("torch.randn", return_value=duration_noise) as draw:
                log_sdp = synthesizer.sdp(
                    hidden,
                    mask,
                    g=speaker,
                    reverse=True,
                    noise_scale=0.8,
                )
            draw_args = draw.call_args
            if (
                draw.call_count != 1
                or draw_args is None
                or draw_args.args != (1, 2, length)
            ):
                raise ValueError("Original SDP noise injection contract changed")
            counts = torch.ceil(torch.exp(log_sdp * 0.2 + log_dp * 0.8)).long()
            frames = int(counts.sum().item())
            if not 1 <= frames <= 1024:
                raise ValueError(f"Unexpected parity fixture frame count: {frames}")
            generator = torch.Generator().manual_seed(731 + length)
            frame_noise = torch.randn(1, 192, frames, generator=generator)
            with (
                patch("torch.randn", return_value=duration_noise) as draw,
                patch("torch.randn_like", return_value=frame_noise) as frame_draw,
            ):
                audio, _, _, expanded = synthesizer.infer(
                    phones,
                    lengths,
                    speaker_id,
                    tones,
                    languages,
                    zero_bert,
                    features,
                    noise_scale=0.6,
                    noise_scale_w=0.8,
                    sdp_ratio=0.2,
                )
            if draw.call_count != 1 or frame_draw.call_count != 1:
                raise ValueError(
                    "Original full inference noise injection contract changed"
                )
            latent_inputs = example_inputs("flow", length, config)
            flow_reference = synthesizer.flow(
                latent_inputs[0],
                torch.ones(1, 1, length),
                g=latent_inputs[1],
                reverse=True,
            )
            decoder_reference = synthesizer.dec(latent_inputs[0], g=latent_inputs[1])
            # A mixed interval explicitly includes both tails and exact endpoints.
            spline_values = (
                torch.tensor([-12, -5, -1, 0, 1, 5, 12]).reshape(1, 1, -1).float()
            )
            widths = torch.randn(1, 1, 7, 10, generator=generator)
            heights = torch.randn(1, 1, 7, 10, generator=generator)
            derivatives = torch.randn(1, 1, 7, 9, generator=generator)
            spline_reference, _ = transform.piecewise_rational_quadratic_transform(
                spline_values,
                widths,
                heights,
                derivatives,
                inverse=True,
                tails="linear",
                tail_bound=5.0,
            )
            references.append(
                {
                    "length": length,
                    "token_inputs": token_inputs,
                    "token_features": token_features,
                    "phone_inputs": tuple(phone_inputs),
                    "encoder": (
                        mean.clone(),
                        logscale.clone(),
                        log_dp.clone(),
                        log_sdp.clone(),
                        speaker.clone(),
                    ),
                    "latent_inputs": latent_inputs,
                    "flow": flow_reference.clone(),
                    "decoder": decoder_reference.clone(),
                    "counts": counts.clone(),
                    "frame_noise": frame_noise,
                    "expanded": tuple(item.clone() for item in expanded),
                    "audio": audio.clone(),
                    "spline_inputs": (spline_values, widths, heights, derivatives),
                    "spline": spline_reference.clone(),
                }
            )
    return references


def expand_prior(
    mean: Any, logscale: Any, log_dp: Any, log_sdp: Any
) -> tuple[Any, Any, Any]:
    """Reconstruct batch-one durations/prior expansion without dense alignment.

    Args:
        mean:
            FP32 prior mean [1,192,P].

        logscale:
            FP32 logarithmic prior scale [1,192,P].

        log_dp:
            FP32 deterministic log duration [1,1,P].

        log_sdp:
            FP32 stochastic log duration [1,1,P], already noise-scaled by encoder.

    """
    # Torch is an optional heavy preparation/parity dependency.
    import torch

    counts = torch.ceil(torch.exp(log_sdp * 0.2 + log_dp * 0.8)).long()
    indices = torch.repeat_interleave(torch.arange(mean.shape[-1]), counts.flatten())
    return counts, mean.index_select(-1, indices), logscale.index_select(-1, indices)


def verify_original_reference(
    graphs: dict[str, Any], references: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compare adapted stages and repeat-based full waveform against original source.

    Args:
        graphs:
            Adapted FP32 Torch BERT/encoder/flow/decoder graph dictionary.

        references:
            Captures taken before neural adaptation using original neural helpers.

    """
    # Torch and the spline graph are optional heavy parity dependencies.
    import torch

    from hoast.tts2.graphs import inverse_rational_spline

    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for case in references:
            errors: dict[str, Any] = {}
            token_features = graphs["bert"](*case["token_inputs"])
            errors["bert_full_hidden_states_minus3"] = compare_arrays(
                token_features.numpy(),
                case["token_features"].numpy(),
                3e-4,
                3e-4,
            )
            encoded = graphs["encoder"](*case["phone_inputs"])
            for name, actual, expected in zip(
                STAGE_CONTRACTS["encoder"]["outputs"],
                encoded,
                case["encoder"],
                strict=True,
            ):
                errors[name] = compare_arrays(
                    actual.numpy(), expected.numpy(), 3e-4, 3e-4
                )
            for stage in ("flow", "decoder"):
                actual = graphs[stage](*case["latent_inputs"])
                errors[stage] = compare_arrays(
                    actual.numpy(), case[stage].numpy(), 3e-4, 3e-4
                )
            counts, mean, logscale = expand_prior(*encoded[:4])
            torch.testing.assert_close(counts, case["counts"], atol=0, rtol=0)
            prior = mean + case["frame_noise"] * torch.exp(logscale) * 0.6
            z = graphs["flow"](prior, encoded[4])
            audio = graphs["decoder"](z, encoded[4])
            for name, actual, expected in zip(
                ("z", "sampled_prior", "expanded_mean", "expanded_logscale"),
                (z, prior, mean, logscale),
                case["expanded"],
                strict=True,
            ):
                errors[name] = compare_arrays(
                    actual.numpy(), expected.numpy(), 3e-4, 3e-4
                )
            errors["original_full_audio"] = compare_arrays(
                audio.numpy(), case["audio"].numpy(), 3e-4, 3e-4
            )
            dense_spline = inverse_rational_spline(*case["spline_inputs"])
            errors["original_spline_tails_and_bounds"] = compare_arrays(
                dense_spline.numpy(),
                case["spline"].numpy(),
                2e-6,
                2e-6,
            )
            results.append(
                {
                    "length": case["length"],
                    "frames": int(counts.sum()),
                    "errors": errors,
                }
            )
    return results


def compress_stage_weights(stage: str, model: Any) -> tuple[Any, dict[str, Any]]:
    """Apply native NNCF INT8_SYM weight-only compression with protected weights.

    No activation quantization or pseudo-data calibration is used. Nonlinear
    spline/normalization/softmax operations remain FP32 graph arithmetic.
    Embeddings, relative tables, DP/SDP, prior heads, coupling post heads, decoder
    boundary/conditioning convolutions and transpose upsampling stay FP32.
    The initial CLI policy compresses BERT only; other stages require opt-in.

    Args:
        stage:
            Contract key used to protect stage-specific sensitive weights.

        model:
            Uncompressed FP32 OpenVINO model, mutated/returned by NNCF.

    """
    # NNCF is a heavy optional preparation dependency; never a runtime import.
    # Its documented CI switch disables telemetry for this offline qualification.
    os.environ["NNCF_CI"] = "1"
    import nncf

    protected: list[str] = []
    for node in model.get_ops():
        name = node.get_friendly_name()
        lowered = name.lower()
        operation = node.get_type_name()
        sensitive = (
            operation
            in {
                "Gather",
                "ConvolutionBackpropData",
                "GroupConvolutionBackpropData",
            }
            or "emb_rel" in lowered
            or "spk_emb_linear" in lowered
        )
        if operation == "MatMul":
            # Constant folding can erase parameter names, so also recognize
            # the pinned nine-position/96-channel relative embedding geometry.
            weight_shape = node.input_value(1).get_partial_shape()
            if weight_shape.is_static:
                dimensions = list(weight_shape.get_shape())
                sensitive = sensitive or (9 in dimensions and 96 in dimensions)
        if stage == "encoder":
            sensitive = sensitive or any(
                token in lowered
                for token in (
                    "sdp",
                    "inverse_flows",
                    ".dp",
                    "/dp",
                    "enc_p.proj",
                    "enc_p.ja_bert_proj",
                    "enc_p.bert_proj",
                )
            )
        elif stage == "flow":
            sensitive = sensitive or ".post" in lowered or "/post" in lowered
        elif stage == "decoder":
            sensitive = sensitive or any(
                token in lowered
                for token in (
                    "conv_pre",
                    "conv_post",
                    ".cond",
                    "/cond",
                )
            )
        if sensitive and operation in {
            "Gather",
            "MatMul",
            "Convolution",
            "GroupConvolution",
            "ConvolutionBackpropData",
            "GroupConvolutionBackpropData",
        }:
            protected.append(name)
    if stage == "encoder" and (
        not any("sdp" in name.lower() for name in protected)
        or not any(".dp" in name.lower() or "/dp" in name.lower() for name in protected)
    ):
        raise ValueError("Cannot identify protected DP/SDP scopes in encoder IR")
    before = sum(
        node.get_type_name() == "Constant"
        and node.get_element_type().get_type_name() in {"i8", "u8"}
        for node in model.get_ops()
    )
    compressed = nncf.compress_weights(
        model,
        mode=nncf.CompressWeightsMode.INT8_SYM,
        ignored_scope=nncf.IgnoredScope(names=sorted(set(protected))),
    )
    after = sum(
        node.get_type_name() == "Constant"
        and node.get_element_type().get_type_name() in {"i8", "u8"}
        for node in compressed.get_ops()
    )
    if after <= before:
        raise ValueError(f"NNCF did not produce INT8 weight constants for {stage}")
    return compressed, {
        "algorithm": "nncf.compress_weights",
        "mode": "INT8_SYM",
        "kind": "weight-only; no activation quantization",
        "nncf": nncf.__version__,
        "protected_nodes": sorted(set(protected)),
        "new_int8_constants": after - before,
    }


def select_stage_precision(
    stage: str,
    precision: str,
    int8_stages: tuple[str, ...] = ("bert",),
    fp16_stages: tuple[str, ...] = DEFAULT_FP16_STAGES,
) -> str:
    """Select stage weight storage, preserving FP32 for unselected sensitive stages.

    Args:
        stage:
            BERT, encoder, inverse flow or decoder contract key.

        precision:
            Root artifact policy FP32, FP16 or INT8; mixed storage is permitted.

        int8_stages:
            Nonempty explicit INT8 opt-ins; other stages retain FP32 weights.

        fp16_stages:
            Nonempty FP16 opt-ins, defaulting to BERT and decoder only.

    """
    if stage not in STAGE_CONTRACTS or precision not in PRECISIONS:
        raise ValueError("Invalid stage or artifact precision")
    for selection in (int8_stages, fp16_stages):
        if not selection or any(name not in STAGE_CONTRACTS for name in selection):
            raise ValueError("Invalid FP16 or INT8 stage selection")
    selected_stages = fp16_stages if precision == "FP16" else int8_stages
    return precision if precision == "FP32" or stage in selected_stages else "FP32"


def export_stage(
    stage: str,
    graph: Any,
    destination: Path,
    config: dict[str, Any],
    threads: int,
    precision: str = "FP32",
    int8_stages: tuple[str, ...] = ("bert",),
    fp16_stages: tuple[str, ...] = DEFAULT_FP16_STAGES,
) -> dict[str, Any]:
    """Convert FP32 reference IR, compress weights and qualify serialized IR.

    Qualification covers three lengths across the relative-window boundary.
    It is numerical stage qualification, not perceptual speech qualification.
    CPU uses one stream, explicit worker budget and FP32 execution hint.

    Args:
        stage:
            Contract key naming the graph and its XML/BIN pair.

        graph:
            Eval-mode FP32 Torch graph with positional input/output contract.

        destination:
            Unpublished temporary artifact directory.

        config:
            Official model configuration.

        threads:
            One or two CPU threads enforced by the caller and OpenVINO.

        precision:
            FP32, FP16 compressed float weights, or experimental INT8 weights.
            External floating tensors and CPU qualification accumulation stay FP32.

        int8_stages:
            Explicit INT8 stage opt-ins; unselected stages retain FP32 weights.

        fp16_stages:
            Explicit FP16 stage opt-ins; default BERT/decoder preserves sensitive
            encoder/flow FP32 weights and their stricter stage qualification.

    """
    # Conversion and numerical libraries are optional preparation dependencies.
    import openvino as ov
    import torch

    selected = select_stage_precision(stage, precision, int8_stages, fp16_stages)
    logger.info(
        "tts2.prepare stage=%s precision=%s phase=convert threads=%d",
        stage,
        selected,
        threads,
    )
    contract = STAGE_CONTRACTS[stage]
    specifications = []
    for name, spec in contract["inputs"].items():
        shape = [
            ov.Dimension(1, 512)
            if axis == "S"
            else ov.Dimension(1, -1)
            if isinstance(axis, str)
            else axis
            for axis in spec["shape"]
        ]
        dtype = ov.Type.i64 if spec["dtype"] == "int64" else ov.Type.f32
        specifications.append((name, ov.PartialShape(shape), dtype))
    graph.eval()
    examples = example_inputs(stage, 5, config)
    with torch.inference_mode():
        converted = ov.convert_model(
            graph, example_input=examples, input=specifications
        )
    if len(converted.inputs) != len(contract["inputs"]) or len(
        converted.outputs
    ) != len(contract["outputs"]):
        raise ValueError(f"Converted {stage} port count differs from contract")
    for port, (name, spec) in zip(
        converted.inputs, contract["inputs"].items(), strict=True
    ):
        port.get_tensor().set_names({name})
        expected_type = ov.Type.i64 if spec["dtype"] == "int64" else ov.Type.f32
        if port.get_element_type() != expected_type:
            raise ValueError(f"Converted {stage}/{name} input dtype mismatch")
        for axis, expected in zip(port.partial_shape, spec["shape"], strict=True):
            if isinstance(expected, str):
                if axis.is_static:
                    raise ValueError(f"Converted {stage}/{name} lost its dynamic axis")
            elif not axis.is_static or axis.get_length() != expected:
                raise ValueError(f"Converted {stage}/{name} changed a fixed dimension")
    for port, name in zip(converted.outputs, contract["outputs"], strict=True):
        port.get_tensor().set_names({name})
        if port.get_element_type() != ov.Type.f32:
            raise ValueError(f"Converted {stage}/{name} output dtype mismatch")
    core = ov.Core()
    settings = {
        "INFERENCE_NUM_THREADS": threads,
        "NUM_STREAMS": 1,
        "PERFORMANCE_HINT": "LATENCY",
        "INFERENCE_PRECISION_HINT": "f32",
    }
    reference_compiled = core.compile_model(converted, "CPU", settings)
    fp32_parity = qualify_stage(stage, graph, reference_compiled, config, "FP32")
    del reference_compiled
    compression: dict[str, Any] = {}
    if selected == "INT8":
        converted, compression = compress_stage_weights(stage, converted)
    path = destination / f"{stage}.xml"
    ov.save_model(converted, path, compress_to_fp16=selected == "FP16")
    # Validate the serialized candidate, retaining FP32 Torch as the reference.
    serialized = core.read_model(str(path))
    constant_types: dict[str, int] = {}
    for node in serialized.get_ops():
        if node.get_type_name() == "Constant":
            dtype_name = node.get_element_type().get_type_name()
            constant_types[dtype_name] = constant_types.get(dtype_name, 0) + 1
    if selected == "FP16" and not constant_types.get("f16", 0):
        raise ValueError(f"Serialized {stage} contains no compressed FP16 constants")
    if selected == "INT8" and not (
        constant_types.get("i8", 0) or constant_types.get("u8", 0)
    ):
        raise ValueError(f"Serialized {stage} contains no compressed INT8 constants")
    compiled = core.compile_model(serialized, "CPU", settings)
    comparisons = qualify_stage(stage, graph, compiled, config, selected)
    logger.info("tts2.prepare stage=%s precision=%s status=verified", stage, selected)
    return {
        **contract,
        "xml": f"{stage}.xml",
        "bin": f"{stage}.bin",
        "weight_precision": selected,
        "qualification_accumulation": "FP32",
        "execution_precision_hints": {
            "cpu": "f32",
            "gpu_candidate": "f16" if selected == "FP16" else "f32",
            "gpu_qualified": False,
        },
        "fp32_reference_parity": fp32_parity,
        "parity": comparisons,
        "serialized_constant_types": constant_types,
        "compression": compression,
    }


def qualify_stage(
    stage: str, graph: Any, compiled: Any, config: dict[str, Any], precision: str
) -> list[dict[str, Any]]:
    """Compare CPU FP32-accumulating stage outputs against FP32 Torch at 4/5/9.

    Args:
        stage:
            Fixed input/output contract key.

        graph:
            Adapted FP32 Torch reference, never compressed or cast to FP16.

        compiled:
            OpenVINO candidate compiled with an explicit FP32 precision hint.

        config:
            Official Melo configuration for bounded synthetic inputs.

        precision:
            Candidate storage precision selecting explicit numerical gates.

    """
    # NumPy/Torch are optional preparation-only numerical dependencies.
    import numpy as np
    import torch

    contract = STAGE_CONTRACTS[stage]
    atol, rtol = PRECISION_TOLERANCES[precision]
    comparisons: list[dict[str, Any]] = []
    for length in (4, 5, 9):
        logger.debug(
            "tts2.stage-parity stage=%s precision=%s length=%d input_seed=%d",
            stage,
            precision,
            length,
            20261005 + length,
        )
        inputs = example_inputs(stage, length, config)
        with torch.inference_mode():
            reference = graph(*inputs)
        outputs = reference if isinstance(reference, tuple) else (reference,)
        result = compiled(
            {
                name: tensor.numpy()
                for name, tensor in zip(contract["inputs"], inputs, strict=True)
            }
        )
        errors: dict[str, dict[str, Any]] = {}
        for (name, spec), tensor in zip(
            contract["outputs"].items(), outputs, strict=True
        ):
            actual = result[compiled.output(name)]
            expected = tensor.numpy()
            wanted_shape = tuple(
                length * 512
                if axis == "512*F"
                else length
                if isinstance(axis, str)
                else axis
                for axis in spec["shape"]
            )
            if actual.shape != wanted_shape or expected.shape != wanted_shape:
                raise ValueError(
                    f"Dynamic {stage}/{name} shape mismatch at length {length}"
                )
            if actual.dtype != np.float32 or not np.isfinite(actual).all():
                raise ValueError(f"Invalid {stage}/{name} output at length {length}")
            errors[name] = compare_arrays(actual, expected, atol, rtol)
        comparisons.append(
            {"length": length, "errors": errors, "atol": atol, "rtol": rtol}
        )
    return comparisons


def qualify_pipeline(
    graphs: dict[str, Any],
    directory: Path,
    config: dict[str, Any],
    precision: str,
    threads: int,
) -> list[dict[str, Any]]:
    """Gate candidate durations/waveforms against FP32 Torch with fixed input noise.

    Token features feed synthetic one-to-one phone fixtures; this tests graph
    composition, not linguistic frontend correctness. Integer durations must
    match exactly. Elementwise waveform budgets and SNR floors are explicit,
    conservative candidate gates, not substitutes for perceptual qualification.
    Quiet references (RMS <1e-4) may bypass only SNR when maximum error is at most
    one PCM16 LSB. Candidate/reference NPZ and JSON metrics are saved before gates
    under .cache/hoast/diagnostics/tts2-qualification/RUN, surviving staging cleanup.

    Args:
        graphs:
            Adapted FP32 Torch reference stages, never weight-compressed.

        directory:
            Unpublished XML/BIN directory, also receiving qualification NPZ files.

        config:
            Official Melo configuration used to construct valid synthetic IDs.

        precision:
            Requested candidate precision selecting error budgets and SNR floor.

        threads:
            Explicit CPU worker budget, one or two; all stages use one stream.

    """
    # OpenVINO/Torch/NumPy are optional heavy preparation dependencies.
    import numpy as np
    import openvino as ov
    import torch

    core = ov.Core()
    settings = {
        "INFERENCE_NUM_THREADS": threads,
        "NUM_STREAMS": 1,
        "PERFORMANCE_HINT": "LATENCY",
        "INFERENCE_PRECISION_HINT": "f32",
    }
    compiled = {
        stage: core.compile_model(str(directory / f"{stage}.xml"), "CPU", settings)
        for stage in STAGE_CONTRACTS
    }
    atol, rtol = PRECISION_TOLERANCES[precision]
    snr_floor = WAVEFORM_SNR_FLOORS[precision]
    diagnostic_root = Path(".cache/hoast/diagnostics/tts2-qualification")
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    durable_directory = Path(
        tempfile.mkdtemp(prefix=f"{directory.name}-", dir=diagnostic_root)
    ).resolve()
    logger.info("tts2.pipeline-parity durable_evidence=%s", durable_directory)
    results: list[dict[str, Any]] = []
    for length in (4, 5, 9):
        tokens = example_inputs("bert", length, config)
        phones = list(example_inputs("encoder", length, config))
        with torch.inference_mode():
            token_features = graphs["bert"](*tokens)
            phones[3] = token_features.transpose(1, 2).contiguous()
            encoded = graphs["encoder"](*phones)
            counts, mean, logscale = expand_prior(*encoded[:4])
            frames = int(counts.sum().item())
            if not 1 <= frames <= 1024:
                raise ValueError(f"Unexpected pipeline fixture frame count: {frames}")
            generator = torch.Generator().manual_seed(911 + length)
            noise = torch.randn(1, 192, frames, generator=generator)
            prior = mean + noise * torch.exp(logscale) * 0.6
            z = graphs["flow"](prior, encoded[4])
            reference_audio = graphs["decoder"](z, encoded[4]).numpy()
        bert_result = compiled["bert"](
            {
                name: tensor.numpy()
                for name, tensor in zip(
                    STAGE_CONTRACTS["bert"]["inputs"],
                    tokens,
                    strict=True,
                )
            }
        )
        features = bert_result[compiled["bert"].output("token_features")]
        inputs = {
            name: tensor.numpy()
            for name, tensor in zip(
                STAGE_CONTRACTS["encoder"]["inputs"],
                phones,
                strict=True,
            )
        }
        inputs["phone_features"] = np.ascontiguousarray(features.transpose(0, 2, 1))
        encoder_result = compiled["encoder"](inputs)
        actual = {
            name: encoder_result[compiled["encoder"].output(name)]
            for name in STAGE_CONTRACTS["encoder"]["outputs"]
        }
        log_duration = actual["log_sdp"] * np.float32(0.2) + actual[
            "log_dp"
        ] * np.float32(0.8)
        if not np.isfinite(log_duration).all():
            raise ValueError("Candidate log durations are nonfinite")
        actual_counts = np.ceil(np.exp(log_duration)).astype(np.int64)
        evidence_arrays = {
            "reference_audio": reference_audio,
            "reference_durations": counts.numpy(),
            "candidate_durations": actual_counts,
            "sample_rate": np.asarray(44100, dtype=np.int64),
        }
        evidence_metadata: dict[str, Any] = {
            "phones": length,
            "frames": frames,
            "precision": precision,
            "input_seed": 20261005 + length,
            "frame_noise_seed": 911 + length,
            "durable_directory": str(durable_directory),
            "status": "duration_gate_pending",
        }
        save_pipeline_evidence(
            directory, durable_directory, length, evidence_arrays, evidence_metadata
        )
        try:
            np.testing.assert_array_equal(actual_counts, counts.numpy())
        except AssertionError as error:
            evidence_metadata.update(status="rejected", reason=str(error))
            save_pipeline_evidence(
                directory, durable_directory, length, evidence_arrays, evidence_metadata
            )
            raise
        indices = np.repeat(np.arange(length), actual_counts.reshape(-1))
        candidate_prior = actual["mean"][..., indices] + noise.numpy() * np.exp(
            actual["logscale"][..., indices]
        ) * np.float32(0.6)
        flow_result = compiled["flow"](
            {
                "latent": candidate_prior,
                "speaker_condition": actual["speaker_condition"],
            }
        )
        decoder_result = compiled["decoder"](
            {
                "latent": flow_result[compiled["flow"].output("z")],
                "speaker_condition": actual["speaker_condition"],
            }
        )
        audio = decoder_result[compiled["decoder"].output("audio")]
        evidence_arrays["candidate_audio"] = audio
        evidence_metadata.update(status="waveform_gates_pending", durations_exact=True)
        evidence = save_pipeline_evidence(
            directory, durable_directory, length, evidence_arrays, evidence_metadata
        )
        try:
            if audio.shape != (1, 1, 512 * frames) or audio.dtype != np.float32:
                raise ValueError(
                    "Candidate waveform geometry/dtype violates native audio contract"
                )
            metrics = array_error_metrics(audio, reference_audio)
            evidence_metadata.update(waveform=metrics, atol=atol, rtol=rtol)
            save_pipeline_evidence(
                directory, durable_directory, length, evidence_arrays, evidence_metadata
            )
            compare_arrays(audio, reference_audio, atol, rtol)
            gate = waveform_snr_gate(
                metrics["signal_rms"],
                metrics["max_abs_error"],
                metrics["snr_db"],
                precision,
            )
        except (ValueError, AssertionError) as error:
            evidence_metadata.update(status="rejected", reason=str(error))
            save_pipeline_evidence(
                directory, durable_directory, length, evidence_arrays, evidence_metadata
            )
            raise
        evidence_metadata.update(status="accepted", waveform_gate=gate)
        save_pipeline_evidence(
            directory, durable_directory, length, evidence_arrays, evidence_metadata
        )
        results.append(
            {
                "phones": length,
                "frames": frames,
                "durations_exact": True,
                "waveform": metrics,
                "snr_floor_db": snr_floor,
                "waveform_gate": gate,
                "atol": atol,
                "rtol": rtol,
                "evidence": evidence,
                "durable_evidence": str(durable_directory / evidence),
                "qualification": "synthetic numerical gate only; perceptual/real-text qualification pending",
            }
        )
    return results


def prepare(
    source: Path,
    assets_manifest: Path,
    output: Path,
    language: str,
    threads: int,
    precision: str = "FP32",
    int8_stages: tuple[str, ...] = ("bert",),
    verify_original: bool = False,
    verify_only: bool = False,
    *,
    fp16_stages: tuple[str, ...] = DEFAULT_FP16_STAGES,
    g2p_resource_dir: Path = DEFAULT_G2P_RESOURCE_DIR,
    nltk_resource_dir: Path = DEFAULT_NLTK_RESOURCE_DIR,
) -> Path:
    """Strict-load pinned local checkpoints, export/qualify and atomically publish.

    Required assets are verified before loading or writing artifacts. Failures
    remove the unpublished staging directory, leaving published output intact.
    The CLI logs complete tracebacks to a durable preparation log.

    Args:
        source:
            Exact pinned offline MeloTTS source checkout.

        assets_manifest:
            Cached JSON inventory of official model and BERT assets with hashes.

        output:
            Root artifact directory; publication target is output/language.

        language:
            EN or ZH, selecting the official Melo and uncased BERT checkpoints.

        threads:
            Explicit one/two-thread budget for preparation and qualification.

        precision:
            FP32, FP16 compressed-weight storage, or experimental NNCF INT8 weights.

        int8_stages:
            INT8 stage opt-ins; default BERT only. Other stages retain FP32 storage.

        verify_original:
            Capture original neural/BERT references before adapting and compare all
            stages, duration expansion, rational-spline tails and full waveform.

        verify_only:
            Run original-source parity without conversion or artifact publication;
            write OUTPUT/LANGUAGE-original-parity.json as the standalone report.

        fp16_stages:
            Explicit FP16 storage opt-ins; default BERT/decoder preserves FP32
            encoder/flow. Root FP16 policy retains its full-waveform SNR gate.

        g2p_resource_dir:
            Explicit offline g2p_en static-data directory; never imported or executed.
            Only required for artifact publication, not standalone neural parity.

        nltk_resource_dir:
            Explicit offline NLTK data root containing unpacked CMU dictionary and
            perceptron tagger resources; no NLTK package import or download.

    """
    if sys.version_info[:2] != (3, 14):
        raise RuntimeError("Melo preparation requires native Python 3.14")
    if language not in REPOSITORIES or threads not in (1, 2):
        raise ValueError("Expected EN/ZH and explicit one/two-thread budget")
    stage_precisions = {
        stage: select_stage_precision(stage, precision, int8_stages, fp16_stages)
        for stage in STAGE_CONTRACTS
    }
    logger.info(
        "tts2.prepare language=%s precision=%s threads=%d source=%s revision=%s verify_original=%s verify_only=%s",
        language,
        precision,
        threads,
        source,
        SOURCE_REVISION,
        verify_original,
        verify_only,
    )
    configure_cpu_budget(threads)
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    # These heavy neural libraries must load only after the process CPU budget.
    import torch
    from transformers import BertModel

    from hoast.tts2.graphs import (
        BertGraph,
        DecoderGraph,
        EncoderGraph,
        FlowGraph,
        adapt_inference_model,
    )

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    target = output / language
    if target.exists() and not verify_only:
        raise FileExistsError(f"Refusing to replace existing artifacts: {target}")
    inventory = json.loads(assets_manifest.read_text())
    melo_repository, bert_repository = REPOSITORIES[language]
    melo_path, melo_assets = resolve_assets(inventory, melo_repository)
    bert_path, bert_assets = resolve_assets(inventory, bert_repository)
    for name in ("checkpoint.pth", "config.json"):
        if name not in {item["name"] for item in melo_assets["files"]}:
            raise ValueError(f"Required Melo asset not inventoried: {name}")
    for name in ("config.json", "pytorch_model.bin", "tokenizer.json"):
        if name not in {item["name"] for item in bert_assets["files"]}:
            raise ValueError(f"Required BERT asset not inventoried: {name}")
    config = json.loads((melo_path / "config.json").read_text())
    validate_config(config)
    models = load_neural_source(source.resolve())
    synthesizer = (
        models.SynthesizerTrn(
            len(config["symbols"]),
            config["data"]["filter_length"] // 2 + 1,
            config["train"]["segment_size"] // config["data"]["hop_length"],
            n_speakers=config["data"]["n_speakers"],
            num_tones=config["num_tones"],
            num_languages=config["num_languages"],
            **config["model"],
        )
        .float()
        .eval()
    )
    checkpoint = torch.load(
        melo_path / "checkpoint.pth", map_location="cpu", weights_only=True
    )
    synthesizer.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    loaded_bert: Any = BertModel.from_pretrained(
        bert_path,
        local_files_only=True,
        add_pooling_layer=False,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        output_loading_info=True,
    )
    bert, bert_loading = loaded_bert
    if any(
        bert_loading[key] for key in ("missing_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"BERT trunk weights did not load completely: {bert_loading}")
    for key in bert_loading["unexpected_keys"]:
        if not key.startswith(("cls.", "bert.pooler.", "pooler.")) and key not in {
            "bert.embeddings.position_ids",
            "embeddings.position_ids",
            "bert.embeddings.token_type_ids",
            "embeddings.token_type_ids",
        }:
            raise ValueError(f"Unexpected non-head BERT checkpoint key: {key}")
    bert.float().eval()
    references = (
        capture_original_reference(synthesizer, bert, models, config)
        if verify_original or verify_only
        else []
    )
    adapt_inference_model(synthesizer)
    graphs = {
        "bert": BertGraph(bert).eval(),
        "encoder": EncoderGraph(synthesizer).eval(),
        "flow": FlowGraph(synthesizer).eval(),
        "decoder": DecoderGraph(synthesizer).eval(),
    }
    original_parity = (
        verify_original_reference(graphs, references) if references else None
    )
    del references
    output.mkdir(parents=True, exist_ok=True)
    if verify_only:
        report = output / f"{language}-original-parity.json"
        with tempfile.NamedTemporaryFile(mode="w", dir=output, delete=False) as stream:
            json.dump(
                {
                    "version": 1,
                    "language": language,
                    "reference_precision": "FP32",
                    "source_revision": SOURCE_REVISION,
                    "assets_manifest_sha256": sha256(assets_manifest),
                    "bert_loading": bert_loading,
                    "original_source_parity": original_parity,
                },
                stream,
                indent=2,
            )
            temporary_report = Path(stream.name)
        os.replace(temporary_report, report)
        logger.info("tts2.original-parity status=ok report=%s", report)
        return report
    temporary = Path(tempfile.mkdtemp(prefix=f".{language}-prepare-", dir=output))
    try:
        shutil.copyfile(melo_path / "config.json", temporary / "config.json")
        shutil.copyfile(bert_path / "tokenizer.json", temporary / "tokenizer.json")
        populate_frontend_assets(
            source,
            temporary,
            language,
            g2p_resource_dir=g2p_resource_dir,
            nltk_resource_dir=nltk_resource_dir,
        )
        stages = {
            name: export_stage(
                name,
                graph,
                temporary,
                config,
                threads,
                precision,
                int8_stages,
                fp16_stages,
            )
            for name, graph in graphs.items()
        }
        waveform_parity = qualify_pipeline(
            graphs, temporary, config, precision, threads
        )
        shutil.copyfile(source / "LICENSE", temporary / "MeloTTS-LICENSE.txt")
        manifest = {
            "version": 1,
            "language": language,
            "backend": "openvino",
            "precision": precision,
            "precision_policy": {
                "requested_weight_precision": precision,
                "int8_stages": list(int8_stages) if precision == "INT8" else [],
                "fp16_stages": list(fp16_stages) if precision == "FP16" else [],
                "stage_weight_precisions": stage_precisions,
                "weight_precision_stage_counts": {
                    value: sum(
                        selected == value for selected in stage_precisions.values()
                    )
                    for value in PRECISIONS
                },
                "graph_float_dtype": "FP32",
                "external_float_dtype": "FP32",
                "reference_precision": "FP32",
                "cpu_execution": "INFERENCE_PRECISION_HINT=f32; dequantize/decompress weights for FP32 accumulation",
                "gpu_candidate_execution": "use per-stage execution_precision_hints; FP32 sensitive stages retain f32; requires separate GPU numeric/audio qualification",
                "gpu_qualified": False,
                "fp16_semantics": "FP16 constant storage with FP32 graph arithmetic; not whole-graph FP16 execution",
                "int8_semantics": "weight-only symmetric INT8; protected sensitive weights remain FP32; not activation quantization",
            },
            "sample_rate": 44100,
            "hop_length": 512,
            "batch_size": 1,
            "dimensions": {
                "S": {"minimum": 1, "maximum": 512},
                "P": {"minimum": 1, "maximum": None},
                "F": {"minimum": 1, "maximum": None},
            },
            "stages": stages,
            "orchestration": {
                "duration_noise": "standard normal float32; encoder multiplies by 0.8 exactly once",
                "noise_scale_w": 0.8,
                "sdp_ratio": 0.2,
                "noise_scale": 0.6,
                "length_scale": "1/speed",
                "duration": "ceil(exp(log_sdp*sdp_ratio + log_dp*(1-sdp_ratio))*length_scale)",
                "prior_expansion": "repeat phone indices by durations; gather mean/logscale; no dense alignment",
                "latent": "expanded_mean + standard_normal*exp(expanded_logscale)*noise_scale",
                "masks": "all ones; batch one; no padding",
                "bert": "embeddings plus first 10 of 12 layers; exact MLM hidden_states[-3]; no MLM head/pooler",
                "zero_bert": "1024-channel zeros folded; bert_proj bias retained",
                "relative_attention": "full quadratic attention, relative embedding window four",
                "frontend": "native text preparation; provenance in frontend-assets.json; recursively inventoried data-only assets",
            },
            "source": {
                "repository": "https://github.com/myshell-ai/MeloTTS",
                "revision": SOURCE_REVISION,
                "hashes": SOURCE_HASHES,
                "ast_modifications": [
                    "rewrite neural melo imports to private package",
                    "remove sole training-only monotonic_align import",
                ],
                "license": "MeloTTS-LICENSE.txt (MIT, Copyright (c) 2024 MyShell.ai)",
            },
            "assets": {melo_repository: melo_assets, bert_repository: bert_assets},
            "assets_manifest_sha256": sha256(assets_manifest),
            "versions": {
                "python": platform.python_version(),
                **{
                    name: importlib.metadata.version(name)
                    for name in ("torch", "openvino", "transformers", "numpy")
                },
                **(
                    {"nncf": importlib.metadata.version("nncf")}
                    if precision == "INT8"
                    else {}
                ),
            },
            "preparation_threads": threads,
            "verification": {
                "dynamic_lengths": [4, 5, 9],
                "atol": 3e-4,
                "rtol": 3e-4,
                "reference": "adapted FP32 Torch inference stages",
                "bert_loading": bert_loading,
                "original_source_parity": original_parity,
                "candidate_waveform_parity": waveform_parity,
                "precision_tolerances": PRECISION_TOLERANCES,
                "limitations": "synthetic fixtures only; real-text frontend/audio and perceptual qualification pending; GPU execution unqualified",
            },
            "files": inventory_files(temporary),
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if target.exists():
            raise FileExistsError(f"Publication target appeared concurrently: {target}")
        os.rename(temporary, target)
    except Exception:
        shutil.rmtree(temporary)
        raise
    logger.info(
        "tts2.prepare status=ok language=%s output=%s threads=%d",
        language,
        target,
        threads,
    )
    return target


def main() -> None:
    """Parse the native offline preparation CLI and preserve complete failure logs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--assets-manifest", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--g2p-resource-dir",
        type=Path,
        default=DEFAULT_G2P_RESOURCE_DIR,
        help="Offline g2p_en static data only; package code is never imported",
    )
    parser.add_argument(
        "--nltk-resource-dir",
        type=Path,
        default=DEFAULT_NLTK_RESOURCE_DIR,
        help="Offline unpacked CMU dictionary and perceptron tagger data",
    )
    parser.add_argument("--language", choices=("EN", "ZH"), required=True)
    parser.add_argument("--threads", type=int, choices=(1, 2), required=True)
    parser.add_argument(
        "--precision",
        type=str.upper,
        choices=PRECISIONS,
        required=True,
        help="Weight storage; CPU qualification always uses FP32 accumulation",
    )
    parser.add_argument(
        "--fp16-stages",
        nargs="+",
        choices=tuple(STAGE_CONTRACTS),
        default=list(DEFAULT_FP16_STAGES),
        help="FP16 storage opt-ins (default bert decoder); other stages remain FP32",
    )
    parser.add_argument(
        "--int8-stages",
        nargs="+",
        choices=tuple(STAGE_CONTRACTS),
        default=["bert"],
        help="Experimental INT8 opt-ins; unselected stages remain FP32",
    )
    parser.add_argument(
        "--verify-original",
        action="store_true",
        help="Compare against unadapted pinned neural/full-BERT/full-waveform references",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Standalone original-source parity report; no conversion/artifact publication",
    )
    args = parser.parse_args()
    configure_logging(log_file=Path(".cache/hoast/diagnostics/prepare-tts2.log"))
    try:
        prepare(
            args.source,
            args.assets_manifest,
            args.output,
            args.language,
            args.threads,
            args.precision,
            tuple(args.int8_stages),
            args.verify_original,
            args.verify_only,
            fp16_stages=tuple(args.fp16_stages),
            g2p_resource_dir=args.g2p_resource_dir,
            nltk_resource_dir=args.nltk_resource_dir,
        )
    except Exception:
        logger.exception(
            "tts2.prepare status=failed language=%s output=%s",
            args.language,
            args.output,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
