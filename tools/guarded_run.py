"""Run a command with a hard memory cgroup, CPU affinity and memory watchdog."""

import argparse
import math
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path
from types import FrameType
from typing import BinaryIO, NoReturn

import psutil

from hoast.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _terminate(signum: int, frame: FrameType | None) -> NoReturn:
    """Convert termination into an exception so the owned scope is cleaned up.

    Args:
        signum:
            Received termination signal number.

        frame:
            Interrupted Python frame, unused.

    """
    raise SystemExit(128 + signum)


def _stop_scope(process: subprocess.Popen[bytes], unit: str, output: BinaryIO) -> None:
    """Kill the owned scope and process group, including detached model descendants.

    Args:
        process:
            Running systemd scope launcher with its own process group.

        unit:
            Generated systemd unit name without its scope suffix.

        output:
            Durable diagnostic stream for shutdown outcomes.

    """
    subprocess.run(
        ["systemctl", "--user", "kill", "--signal=SIGKILL", f"{unit}.scope"],
        check=False,
        stdout=output,
        stderr=output,
    )
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_guarded(
    command: list[str],
    threads: int,
    memory_gib: float,
    reserve_gib: float,
    timeout: float,
    log_file: Path,
) -> int:
    """Run a bounded child and stop its scope on resource pressure or interruption.

    Diagnostic RSS sums can double-count shared pages across child processes;
    the kernel enforces MemoryMax against the cgroup's charged memory instead.

    Args:
        command:
            Executable and arguments, passed without a shell. Arguments are not logged.

        threads:
            One or two permitted CPU cores and inference workers.

        memory_gib:
            Hard cgroup memory limit in GiB; swap is disabled for this scope.

        reserve_gib:
            Minimum system available memory in GiB before the watchdog stops the job.

        timeout:
            Maximum runtime in seconds.

        log_file:
            Durable child stdout/stderr log.

    """
    if (
        not command
        or threads not in (1, 2)
        or not all(math.isfinite(value) for value in (memory_gib, reserve_gib, timeout))
        or min(memory_gib, reserve_gib, timeout) <= 0
    ):
        raise ValueError("Invalid command or resource budget")
    available = psutil.virtual_memory().available
    if available < (memory_gib + reserve_gib) * 1024**3:
        raise RuntimeError("Insufficient memory headroom to start guarded command")
    cpus = sorted(os.sched_getaffinity(0))[-threads:]
    if len(cpus) != threads:
        raise RuntimeError("Requested CPU budget unavailable")
    os.sched_setaffinity(0, cpus)
    environment = os.environ.copy()
    environment.update(
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS="1",
        OMP_WAIT_POLICY="PASSIVE",
        TOKENIZERS_PARALLELISM="false",
        TORCHINDUCTOR_COMPILE_THREADS="1",
        MAX_JOBS="1",
    )
    unit = "hoast-job-" + uuid.uuid4().hex[:12]
    wrapped = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        f"--unit={unit}",
        "-p",
        f"MemoryMax={int(memory_gib * 1024**3)}",
        "-p",
        "MemorySwapMax=0",
        "taskset",
        "-c",
        ",".join(map(str, cpus)),
        *command,
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    peak_rss = 0
    with log_file.open("ab") as output:
        process = subprocess.Popen(
            wrapped,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        logger.info(
            "Started executable=%s args=%d scope=%s cpus=%s memory_gib=%.2f",
            command[0],
            len(command) - 1,
            unit,
            cpus,
            memory_gib,
        )
        reason: str | None = None
        try:
            while process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    rss = sum(
                        child.memory_info().rss
                        for child in [parent, *parent.children(recursive=True)]
                    )
                    peak_rss = max(peak_rss, rss)
                except psutil.NoSuchProcess:
                    pass
                if psutil.virtual_memory().available < reserve_gib * 1024**3:
                    reason = "system_memory_reserve"
                elif time.monotonic() - started > timeout:
                    reason = "timeout"
                if reason is not None:
                    logger.error(
                        "Stopping scope=%s reason=%s peak_rss=%d",
                        unit,
                        reason,
                        peak_rss,
                    )
                    _stop_scope(process, unit, output)
                    break
                time.sleep(0.1)
            status = process.wait()
        except BaseException:
            logger.exception("Guard interrupted or failed; stopping scope=%s", unit)
            _stop_scope(process, unit, output)
            process.wait()
            raise
    logger.info(
        "Finished status=%d seconds=%.3f peak_process_rss_sum_gib=%.3f reason=%s",
        status,
        time.monotonic() - started,
        peak_rss / 1024**3,
        reason,
    )
    return status if reason is None else 125


def main() -> None:
    """Apply conservative defaults to a noninteractive command."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--threads",
        type=int,
        choices=(1, 2),
        default=2,
        help="CPU core and worker budget",
    )
    parser.add_argument(
        "--memory-gib",
        type=float,
        default=4,
        help="Hard memory limit for the child scope",
    )
    parser.add_argument(
        "--reserve-gib", type=float, default=3, help="System available-memory reserve"
    )
    parser.add_argument(
        "--timeout", type=float, default=600, help="Maximum command seconds"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path(".cache/hoast/diagnostics/guarded-run.log"),
        help="Durable child stdout/stderr log",
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="Executable and arguments after --"
    )
    args = parser.parse_args()
    configure_logging(log_file=args.log_file.with_suffix(".guard.log"))
    signal.signal(signal.SIGTERM, _terminate)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        status = run_guarded(
            command,
            args.threads,
            args.memory_gib,
            args.reserve_gib,
            args.timeout,
            args.log_file,
        )
    except Exception:
        logger.exception("Guarded experiment failed to start")
        raise SystemExit(1) from None
    raise SystemExit(status if status >= 0 else 128 - status)


if __name__ == "__main__":
    main()
