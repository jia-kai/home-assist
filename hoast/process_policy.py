"""Early Linux process policy, independent of numerical libraries and models."""

import ctypes
import os
import sys

from .logging import get_logger

logger = get_logger(__name__)
_PR_SET_THP_DISABLE = 41
_PR_GET_THP_DISABLE = 42


def disable_transparent_huge_pages() -> None:
    """Disable Linux THP for this process and verify the effective process policy.

    Call before importing inference libraries or allocating model buffers. The
    policy is inherited by child processes and across exec; it does not split
    existing huge pages or change host policy. No privileges are required.
    Unsupported platforms, failed syscalls and unexpected readback fail startup
    rather than silently running with a different memory policy.
    """
    if sys.platform != "linux":
        raise RuntimeError("Hoast's process THP policy requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    prctl.restype = ctypes.c_int
    if prctl(_PR_SET_THP_DISABLE, 1, 0, 0, 0) == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), "prctl(PR_SET_THP_DISABLE)")
    result = prctl(_PR_GET_THP_DISABLE, 0, 0, 0, 0)
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), "prctl(PR_GET_THP_DISABLE)")
    if result != 1:
        raise RuntimeError(f"Unexpected process THP-disable readback: {result}")
    logger.info(
        "memory.thp status=disabled scope=process inherited=true readback=%d", result
    )
