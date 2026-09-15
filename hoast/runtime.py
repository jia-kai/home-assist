"""Shared process CPU budgeting for local inference entrypoints."""

import os

import psutil
from threadpoolctl import ThreadpoolController, threadpool_limits

from .logging import get_logger

logger = get_logger(__name__)
_initial_affinity: frozenset[int] | None = None
_limits: list[object] = []


def configure_cpu_budget(threads: int) -> tuple[int, ...]:
    """Bound inference workers, existing native pools and Linux process CPU affinity.

    Entry points call this before loading models. Reconfiguration can select one
    or two of the CPUs allowed at the first call, never CPUs outside that mask.
    Existing threads are included because numerical libraries may create pools
    during import. Driver/helper threads may exist, but execute within this budget.

    Args:
        threads:
            One or two inference threads and permitted CPUs within the initial mask.

    """
    global _initial_affinity
    if threads not in (1, 2):
        raise ValueError("CPU budget must be one or two")
    os.environ.update(
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        OMP_WAIT_POLICY="PASSIVE",
        TOKENIZERS_PARALLELISM="false",
        TORCHINDUCTOR_COMPILE_THREADS="1",
        MAX_JOBS="1",
    )
    controller = ThreadpoolController()
    _limits[:] = [
        threadpool_limits(limits=threads, user_api="openmp"),
        controller.select(internal_api="openblas").limit(limits=1),
        controller.select(internal_api="mkl").limit(limits=threads),
    ]
    if _initial_affinity is None:
        _initial_affinity = frozenset(os.sched_getaffinity(0))
    selected = tuple(sorted(_initial_affinity)[-threads:])
    if len(selected) != threads:
        raise RuntimeError("Requested CPU budget is unavailable")
    os.sched_setaffinity(0, selected)
    for thread in psutil.Process().threads():
        try:
            os.sched_setaffinity(thread.id, selected)
        except ProcessLookupError:
            logger.debug("Thread %d exited during CPU-budget configuration", thread.id)
    logger.info(
        "cpu.budget workers=%d affinity=%s openblas_workers=1", threads, selected
    )
    return selected
