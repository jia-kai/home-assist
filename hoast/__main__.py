"""Run the configured voice assistant; use --text for the stdin/stdout interface."""

from .logging import configure_logging, get_logger
from .process_policy import disable_transparent_huge_pages

logger = get_logger(__name__)


def main() -> None:
    """Apply checked process memory policy before importing native inference code."""
    configure_logging()
    try:
        disable_transparent_huge_pages()
    except OSError, RuntimeError:
        logger.exception("memory.thp status=failed scope=process")
        raise SystemExit(1) from None

    # Keep heavy imports lazy so native allocations inherit the THP policy.
    from .llm_cli import main as run_assistant

    run_assistant()


if __name__ == "__main__":
    main()
