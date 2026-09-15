# Smart plug CLI

Configure a local Kasa smart plug in the system `config.toml`:

```toml
[switch]
ip = "192.168.42.20"
```

Run from the project directory after `uv sync`:

```bash
uv run python -m hoast.switch --on
uv run python -m hoast.switch --off
uv run python -m hoast.switch --config /path/to/config.toml --on
```

Exactly one of `--on` or `--off` is required. The CLI reads the configured
address, sends the action, and checks the observed power state before confirming
success. The system configuration also requires its usual `[weather]` settings.
The command exits with 0 on confirmed success, 1 on failure, or 2 for invalid
arguments. Diagnostics are logged to `.cache/hoast/diagnostics/switch-cli.log`.

The CLI is independent of model loading and is not registered as an LLM tool.
Use a DHCP reservation to keep the configured address stable.
