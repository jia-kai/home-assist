# Home Assistant

A new project for a home assistant.

## Voice Responses

- **Voice-first:** all user-facing tool responses must be brief, natural to say
  aloud, and easy to understand by listening. Prefer a short confirmation or a
  concise clarification with one actionable next step.
- Do not read out candidate lists, player inventories, raw tool JSON, technical
  diagnostics, or lengthy metadata. Write technical details to logs rather than
  spoken responses; retaining them only in tool results is not sufficient.
- Log tool names, relevant arguments, outcome statuses, and diagnostic reasons
  needed to investigate failures or ambiguity. Use `debug` for detailed results
  and candidate lists, `warning` for recoverable problems, and `error` with
  exception tracebacks for failures. Never log credentials or access tokens.
- For unknown or ambiguous locations, say something like: “I don't know
  ‘Springfield’. Please tell me the exact location.” Do not enumerate matches.
- Bound spoken location, song, and artist labels. Use a short reference such as
  “that location” or “your selection” for long labels; mention at most the first
  artist in music confirmations.
- Keep weather answers in the compact condition, integer temperature, and rain
  chance format. Preserve unavailable-data semantics and distinguish requested
  actions from confirmed outcomes when shortening responses.
- Apply these requirements to every formatted tool outcome, including success,
  ambiguity, missing results, and playback-control failures.

## Coding Style

- **Type annotations:** annotate all function parameters, return values, and
  class/dataclass member variables.
- Avoid `typing.cast` when a value can be constructed with its declared type.
  Construct `TypedDict` values with `DictType(key=value, ...)` instead of casting
  a plain dictionary.
- Prefer `@dataclass(slots=True)` for stateful value objects.
- **No backslash line continuation:** wrap long statements with parentheses
  for implicit continuation.
- **Import at module level.** Local imports are only appropriate to break a
  genuine circular import or keep a heavy optional dependency lazy. Explain
  the reason in a concise comment.
- **Logging, not `print`:** use module-level loggers and configure logging once
  per process in the entry point. Use `debug` for verbose detail, `info` for
  progress, `warning` for recoverable anomalies, and `error` for failures.
  Do not use bare `print()` in library code.

## Documentation

- Give every new or changed function and method a concise, meaningful docstring.
  Document behavior, side effects, return semantics, units where relevant, and
  non-obvious edge cases. Reserve inline comments for explaining why.
- Document every new or changed parameter in a Google-style `Args:` block,
  with its description on the following line and a blank line after each
  parameter.
- Declare every class/dataclass field in the class body as `field_name: Type`,
  including fields initialized later in `__init__`. Follow each declaration
  with its field docstring and a blank line.
- For tensors and arrays, document shapes and axis meanings. Include data types,
  units, and coordinate frames when they affect the contract.
- Describe current behavior and exact semantics rather than change history.
  Avoid phrases such as "no longer", "old", "legacy", or "now" unless
  explicitly documenting migration behavior.
- Keep interface documentation and type stubs consistent with implementation
  changes.
- Pad Markdown table columns so the raw source is aligned and readable.

## Error Handling and Validation

- **Fail loudly:** do not silently ignore errors or supply defaults for required
  inputs. Missing keys and invalid inputs must raise appropriate exceptions.
  Only catch exceptions you genuinely expect and intend to handle.
- **Program defensively:** when information has multiple local representations,
  assert the invariants tying them together. Prefer cheap checks that catch
  inconsistencies early.
- **Keep runtime validation cheap:** avoid full-array scans, per-element checks,
  and other expensive validation unless required at an external/untrusted
  boundary or correctness cannot be guaranteed by construction. Prefer
  constant-time type, shape, and metadata checks. Document and benchmark
  expensive validation on hot paths.

## Testing and Development Workflow

- Tests must not depend on external datasets or machine-local paths. Embed
  minimal fixture data in tests or use fixture files tracked in the repository.
- Before attempting to reproduce an error, preserve the complete exception
  output in a durable file, including tracebacks, chained exceptions, exception
  notes, and relevant input metadata or random-number-generator state. Do not
  rely on terminal history as the only copy of failure diagnostics.
- For non-trivial one-off analysis, write a patchable script in a temporary
  directory rather than embedding a long program in a shell command. Preserve
  reproducibility and ensure project imports resolve when running scripts
  outside the project root.
- After coding, run applicable checks when available: `clang-format` and clangd
  diagnostics on touched C++ files, and `pyright` on touched Python files.
