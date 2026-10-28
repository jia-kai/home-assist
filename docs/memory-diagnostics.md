# Hoast memory policy and OOM diagnostics

## Process policy

The supported assistant entrypoint, `python -m hoast`, disables Linux transparent
huge pages (THP) before importing numerical/inference libraries. A checked
`prctl(PR_SET_THP_DISABLE, 1)` call and readback must succeed; otherwise startup
fails with a logged exception. The policy applies to this process and descendants,
including across exec, without privileges or changes to the host or other
containers. It does not split existing huge pages. Direct library imports and
other module entrypoints do not apply this policy automatically.

Expected startup record:

```text
memory.thp status=disabled scope=process inherited=true readback=1
```

Verify the actual main process on the Docker host:

```sh
pid=$(docker inspect hoast --format '{{.State.Pid}}')
grep '^THP_enabled:' /proc/"$pid"/status
```

The value must be `0`. Disabling THP is a targeted memory-policy experiment, not
a demonstrated cure for native leaks or an assurance that the 5.5 GiB limit is
sufficient. Compare end-to-end latency and memory over long sparse-command runs.

## Observations and cadence

- `memory.usage`: cheap RSS, thread, Python-block and cgroup counters at voice
  stage boundaries and every 60 seconds. Cgroup records include limit events,
  anonymous/file/shared memory, native huge-page allocation/collapse counters,
  kernel overhead and reclaim activity. Missing optional counters remain `None`.
- `memory.details`: startup and every 60 seconds, with process `smaps_rollup`
  accounting and cgroup memory pressure. These reads use a dedicated diagnostic
  thread rather than the event loop or inference worker.
- `memory.mapping`: bounded largest-anonymous-mapping breakdown at startup and
  every 15 minutes, collected by the same diagnostic thread. Mapping scans are
  intentionally not performed at every inference boundary. Output identifies
  address ranges and backing labels, not allocation contents or call stacks.

Periodic `voice.idle` observations are not proof the process is idle: correlate
with stage/activity records. There is no forced GC/heap trim, allocation tracing,
audio recording, or unbounded in-process sample history. Slow observations can
perturb execution and include timing; memory counter reads are not atomic.

Interpretation:

- Anonymous THP is a **subset** of anonymous memory; do not add them.
- Shared memory overlaps file accounting; do not sum those as disjoint data.
- RSS and cgroup charge differ because of shared/file accounting and other
  processes. Neither alone is a count of live application allocations.
- `memory.events:max` counts limit/reclaim encounters, not OOM kills.
- Lifetime peak at the limit does not measure individual call peaks.
- Constant Python-block counts do not exclude growing native allocations.
- Missing post-stage records can result from SIGKILL; a killed process cannot
  log its own final OOM counter or traceback. This in-process monitor cannot
  survive the kill or reliably capture transient sub-second peaks.

## Preserve the next failure

The user service attaches Compose output to the journal, preserving observations
across container recreation. Docker logs alone cover only the current container.
Journal retention is finite. Export evidence promptly, including the preceding
idle period, and keep exports private: existing application logs may include
spoken requests and local device information.

```sh
mkdir -p .local/diagnostics/oom-next
journalctl --user -u hoast.service --since '2 days ago' --no-pager > .local/diagnostics/oom-next/service.log
journalctl _TRANSPORT=kernel --since '2 days ago' --no-pager > .local/diagnostics/oom-next/kernel.log
```

For a suspected delayed huge-page effect, compare anonymous bytes, huge-page
bytes and collapse-counter **deltas over the same interval**, then inspect which
mappings changed. For growth confined to inference, compare the stage boundaries
with existing capture durations, synthesis metrics and generation records.
A bounded mapping report does not identify allocator ownership by itself; a
subsequent controlled run may require complete `smaps` exports or native profiling.
