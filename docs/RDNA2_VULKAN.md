# RDNA2 Vulkan PQ2 branch

This branch is an experimental RDNA2/Vulkan specialization of the PrismML
`prism` branch. It targets the PQ2_0 Bonsai path on AMD GPUs using Vulkan; it
does not change model weights, quantization, or the GGUF format.

Base revision: `PrismML-Eng/llama.cpp@9a9394a895b96003ca842a6041cb28ac49a108f7`

## What is optimized

- Adds the missing PQ2_0 Vulkan integer-dot MMVQ path for Q8 activations.
- Uses a PQ2-specific packed-16 load path, because PQ2's scale/payload layout
  is not naturally safe for a packed-32 load.
- Uses a wave32 PQ2 decode pipeline with two output rows per wave on RDNA2.
- Uses vector Q8 activation loads in the PQ2 decode loop.
- Adds Vulkan GatedDeltaNet rows mode. The fused Vulkan kernel reads each
  sequence's physical recurrent-state row directly from the cache instead of
  consuming a per-layer gathered temporary.
- Adds the inverse Hadamard transform to the Qwen3.5 MTP embedding lookup. This
  fixes MTP initialization for Prism's latent `token_embd.weight` models;
  without it, the graph verifier correctly rejects the untransformed lookup.

The CUDA PTQ1 kernels from the related Bonsai research repositories were not
ported: their trit unpacking, CUDA warp geometry, and `v_perm_b32`/DP4A
assumptions do not map directly to Vulkan on RDNA2.

## RX 6700 XT GDN rows-mode A/B

Measured on 2026-09-19 with Chrome closed and the working tree rebuilt after the
rows-mode implementation:

- GPU: AMD Radeon RX 6700 XT, Vulkan0, 12,272 MiB
- Model: `Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf`
- Context: 65,536 tokens
- Target KV: `q8_0`, offloaded to Vulkan
- Placement: `--gpu-layers all`, `--fit off`
- Flash attention: enabled
- CPU: 8 threads; one server slot
- Reasoning: medium; deterministic temperature 0 benchmark request
- No-MTP generation: 256 tokens from the same deterministic prompt
- MTP generation: 128 tokens from the same deterministic prompt
- MTP: `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0`
- Draft K/V: F16

The CLI run is one sequence and uses the same effective concurrency as
`--parallel 1`. Each A/B mode was run four times in a fresh process; run 1 was
discarded and the table reports runs 2-4. Standard deviation is the sample SD
of the displayed decode rates. The wall time is the generated-token count
divided by the reported decode rate, so it excludes model load and is a
reproducible generation-time estimate rather than a process wall-clock time.

| Path | Mode | Runs 2-4 decode tok/s | Mean ± SD | Prompt mean | Generation time | Output hash |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Legacy `GET_ROWS` | no MTP | 33.8, 33.9, 33.8 | 33.83 ± 0.06 | 76.4 | 7.57 s / 256 | `927933b73f80c04f` |
| Direct state rows (default) | no MTP | 34.7, 34.7, 34.6 | 34.67 ± 0.06 | 77.97 | 7.38 s / 256 | `927933b73f80c04f` |
| Legacy `GET_ROWS` | MTP n-max 2 | 46.7, 46.7, 46.7 | 46.70 ± 0.00 | 68.63 | 2.74 s / 128 | `d15d373c288580ba` |
| Direct state rows (default) | MTP n-max 2 | 47.4, 47.4, 47.4 | 47.40 ± 0.00 | 68.37 | 2.70 s / 128 | `d15d373c288580ba` |

Relative to the forced legacy path, direct rows improves measured decode by
**+2.46% without MTP** and **+1.50% with MTP**. The identical deterministic
hashes show no observed generation difference in these runs.

The verbose MTP receipt on the direct path reported **70 accepted / 112
generated draft tokens (62.5%)**, mean accepted length **2.25**, and the same
acceptance counters as the legacy path.

### Profiling receipt

`GGML_VK_PERF_LOGGER=1` was used for a short post-warmup run. Times below are
the aggregate operation times from the main timing group; they are not
wall-clock totals for the full model load.

| Mode | Path | `CPY` | `GATED_DELTA_NET` | `GET_ROWS` | `SET_ROWS` | Graph total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| no MTP | Legacy | 11.677 ms | 15.491 ms | 23.939 ms | 2.430 ms | 888.770 ms |
| no MTP | Direct rows | 11.260 ms | 21.400 ms | 2.884 ms | 2.461 ms | 880.185 ms |
| MTP | Legacy | 16.941 ms | 15.653 ms | 12.444 ms | 1.295 ms | 669.605 ms |
| MTP | Direct rows | 16.180 ms | 17.377 ms | 1.614 ms | 1.378 ms | 654.538 ms |

The remaining direct-path `GET_ROWS` is the extra recurrent-cache relocation
needed to preserve read-before-write ordering during cache reorders. The main
per-layer live-state gather is the work removed by this change. Direct state
addressing makes the GDN shader itself slower (about +38% in the short no-MTP
profile and +11% in the MTP profile), but the eliminated gather/cache traffic
still produces the positive end-to-end result.

### A/B switch

Direct rows mode is the default for Vulkan Qwen3.5 graphs. Set the environment
variable before starting the process to force the old implementation:

```text
GGML_GDN_STATE_GATHER=1
```

The switch is read while the graph is built, so use a new CLI/server process
for each A/B run. The direct shader uses a separate `_rows` pipeline variant;
the existing wave32 GDN geometry is unchanged.

### Correctness and limitations

- Four no-MTP runs and four MTP runs exited successfully; deterministic output
  hashes matched within each A/B pair.
- A Vulkan validation-layer smoke run completed with no `VUID`, validation,
  or error messages.
- The repository's proper recurrent-state rollback test was also run on CPU
  with the direct rows path and with `GGML_GDN_STATE_GATHER=1`. Both modes
  fail at the same existing dirty-context check (`position 6`, token 0:
  `4.9774 != 4.21362`). This makes the failure a baseline rollback issue,
  not evidence of a rows-mode-only regression. The older one-token-at-a-time
  rollback harness fails identically in both modes as well; its snapshot
  semantics are not a sufficient validation of the multi-token recurrent
  graph.
- The existing `llama-rs-rollback-multi` harness does not currently pass for
  this model even when forced to the legacy path. With `-n 16 -S 3 -s 5 -r 2
  -k 8`, both legacy and direct produce 4 mismatches in rolled-back sequence
  0 (first mismatch at position 12, `ref=1204`, `rb=264`), while untouched
  sequences 1 and 2 have zero mismatches. This is a baseline limitation of
  the current rollback path, not a new direct-row-only mismatch; it remains a
  follow-up before claiming full multi-sequence rollback correctness.
- GPU clock and power telemetry was not available through the installed
  Windows Vulkan tooling, so no power/clock conclusion is claimed.

### Decision and follow-up

Keep direct rows enabled by default on Vulkan. The improvement is reproducible
above the observed run-to-run noise and the old path remains available for
regression testing. A subgroup-broadcast experiment for `rows[seq]` was
implemented and measured, then reverted: in the short no-MTP profiler run it
increased the direct graph total from **880.185 ms** to **901.566 ms**
(**+2.43%**, slower), with no compensating end-to-end gain. The existing
wave32 geometry and address calculation are therefore retained. The next
useful optimization work is targeted SPIR-V/address-arithmetic inspection or
full-generation timestamp profiling, rather than adding another unmeasured
shader branch.

## Earlier fork-vs-official baseline

For historical context, the first branch comparison (before this GDN rows-mode
A/B) was also measured on 2026-09-19 with Chrome closed. It is retained here
because it documents the benefit of the earlier PQ2 Vulkan work.

| Build | Mode | Decode tok/s | Prompt tok/s | Result |
| --- | --- | ---: | ---: | --- |
| Official Prism, unmodified | no MTP | 0.91 | 1.05 | Unoptimized PQ2 Vulkan path |
| This branch | no MTP | 32.57 | 60.06 | 35.85x official |
| Official Prism, MTP n-max 2 | MTP | — | — | Fails graph validation before serving |
| This branch | MTP n-max 2 | 48.67 / 48.88 warm | 56.20 / 56.47 warm | 73.5% draft acceptance |

The official MTP failure is:

```text
Hadamard-latent table 'token_embd.weight' is read without the inverse transform
```

That is a functional compatibility gap in the unmodified Prism graph, not a
Vulkan shader crash. The fork fixes the MTP graph and keeps the verifier enabled.

## Logged allocation receipt

The verbose fork run reported these allocations for the target context and MTP
draft on the same hardware/configuration:

| Allocation | MiB |
| --- | ---: |
| Vulkan model buffer | 6,970.07 |
| CPU-mapped model buffer | 322.07 |
| Target Vulkan q8 KV, 64K | 2,176.00 |
| Target recurrent state | 448.88 |
| Target Vulkan compute | 198.28 |
| MTP draft Vulkan KV, 64K F16 | 256.00 |
| MTP draft Vulkan compute | 136.02 |
| Host output | 0.95 |

The logged Vulkan allocation subtotal is about **10,185 MiB**. The driver
reported 11,474 MiB free at load start. These are allocator-reported buffers,
not a replacement for a driver-level peak-memory trace.

## Reproduce

Build with Vulkan enabled, then launch the server with the model and the
settings above. The important options are:

```text
--device Vulkan0 --gpu-layers all --fit off
--ctx-size 65536 --kv-offload --cache-type-k q8_0 --cache-type-v q8_0
--flash-attn on --threads 8 --parallel 1
--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0
--spec-draft-type-k f16 --spec-draft-type-v f16
```

## Caveats

- This is an experimental hardware-specific fork, not a general Vulkan
  performance claim for every quantization or GPU.
- The wave32 path can change floating-point reduction order. The optimized
  deterministic response is stable within the run, but it is not claimed to be
  byte-identical to the old wave64 path.
- Run the complete BenchLocal quality suite before treating the speed result as
  a production default.
- No prebuilt Windows binaries are published by this branch yet.
