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
- Adds the inverse Hadamard transform to the Qwen3.5 MTP embedding lookup. This
  fixes MTP initialization for Prism's latent `token_embd.weight` models;
  without it, the graph verifier correctly rejects the untransformed lookup.

The CUDA PTQ1 kernels from the related Bonsai research repositories were not
ported: their trit unpacking, CUDA warp geometry, and `v_perm_b32`/DP4A
assumptions do not map directly to Vulkan on RDNA2.

## RX 6700 XT benchmark

Measured on 2026-09-19 with Chrome closed:

- GPU: AMD Radeon RX 6700 XT, Vulkan0, 12,272 MiB
- Model: `Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf`
- Context: 65,536 tokens
- Target KV: `q8_0`, offloaded to Vulkan
- Placement: `--gpu-layers all`, `--fit off`
- Flash attention: enabled
- CPU: 8 threads; one server slot
- Reasoning: medium; deterministic temperature 0 benchmark request
- Generation: 128 tokens from a 22-token prompt
- MTP: `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0`
- Draft K/V: F16

The first generation includes startup/warm-cache effects. The table uses the
completed matched artifacts; the official no-MTP and optimized no-MTP rows are
single 128-token runs because the official baseline took about 140 seconds to
decode each run.

| Build | Mode | Decode tok/s | Prompt tok/s | Result |
| --- | --- | ---: | ---: | --- |
| Official Prism, unmodified | no MTP | 0.91 | 1.05 | Loads, but uses the unoptimized PQ2 Vulkan path |
| This branch | no MTP | 32.57 | 60.06 | **35.85× official** |
| Official Prism, unmodified | MTP n-max 2 | — | — | Fails graph validation before serving |
| This branch | MTP n-max 2 | 48.67 / 48.88 warm | 56.20 / 56.47 warm | 73.5% draft acceptance |

The two warm MTP runs average **48.78 tok/s**. A separate verbose single-run
receipt measured 49.17 tok/s and confirmed the same load and placement.

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
