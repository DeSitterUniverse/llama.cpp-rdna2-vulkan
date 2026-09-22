# RDNA2 Vulkan PQ2

Experimental Vulkan support for Prism PQ2_0 on AMD RDNA2. No model weights or
quantization formats are changed.

## Implementation

- Adds a PQ2_0 Vulkan Q8-activation integer-dot MMVQ path with wave32 scheduling.
- Adds direct recurrent-state row reads for Qwen3.5 GatedDeltaNet; legacy
  `GET_ROWS` remains available with `GGML_GDN_STATE_GATHER=1`.
- Adds the inverse Hadamard transform used by Prism Qwen3.5 MTP models.
- Fixes PQ2 fallback dequant coverage and gates the wave32 specialization on
  subgroup-32 support; unsupported devices use the existing fallback.

## RX 6700 XT benchmark

Model: `Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf`; RX 6700 XT, Vulkan0, 65,536
context, Q8_0 K/V, all layers and KV on GPU, flash attention, 8 threads, one
slot. Each path/mode used a separate server, one discarded 32-token warm-up,
then three repeats of each prompt. Every measured response hit the 256-token
cap (`--ignore-eos`, reasoning effort `medium`, temperature 0, top-p 1, seed
5). Ranges are the three repeated samples, not confidence intervals.

No-MTP decode tok/s:

| Prompt | Legacy mean [range] | Direct rows mean [range] | Change |
| --- | ---: | ---: | ---: |
| Database transaction | 33.833 [33.803–33.861] | 34.170 [34.127–34.214] | +1.00% |
| Keyboard event to pixels | 33.875 [33.874–33.877] | 34.211 [34.170–34.233] | +0.99% |
| Bandwidth vs. compute | 33.863 [33.858–33.869] | 34.181 [34.123–34.228] | +0.94% |
| Mean across prompts | 33.857 | 34.187 | +0.98% |

MTP decode tok/s (`--spec-type draft-mtp --spec-draft-n-max 2
--spec-draft-p-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16`):

| Prompt | Legacy mean [range] | Direct rows mean [range] | Change | Accepted / drafted (rate)* |
| --- | ---: | ---: | ---: | ---: |
| Database transaction | 45.235 [45.203–45.293] | 45.895 [45.842–45.923] | +1.46% | 411/708 (58.1%) |
| Keyboard event to pixels | 43.010 [42.794–43.131] | 43.664 [43.424–43.790] | +1.52% | 392/743 (52.8%) |
| Bandwidth vs. compute | 50.933 [50.921–50.943] | 51.687 [51.661–51.704] | +1.48% | 450/627 (71.8%) |
| Mean across prompts | 46.393 | 47.082 | +1.49% | — |

*Counts sum the three repeats; matched requests had the same acceptance counts
and output hash in both paths.*

Prompt inputs:

1. `Describe how a database transaction moves from request to durable commit, step by step.`
2. `Explain what happens between a keyboard input event and pixels appearing in a desktop application.`
3. `Compare memory bandwidth and compute throughput for neural-network inference, with concrete examples.`

Identical prompt repeats reuse cached prefixes, so prompt-evaluation tok/s is
omitted; it is not comparable across repeats. Decode tok/s is the server's
timed generation rate for each 256-token response.

Server options:

~~~text
--device Vulkan0 --gpu-layers all --fit off --ctx-size 65536 --kv-offload
--cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --threads 8 --parallel 1
~~~

For MTP, add:

~~~text
--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0
--spec-draft-type-k f16 --spec-draft-type-v f16
~~~

## Correctness

- PQ2 GPU dequant regression: 16,384 values matched the CPU reference exactly
  (`max_err=0`) on RX 6700 XT. The test reproduced the old omission before the
  shader fix. PQ2 Vulkan MMVQ backend test passed 28/28 cases.
- `test-recurrent-state-rollback`: checkpoint restore, multi-sequence split
  replay (40 tokens across ubatches), and sequence isolation passed in both
  direct and legacy modes; maximum replay difference was 0.
- Deterministic no-MTP and MTP server outputs matched between direct and legacy
  paths. A separate 32-token MTP request also matched ordinary greedy output
  exactly; MTP accepted 18/23 draft tokens.
- Vulkan validation-layer smoke test reported no validation errors.
- The experimental `llama-rs-rollback-multi` helper's repeated one-token draft
  calls fail after a later rollback in both paths. That pattern is outside the
  recurrent snapshot test contract, which covers batched windows and ubatch
  splits within a decode call; it is not used as evidence of a path regression.

## Profile and decision

Direct addressing makes the GDN kernel itself slower, but removes gather and
cache traffic. This separate graph-level profile is not request wall time and
is not included in the token-rate means above:

| Mode | Legacy graph | Direct rows | Change |
| --- | ---: | ---: | ---: |
| No MTP | 888.770 ms | 880.185 ms | -0.97% |
| MTP | 669.605 ms | 654.538 ms | -2.25% |

Keep direct rows as the Vulkan default: in all six prompt/mode comparisons, the
three-run direct range is above the matching legacy range. Mean decode gain is
0.98% without MTP and 1.49% with MTP. The legacy path remains available for A/B
checks. Results apply to this RX 6700 XT and tested model/configuration, not
other Vulkan devices or quantizations.
