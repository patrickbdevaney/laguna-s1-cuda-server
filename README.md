# laguna-s1-cuda-server

A pure-CUDA inference engine and OpenAI-compatible server for
[`poolside/Laguna-S-2.1-NVFP4`](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4), a 117.6B-parameter
mixture-of-experts model with 8.5B active, on one NVIDIA Jetson AGX Thor (`sm_110a`, 122 GB unified memory).
Speculative decoding uses poolside's
[`Laguna-S-2.1-DFlash-NVFP4`](https://huggingface.co/poolside/Laguna-S-2.1-DFlash-NVFP4) drafter. Everything from the
checkpoint on disk to the HTTP response is C++/CUDA. Python exists only in `oracle/`, for validation.

## Status (2026-07-26)

This is a research release from a single maintainer. It serves one stream at a time with a 262K-token default
context, and its greedy output is bit-exact against a reference transcription of poolside's own modeling code.

| Component | State |
|---|---|
| 48-layer engine: NVFP4 experts, FP8 attention and dense weights, sliding and global attention | Complete; greedy output bit-exact against the oracle |
| DFlash speculative decoding | Complete; greedy-exact at every draft length |
| Server: OpenAI endpoint, prefix cache, tool calls, reasoning split, WebUI, terminal client | Complete |
| Gates: capacity, roofline, oracle, loader, kernels, batched ≡ sequential, DFlash, server | All pass |
| NVFP4 attention weights (+19.9% measured available) | Validated, not built |
| Capability benchmark | Not completed (see [Limitations](#limitations)) |

## Results

All numbers were measured on one Jetson AGX Thor.

**Decode**

| Workload | tok/s | Tokens accepted per target forward |
|---|---:|---:|
| Autoregressive, ctx 4,096 | **33.2** | 1 |
| Served, prose | 33.6 | 1.00 (the policy declines to speculate) |
| Served, code | 40.1 | 3.67 |
| Served, repetitive or edit-style | **49.7** | 14.6 |

- **Bandwidth efficiency.** Plain decode reads 6.25 GB per token, about 85% of the bandwidth wall measured on this
  box. The best published batch-1 efficiency for a whole decode step is 82% (FlashFormer, on H100).
- **Reference point.** poolside reports 13–14 tok/s for this model on a DGX Spark, the closest hardware in the same
  bandwidth class, and 22–24 tok/s with DFlash. Those runs were under vLLM.

**The path from first forward to 33.2 tok/s.** Every step was measured end to end and kept greedy output 8/8 against
the oracle ([`docs/OPTIMIZATION_LOG.md`](docs/OPTIMIZATION_LOG.md)):

| Step | tok/s |
|---|---:|
| First working forward | 3.66 |
| Hardware FP4 converter, flash-decoding attention split, MoE grid sized by actual M | 9.23 |
| Offline expert repack, thread per output | 16.57 |
| Whole-step CUDA graph | 18.74 |
| FP8 attention weights (W8A16, per-row scale) | 20.44 |
| 4 warps per block: the 24-CTA limit had left half the warp slots unreachable | 24.59 |
| 16-byte FP8 loads; segmented GEMMs (q\|k\|v\|g, gate\|up) | 27.35 |
| NVFP4 scale layout reordered for coalesced reads | 30.8 |
| FP8 `lm_head`, shared experts and layer-0 MLP | 33.0 |
| MoE inversion fused (7 launches per layer → 1) | **33.2** |

**Context.** Only the 12 global-attention layers keep a growing KV cache, at 24,576 bytes per token. The 36
sliding-window layers use a 512-slot ring, a constant 1.05 MB per sequence.
- At the 262K default, KV takes 6.48 GB and the process 78 GB of 122.
- Allocated context costs no decode speed; only context actually used does. At 175 GB/s, 128K of used context
  keeps 89% of peak decode and 256K keeps 80%.

## Findings

**1. Only the routed experts shipped quantized, so the per-token bytes were twice the naive estimate.**
- Attention, shared experts, router, layer-0 MLP and `lm_head` shipped in BF16. That is 8.03 GB of the checkpoint
  and 5.61 GB of every decode step, so bytes per token started at 10.04 GB, not the ~4.5 GB a uniformly quantized
  model would give.
- Converting those weights to FP8 here brought it to 6.25 GB.
- **What is left:** attention projections are now 44.9% of the bytes. Simulated NVFP4 on them keeps greedy output
  bit-identical, 8/8, while the same gate fails 2.81-bit experts 0/8. A microbenchmark converts 95% of the saving
  to speed, for +19.9% projected ([`docs/ATTENTION_NVFP4.md`](docs/ATTENTION_NVFP4.md)).

**2. Decode, prefill and speculative verify must be the same arithmetic.**
- The flash-decoding split count originally depended on M. That changed the fp32 rescale chain, so the same token
  rounded differently in decode and prefill. 1 ulp at layer 0, amplified ~1.4× per layer, flipped the argmax by
  layer 48.
- Sizing the split from key length alone makes all three paths bit-identical (Gate B1c, 10/10). Without that, a
  DFlash acceptance rate would not mean anything.

**3. The speculation controller, not the drafter, is the live defect** ([`docs/DECODE_FINAL.md`](docs/DECODE_FINAL.md)).
- A fixed `SPEC_K=2` reaches 40.99 tok/s on code, +21% over no speculation. On prose, speculation off wins every
  cell (33.9 tok/s).
- The adaptive policy's boot-time estimate of the plain-decode rate swung between 20.1 and 33.3 across restarts,
  against a true 33.7–33.9. So it sometimes speculates on prose and sometimes gives up the gain on code.
- A static rule (speculate on code, not on prose) currently beats it on both workloads.

**4. Acceptance rate cannot grade quantization quality.** Coarsening the experts to 2.81 bits *raised* τ at every
draft length and gave an apparent +7.6% speed, without saving a byte. It also failed the greedy gate 0/8.
Degrading the target can make it easier to predict ([`docs/EXPERT_BITS_EVAL.md`](docs/EXPERT_BITS_EVAL.md)).

**5. Several frontier levers were priced against a profile and rejected with arithmetic**, not left as
aspirations:
- a 3-bit trellis expert format measured +7% end to end, not the projected +15.2%
  ([`docs/TRELLIS_VERDICT.md`](docs/TRELLIS_VERDICT.md));
- activation sparsity ([`docs/ACTIVATION_SPARSITY.md`](docs/ACTIVATION_SPARSITY.md));
- lossless entropy coding of the weights (FP8 weights measure 82.2% entropy, against a 72% break-even);
- a grid-cooperative megakernel (35% of token time spent in `grid.sync`);
- tree verification, Lookahead decoding, and certified expert skipping.

The full list, with the measurement for each, is [`docs/WIKI.md`](docs/WIKI.md) §5.

## Engineering

**The checkpoint's layout is unusual, and each fact below cost a failed gate to learn.**
- **Global scale.** `weight_global_scale` is a reciprocal (`2688/amax`), so dequantization divides.
  Multiplying gives absmax ≈ 3e7 and garbage experts, silently.
- **Router.** It is sigmoid with top-10 of 256. `e_score_correction_bias` affects selection only, never the
  returned weights.
- **Per-layer shapes.** Head counts differ by layer type: 48 on global layers, 72 on sliding layers. Rotary
  differs too: YaRN θ=5e5 on half the dims for global, θ=1e4 on all of them for sliding.
- **Output head.** `lm_head` is not tied to the embedding, and the vocabulary is 100,352.

**Gate methodology**
- Kernel gates compare against a bf16-activation reference, because the deployed model is bf16. fp32 activations
  flip 15 of 540 router selections between experts whose scores differ by 1e-5.
- Gates on random token ids are the strictest form, since an off-distribution router turns 1 ulp into a different
  expert. Two paths that round differently are compared against the oracle, never against each other.

**Kernel lessons**
- A small lookup table indexed at runtime inside an inner loop lives in local memory. The hardware FP4 converter
  was worth 2.5× on that kernel.
- An offline expert repack won +70% on the MoE and lost 33% on the BF16 GEMMs. Trading warps for iterations per
  warp pays only when the iteration count is small.

**Measurement on Thor**
- Short microbenchmarks measure the idle clock. Spin 300 ms first and discard the first sample.
- The first kernel over a fresh allocation is dominated by page-table warmup.
- `cudaMalloc` memory streams at 227–254 GB/s; a `cudaHostRegister`'d mmap only at 160, so weights are never
  mapped.

**Speculation**
- A correctness gate cannot see a broken speculator. When a buffer overflow made the drafter emit one constant
  token, every draft was rejected and output stayed correct. τ = 1.000 was the only signal, so served
  measurements report accepted tokens per forward.

## Getting started

Requirements:
- Jetson AGX Thor with CUDA for `sm_110a`.
- Both checkpoints in `models/`: `Laguna-S-2.1-NVFP4`, which loads as 71.9 GB in one arena, and
  `Laguna-S-2.1-DFlash-NVFP4`.
- Run nothing else GPU-heavy alongside.

```bash
huggingface-cli download poolside/Laguna-S-2.1-NVFP4        --local-dir models/Laguna-S-2.1-NVFP4
huggingface-cli download poolside/Laguna-S-2.1-DFlash-NVFP4 --local-dir models/Laguna-S-2.1-DFlash-NVFP4

nvcc -O3 -std=c++17 -arch=sm_110a -I. -o build/lgserve src/server.cu kernels/*.cu -lpthread
g++  -O2 -std=c++17 -I. -o build/lgchat tools/lgchat.cc -lpthread

CTX=262144 ./build/lgserve        # OpenAI endpoint and WebUI on :8080; loads in ~45 s
./build/lgchat                    # terminal client
```

**API**
- **Endpoints:** `POST /v1/chat/completions` (streaming and not), `GET /v1/models`, `GET /healthz`, and a WebUI at
  `/`.
- **Reasoning and tools:** reasoning arrives as `reasoning_content` until the model closes `</think>`, then as
  `content`. Tool calls are parsed with the same `poolside_v1` grammar the prompt is rendered with.
- **Speculation control:** `SPEC_K=0` disables speculation; `SPEC_K=2` is the measured best on code.
- **Policy visibility:** `timings.spec_arms` reports what the speculation policy currently believes about each arm.

**Gates and benchmarks**

```bash
nvcc -O3 -std=c++17 -arch=sm_110a -o build/gate_kernels tests/gate_kernels.cu kernels/*.cu && ./build/gate_kernels
nvcc -O3 -std=c++17 -arch=sm_110a -o build/bench_decode tests/bench_decode.cu kernels/*.cu
CTX=4096 N=40 ./build/bench_decode              # median decode plus a greedy re-check
CTX=4096 N=40 LG_PROF=1 ./build/bench_decode    # per-category profile
oracle-venv/bin/python oracle/validate_matrix.py   # oracle validation (needs its venv)
```

## Compatibility

| Platform | State |
|---|---|
| Jetson AGX Thor (`sm_110a`), 122 GB unified | Tested |
| DGX Spark / GB10 (`sm_121`) | Not tested; same bandwidth class, different architecture target |
| Other GPUs | Not supported: kernels and occupancy choices are tuned to Thor's 20 SMs |

## Limitations

- **Single stream.** No batching across requests.
- **No completed capability benchmark.** The GSM8K sweep's baseline run lost its server partway through and is
  void. Quality evidence is bit-exactness against the reference implementation, which inherits whatever the
  published NVFP4 checkpoint loses against BF16.
- **The adaptive speculation policy is unreliable** (Finding 3). Pin `SPEC_K` by workload until it is fixed.
- **Acceptance degrades with prompt length.** τ falls from 0.817 at 67 prompt tokens to 0.533 at 15,267.
- **Context past 262,144** needs a documented RoPE override and is untested for quality.

## Roadmap

1. **NVFP4 attention projections:** validated on both axes, +19.9% projected to ~39.8 tok/s base.
2. **Fix the speculation controller:** more samples per arm and periodic re-measurement of unselected arms, or a
   content-aware static rule. Worth up to +21% on code.
3. **Complete the capability sweep** against a supervised server.

## Repository map

| Path | Contents |
|---|---|
| [`kernels/`](kernels/) | GEMMs (FP4, FP8, BF16), attention with FP8 KV and sliding rings, MoE, elementwise |
| [`src/`](src/) | Forward pass and the server |
| [`include/`](include/) | Config parser (all constants read from `config.json`), arena loader, safetensors reader |
| [`tests/`](tests/) | `gate_*` (correctness), `bench_*` (performance), bandwidth probes |
| [`oracle/`](oracle/) | Reference transcription of `modeling_laguna.py` and golden-tensor generation (validation only) |
| [`tools/`](tools/) | Roofline model, sweeps, terminal client, requantization and evaluation tools |
| [`docs/`](docs/) | Start at [`WIKI.md`](docs/WIKI.md). Also [`ROOFLINE`](docs/ROOFLINE.md), [`MODEL_INVENTORY`](docs/MODEL_INVENTORY.md), [`ARCH_DELTA`](docs/ARCH_DELTA.md), [`RESCOPE`](docs/RESCOPE.md) (corrections to the original brief), [`LOOP_LOG`](docs/LOOP_LOG.md) (gate record), [`OPTIMIZATION_LOG`](docs/OPTIMIZATION_LOG.md), [`DECODE_FINAL`](docs/DECODE_FINAL.md), cross-model comparison ([`APPENDIX_CROSS_MODEL`](docs/APPENDIX_CROSS_MODEL.md)) |
| [`dev/archive/`](dev/archive/) | The original directive, superseded plans and status, research prompts and findings, sweep logs |

## Contributing and contact

Issues are welcome for reproducibility problems and wrong numbers. Pull requests are not being accepted yet.

## License

- **Code in this repository:** MIT ([`LICENSE`](LICENSE)).
- **Third-party code keeps its own license:** `include/third_party/httplib.h` and `json.hpp` (both MIT).
- **Weights** are not distributed here. Laguna-S-2.1 is released by poolside under OpenMDW-1.1; this work is not
  affiliated with or endorsed by poolside.

## Citation

```bibtex
@misc{devaney2026lagunacuda,
  title  = {laguna-s1-cuda-server: a pure-CUDA server for Laguna-S-2.1 NVFP4 with DFlash speculation on Jetson AGX Thor},
  author = {Devaney, Patrick},
  year   = {2026},
  url    = {https://github.com/patrickbdevaney/laguna-s1-cuda-server}
}
```
