# Design decisions

Each entry records the context, the options considered, the decision, and the evidence. Newest
phases at the bottom. Entries that later change get an **Outcome** line instead of being rewritten.

Template:

```
### <Decision title>
- **Context:** what forced a choice.
- **Options considered:**
  1. ...
  2. ... (chosen)
- **Decision:** what was done, concretely.
- **Evidence:** links to code, tests, measurements, or source lines.
- **Why:** the reasoning.
- **Outcome:** (added later, if it changed)
```

## GPU spend

| Date | GPU | Minutes | Cost (USD) | Purpose |
|---|---|---|---|---|
| — | — | — | — | No GPU used yet |

Prices (Modal, checked 2026-11-15): L4 $0.000222/s (≈ $0.80/h), L40S $0.000542/s (≈ $1.95/h),
H100 $0.001097/s (≈ $3.95/h). CPU $0.0000131/core/s, memory $0.00000222/GiB/s.

## Phase 0 — Research and plan

### What the reference implementations do
- **Context:** The project should cover ground the well-known minimal engines don't. Both were read
  before choosing the feature set: nano-vllm at `bb823b3` (2026-04-26) and mini-sglang at
  `9a91cfa` (2026-05-17). Both are MIT-licensed.
- **Findings, nano-vllm:**
  - Offline only: `LLM.generate(prompts)`. No server, no streaming, no asyncio. Its benchmark
    reports total throughput.
  - Chunked prefill was added in April 2026 (PRs #204 and #218). Only the first waiting sequence
    may be chunked (`scheduler.py`: `if remaining < num_tokens and scheduled_seqs: break`).
  - Prefill and decode never share a step: `schedule()` returns as soon as any prefill is
    scheduled, and decode runs only when nothing is waiting. A long prompt stalls every running
    request for its whole prefill.
  - Preemption is recompute: `preempt()` frees the blocks and puts the sequence at the front of
    `waiting`. Victim is the last running sequence. No swapping.
  - Prefix cache: xxhash over each full block's token IDs chained with the previous block's hash;
    token IDs are stored and compared to rule out collisions. Block size must be a multiple of 256.
  - K/V are written to the cache with a Triton kernel.
- **Findings, mini-sglang:**
  - Online server with an OpenAI-compatible `/v1/chat/completions` and SSE.
  - Chunked prefill across multiple requests (`PrefillAdder`, `ChunkedReq`) with a prefill token
    budget (`max_extend_tokens`, default 8192).
  - A step is either prefill or decode (`Batch(phase="prefill" | "decode")`), prefill first. No
    mixing.
  - No preemption. Admission reserves room for the whole prompt plus `max_tokens`, so a request
    never runs out of KV cache mid-generation, at the cost of admitting fewer requests.
  - Radix-tree prefix cache, overlap scheduling (CPU scheduling of step N+1 during GPU step N),
    FlashInfer and custom kernels.
- **Decision:** Build an online engine whose distinguishing feature is mixing prefill chunks and
  decodes in one forward pass (next entry), with recompute preemption, block-hash prefix caching (later made optional; see below),
  and a correctness suite against Hugging Face. Neither reference has the mixed batch or tests
  against HF.

### Third feature: stall-free mixed batching instead of speculative decoding
- **Context:** The original plan was "chunked prefill". nano-vllm already has a form of it, so the
  feature had to be either something stronger in the same area or speculative decoding.
- **Options considered:**
  1. Chunked prefill alone, as in nano-vllm. Already done there; not distinctive.
  2. Speculative decoding. Qwen3 has no model smaller than 0.6B to draft with, so it would be
     n-gram (prompt lookup) drafting. Its speedup shrinks as batch size grows, which is exactly the
     loaded regime the benchmark targets, and it adds verification and rollback paths to the
     correctness suite.
  3. Stall-free mixed batching (chosen): every step has a token budget; running decodes are
     scheduled first, then prefill chunks fill the rest of the budget, and all of it runs in one
     forward pass.
- **Decision:** Option 3, behind `enable_mixed_batching`. With it off, the scheduler is
  prefill-priority (prefill-only steps whenever something can be admitted), which is the baseline
  for the ablation.
- **Why:** Prefill-priority scheduling makes inter-token latency spike for every running request
  whenever a long prompt arrives; mixing removes the stall at a small cost to TTFT. That trade-off
  is what the latency benchmark measures, so the feature and the evaluation support each other.
  FlashAttention-2's varlen kernel with `block_table` handles decodes (query length 1) and prefill
  chunks (query length > 1) as segments of the same call, so no custom kernel is needed.
- **Reference:** A. Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with
  Sarathi-Serve", OSDI 2024 (arXiv:2403.02310). Earlier version: "SARATHI: Efficient LLM
  Inference by Piggybacking Decodes with Chunked Prefills" (arXiv:2308.16369).

### Correctness: strict fp32 on CPU plus a margin rule in bf16 on GPU
- **Context:** Greedy output should match Hugging Face transformers token for token. In bf16 that
  is not well defined: different attention kernels and batch shapes reduce in different orders,
  so when the top two logits are nearly tied, two correct implementations can pick different
  tokens, and from then on the sequences differ.
- **Options considered:**
  1. Strict token equality in bf16 on GPU. Flaky by construction; failures would not mean bugs.
  2. Compare logits within an absolute tolerance only. Does not test the generation loop
     (scheduling, preemption, KV reuse) end to end.
  3. Both of the following (chosen):
     - A pure-PyTorch attention backend that runs in fp32 on CPU, compared strictly against HF
       (`attn_implementation="eager"`, fp32) on a tiny randomly initialised Qwen3 config built in
       the test. No downloads, runs in CI.
     - On GPU in bf16 with the real model: identical outputs, or a first divergence where the
       reference's top-1 minus top-2 logit gap is at most τ; plus a teacher-forced pass requiring
       top-1 agreement wherever the reference gap exceeds τ.
- **Decision:** Option 3. τ is provisionally 0.5 (logit units, on fp32-upcast logits).
- **Calibration plan (Phase 1b):** run HF against itself in bf16, batch 1 vs batch 8, on the same
  prompts and record the gap at every self-divergence. τ must be at least the largest of these,
  since a smaller τ would fail HF against itself. Record the data and the final τ here.
- **Not done:** comparing HF's SDPA and FlashAttention-2 paths as a second calibration source. It
  would widen the noise estimate but adds a flash-attn dependency to the reference environment;
  batch-shape variation alone already exercises different reduction orders.
- **Risk:** a randomly initialised model can have small logit gaps, so even fp32 might expose
  batch-shape rounding differences on CPU. If that happens, the strict suite moves to float64
  (the PyTorch backend supports any dtype) rather than gaining a tolerance.

### torch 2.10 with the prebuilt flash-attn 2.8.3 wheel; vLLM in its own image
- **Context:** Current releases (2026-11-15): torch 2.15.0, flash-attn 2.8.3.post2, transformers
  5.23.0, vLLM 0.33.0 (pins `torch==2.14.0`, `transformers>=5.16.1,<5.21.0`). FlashAttention-2
  publishes prebuilt Linux wheels only up to torch 2.10: the newest is
  `flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl` on the v2.8.3
  release; v2.8.3.post2 ships wheels only up to torch 2.8.
- **Options considered:**
  1. torch 2.14 to match vLLM, building flash-attn from source in the Modal image. Long, fragile
     image build; any cache miss repeats it.
  2. torch 2.10.0 + the prebuilt flash-attn 2.8.3 wheel (chosen).
  3. Use vLLM's bundled `vllm_flash_attn`. Couples the engine to vLLM's internals.
- **Decision:** Option 2 for stallfree. vLLM 0.33.0 runs in a separate Modal image with its own
  torch. The two systems never share a Python environment.
- **Why:** A reproducible image in minutes. The benchmark compares serving systems, each with its
  supported stack, on the same GPU, model revision, and driver.
- **To verify in Phase 1b:** the CUDA 13 wheel loads under Modal's driver on L4 and L40S.

### KV block size 256, shared by both attention backends
- **Context:** vLLM uses 16-token blocks; nano-vllm requires multiples of 256.
- **Evidence:** FlashAttention-2 v2.8.3 and v2.8.3.post2, `csrc/flash_attn/flash_api.cpp`, lines
  583 (`mha_varlen_fwd`) and 1264 (`mha_fwd_kvcache`):
  `TORCH_CHECK(!paged_KV || page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256")`.
- **Decision:** `block_size = 256` for the cache layout, the block manager, and both attention
  backends. The PyTorch backend could use any size, but it has to read the same cache layout to be
  a meaningful cross-check.
- **Trade-offs:**
  - Prefix caching works on full blocks only, so a shared prefix shorter than 256 tokens is never
    reused, and reuse is rounded down to a multiple of 256.
  - Internal fragmentation: on average half a block (128 tokens ≈ 14 MiB for Qwen3-0.6B) per
    running sequence is reserved but unused. With ~40 GB of KV on an L40S (~1,400 blocks) and at
    most a few hundred running sequences this is acceptable; it is noted in the results if it
    limits batch size.
  - Preemption frees and reallocates in larger units.
- **Not chosen:** FlashInfer (supports small pages) adds a second attention dependency.

### KV cache write without Triton
- **Context:** nano-vllm writes K/V into the cache with a Triton kernel. This project uses no
  custom kernels.
- **Decision:** `index_copy_` on the flattened cache, `[num_blocks * block_size, n_kv_heads,
  head_dim]`, using `slot_mapping`.
- **Why:** One PyTorch op per layer for K and one for V; correctness is easy to test on CPU.
  Overhead to be measured in Phase 1b.

### Development and benchmark GPUs
- **Options (Modal, 2026-11-15):** T4 $0.59/h (Turing; FlashAttention-2 does not support it, and no
  bf16), L4 $0.80/h (Ada, 24 GB), A10 $1.10/h, L40S $1.95/h (Ada, 48 GB), H100 $3.95/h.
- **Decision:** L4 for development and GPU tests; L40S for the final benchmark only. H100 not used.
- **Why:** L4 is the cheapest card that runs the same kernels (Ada, sm_89) as the benchmark card.

### Phase 1 split into a CPU phase and a GPU phase
- **Context:** Development happens on a Mac without CUDA, and GPU time costs money. Most of the
  engine (model math, paging, scheduling, preemption) can be proven correct without a GPU.
- **Decision:** Phase 1a is CPU only: Qwen3 forward, the PyTorch attention backend, block manager,
  scheduler, synchronous engine, and the strict fp32 suite against HF in CI. Phase 1b adds the
  FlashAttention-2 backend, KV sizing from GPU memory, the bf16 margin suite on L4, and τ
  calibration.
- **Why:** Bugs in paging and scheduling are found in CI for free, and the first GPU session starts
  from an engine already token-exact on CPU, so any GPU-only difference points at the flash backend
  or bf16 numerics.

### Prefix caching moved to an optional phase after Phase 3
- **Context:** Prefix caching adds hashing, sharing, and eviction to the block manager, the module
  with the most subtle invariants. Getting allocation, reference counts, and preemption
  right first matters more.
- **Decision:** The Phase 1a block manager does allocation, reference counts, and free only.
  Reference counts are there from the start (always 1 until sharing exists) so the interface does
  not change later. Prefix caching becomes an optional phase after Phase 3, with its interface and
  tests written at the start of that phase. The benchmark runs with prefix caching off in every
  system, so it does not depend on this phase.
- **Design notes for that phase:**
  - Hash only full blocks. Block *i*'s hash covers block *i−1*'s hash and block *i*'s token IDs,
    so equal hashes mean equal full prefixes, not just equal blocks.
  - Store each cached block's token IDs and compare them on a hit, so a hash collision costs a
    miss instead of wrong output (nano-vllm does the same).
  - A freed block keeps its hash and stays in the free list; it can be revived by a later hit until
    it is reallocated. Allocation takes from the least recently freed end (LRU).
  - Never serve the last prompt token from cache: at least one token must be computed to produce
    logits. A fully cached prompt recomputes its last token.
  - Hash a block once it is full and its K/V are written (after the step that fills it), not at
    allocation.
  - With 256-token blocks, prefixes shorter than 256 tokens never hit (see the block size entry).
  - Alternative considered: a radix tree over tokens (mini-sglang, SGLang). Finer-grained sharing,
    but much more code; block hashing fits a paged allocator directly.

### CUDA graphs and `torch.compile` out of scope
- **Context:** Without them, decode for a 0.6B model is dominated by Python and kernel-launch
  overhead (28 layers × a dozen-plus ops per step). vLLM uses both by default.
- **Decision:** Out of scope for the main phases. The benchmark includes vLLM with
  `--enforce-eager` so the effect of graphs and compilation can be separated from scheduling.
  CUDA graphs for decode-only steps are a stretch goal after Phase 4.
- **Why:** Graph capture for variable mixed batches (padding buckets, static buffers) is a large
  piece of work that does not change the scheduling story this project is about.

### Scope trims in Phase 2
- **Decision:** No stop strings, no `seed`, and no server-side metrics field in API responses
  (metrics go to structured logs). Requests that use `stop` or `seed` get a 400 rather than being
  silently ignored. The concurrency acceptance test uses 8 clients.
- **Why:** None of these affect the scheduler, the correctness suite, or the benchmark (which
  measures latency on the client). Rejecting the fields keeps behavior honest.

### GPU Reserve compatibility check
- **Context:** GPU Reserve's `vllm:` provider calls chat completions through LangChain with a
  bearer key and sends `chat_template_kwargs: {"enable_thinking": ...}` in the request body. Its
  agent binds tools, which this server does not support.
- **Decision:** The check is a small script that builds GPU Reserve's chat model through its own
  provider code, points it at this server, and makes a plain chat call without tools, with thinking
  off and on. The server passes `chat_template_kwargs` to the chat template.
- **Why:** It proves drop-in compatibility for the part of the API this server implements, without
  changing GPU Reserve or adding tool calling.

### Benchmark workload mixture
- **Context:** The comparison needs a workload where prefill stalls happen often enough to measure.
- **Decision:** Poisson arrivals; prompt lengths 80% uniform in [128, 512] tokens and 20% uniform
  in [1536, 3072]; output lengths uniform in [64, 512] with `ignore_eos`. Fixed before the first
  comparison run.
- **Why:** With one long prompt in five, a long prefill arrives every second or two at a few
  requests per second, while many short requests are decoding, which is the case mixed batching targets. The longest
  request (3072 + 512 tokens plus the chat template) fits `max_model_len = 4096`.
- **Caveat:** The mixture is designed to expose prefill stalls, not to model real traffic.
  `benchmarks/results.md` states this next to the results.

### Benchmark rates and repeats
- **Decision:** 5 rates (0.2, 0.5, 0.8, 1.0, 1.2 × stallfree-off's saturation throughput), 4
  systems, 3 repeats. Estimated ≈ $11 on L40S including CPU and memory.
- **Why:** Five points still cover light load, the knee, saturation, and overload, at about 70% of
  the cost of seven.

### Explicit CPU allocation and CPU logging in benchmark runs
- **Context:** Online serving latency depends on CPU as well as GPU: HTTP handling, tokenization,
  detokenization, and the Python engine loop. The load generator shares the container. Modal's
  default CPU allocation is small, so unspecified CPU would make results depend on scheduling noise.
- **Decision:** The benchmark container requests `cpu=8.0` and 32 GiB for every system. Server and
  load generator run as separate processes, and per-process CPU utilization is sampled every
  second with `psutil` and stored with each run. A run where the load generator exceeds 80% of
  one core is flagged.
- **Risk noted in the methodology:** stallfree runs its API and engine loop in one Python process,
  while vLLM separates the API server from the engine core. CPU contention inside stallfree's
  process is part of what is measured and will be discussed as a source of any gap.

## Phase 1a — CPU engine

### Block manager interface
- **Context:** The block manager has to serve three callers with different needs: admission of a
  new prompt, growth by one prefill chunk, and growth by one decode token. It also has to be
  testable without tensors.
- **Options considered:**
  1. Block tables stored on the sequence object and mutated by the block manager (nano-vllm).
     Ownership of a table is then split between two classes.
  2. The block manager owns the tables, keyed by an integer sequence id (chosen).
- **Decision:**
  - One growth call, `allocate(seq_id, num_tokens)`, where `num_tokens` is the total the sequence
    must hold afterwards. A new prompt, a prefill chunk, and a decode token are the same operation
    with different totals, and a decode needs a new block exactly when it crosses a block
    boundary.
  - `blocks_needed` and `can_allocate` are pure queries so the scheduler can plan a whole step
    before changing anything.
  - Allocation is all-or-nothing and raises `OutOfBlocksError`; `free` of an unknown or already
    freed sequence raises `KeyError`, because a double free is a scheduler bug.
  - Tables are returned as tuples, so callers get snapshots and cannot corrupt the manager.
  - Reference counts exist from the start (always 1 until prefix caching shares blocks).
  - Allocation is deterministic for a given call history, so failures reproduce.
  - The admission watermark and preemption are scheduler policies; the block manager only grants
    or refuses.
  - No torch import; a test enforces it.
- **Evidence:** `tests/unit/test_block_manager.py`, including a randomized test (20 seeds × 300
  operations) that checks every invariant after each call against a simple model of the expected
  table sizes.

### CPU-only torch for CI
- **Decision:** On Linux, `torch` resolves from the PyTorch CPU wheel index (`[tool.uv.sources]`).
  macOS uses the regular PyPI wheel. The CUDA build and the flash-attn wheel are installed in the
  Modal GPU image instead (Phase 1b).
- **Why:** The default Linux wheel bundles CUDA libraries of several GB that CPU tests never use.

### Verifying the runner before the scheduler exists
- **Context:** The model, KV cache, and runner can be wrong in ways unrelated to scheduling (slot
  mapping, positions, causal masks over paged context). Debugging them together with a new
  scheduler and block manager makes failures hard to attribute.
- **Decision:** A scripted scheduler double under `tests/` replays fixed, hand-written step plans
  with fixed block tables. It makes no decisions. The runner is first made token-exact against HF
  with it; then the same end-to-end tests run against the real block manager and scheduler, and
  Phase 1a is done only when they pass there.
