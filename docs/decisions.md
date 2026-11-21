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
| 2026-11-16 | L4 | < 1 (plus ~12 queued for capacity, not billed) | 0.00 (no billed line) | Driver check (driver 620.70.06) |
| 2026-11-16 | CPU only | ~3 (image builds + weight download, 8 s for 1.5 GB) | 0.006 | Weights into the volume; image builds have no billed line |
| 2026-11-16 | A10 (no L4 free) | ~4 (≈ 160 GPU-seconds billed) | 0.063 | Smoke run, flash vs reference, bf16 noise calibration |
| 2026-11-17 | A10, then A10G, then A10 | ~10 (≈ 545 GPU-seconds billed) | 0.215 | Margin suite (A10), rerun after the preemption fix (landed on A10G), calibration + margin suite in one container (A10) |
| | | | **0.284** | **Phase 1b total** (budget 4.00); billed figures from `modal billing report -r h`, which lists the A10 class as "A10G" |
| 2026-11-17 | A10 | ~1 (≈ 23 GPU-seconds billed) | 0.010 | Phase 2: eight concurrent streaming clients on the real server |
| 2026-11-17 | A10 | ~2 (≈ 88 GPU-seconds billed, incl. cold start) | 0.035 | Phase 2: `modal serve` endpoint for the GPU Reserve check |
| | | | **0.045** | **Phase 2 total** (estimate 0.25); running total 0.329 |
| 2026-11-17 | A10G (A10 requested) | ~6 (366 s function wall time) | 0.139 | Phase 3: calibration + margin suite with mixed batching on, budget 64 |
| 2026-11-18 | A10 | ~5 (265 s margin + 40 s micro-benchmark) | 0.121 | Phase 3: calibration + margin suite with every prompt chunked (budget 16), then the stall micro-benchmark |
| | | | **0.260** | **Phase 3 total** (estimate 0.35); running total 0.589 |
| 2026-11-20 | L40S | ~5 (268 s function time) | 0.18 | Phase 4: smoke, calibration, margin suite flag off and with every prompt chunked, in one container |
| 2026-11-20 | L40S (8 CPU, 32 GiB) | ~31 billed GPU minutes | 1.31 | Phase 4: saturation closed loop, four-system dry run, and nine short vLLM start-up failures (flag spelling 3, slim-image FlashInfer JIT 5 while collecting its error, package shadowing 1) before the official image worked |
| | | | **1.49** | **Phase 4 stage 1 total** (cap 3.00); running total 2.08 |

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

### Scheduler interface and the prefill-priority baseline
- **Context:** The scheduler has to express full prefills now, chunked prefills and mixed steps
  later, decodes, and recompute after preemption, without the runner knowing which is which.
- **Decision:**
  - Per sequence, only two counters: `num_tokens` (prompt + generated) and
    `num_computed_tokens` (positions whose KV is cached). A step entry computes positions
    `num_computed_tokens .. +num_new_tokens-1`. A decode is an entry with one new token; a
    recompute after preemption is a prefill of prompt + generated tokens from position 0. This is
    the same model vLLM V1 uses, and it makes chunked and mixed steps a scheduling change only.
  - Each entry carries its token ids, block table, start position, sampling parameters, and a
    `sample` flag (false only for a non-final prefill chunk). The runner needs nothing else, so a
    scripted plan can drive it in tests.
  - `schedule()` allocates and preempts; `update()` advances `num_computed_tokens`, appends
    sampled tokens, checks EOS before `max_tokens`, and frees finished sequences.
  - Prefill-priority with mixed batching off: whenever the head of the waiting queue can be
    admitted, the step is prefill-only; admission is strict FIFO and stops at the first request
    that does not fit (`max_num_seqs`, token budget, or blocks minus the watermark). A prompt
    longer than the budget runs alone. Otherwise all running sequences decode.
  - Preemption victim: the most recently admitted running sequence not already in the step; if
    none is left, the sequence that needs the block preempts itself. Preempted sequences return to
    the front of the waiting queue in admission order, keeping their generated tokens.
  - The watermark applies to admission only. Decodes may use the reserve.
  - `add()` rejects a request that could not finish when running alone (prompt + `max_tokens`
    over `max_model_len`, or more blocks than the pool minus the watermark).
- **Why the `add()` check matters:** with newest-first preemption, the oldest running sequence is
  never a victim while others exist, and a request that fits alone always completes once it is the
  only one left. Without the check, a request too large for the pool would be admitted, preempt
  everyone, then preempt itself forever.
- **Evidence:** `tests/unit/test_scheduler.py`. A test runner keeps its own per-sequence record of
  computed positions and checks each entry continues exactly where that record ends; generated
  tokens depend on the whole history, so any wrong recompute changes the output and fails the
  comparison with a reference produced without a scheduler. 15 random workloads with small block
  pools (including preemptions) must all match the reference.

### Continuous integration
- **Decision:** One GitHub Actions job on every push and pull request (`.github/workflows/ci.yml`):
  `uv sync --frozen`, `ruff check`, `ruff format --check`, `mypy src`, `pytest -m "not gpu"` on
  `ubuntu-24.04` with Python 3.12.
- **Pins:** uv 0.11.6, the version that wrote `uv.lock`, so the lockfile is read exactly as written.
  Third-party actions are pinned by commit SHA with the release tag in a comment, so a moved tag
  cannot change what runs. `--frozen` fails the job if `uv.lock` is out of date instead of
  silently re-resolving.
- **Why CPU-only:** the lockfile resolves `torch==2.10.0+cpu` from the PyTorch CPU index on Linux and
  contains no CUDA packages, so the job installs no GPU libraries and downloads no model weights.
  GPU tests run on Modal by hand.
- **Not added yet:** a tokenizer cache. When a test needs the Qwen3 tokenizer, it will be cached by
  its pinned revision.

### Qwen3 forward over a flat token batch
- **Context:** One step can mix prefills of different lengths and single-token decodes. The model
  must not care which is which.
- **Decision:**
  - The model takes `input_ids [T]` and `positions [T]` for all tokens of the step concatenated,
    and returns hidden states `[T, hidden]`. Everything except attention is per token, so
    sequences only matter inside attention.
  - The attention backend is passed into `forward` explicitly (an `AttentionBackend` callable
    taking the layer index and that layer's Q, K, V). nano-vllm uses a module-level global context
    instead; passing it makes the dependency visible and lets tests substitute a simple dense
    attention.
  - Logits are computed separately (`compute_logits`) and only for rows that will be sampled,
    because the vocabulary projection (151,936 columns for Qwen3-0.6B) is wasted on prefill
    positions whose output is never used.
  - The arithmetic follows Hugging Face transformers' Qwen3 module operation by operation:
    RMSNorm statistics in fp32, per-head q/k RMSNorm before RoPE, RoPE angles computed on the fly
    from `positions * inv_freq`, SwiGLU MLP, tied embeddings. Parameter names match the
    checkpoint, so weights load by name with no mapping table.
  - RoPE inverse frequencies are kept as a plain fp32 tensor rather than a module buffer, so
    casting the model to bf16 does not round them.
  - The loader fails on any missing, unexpected, or mis-shaped tensor instead of skipping it.
- **Correctness contract:** against HF (`attn_implementation="eager"`, fp32) on a tiny random
  Qwen3 (3 layers, hidden 64, 4 query heads, 2 KV heads, head dim 32, so heads × head_dim ≠ hidden
  as in the real model), the predicted token at every position must be identical and logits must
  pass `torch.testing.assert_close` at its default fp32 tolerances. Norm weights are randomised
  in the test model so a norm that fails to load cannot hide behind its default value of 1.
- **Observed:** a single sequence gives logits bitwise identical to HF on the development machine;
  flattening several sequences into one batch changes the last bits (max difference 1.5e-6),
  because matrix multiplications of different shapes accumulate in different orders. The
  predicted tokens do not change. Bitwise equality is not part of the contract, since it depends
  on the CPU's math library.
- **Evidence:** `tests/unit/test_qwen3_model.py`. Seven deliberate bugs (RoPE sign, missing q
  norm, RoPE skipped on keys, missing residual, RMSNorm epsilon outside the square root, missing
  final norm, wrong RoPE frequencies) each fail it. The epsilon bug leaves greedy tokens
  unchanged and is caught only by the logit comparison, which is why both checks are kept.

### KV cache layout, slot mapping, and the reference paged attention
- **Layout:** one tensor `[num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]`
  (index 0 = K, 1 = V). Per layer, the `[num_blocks, block_size, Hkv, D]` part is exactly the
  paged layout FlashAttention-2 takes with a `block_table`, so both backends read the same memory.
  A *slot* is `block_id * block_size + offset`; writes go through a flat
  `[num_blocks * block_size, Hkv, D]` view with `index_copy_`, one call for K and one for V per
  layer, no custom kernel.
- **Per-step metadata, built once and shared by all layers:** `slot_mapping` (where each new
  token's K/V goes), `cu_seqlens_q` (which query rows belong to which segment), `seq_lens`
  (context length per segment = start position + new tokens), padded `block_tables`, and the
  longest query and context lengths. These are the same arguments FlashAttention's varlen call
  takes, so the GPU backend can reuse them unchanged.
- **Reference backend (`torch_ref`):** writes the new K/V first, then, per segment, gathers the
  whole context from the cache through the block table and runs causal attention with the mask
  aligned to the end of the context (query `j` of a segment that starts at position `s` sees keys
  `0 .. s + j`). It never reads keys or values from its arguments. A Python loop over segments
  makes it slow; it exists for exact CPU tests and as a cross-check of the GPU backend.
- **Evidence:** `tests/unit/test_paged_attention.py`:
  - attention alone against dense attention on the full sequence: prefills at block boundaries
    (block size 4 with lengths 3/4/5/7/8/9, block size 256 with 255/256/257), a prefill split into
    chunks that start mid-block, token-by-token decode, one call mixing a chunk continuation, a
    decode, and a fresh prefill, and two sequences on interleaved, non-contiguous blocks where
    every block the checked sequence does not own is overwritten with 1e4 before its last step;
  - the tiny Qwen3 with paged attention against Hugging Face in fp32: prefill then decode (prompts
    of 3, 4, 5, 9 tokens on scattered blocks), a 23-token prompt prefilled in three chunks, and
    two sequences decoded together on interleaved blocks. Generated tokens equal HF `generate()`
    and every row's logits match HF's within the default fp32 tolerances.
  - Twelve deliberate bugs (slot offset, slot block index, K/V halves swapped, causal mask not
    shifted by the start position, mask off by one, context missing the newest token, wrong GQA
    head mapping, missing 1/sqrt(d), reading new K from the argument, wrong `seq_lens`,
    non-cumulative `cu_seqlens`, accepting a too-short block table) each fail at least one test.
    The unshifted mask and the argument-read bug only show up in chunk and decode cases, where a
    segment starts after position 0.
- **Near-ties:** the smallest top-1 minus top-2 logit gap of the reference over every compared
  position is 1.2e-3 (printed by the test run), about a hundred times larger than fp32 rounding
  noise. If a later scenario flips a token, the gap at that position decides: near 1e-5 means a
  near-tie on the random model and the strict suite moves to float64; a large gap means a bug.

### Model runner
- **Decision:** The runner takes a `SchedulerOutput` and nothing else. For each entry it appends
  the entry's tokens to one flat batch, gives each token its position in its own sequence, adds a
  segment (start position, token count, block table) to the attention metadata, and, if the entry
  samples, remembers the batch row of the entry's last token. After the forward pass it computes
  logits for those rows only and returns them with their sequence ids. Sampling is a separate
  step.
- **Why it knows so little:** whether an entry is a prefill, a chunk, a decode, or a recompute
  after preemption is invisible to it; all four are "compute these positions for this sequence".
  So the scheduling policy can change (mixed batching in Phase 3) without touching the runner.
- **Non-sampled entries still run:** a non-final prefill chunk produces no logits but its K/V are
  written, which the next chunk needs.
- **Attention backend as a factory:** the runner builds a backend per step from the cache and
  that step's metadata. The PyTorch reference is the default; the FlashAttention backend plugs in
  the same way.
- **Tested before the real scheduler existed in the engine:** a scripted scheduler under `tests/`
  replays hand-written plans (start, count, block table, sample flag) with HF's greedy
  continuation as the tokens, so each sampled row must predict HF's next token and match HF's
  logits at that position. Plans cover prefill then decode, a prompt in three chunks where the
  first two do not sample, one step mixing a chunk continuation, a decode, and a fresh prefill,
  and a recompute: after some decodes the same sequence is prefilled again from position 0 into
  different blocks while its old blocks are overwritten with garbage.

### Sampler
- **Decision:** One sampler per engine with its own seeded `torch.Generator`. Per row:
  temperature 0 is greedy (`argmax`, ties to the lowest id, as Hugging Face); otherwise logits are
  divided by the temperature and turned into probabilities in fp32, nucleus (top-p) filtering
  keeps every token whose more-likely tokens hold less than `top_p` of the mass (so the most
  likely token is always kept and the kept set is the smallest reaching `top_p`), and one token is
  drawn with `torch.multinomial` from the filtered weights.
- **Why fp32:** with bf16 logits the softmax of a 151,936-way distribution loses most small
  probabilities to rounding; computing it in fp32 costs little because it runs on the sampled rows
  only.
- **Why a per-engine generator:** a fixed seed makes a whole run reproducible for debugging, and
  greedy rows consume no random numbers, so adding a greedy request does not change the draws of
  sampled ones. Per-request seeds are deferred along with the `seed` API parameter.
- **Not implemented:** top-k, min-p, repetition and frequency penalties, logprobs.
- **Evidence:** `tests/unit/test_sampler.py` checks greedy and ties, that greedy draws no random
  numbers, frequencies against the expected softmax at temperature 1 and 0.5 (20,000 draws,
  tolerance 0.015, about four standard deviations), the top-p kept set at boundary values and in
  shuffled token order, that sampling never leaves the nucleus, reproducibility by seed, and mixed
  greedy and random rows. Nine deliberate bugs each fail at least one test.

### Synchronous engine
- **Decision:** `LLMEngine` owns the model, KV cache, block manager, scheduler, runner, and
  sampler, built from one frozen `EngineConfig` validated at construction (block count, block
  size, dtype, backend; the flash backend requires a block size divisible by 256 and is not
  available yet). `step()` = schedule, run, sample, update; it returns one `StepOutput` (sequence,
  token, finished, reason) per sequence that produced a token. `generate()` is an offline
  convenience that runs prompts to completion. Requests are token ids in and out; tokenizers and
  chat templates belong to the API layer.
- **KV cache size is explicit for now** (`num_blocks`). Sizing it from free GPU memory after a
  profiling forward pass comes with the GPU backend, where it can be measured.
- **Block size:** the engine default is 256 (FlashAttention-2's requirement). The reference
  backend accepts any size, and the CPU tests use 4 so short prompts cross many block
  boundaries; the same code paths run either way.
- **Counters:** steps and preemptions are counted on the engine so tests (and later metrics) can
  assert that a scenario really preempted.
- **Evidence:** `tests/unit/test_engine.py`. A first mutation pass found that swapping which
  sequence a sampled token or a sampling setting belongs to went unnoticed, because no engine test
  ran several requests in one step; two tests were added (a batch of greedy requests against HF,
  and a greedy request decoded in the same steps as a temperature-5 request), after which all six
  deliberate bugs fail at least one test.
- **No-progress guard (added after the end-to-end mutation run):** if the scheduler returns an
  empty plan while requests are still waiting or running, `step()` raises instead of returning
  nothing. With a correct scheduler and block manager this cannot happen (admission control
  guarantees the oldest request fits once nothing else holds blocks); without the guard, a block
  leak made `generate()` loop forever, which a test run only reported as a timeout.

### End-to-end correctness with the real scheduler and block manager
- **Scenarios** (`tests/integration/test_engine_vs_hf.py`, tiny Qwen3, fp32, PyTorch paged
  attention, block size 4), each compared token for token with Hugging Face greedy `generate()`:
  a single request; eight requests of 1 to 20 tokens at once; requests arriving at steps 0, 3, 7,
  and 12 while others decode; forced preemption (10 blocks, four requests that may each reach
  17 tokens, with watermark 0 and 0.2; at least one preemption asserted, two observed in each);
  prompts of 3, 4, 5, 7, 8, 9 tokens (block size minus one, equal, plus one, around one and two
  blocks), alone and together; a token budget of 8 with at most 2 running, so a 20-token prompt
  runs alone; and five random workloads with 8 to 14 blocks (11 preemptions observed across them).
  Every run must end with all blocks free.
- **Result:** all token-exact; no token flipped, so the strict suite stays in fp32.
- **Smallest reference gap** (top-1 minus top-2 logit over every compared position, all tests):
  1.9e-4, in the token-budget scenario; about 18 times the ~1e-5 near-tie threshold.
- **The end-to-end tests catch component bugs on their own:** replaying earlier mutants against
  this file only, a preemption that keeps `num_computed_tokens`, an oldest-first victim, a decode
  that reserves one slot too few, scheduler entries missing the last token, block counts rounded
  down, per-entry positions starting at 0, and a block leak all fail it. The leak first showed up
  as a hang, which led to the engine's no-progress guard.

## Phase 1b — GPU path

### Driver and development GPU
- **Driver check (2026-11-16, before any image build):** a Modal L4 reported NVIDIA L4, driver
  620.70.06, 23,034 MiB. CUDA 13 wheels need driver 580 or newer, so the planned torch 2.10.0
  (CUDA 13) + flash-attn 2.8.3 stack stays; the CUDA 12.8 fallback was not needed.
- **Capacity:** the L4 request waited about 12 minutes for capacity. Development functions now
  request `["L4", "A10"]`: L4 preferred, A10 (Ampere, sm_86, 24 GB, $1.10/h) accepted when no
  L4 is free. Every result records which GPU ran. Calibration of the bf16 tolerance and the
  bf16 correctness suite run on one GPU type end to end; if they would land on different types,
  calibration is repeated on the second. If A10 is used, it is development only: the benchmark
  target stays L40S (Ada, sm_89).

### FlashAttention backend
- **Decision:** `attention/flash.py` writes the step's new K/V with the same `index_copy_` as the
  reference backend, then calls `flash_attn_varlen_func` once per layer with the cache's
  `[num_blocks, block_size, Hkv, D]` slices, `cu_seqlens_q`, `cu_seqlens_k` (cumulative context
  lengths, newly added to the metadata), the longest query and context lengths, `causal=True`,
  and the padded block tables. Every segment (prefill, chunk, decode) is one variable-length
  sequence in a single call; FlashAttention aligns the causal mask to the end of each context,
  which is what a segment starting at position `s > 0` needs.
- **Guards:** the backend refuses a non-CUDA cache and a block size that is not a multiple of 256;
  the engine refuses the flash backend on a non-CUDA device at construction.
- **`flash_attn` stays optional:** it is imported when the backend is built, never at module
  import. A CPU test blocks the module with an import hook and checks that `import stallfree` and
  a reference-backend engine run end to end without it. mypy ignores the missing module only for
  `flash_attn`.

### KV cache sizing from GPU memory
- **Decision:** with `num_blocks=None` (CUDA only), the engine loads the weights, runs one forward
  of the largest possible step (`max(max_num_batched_tokens, max_model_len)` tokens as one prefill,
  plus logits for `max_num_seqs` rows) on a throwaway cache, and reads the allocator's peak above
  the post-load baseline, minus the throwaway cache itself. The cache then gets
  `gpu_memory_utilization * total - (total - free) - activation_peak`, in whole blocks of
  `bytes_per_token * block_size` (28 MiB for Qwen3-0.6B in bf16). `total - free` is measured from
  the device, so it counts weights, the CUDA context, and anything else on the GPU.
- **Why profile instead of estimating:** activation memory depends on kernels and on the largest
  step, and the CUDA context is not visible to PyTorch's allocator. Measuring both avoids running
  out of memory on the first long prompt.
- **Evidence (CPU):** `tests/unit/test_memory_and_flash_cpu.py` checks the arithmetic (worked
  24 GiB example, rounding down, failure when no block fits) and the 28 MiB block; the measured
  values on L4 are recorded below once a GPU session has run.
- **First GPU runs (A10, driver 610.57.04, torch 2.10.0+cu130, flash-attn 2.8.3):** the smoke run
  generated correct answers with the flash backend and sized the cache automatically: 22.06 GiB
  total, 2.00 GiB in use after loading, 0.11 GiB activation peak for a 4,096-token step,
  649 blocks = 17.75 GiB of KV cache at `gpu_memory_utilization=0.9`.
- **Flash vs reference (`tests/gpu/test_flash_vs_ref.py`, 14 cases on A10):** criterion fixed
  before the run: against an fp32 reference on the same paged cache, flash's largest error must be
  at most twice the reference backend's own error in the same dtype plus one rounding unit at 1.0.
  Cases: prefill lengths 1/255/256/257/513, chunks crossing 256 and 512 then decodes, and one step
  mixing a chunk, a decode, and a fresh prefill on interleaved blocks, each in bf16 and fp16. All
  pass; flash's error never exceeded the reference's (bf16 worst 1.4e-2 vs 2.9e-2, fp16 2.1e-3 vs
  2.5e-3).

### bf16 margin threshold τ = 0.5
- **Calibration** (`docs/data/tau_calibration.json`, A10, 32 chat prompts, greedy, up to 64 new
  tokens, 1,702 generated positions). Reference R: Hugging Face bf16, eager attention, one prompt
  at a time. Three noise sources, each changing one thing:
  - B, bf16 eager in batches of 8 (left padding): 15/32 prompts diverge from R; R's top-1 minus
    top-2 gap at the first differing token: 0.0 ×2, 0.125 ×9, 0.25 ×4 (largest 0.25);
  - S, bf16 SDPA attention instead of eager: 18/32; 0.0 ×5, 0.0625 ×1, 0.125 ×7, 0.25 ×4, 0.5 ×1
    (largest 0.5);
  - F, fp32 eager: 12/32; 0.0 ×3, 0.125 ×5, 0.25 ×3, 0.5 ×1 (largest 0.5).
  Over all positions R's gap has minimum 0 (exact ties at 24 positions), 5th percentile 0.25,
  median 2.6; 11.5% of positions are below 0.5.
- **Observation: the gaps are quantized.** Every divergence gap is 0, 0.0625, 0.125, 0.25, or 0.5:
  the logits come out of the vocabulary projection in bf16, whose spacing at logit magnitudes of
  about 16 to 32 is 0.125 to 0.25. A gap is a whole number of those steps, and 0 is an exact tie.
- **Decision: τ = 0.5** (`tests/gpu/margin.py`, `BF16_MARGIN_TAU`), the largest gap at which HF
  disagreed with itself across all three sources, about two bf16 steps.
- **Why not 0.25 (batch shape alone):** this engine differs from HF in more than batch shape. It
  uses a different attention kernel (FlashAttention; S models a kernel change) and different
  matrix-multiplication shapes (a flat batch; F shows how far exact arithmetic moves the argmax).
  Both reach 0.5. Batch shape alone underestimates the noise a correct implementation can show,
  and a τ of 0.25 would fail correct code on some prompts.
- **The rule stays strict where the model is confident:** a divergence passes only if the
  reference itself was within τ at that position; the teacher-forced check (the reference's whole
  sequence fed through the engine in one prefill) must give the same top-1 token at every position
  where the reference's gap exceeds τ.

### bf16 correctness on the real model (margin rule, teacher forcing, preemption)
- **Same GPU for calibration and suite.** Modal's `A10` request lands on either an "NVIDIA A10" or
  an "NVIDIA A10G" host. A margin run that landed on an A10G
  (`docs/data/margin_suite_run2_a10g.json`) passed every check but could not be compared with a
  calibration done on an A10, and showed a gap value (0.375) the A10 calibration never produced.
  The final numbers therefore come from one function that runs calibration and then the margin
  suite in the same container (`modal_apps/dev.py::calibrate_and_margin`), so both share one GPU.
  That run was on an NVIDIA A10 and reproduced the first A10 calibration exactly
  (`docs/data/tau_calibration_run1_a10.json`), so τ = 0.5 stands.
- **Results** (`docs/data/margin_suite.json`, NVIDIA A10, Qwen3-0.6B bf16, flash backend, cache
  sized automatically, 32 chat prompts, greedy, up to 64 tokens):

  | | diverged (of 32) | gaps at divergence | failed the rule |
  |---|---|---|---|
  | engine, one prompt at a time | 15 | 0.0 ×3, 0.125 ×8, 0.25 ×3, 0.5 ×1 | 0 |
  | engine, all 32 batched | 14 | 0.0 ×4, 0.125 ×6, 0.25 ×3, 0.5 ×1 | 0 |
  | HF bf16 eager, batch 8 (same run) | 15 | largest 0.25 | n/a |
  | HF bf16 SDPA (same run) | 18 | largest 0.5 | n/a |
  | HF fp32 (same run) | 12 | largest 0.5 | n/a |

  The engine diverges no more often than HF does against itself with a different kernel, and the
  divergences do not cluster at τ: the single 0.5 case (prompt 27, position 19) is a position
  where HF's SDPA and fp32 runs also diverge.
- **Teacher-forced check:** the reference's 32 sequences (1,702 positions) through the engine's
  model in one prefill each: at all 1,451 positions where the reference's gap exceeds τ, the top-1
  token matches, with both the flash and the PyTorch backend. Their logits differ by at most 0.91
  (bf16 logits of magnitude up to about 30 have a spacing of 0.125 to 0.25).
- **Forced preemption on GPU:** four requests of 202 to 215 tokens (one 256-token block each)
  generating 100 tokens (into a second block) on a 6-block cache: one preemption, all four pass
  the rule (two exact, divergences at gaps 0.125 and 0.0). A first version used prompts of about
  100 tokens, which never left their first block, so nothing was preempted and the test failed
  its own assertion; the prompts were lengthened (measured with the tokenizer) before the result
  above.
- **KV sizing in that run:** 509 blocks (13.92 GiB), lower than the smoke run's 649 because the
  Hugging Face reference model was still resident in the same process (5.81 GiB in use).

## Phase 2 — Online serving

### API protocol, chat templates, and incremental text
- **Request handling:** `/v1/chat/completions` accepts the OpenAI fields this server can honour
  (`model`, `messages`, `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `stream`,
  `stream_options.include_usage`) plus vLLM's `chat_template_kwargs` and `ignore_eos`. Fields that
  would change the response but are not implemented (`n > 1`, `stop`, `seed`, `tools`,
  `tool_choice` other than "none", `logprobs`, `top_logprobs`, `response_format`, non-zero
  penalties, `logit_bias`) are rejected with an OpenAI-shaped 400 naming them, so a client never
  silently gets something it did not ask for. Other unknown fields are ignored, as OpenAI-
  compatible servers commonly do.
- **Tokenizer boundary:** the engine stays token ids in and out. A small `ChatTokenizer` interface
  (render a chat with the model's template and tokenize it; decode without special tokens) has a
  Hugging Face implementation for serving and a byte-level stub for CPU tests whose 256 ids match
  the tiny model's vocabulary, so API tests need no tokenizer download. `chat_template_kwargs` go
  straight to the template; with Qwen3, `enable_thinking=False` pre-fills an empty think block.
- **Incremental decoding:** each new token decodes a short window ending at it and emits only the
  text past what was already sent; a window ending in an incomplete UTF-8 character emits nothing
  until a later token completes it. Concatenated deltas equal the full decode (tested with random
  multi-byte text and with Qwen3's tokenizer), and no partial character is ever sent.
- **Admission check before streaming:** `Scheduler.validate()` holds the "could this request ever
  finish alone" check from `add()`. It reads only constants, so the API can reject an impossible
  request with a 400 before the stream starts, without touching scheduler state while a step runs
  in the worker thread. The scheduler's contract tests pass unchanged.
- **CI:** the Qwen3 tokenizer files (no weights) are cached by the pinned revision for the
  chat-template tests.

### Async engine
- **Ownership:** one background asyncio task (the loop) owns the synchronous engine; nothing else
  calls it. Each `engine.step()` runs in a worker thread via `asyncio.to_thread`, so while the GPU
  works the event loop keeps accepting connections and streaming tokens. The engine is never
  called from two threads at once: new requests and aborts are appended to an inbox and applied by
  the loop between steps.
- **Per-request queues:** each request gets an `asyncio.Queue` of step outputs that its HTTP
  handler reads. After a step, the loop puts each sequence's token in its queue and, if the request
  finished, records it as done.
- **Cancellation:** `generate()` is an async generator; if its consumer stops early (client
  disconnect, task cancellation, `break`), its `finally` posts an abort, which the loop applies at
  the next step boundary: the scheduler frees the blocks and the sequence is never scheduled
  again. A step already running when the consumer stopped still completes, so an aborted request
  may produce a token or two nobody reads.
- **Backpressure:** at most `max_queued_requests` requests (waiting or running) at once; beyond
  that `generate()` raises `QueueFullError` immediately and the API returns 503, instead of
  queueing without bound. Requests that could never fit are rejected in `generate()` itself
  (`Scheduler.validate()`), before the loop sees them.
- **Failure:** if a step raises, the loop records the error, delivers `EngineDeadError` to every
  waiting request, and refuses new ones, rather than leaving clients hanging.
- **Metrics** (one JSON log line per request on `stallfree.metrics`): `queue_s` from arrival to the
  start of the step that produced the first token, `ttft_s` to that token, `tpot_s` as the mean
  gap between later tokens, `e2e_s`, token counts, and finish reason. With mixed batching off, the
  first token comes from the request's prefill step, so `queue_s` is the time before its prefill
  started; once prefills are chunked (Phase 3) it becomes the time before its last chunk started,
  which will need its own field.
- **Evidence:** `tests/unit/test_async_engine.py` (each event loop wrapped in a 30 s deadline, so a
  broken engine fails rather than hangs). A first mutation pass left three bugs undetected: an
  abort that never reached the engine (the request quietly ran to its limit and then freed its
  blocks), skipping the up-front validation (the loop's safety net still raised, only later), and
  queue time recorded as time to first token. Tests now check that generation stops within two
  steps of an abort, that impossible requests fail without the loop running, and that with a
  50 ms step TTFT exceeds queue time by at least 40 ms. All nine mutants fail.

### HTTP server
- **Decision:** FastAPI with `/health` (open), `/v1/models`, and `/v1/chat/completions` (bearer key
  checked in constant time when configured; 401 otherwise). The handler validates the request,
  renders and tokenizes the chat, checks admission (`Scheduler.validate()`) and capacity, and only
  then either collects the whole completion (JSON) or returns an SSE stream. Every error uses
  OpenAI's `{"error": {message, type, param, code}}` shape: 400 for invalid or unsupported input,
  404 for another model name, 401, 503 when at capacity, 500 if the engine loop died.
- **Streaming:** the first event carries the assistant role, then one event per text delta from the
  incremental detokenizer, then an empty delta with the finish reason, then (if
  `stream_options.include_usage`) a usage event with empty `choices`, then `data: [DONE]`.
- **Why the capacity check happens before the stream starts:** once a 200 SSE response has begun,
  an error can only be sent as an event inside it, which clients handle inconsistently. Checking
  first gives a plain 503. A mutation test removing that check went unnoticed until a test sent a
  streaming request to a full server.
- **Disconnects:** when a streaming client disconnects, Starlette cancels the response generator;
  the cancellation lands inside the engine's `generate()`, whose `finally` posts the abort. The
  response generator also closes the engine stream explicitly in its own `finally`. Removing that
  close did not fail any test, because cancellation and asyncio's finalizer for abandoned async
  generators reach the same abort; it is kept so the abort does not depend on garbage collection.
  A non-streaming request whose client disconnects runs to completion (its blocks are freed when
  it finishes).
- **Defaults:** temperature 1.0 and top_p 1.0 when omitted (OpenAI's defaults); `max_tokens`
  defaults to the room left in `max_model_len`.
- **Evidence:** `tests/integration/test_api.py` runs a real uvicorn server in a thread with the
  tiny model and a byte-level tokenizer and drives it with the `openai` client: greedy output
  equals Hugging Face's for the same prompt ids, streamed text equals non-streamed text, usage is
  exact, 8 concurrent async streams each match their own reference and never mix chunk ids,
  `chat_template_kwargs` change the prompt, unsupported parameters get a 400 naming them, a
  disconnect mid-stream frees all blocks and stops generation, a full server answers 503 (JSON and
  streaming), and `ignore_eos` overrides an end-of-sequence token the model actually produces.

### Serving on Modal and the GPU checks
- **Endpoint:** `modal_apps/serve.py` serves the FastAPI app as a Modal ASGI app (flash backend,
  bf16, cache sized automatically, one container, up to 256 concurrent requests per container,
  scaled to zero after 60 idle seconds). The bearer key is read from the local environment when
  the app is created and attached as an ephemeral secret, so no named secret is stored in the
  workspace. For the checks below it ran under `modal serve` (a temporary URL that disappears
  when the command ends) rather than `modal deploy`. Stopping the local `modal serve` process left
  the app "ephemeral" until `modal app stop -y` was run; the session checklist now ends with that
  and `modal app list`.
- **GPU concurrency run** (`docs/data/concurrency_gpu.json`, NVIDIA A10): the real server with the
  Qwen3 tokenizer, eight `openai` async clients streaming at once (greedy, up to 64 tokens,
  thinking off). Eight distinct response ids, one per stream, every stream with a finish reason
  and correct usage. Content against Hugging Face bf16 greedy under the margin rule: five
  identical, three diverging at gaps 0.25, 0.25, and 0.125 (the same prompts and positions as in
  the single-request margin suite), all passing.
- **GPU Reserve compatibility** (`docs/data/gpu_reserve_check.json`): `scripts/gpu_reserve_check.py`,
  run with GPU Reserve's own Python, builds its chat model through its unmodified provider code
  (`ASSISTANT_MODEL=vllm:Qwen/Qwen3-0.6B`, a LangChain `ChatOpenAI` on chat completions with
  `stream_usage`) with only the base URL and key pointing here, from an empty directory so its
  `.env` is never read. With thinking off, `invoke` and `stream` both return "The capital of France
  is Paris." with no reasoning; the prompt is 4 tokens longer than with thinking on (the empty think
  block the template adds), so `chat_template_kwargs` reached the template. With thinking on, the
  reply contains a `<think>...</think>` block before the answer. GPU Reserve sends no temperature, so
  the OpenAI default of 1.0 applies and its `invoke` and `stream` texts differ. Tool calling stays
  out of scope.

## Phase 3 — Stall-free mixed batching

### Mixed-batching scheduler
- **Policy** (`enable_mixed_batching=True`; full statement in `scheduler.py`): each step has a
  strict token budget and three passes. (1) Every running sequence with one uncomputed token
  (a decode, or the last one-token piece of a chunked prefill) gets it, oldest first, with the
  same newest-first preemption as before. (2) Prefills already in progress continue, oldest first,
  with `min(uncomputed, budget left)` tokens. (3) Waiting sequences are admitted in strict FIFO
  order with a first chunk sized to the budget left and to the free blocks minus the watermark.
  This is Sarathi-Serve's stall-free batching: decodes are never held back by a prefill, and a
  long prompt is spread over several steps that each also carry every decode.
- **Shortage during a continuation:** take blocks from the newest running sequence not yet in the
  step; if there is none, shrink the chunk to what fits in the sequence's own last block and the
  free blocks; only if nothing fits does it preempt itself. Shrinking avoids a recompute: in the
  tested scenario it completes with no preemption where giving up would have preempted once.
- **No admissions in a step that preempted:** a preempted sequence goes to the front of the waiting
  queue, and pass 3 would otherwise readmit it into the blocks it just released.
- **Invariant that makes one guard redundant:** a sequence in the middle of its prefill is always
  newer than every sequence that is decoding. An older prefill takes all the budget left in
  pass 2 unless it finishes, so a newer request cannot complete its prefill first; the exception,
  a chunk shrunk for lack of blocks, only happens when there is no newer unscheduled sequence.
  So when pass 2 looks for a victim, no already-scheduled sequence can come after the current one.
  The "not already scheduled" check in the victim search is kept as a defensive guard; the mutant
  that removes it is the one survivor of the mixed-mode mutation run, for this reason.
- **Off mode is unchanged:** every off-mode scheduler test passes as before; the only edit to that
  file replaced the test asserting mixed mode was unimplemented with one that it constructs and
  schedules.
- **Evidence:** `tests/unit/test_scheduler_mixed.py` (26 tests): step-by-step plans for the policy,
  the SPEC test (three decodes keep running every step while a 40-token prompt is fed in 5-token
  chunks under a budget of 8), FIFO with a partial chunk filling leftover budget, a budget smaller
  than every prompt, `max_num_seqs` counting partial prefills, the watermark shrinking an admitted
  chunk, chunk shrinking instead of preempting, and 15 random workloads with small caches checked
  against a scheduler-free reference, which include preemptions in the middle of a prefill. Twelve
  deliberate bugs: eleven fail at least one test; the remaining one is the redundant guard above.

### Queue time with chunked prefill
- **Definition:** `queue_s` runs from arrival to the start of the first step that scheduled any
  token of the request (its first chunk, or its whole prefill with mixed batching off). TTFT still
  ends at the first generated token, which with chunked prefill comes from the last chunk, so
  `ttft_s - queue_s` is the time spent prefilling. With mixed batching off the two definitions
  coincide, since the first scheduled step is the prefill that produces the first token.
- **Mechanism:** the engine records which sequences each step scheduled
  (`last_scheduled_seq_ids`); the async loop reads it between steps.
- **Evidence:** with an 8-token budget and 20 ms steps, a 40-token prompt is scheduled in the
  first step after arrival (`queue_s` < 20 ms) and its first token comes about four steps later.

### End-to-end correctness with mixed batching on
- The engine unit tests and the end-to-end file run every scenario with mixed batching off and on
  (an autouse fixture with ids `off`/`on`; assertions unchanged): 39 existing tests per mode, all
  token-exact against Hugging Face in fp32, including forced preemption with an asserted
  preemption and lengths at block boundaries ±1.
- New end-to-end scenarios: every prompt chunked (budget 4, prompts of 5 to 20 tokens) and
  preemption in the middle of a chunked prefill (asserted to happen with mixed batching on), both
  token-exact. The smallest reference top-1/top-2 gap over the whole run stays at 1.9e-4.

### bf16 correctness on GPU with mixed batching on
- **Run** (`docs/data/margin_suite_mixed64.json`, calibration in the same container in
  `docs/data/tau_calibration_mixed64.json`): the margin suite with mixed batching on and a budget
  of 64 tokens (`modal_apps/dev.py::calibrate_and_margin --mixed-budget 64`). Modal's `A10`
  request landed on an NVIDIA A10G; τ still holds there (HF's largest self-divergence gap is
  0.375).

  | | diverged (of 32) | gaps at divergence | failed the rule |
  |---|---|---|---|
  | engine, one prompt at a time | 17 | 0.0 ×5, 0.125 ×9, 0.25 ×2, 0.375 ×1 | 0 |
  | engine, all 32 batched | 13 | 0.0 ×8, 0.125 ×5 | 0 |
  | HF bf16 eager, batch 8 (same run) | 17 | largest 0.375 | n/a |
  | HF bf16 SDPA (same run) | 12 | largest 0.375 | n/a |
  | HF fp32 (same run) | 16 | largest 0.25 | n/a |

  Teacher-forced: all 1,451 positions with a reference gap above τ agree, flash and the PyTorch
  backend. Forced preemption: one preemption, all four requests pass.
- **Which baseline:** at first the single-request rate (17) was compared with HF's SDPA rate (12)
  and treated as clearly higher. That is the wrong baseline on its own: the right comparison is
  the whole range of HF's self-divergence on the same GPU, and HF against itself at batch 8 also
  diverges on 17 of 32 here.
- **This run did not chunk single requests:** the 32 prompts are 17 to 27 tokens, below the
  64-token budget, so one prompt alone is never split. The 17 single-request divergences are the
  same prompts, positions, and gaps as the earlier A10G run with mixed batching off
  (`margin_suite_run2_a10g.json`), which is expected: the two runs execute identical steps.
  Chunking happened only in the batched run. A separate run with a budget small enough to chunk
  every prompt covers the single-request case (next entry).
- **Every prompt chunked** (`docs/data/margin_suite_mixed16.json`, calibration in
  `docs/data/tau_calibration_mixed16.json`; `modal_apps/dev.py::phase3_gpu`, budget 16 and
  `max_num_seqs` 16, NVIDIA A10). The suite counts, per prompt, the steps that computed part of
  its prompt. Run alone, all 32 prompts were split into 2 chunks; in the batched run all 32 were
  chunked, into 2 to 11 chunks (2 ×3, 3 ×6, 4 ×8, 5 ×9, 6 ×3, 9, 10, 11 ×1). The calibration in
  this container reproduced the earlier A10 calibration exactly.

  | | diverged (of 32) | gaps at divergence | failed the rule |
  |---|---|---|---|
  | engine, one prompt at a time, 2 chunks each | 17 | 0.0 ×4, 0.125 ×7, 0.25 ×5, 0.5 ×1 | 0 |
  | engine, all 32 batched, 2-11 chunks | 17 | 0.0 ×5, 0.125 ×8, 0.25 ×3, 0.5 ×1 | 0 |
  | HF bf16 eager, batch 8 (same run) | 15 | largest 0.25 | n/a |
  | HF bf16 SDPA (same run) | 18 | largest 0.5 | n/a |
  | HF fp32 (same run) | 12 | largest 0.5 | n/a |
  | engine, mixed batching off (earlier A10 run) | 15 single, 14 batched | one 0.5 each | 0 |

  The rates sit inside HF's own same-GPU range (12 to 18). The one divergence at 0.5 is prompt 27
  at position 19 in both modes, the same position as with mixed batching off, where HF's SDPA and
  fp32 runs also diverge; nothing else is above 0.25. Chunking changes which near-ties flip
  (10 of the 17 single-request divergences are at the same position as without chunking), as
  expected from different kernel shapes and attention split across steps. Teacher-forced: all
  1,451 positions with a gap above τ agree, flash and the PyTorch backend (largest logit
  difference 0.91). Forced preemption with 16-token chunks: one preemption, all four requests
  pass (three exact, one divergence at gap 0.0).

### Stall micro-benchmark
- **What it measures** (`benchmarks/microbench.py`): the engine driven in-process (no HTTP, no
  async loop), so only the scheduling policy differs between configurations; the end-to-end
  comparison over HTTP belongs to the Phase 4 benchmark. Sixteen requests with 128-token prompts
  decode steadily; then one 3,000-token prompt arrives. Measured: the decodes' inter-token gaps
  from the one spanning the arrival onward (64 per decode), the long request's TTFT, steps, and
  output tokens per second over the window (until the long request has 64 tokens and every decode
  64 more).
- **Fairness:** one engine (model, KV cache, block manager) serves every run; only the scheduler
  object is swapped between runs. All configurations run in the same container on the same GPU,
  one discarded warm-up each, then three repeats interleaved (off, 256, 512, 1024, off, ...).
  Random prompt ids with a fixed seed, greedy, `ignore_eos`. Raw per-step and per-token
  timestamps are committed (`benchmarks/data/microbench/stall_a10_d93b769.json`); the table and
  plots in `benchmarks/results/` are generated from them by `python -m benchmarks.microbench`.
- **Results** (NVIDIA A10, Qwen3-0.6B bf16, flash backend; median of 3 repeats; every repeat
  is within 2.3% of its median on every metric):

  | config | max ITL | p99 ITL | median ITL | long TTFT | long prefill steps | steps | output tok/s |
  |---|---|---|---|---|---|---|---|
  | off | 143.3 ms | 143.3 ms | 29.5 ms | 112.9 ms | 1 | 65 | 543 |
  | budget 256 | 30.0 ms | 30.0 ms | 29.7 ms | 366.4 ms | 13 | 76 | 572 |
  | budget 512 | 30.6 ms | 30.6 ms | 29.7 ms | 201.9 ms | 7 | 70 | 572 |
  | budget 1024 | 46.8 ms | 46.8 ms | 29.7 ms | 125.1 ms | 3 | 66 | 562 |

- **Reading:** with mixed batching off, the 3,000-token prefill runs alone for 113 ms and every
  decode waits one extra step (max ITL 143 ms against a normal 29.5 ms). With a 512-token budget
  the prompt is spread over 7 steps that each also carry all 16 decodes; the largest gap is
  30.6 ms, 4.7 times lower, and the long request's TTFT rises from 113 ms to 202 ms (1.8 times).
  A 256 budget buys no further ITL and costs 3.2 times the TTFT; 1024 keeps TTFT within 11% of
  off but lets gaps reach 47 ms. 512 is the default recommendation for this GPU and model.
- **Why chunks are almost free here, and what that means:** a step with the 16 decodes takes
  29.5 ms, and a step with the decodes plus a 496-token chunk takes 28.5 to 29 ms. Steps are
  dominated by per-step overhead (Python and kernel launches for 28 layers; no CUDA graphs), not
  by GPU arithmetic, so up to about 500 prompt tokens ride along at no visible cost; 1,000-token
  chunks cost 36 to 47 ms. The higher output throughput with mixed batching on (572 against
  543 tokens/s) comes from the same effect: the prefill is absorbed into steps that would cost the
  same anyway, while off mode spends a whole step on it. An engine with fast decode steps (CUDA
  graphs, as in vLLM) would see chunks cost relatively more, so the size of these gains is
  specific to this engine and GPU; this run does not show how stallfree compares with vLLM.
- **p99 equals max** because each decode has exactly one stalled gap among its 64 (16 of 1,024
  gaps, above the 1% tail) with mixed batching off, and with it on the elevated gaps are the
  chunk steps, which are close to every other step.

### Token budget defaults
- **Context:** `max_num_batched_tokens` means different things in the two modes. With mixed
  batching off it caps a prefill step (a prompt longer than it is still admitted alone); with it
  on it is the hard size of every step, so it trades inter-token latency against TTFT.
- **Decision:** the default stays 4096 with mixed batching off. With it on, 512 is the
  recommended budget for Qwen3-0.6B on an A10: in the micro-benchmark it gave the same largest
  gap as 256 (30.6 against 30.0 ms) at 0.55 of 256's TTFT, while 1024 let gaps reach 47 ms. The
  server takes it as `--enable-mixed-batching --max-num-batched-tokens 512`. The value has to be
  measured again on the benchmark GPU (L40S), where steps are cheaper relative to their fixed
  overhead.
- **Evidence:** "Stall micro-benchmark" above; `tests/unit/test_server_cli.py` for the flags.
- **Not done:** the Modal endpoint (`modal_apps/serve.py`) still runs with mixed batching off; it
  has not been run on GPU with the flag on.

### Known gaps to vLLM before the benchmark
These are design differences known before any comparison is run; the Phase 4 results will say
how much each one costs.
- **No CUDA graphs or `torch.compile`:** every step pays Python and kernel-launch overhead for 28
  layers. On the A10 a decode step for 16 sequences takes about 29.5 ms, and up to about 500
  prefill tokens fit into it with no visible cost, which means the GPU is idle for much of the
  step. vLLM captures decode steps in CUDA graphs.
- **No custom kernels:** K/V are written with `index_copy_` and RoPE, RMSNorm, and sampling are
  plain PyTorch ops; vLLM fuses these.
- **Block size 256** (FlashAttention's paged requirement) against vLLM's 16: more memory is
  reserved but unused in each sequence's last block.
- **No prefix caching**, no speculative decoding, no tensor parallelism, no quantization.
- **API subset:** chat completions only; `n > 1`, `stop`, `seed`, `tools`, logprobs and penalties
  are rejected with a 400. A non-streaming request is not cancelled when its client disconnects.
- **Single-threaded engine loop:** scheduling, detokenization, and HTTP handling share one Python
  process with the model's step running in a worker thread.

## Phase 4 — Benchmark

### bf16 correctness on the benchmark GPU (L40S)
- **Context:** τ = 0.5 was calibrated on an A10. The benchmark runs on an NVIDIA L40S (driver
  610.57.04 for this run), so calibration and the margin suite were repeated there, in one
  container: smoke run, calibration, margin suite with mixed batching off, then again with mixed
  batching on and a 16-token budget so every prompt is chunked.
- **Result:**

  | Run | Engine diverged (single / batched) | Rule failures | Largest gap at a divergence |
  |---|---|---|---|
  | Flag off | 16 / 15 of 32 | 0 | 0.375 |
  | Flag on, budget 16 (all 32 prompts chunked; batched: 2-11 chunks) | 15 / 14 of 32 | 0 | 0.375 |
  | Hugging Face against itself (B eager batch 8 / S SDPA / F fp32) | 14 / 16 / 13 of 32 | - | 0.25 |

  Teacher-forced: the top-1 token matched at all 1,437 positions whose gap exceeds τ (of 1,689).
  The preemption scenario passed. The engine's divergence rate is within the range of Hugging
  Face's own self-divergence on the same GPU, so τ = 0.5 stands on the L40S.
- **Evidence:** `docs/data/smoke_l40s.json`, `tau_calibration_l40s.json`,
  `margin_suite_l40s.json`, `margin_suite_l40s_mixed16.json`.

### Prompt text and prompt lengths
- **Context:** Prompts need realistic tokenization, no shared prefixes, and exact lengths that
  both servers count the same way.
- **Decision:** Token ids are sampled at random offsets from *Pride and Prejudice* (public domain,
  1813; Project Gutenberg #1342 with every Project Gutenberg, edition, and illustration text
  removed; SHA-256 pinned in `benchmarks/workload.py`, `benchmarks/data/text/README.md`, a test,
  and every `run.json`). A prompt's length is the length of the **templated** prompt, the number
  the server reports as `usage.prompt_tokens`: the slice is decoded, re-encoded through the chat
  template, and lengthened or shortened until the count is exact. Every request is checked against
  the server's count.
- **Why templated length:** It is the number the engine actually prefills, and the one both
  servers report, so the check is end to end. The Qwen3 template adds 8 tokens.
- **Result:** 0 mismatches in every run so far on both servers, including vLLM, whose image has
  transformers 5.20.2 against stallfree's 5.23.0.

### Requests: greedy, `ignore_eos`, streamed with usage
- **Decision:** Every request streams with `temperature: 0`, `ignore_eos: true`, `max_tokens` from
  the trace, and `stream_options.include_usage`. Greedy decoding removes sampling as a variable;
  `ignore_eos` makes every system generate exactly the same number of tokens; usage gives the
  server-side token counts. The analysis checks `completion_tokens == max_tokens` for every
  request.

### Trace replay, warm-up, and the measured window
- **Decision:** One trace per seed (seeds 1, 2, 3), generated locally with the pinned tokenizer and
  sent to every container, so all systems replay byte-identical requests (the trace's SHA-256 is
  in each `run.json` and checked to match across systems). Gaps are unit-mean exponentials scaled
  by 1/λ, so every rate replays the same requests in the same order. Requests arriving in the first
  30 s are warm-up; then the longer of 300 requests and 120 s of arrivals is measured; the run
  waits for every request to finish. Each (system, seed) gets its own container, which starts the
  server once and runs the rates from lowest to highest.
- **Throughput accounting:** request and output-token throughput are measured requests (and their
  tokens) divided by the window from the first measured send to the last measured completion.
  That window includes the drain at the end, so below saturation the request throughput reads
  lower than the offered rate, more so for a system with long end-to-end latency.

### Saturation throughput (closed loop)
- **Decision:** 256 concurrent clients (= `max_num_seqs`), each sending its next request as soon
  as its last one finishes, cycling through the seed-1 trace; 30 s warm-up, then requests that
  finish within 120 s are counted. Requests still running at the end are cancelled.
- **Result:** stallfree-off saturates at **8.02 req/s** (962 requests, 2,259 output tok/s). The
  sweep rates are therefore 1.60, 4.01, 6.41, 8.02, and 9.62 req/s.
- **Observation:** during the closed loop the stallfree server process used 100-113% of one core
  (mean 102%) while the GPU was busy 48% of the time, at a TPOT of 116 ms: with 256 sequences the
  engine is limited by Python on the CPU, not by the GPU.
- **Evidence:** `benchmarks/data/saturation/`.

### Labelling saturated points
- **Decision:** An open-loop point is labelled saturated when the median TTFT of the last fifth of
  measured requests (in send order) is more than twice that of the first fifth.
- **Why:** Below saturation the queue is stationary and the ratio stays near 1; above it the queue
  grows for the whole run. Comparing achieved throughput with the offered rate does not work here,
  because the measured window includes the drain (see above).

### vLLM configuration
- **Context:** The comparison should be against vLLM as it is normally deployed, with only the
  settings that change what is measured matched.
- **Decision:** vLLM 0.33.0 runs in its official server image (`vllm/vllm-openai:v0.33.0`; torch
  2.14.0+cu130). Matched with stallfree: `--max-model-len 4096`, `--max-num-seqs 256`,
  `--gpu-memory-utilization 0.9`, `--dtype bfloat16`, and `--max-num-batched-tokens 512`, the same
  token budget as stallfree with mixed batching on (vLLM always chunks prefills). Prefix caching
  is off (`--disable-prefix-caching`; stallfree has none and the workload shares no prefixes).
  Everything else is vLLM's default: CUDA graphs, `torch.compile`, its own sampler kernels, and
  separate API-server and engine-core processes. `vllm-eager` adds only `--enforce-eager`.
- **Why the official image:** A pip-only install on the slim base image failed at startup: vLLM's
  default sampler JIT-compiles a FlashInfer kernel with `nvcc`, which that image does not have.
  Turning the FlashInfer sampler off would have moved vLLM away from its defaults; the official
  image ships what the defaults need.
- **Safeguard:** Before starting vLLM, the container checks that every flag in the command appears
  in `vllm serve --help=all` and stops otherwise. This caught that 0.33.0 spells the prefix-cache
  switch `--disable-prefix-caching` (not `--no-enable-prefix-caching`).

### stallfree configuration in the benchmark
- **Decision:** `--max-num-seqs 256 --max-model-len 4096`, automatic KV sizing at 0.9 of GPU
  memory (1,474 blocks of 256 tokens on the L40S), flash attention, bf16. Mixed batching off uses
  `--max-num-batched-tokens 4096` (the prefill cap); on uses 512. `--max-queued-requests` is set
  far above any load so that stallfree never answers 503 (vLLM has no such limit).

### Raw data format
- **Decision:** Per-request records are gzipped JSON lines with timestamps rounded to the
  microsecond (`<seed>.jsonl.gz`), next to the CPU samples, the load generator's summary, and a
  `run.json` with versions, GPU and driver, clocks before and after, CPU and memory requests,
  flags, trace seed and SHA-256, the source text's SHA-256, and the commit. The server's log is
  stored per container (gzipped).
- **Why:** Every chunk's arrival time is kept, so any metric can be recomputed. Uncompressed, one
  saturation run was 5.8 MB; gzipped with microsecond times about 1.1 KB per request, which puts
  the full sweep at an estimated 60 MB.

### CPU sampling
- **Decision:** The driver process (neither server nor load generator) samples, every second, the
  CPU use of each process in the server's tree and of the load generator with `psutil`, plus GPU
  utilization and SM clock with `nvidia-smi`. Samples are aligned to each run's measured window by
  wall-clock time. A run is flagged when the load generator averages over 80% of one core.
- **Result so far:** the load generator averaged 15-20% (max 31%) in every run, so no run is
  flagged.

### Driver versions on Modal
- **Observation:** Modal places containers on different hosts, and the dry run landed on two NVIDIA
  driver versions (610.57.04 and 620.70.06), both with SM clocks at their 2,520 MHz maximum before
  and after each run. The driver cannot be pinned on Modal; it is recorded in every `run.json` and
  in the per-run table.

### Dry run (one rate, one repeat)
- **Result** (6.41 req/s = 0.8 x saturation, seed 1; `benchmarks/results/dryrun.md`): every
  system completed all 721 measured requests with no errors, no prompt-token mismatches, and no
  `ignore_eos` violations. Median TPOT: stallfree-off 87.8 ms, stallfree-on 55.7 ms, vLLM 3.9 ms,
  vLLM eager 22.8 ms. Median TTFT: 90, 258, 20, and 80 ms. These are single runs; the full sweep
  repeats each point three times.
- **Early reading of the gap to vLLM:** vLLM eager is 2.4 times faster per output token than
  stallfree with mixed batching on, and CUDA graphs make vLLM another 5.8 times faster. stallfree's
  server process sits at about one full core while the GPU is 40% busy; vLLM's engine core uses
  111% and keeps the GPU 99% busy. The full sweep will quantify this per rate.

### Full sweep, budget sweep, and saturation of vLLM
- **What ran:**
  - 4 systems x 5 rates x 3 seeds, one container per (system, seed): 60 runs.
  - stallfree-on with budgets of 256, 1,024 and 2,048 tokens at 0.8 x, with 3 seeds each.
  - A closed-loop saturation run of vLLM.
  - All 60 sweep runs passed every sanity check (no errors, no prompt-token mismatches, no
    `ignore_eos` violations). The load generator averaged at most 27% of a core (peak 61%).
- **Trace identity:** all four systems share one trace hash per seed. The budget runs used a
  shorter trace from the same seeds (sized for 0.8 x instead of 1.2 x), so their hashes differ.
  The replayed requests were checked to be identical: same indices, prompt lengths, output lengths,
  and send times, request by request.
- **Results** are in `benchmarks/results.md`; the tables come from `benchmarks/summarize.py` and
  the plots from `benchmarks/plot.py`.

### Host types on Modal
- **Observation:** containers landed on two host types, told apart by their NVIDIA driver version
  (610.57.04 and 620.70.06). stallfree is 1.8 to 2.4 times faster per token on the 610 hosts.
  vLLM with CUDA graphs is the same on both, and every vLLM-eager run landed on 620 hosts.
  Driver version and host CPU are confounded; these runs did not record the CPU model.
- **Decision:** report the host split instead of hiding it.
  - A table split by host type is generated with the results.
  - The stallfree medians are 620-host values (two of three repeats), the same host type as vLLM
    eager.
  - The saturation run that set the rates was on a 610 host; this is stated next to the
    saturated points.
  - From now on every `run.json` records the CPU model and kernel version.
- **Not done:** a rerun pinned to one host type. Modal does not let a function choose the host
  CPU directly; pinning would need probing which cloud or region gives which host type.

### Token budget on the L40S
- **Result:** at 0.8 x on the same host type (620), 1,024 tokens gave half the TTFT of 512 (p50
  107-113 against 207-265 ms) with a similar ITL p99 (64-84 against 69-76 ms). 256 saturates at
  this rate (TTFT p50 14.5 s). 2,048 let ITL p99 reach 97 ms.
- **Decision:** the sweep keeps 512, which was fixed before it ran. 1,024 is the better budget for
  this GPU and workload, and the README says so.
- **Outcome for "Token budget defaults":** the A10 recommendation (512) does not carry over to the
  L40S.

### Profiling the stallfree server
- **Context:** stallfree's server uses one core while the GPU is half idle, so the gap to vLLM
  needs evidence of where that core goes.
- **Getting py-spy to attach:** in the benchmark sandbox a process may only be traced by its own
  ancestors. The profiling run therefore starts the server through `benchmarks/ptraceable.py`,
  which calls `prctl(PR_SET_PTRACER, ANY)` and then runs the server module unchanged. py-spy
  (0.4.3, in the stallfree benchmark image only) then attaches by PID 40 s into a normal run at
  0.8 x, after warm-up, for 60 s. The profiling runs are separate from the results.
- **Attribution (`benchmarks/profile_summary.py`):**
  - A sample's component is decided by its innermost stallfree frame; samples without one are
    assigned by library.
  - Its state comes from the innermost native frames: `epoll_wait` is the event loop waiting;
    a condition wait or lock acquire is waiting for the GIL or a lock; anything else is running.
    This sandbox reports blocked threads as active, so this step is needed.
  - Running samples / (rate x duration) gave 0.99 cores, matching psutil's ~100%.
- **Findings** (native profile, `benchmarks/results/profile.md`):
  - Shares of the server's CPU: model forward 34%; batch metadata, sampling and scheduling 8%;
    prompt tokenization 22%; HTTP/SSE 17%; event loop 16.5%.
  - Within model forward, the innermost frame is PyTorch C++ dispatch in 42% of samples,
    malloc/locks in 19%, the CUDA driver in 18%, and Python in 17%.
  - The step threads wait for the GIL in 12% of their samples.
- **Caveat:** native-frame sampling pauses the process and slowed the profiled run from 28 to
  57 ms TPOT, pushing it into saturation. A non-blocking, Python-only profile slowed the server
  less and agrees on the model-forward share (35%). In this sandbox, though, it cannot see Rust
  tokenizer time and counts idle `epoll` waits as running, so it is not used for the other shares.

### Gaps to vLLM, measured
These are the measured counterparts of "Known gaps to vLLM before the benchmark".
- **Overall, at 0.8 x:** vLLM's TPOT is 3.9 ms against stallfree-on's 54.0 ms (14x), and its TTFT
  is 20 against 207 ms. Its closed-loop saturation is 18.2 against 8.0 req/s on the same host type
  (2.3x).
- **CUDA graphs:** 5.4x within vLLM (eager 20.9 ms against 3.9 ms). The profile shows stallfree's
  model forward as host-side dispatch and launch work, which is what graphs remove.
- **The rest of the engine:** vLLM eager against stallfree-on is 2.6x.
  - vLLM runs the API server (tokenization, detokenization, HTTP) in its own process (8-28% of a
    core) beside an engine core that uses 128-138%.
  - stallfree's step threads get about 42% of one GIL-bound core, because 58% goes to the asyncio
    thread.
  - This is consistent with a factor of about two but is not separated from vLLM's other per-step
    differences by this benchmark.
- **Unexplained:** stallfree's global stalls of 150-190 ms (47 ms on the faster host type) do not
  line up with long-prompt arrivals. Finding the cause needs per-step timing inside the server.
