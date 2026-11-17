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
