# Primus DLRM Benchmark Suite Design


## 1. Goals and constraints

### Primary goal
Build a reference training benchmark suite (**TorchRec-based**) for **AMD MI350+ + HIP/ROCm**, centered on:

- **DLRM-like baseline** (dense + sparse + configurable interaction module)
- **OneTrans-like SOTA** (unified transformer for sequence + interactions, with caching)

### Hard constraints you gave
- **Hardware:** MI350+
- **Scale:** 8 - 16 nodes typical, 32 nodes for large scale press test
- **Mode:** training only
- **Candidates per request:** 100–1000

### Benchmark suite must stress
- step time / throughput and peak compute utilization
- distributed embedding lookup + updates
- Adam + distributed Shampoo (and stability)
- multi-stream / queue saturation
- DDP/FSDP/multi-node stability
- convergence + numerical stability

---

## 2. Feature schema

### Yambda gives
- interactions with `uid`, `item_id`, `timestamp`, `is_organic`, `event_type`
- playback fields for listens: `played_ratio_pct`, `track_length_seconds`
- audio embeddings for most items
- artist and album mappings in separate parquet files

### 2.1 Core categorical (embedding tables)

#### User-side
- `uid` (1M users)
- time buckets derived from `timestamp` (hour-of-day, day-of-week, recency bucket)
- user history aggregates (hashed): e.g., “recent artist histogram bucket id”, “recent genre proxy”  
  (if genre not available, use artist/album)

#### Item-side
- `item_id` (9.39M tracks)
- `artist_id` (via `artist_item_mapping.parquet`)
- `album_id` (via `album_item_mapping.parquet`)

#### Context / event
- `event_type` ∈ {`listen`, `like`, `dislike`, `unlike`, `undislike`}
- `is_organic` (organic vs recommendation-driven)

### 2.2 Dense / continuous features
- `track_length_seconds` (normalized + bucketized version)
- `played_ratio_pct` (for listen events; normalized + bucketized + raw regression target)
- audio embedding vector (`embed` or `normalized_embed`) as dense input

### 2.3 Sequence features
We should represent multi-behavior sequences (listen/like/dislike + reversals) in at least two ways:

#### (A) Baseline-friendly sequence features (cheap but lookup-heavy)
- last N `item_ids` (N = 100/300/1000 configs)
- last N `artist_ids`
- last N `album_ids`
- last N `event_types`
- last N time-gap buckets

Use `EmbeddingBag` / pooled representations (mean/sum/attention pooling) to keep compute light but stress embeddings.

#### (B) OneTrans-style token sequence
- Build a unified sequence of “event tokens” that include `item_id` + side info (artist/album + maybe time bucket) and optionally `event_type`.
- Insert learned separators between behavior types if you keep separate streams; otherwise interleave timestamp-aware.  
  (OneTrans supports both and reports timestamp-aware works better when timestamps exist.)

---

## 3. Training sample construction

### 3.1 “Request” definition (training-only but request-structured)
Each training example = one request with:
- **user history:** last L events (L = 100–1000)
- **candidate set:** C items (C = 100–1000)
  - 1 positive (the next event item)
  - C-1 negatives (mixture of uniform + popularity + in-batch)

This is critical because it enables the OneTrans “S-side once per request, NS-side per candidate” computation structure (KV caching across candidates).

### 3.2 Negatives
Support at least:
- no negative sampling
- uniform negatives
- popularity-weighted negatives (for hardening)
- in-batch negatives (cheap and very stressful for collectives)

### 3.3 Multi-task label assignment
Per request, define the positive’s labels as described earlier, and set all negatives’ labels to 0 (or masked for regression).  
This keeps the training step vectorized as `[B, C, heads]`.

---

## 4. Model 1: Baseline “DLRM++” (TorchRec-first, interaction modular)

### 4.1 Baseline model architecture (high-level)

#### Inputs
- Sparse categorical embeddings (TorchRec `EmbeddingCollection` / `EmbeddingBagCollection`)
- Dense features MLP (audio embedding projection + numeric features)

#### User representation
- `user_id` embedding
- pooled sequence embedding(s) from last L events (cheap pooling by default)
- dense context projection

#### Item representation (per candidate)
- `item_id` embedding + artist/album embeddings
- audio embedding projection
- candidate context embeddings

#### Interaction module options (configurable)
Implement multiple interaction operators under a common interface:
- DLRM dot-interaction (pairwise dot among embedded features + dense)
- Concatenate + MLP (simple but strong baseline)
- DCN / DCNv2-style cross network (explicit crosses)

#### Heads
Multi-task heads:
- `listen_plus` (logit)
- `like` (logit)
- `dislike` (logit)
- `played_ratio` (regression)

### 4.2 Why baseline should be clearly weaker than OneTrans (by design)
The baseline will:
- only get pooled sequence summaries (no deep token-level joint modeling),
- do interactions “after encoding,”
- won’t have OneTrans’s unified token mixing across sequence+features.

This creates a realistic quality gap that OneTrans should close.

---

## 5. Model 2: OneTrans-on-Yambda (adaptation plan)

### Paper link (for implementation reference)
- HTML: https://arxiv.org/html/2510.26104v1
- PDF: https://arxiv.org/pdf/2510.26104.pdf

### 5.1 Key OneTrans ingredients we must preserve

#### Unified tokenization
- Sequential tokens (S-tokens) from user behavior sequences
- Non-sequential tokens (NS-tokens) from user/item/context features

OneTrans supports:
- group-wise tokenizer (feature groups each → token via group MLP)
- auto-split tokenizer (single MLP then split into multiple tokens; fewer kernel launches)

#### Mixed parameterization in the Transformer block
- S-tokens share Q/K/V projection weights and FFN weights
- NS-tokens get token-specific Q/K/V and FFN parameters

#### Single causal mask
- NS-tokens appear after S-tokens and can attend to the full S history, enabling cross interactions inside one stack

#### Pyramid stack
- progressively shrink the query set for S-tokens by keeping only the most recent tail as depth increases, reducing compute/memory

#### Cross-candidate KV caching (the training-critical part)
- **Stage I:** compute S-side once per request and cache K/V + outputs
- **Stage II:** per candidate compute NS-tokens and attend to cached S K/V

(They also extend caching “across requests,” but for training-only we can treat this as optional.)

### 5.2 Mapping OneTrans to Yambda features

#### Sequential S-tokens
- last L events; each event token embeds:
  - `item_id` embedding
  - plus side info (`artist_id`, `album_id`, `event_type`, time bucket, `is_organic`)
- timestamp-aware interleaving should be default since timestamps exist

#### Non-sequential NS-tokens
- candidate item features (`item_id`, artist/album, audio embed projection, track length buckets)
- user/static features (`uid` embedding, time context buckets)
- “request context” tokens (e.g., `is_organic`?; though in Yambda it’s per event—still usable)

#### Heads
Same multi-task heads as baseline, produced from final NS-token states.

### 5.3 Ensuring OneTrans beats baseline
To maximize OneTrans quality (without turning this into a pure research project), add these “quality levers” early:
- longer history L (300–1000) for OneTrans vs smaller L for baseline
- use timestamp-aware sequence fusion
- use auto-split tokenizer to reduce overhead and allow more tokens
- tune loss weights so Listen+/Like tasks actually lift ranking metrics

---

## 6. TorchRec + ROCm/HIP stack plan

### 6.1 Embedding backend strategy (practical ROCm reality)
TorchRec’s recommended fast path on GPU is **FBGEMM_GPU Table Batched Embedding (TBE)**:
- It batches embedding tables into fewer kernel calls and supports optimizer fusion (module updates itself like fused optimizer).
- FBGEMM_GPU explicitly documents ROCm build variants and “supports running on AMD (ROCm) devices”.

Plan: build an abstraction so we can run:
- **Correctness mode:** plain PyTorch `EmbeddingBag` / TorchRec CPU-ish fallback if needed
- **Perf mode:** FBGEMM_GPU TBE on ROCm (the target stress path)

### 6.2 Distributed embeddings + train pipeline stress knobs
We will use TorchRec distributed primitives:
- sharding plans for embedding tables (row-wise, table-wise, column-wise; plus 2D sharding where supported)
- all-to-all comm patterns
- train pipelines that enable overlap

Notably, TorchRec has introduced **Fused SDD** train pipelines designed to overlap optimizer work with embedding lookup, explicitly calling out gains for “heavy optimizers (e.g., Shampoo)”.

### 6.3 Optimizers: Adam + Distributed Shampoo (what applies to what)
- **Embedding parameters:** use fused optimizers if available via TBE; otherwise Adam/RowWise variants depending on memory.
- **Dense parameters (MLPs + OneTrans transformer):**
  - AdamW baseline
  - Distributed Shampoo for stress testing (dense-only per the reference implementation):  
    “Distributed Shampoo currently only supports dense parameters.”
  - Shampoo has DDP/FSDP/HSDP support modes (relevant to stage 2).

---

## 7. Staged execution plan

### Stage 1A — Baseline DLRM++ (shared infra + single-node correctness)

#### Deliverables
- Data loader that yields `(history, candidates, labels)` with:
  - L = 100/300/1000 configs
  - C = 100/500/1000 configs
- Metric runner that matches Yambda style:
  - NDCG@{10,50,100}, Recall@{10,50,100}
  - top-100 ranking evaluation
- DLRM++ model implementation:
  - sparse categorical embeddings (TorchRec `EmbeddingCollection` / `EmbeddingBagCollection`)
  - dense features MLP (audio embedding projection + numeric features)
  - user representation (user_id embedding + pooled sequence embeddings + dense context)
  - item representation per candidate (item_id + artist/album + audio projection)
  - configurable interaction module (dot-interaction, concat+MLP, DCN/DCNv2)
  - multi-task heads: `listen_plus`, `like`, `dislike`, `played_ratio`
- Single-node training loop that converges (loss down; metrics up)

#### Acceptance targets
- Basic sanity: loss decreases; no NaNs in BF16
- Reasonable ranking metrics on Yambda (establishes the baseline numbers for comparison)

---

### Stage 1B — OneTrans (single-node correctness + convergence)

#### Deliverables
- OneTrans tokenization:
  - S-tokens from user behavior sequences (item_id + artist/album + event_type + time bucket + is_organic)
  - NS-tokens from candidate item features + user/static features + request context
  - timestamp-aware interleaving (default)
  - group-wise or auto-split tokenizer
- Transformer with mixed parameterization:
  - S-tokens share Q/K/V and FFN weights
  - NS-tokens get token-specific Q/K/V and FFN parameters
  - single causal mask (NS-tokens after S-tokens, attending to full S history)
- Cross-candidate KV caching (Stage I / Stage II split in training):
  - Stage I: compute S-side once per request, cache K/V + outputs
  - Stage II: per candidate, compute NS-tokens and attend to cached S K/V
- Same multi-task heads as baseline, produced from final NS-token states
- Single-node training that converges

#### Acceptance targets
- Basic sanity: loss decreases; no NaNs in BF16
- Quality: OneTrans > baseline on Listen+ NDCG@100 and Recall@100  
  (pick a meaningful delta threshold, e.g. +5–10% relative)
- Stretch: OneTrans approaches / exceeds SASRec reference metrics from the Yambda paper

---

### Stage 2 — Distributed training correctness (2–8 nodes)

#### Deliverables
- TorchRec sharded embeddings + multi-node training script
- DDP and FSDP mode
- Convergence parity checks:
  - 1 node vs 2 nodes vs 8 nodes with same global batch (within tolerance)

#### Stress axes
- embedding table sharding strategies (row-wise vs table-wise)
- all-to-all patterns from JaggedTensor/embedding comms
- optimizer variants: Adam vs distributed Shampoo

#### Acceptance
- stable multi-node runs (no hangs, NCCL/RCCL issues, no silent divergence)
- reproducible evaluation metrics within tolerance

---

### Stage 3 — Performance + stability stress suite (MI350+, “torture modes”)

This stage is about turning knobs that specifically saturate:
- HBM bandwidth (embedding lookup + optimizer updates)
- collectives bandwidth/latency (all-to-all, reduce-scatter, all-reduce)
- kernel launch / stream scheduling

#### Workloads (each becomes a benchmark test case)

**Embedding lookup pressure sweep**
- L ∈ {100, 300, 1000}
- C ∈ {100, 500, 1000}
- #tables ∈ {8, 32, 64, 128} (see “synthetic feature expansion” below)

**Optimizer sweep**
- Adam (baseline)
- row-wise Adam/Adagrad
- distributed Shampoo (aggressive stress)
- measure update time, memory, and stability

**Comm overlap modes**
- baseline synchronous
- overlap embedding all-to-all with dense compute
- overlap optimizer step where possible

**Multi-stream / queue stress**
- explicit use of multiple streams to overlap:
  - embedding forward
  - dense forward/backward
  - all-to-all
  - optimizer updates
- measure throughput and tail latency of step time distribution

**Numerical stability tests**
- BF16 everywhere vs selective FP32 accum
- large-batch stress
- gradient clipping on/off
- intentional extreme logits tests (to catch softmax/CE overflows)

#### Synthetic feature expansion (to get “as much feature as we can”)
Beyond Yambda’s native features, add configurable hashed categorical features to create realistic multi-table pressure:
- hashed(user_id, time_bucket) → table A
- hashed(item_id, time_bucket) → table B
- hashed(artist_id, album_id) → table C
- hashed(recent_item_id_k) for k in 1..N → tables

This is a standard way to create 32–128 embedding tables without inventing semantics, and it stresses exactly what you care about (lookup + update).

#### Acceptance
- stable, repeatable perf numbers
- no crashes/hangs over long runs
- clear profiling artifacts (bandwidth, collective times, kernel occupancy)

---

### Optional Stage 4 — OneTrans system optimizations (paper-aligned)

If you want to push “systems realism” further, add:
- Pyramid stack schedule (shrink S-token query tail by layer)
- FlashAttention / memory-efficient attention (if available on ROCm)
- activation recomputation
- cross-request KV caching (more inference-serving aligned; optional for training-only)