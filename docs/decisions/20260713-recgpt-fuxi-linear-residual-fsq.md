# Switch the recommender to RecGPT / FuXi-Linear over Residual FSQ semantic IDs

**Status:** Accepted · 2026-07-13 · **Supersedes the recommender core of** [Holographic HRR sequential recall](../sequential-recall.md) (kept as the training-free baseline)

## Context and problem statement

This project recommends the next item in a session over **ResidualFSQ semantic IDs**
(`{item_id, [t0, t1, t2]}`, `levels = [8,8,8,8]`, `num_quantizers = 3`). Today the recommender is
`Holo.Core.Memory` — a training-free Holographic Reduced Representation (HRR) memory: bind/unbind/
bundle over phase vectors, no neural network.

The HRR path is a strong *floor* but a first-order one. Measured on MovieLens-100K
(leave-one-out, 100 sampled negatives, `docs/sequential-recall.md §5`):

| | Recall@10 | NDCG@10 |
|---|---|---|
| HRR transition-only (ours) | 34.4 | 18.7 |
| popularity floor | 31.8 | 17.1 |
| trained transformer ceiling (SASRec/BERT4Rec) | ~55–80 | ~35–56 |

HRR clears popularity but sits at roughly half a trained sequence model's NDCG@10, and most of that
gap is structural: HRR models one Markov step, not the whole session. We want the trained ceiling
**without abandoning the semantic-ID substrate** the whole system (and the Lean proofs) rests on.

The archived sibling [`weftspun/elixir-sequential-recommendation`](https://github.com/weftspun/elixir-sequential-recommendation)
already implements a RecGPT-style generative recommender in Elixir: data pipeline, FSQ tokenizer,
**FuXi-Linear** inference (linear-attention), constrained trie decode, and an MCP server. Adopting it
gets us a trained model over the same token idea — but its FSQ is the wrong *kind* of FSQ for us.

## Decision drivers

- **Keep the residual semantic-ID contract.** `Holo.Core.Memory` and the Lean model
  (`formal/HoloModel.lean`: `stage_bound`, `stage_roundtrip`, `stage_injective`, `itemKey_injective`)
  certify a *residual* codec: 3 coarse-to-fine tokens, base-4096, injective. Any tokenizer we ship
  must honor that, so HRR and the trained model consume **identical** IDs.
- **Linear complexity for long trajectories.** The RecGPT datasets are long — MerRec has ~1B events /
  200M sessions, KuaiRand is long feed logs. Quadratic attention (T5/SASRec) does not scale here;
  FuXi-Linear's Retention + Linear Temporal + Linear Positional channels are O(n).
- **Decoder-only, no encoder-decoder autoencoders.** FSQ is parameter-free quantization (no RQ-VAE
  encoder/decoder, no codebook collapse, no commitment loss); FuXi-Linear is a single causal stream
  (no T5 encoder-decoder). This matches the standing "decode-only" constraint on this project.
- **Coarse-to-fine generalization.** Residual quantization puts the most semantic mass in `t0`, so
  similar items share ID prefixes — the property TIGER relies on for cold-start / long-tail, and the
  property grouped/product FSQ does **not** have.

## Considered options

1. **Keep HRR-only (status quo).** ✅ zero training, formally bounded, cheap, cold-start-native.
   ❌ first-order Markov; ~half the trained ceiling; no full-session context.
2. **TIGER (RQ-VAE + T5 encoder-decoder).** ✅ proven SOTA generative retrieval. ❌ RQ-VAE codebook
   collapse / commitment loss; T5 is an encoder-decoder with O(n²) attention; two autoencoders to train.
3. **RecGPT + FuXi-Linear, grouped FSQ (the reference repo as-is).** ✅ working Elixir code, linear
   attention, decoder-only. ❌ grouped/product FSQ (`reshape → 4×192`, quantized independently) is
   **not** coarse-to-fine: tokens are parallel and equal-weight, no shared-prefix generalization, and
   it violates the residual contract the Lean proofs certify.
4. **RecGPT + FuXi-Linear over *Residual* FSQ (chosen).** ✅ linear-attention decoder-only generative
   model *and* the coarse-to-fine residual semantic IDs this repo already speaks; reuses the archived
   implementation with one tokenizer swap. ❌ requires training (GPU), a Torchx/CUDA build, and a
   larger dependency surface (Axon, Bumblebee, Torchx).

## Decision outcome

**Chosen: option 4.** Port the RecGPT + FuXi-Linear stack into this project and **replace grouped FSQ
with residual FSQ**. `Holo.Core.Memory` (HRR) is retained as the training-free baseline; both
recommenders consume the same residual IDs, giving a clean apples-to-apples comparison
(*same tokens → { trained FuXi-Linear generation vs training-free HRR memory }*).

### The core technical change: grouped FSQ → residual FSQ

Both emit `S` tokens per item; they differ in **how** the tokens partition the content embedding `e`.

| | Grouped / product FSQ (reference) | Residual FSQ (this decision) |
|---|---|---|
| decomposition | split `e` into `S` disjoint sub-vectors, quantize each | quantize all of `e`, subtract reconstruction, quantize the **residual**, repeat `S` times |
| token semantics | parallel, order-agnostic, equal magnitude | hierarchical, coarse→fine, decreasing magnitude |
| shared prefixes | none (independent groups) | yes — similar items agree on `t0`, then `t1`, … |
| contract match | `levels [8,8,8,6,5]`, 4 tokens, 15 360 codes | `levels [8,8,8,8]`, 3 tokens, 4096 codes/stage (repo + Lean) |

Residual stage `s`: `code_s = FSQ(project(r_{s-1}))`; `r_s = project(r_{s-1}) − dequant(code_s)`;
`r_0 = project(e)`. FSQ's `bound`/`round_ste`/`codes_to_indices` (from `Core.FSQ`) are reused per
stage unchanged — only the *driver* changes from "slice into groups" to "iterate on the residual".

## Architecture / layout

Adopt the reference's hexagonal split (see the sibling
[hexagonal core/ports/adapters ADR](https://github.com/weftspun/elixir-sequential-recommendation/blob/main/docs/decisions/20260610-hexagonal-core-ports-adapters.md)),
namespaced to this app:

```
lib/holo/
  core/
    memory.ex  hrr.ex                 # RETAINED — training-free baseline over the same IDs
    residual_fsq.ex                   # NEW — residual driver over Core.FSQ stage math
    fuxi_linear_inference{,_defn,_params}.ex   # PORT — linear-attention generative body
    decode.ex  trie.ex  training.ex  eval.ex   # PORT — constrained decode + metrics
    trajectories/split.ex             # PORT — chrono train/test split
  ports/    embedding_source · checkpoint_{source,sink} · fsq_params_source · test_case_source
  adapters/ bumblebee_embedding · axon_trainer · {npy,pt}_checkpoint_* · mcp/server · serve
```

New deps (from the reference `mix.exs`): `torchx`, `axon`, `bumblebee`, `npy`, `unpickler`,
`unzip`, `nimble_csv` (Explorer and Nx already present).

## Migration / rollout

1. **`Holo.Core.ResidualFSQ`** — residual driver over the ported `Core.FSQ` stage math; emit
   `[t0,t1,t2]` in `0..4095`. Property-test injectivity against the Lean `itemKey_injective` contract.
2. **Tokenize a controlled corpus** — MovieLens-20M genome-tag embeddings → residual IDs; feed into
   the existing `Holo.Core.Memory` first (no training) to confirm the substrate before any GPU work.
3. **Port FuXi-Linear** (`fuxi_linear_inference*`) + decode/trie/eval; wire `wte`/`ae`/`pred_head` to
   the 3-token / 4096-code residual vocabulary.
4. **Train** on a MovieLens-20M subset (CPU-smoke / single-GPU), then scale the recipe to MerRec where
   the 80M-item long tail makes residual semantic IDs indispensable.
5. Report FuXi-Linear vs HRR on the identical IDs under the existing LOO / 100-negative harness.

## Consequences

- **Positive:** trained full-session model at linear cost; same residual IDs across HRR and
  FuXi-Linear (Lean proofs still apply); coarse-to-fine tokens enable cold-start and long-tail;
  reuses a working Elixir implementation instead of a from-scratch build.
- **Negative:** training infrastructure (GPU, Torchx/CUDA), a heavier dependency surface, and a larger
  codebase than the two-module HRR core. Grouped FSQ's cheaper independent-group encode is given up.
- **Neutral:** HRR is not deleted — it is reframed as the training-free floor and the fixture-level
  test oracle, which is where it is strongest.

## References

- FuXi-Linear — *Unleashing the Power of Linear Attention in Long-term Time-aware Sequential
  Recommendation*, arXiv 2602.23671.
- TIGER — Rajput et al., *Recommender Systems with Generative Retrieval*, NeurIPS 2023 (RQ-VAE
  semantic IDs + T5; the lineage this project consumes).
- FSQ — Mentzer et al., *Finite Scalar Quantization: VQ-VAE Made Simple*, ICLR 2024.
- Reference implementation: `weftspun/elixir-sequential-recommendation` (archived 2026-07-13).
- This repo's residual-codec proofs: `formal/HoloModel.lean` (`stage_*`, `itemKey_injective`).
