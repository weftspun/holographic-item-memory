# Multimodal recommender roadmap

Ordered by the taskweft HTN planner (`priv/plans/domains/multimodal-recommender-roadmap.jsonld`,
`taskweft plan`). Baseline-first on a bootstrap 768-d embedding → build the multimodal embedder
(two independent tracks → fuse) → upgrade the recommender onto the fused embedding.

The one load-bearing constraint: the recommender needs **one 768-d content embedding** feeding
*both* `project_in` (768 → 4 ID tokens) and the aux (768 → 4×192), so a first baseline can start
on any single 768-d embedding, and the full multimodal embedder is an upgrade, not a prerequisite.

## Phase 1 — Recommender baseline (`residual-fsq-recommender`, no multimodal deps)

| # | action | work |
|---|---|---|
| 1 | `bootstrap_embed_mpnet_768` | one 768-d embedding (MPNet, or Python-Qwen3-VL offline) to unblock tokenization |
| 2 | `ingest_amazon_cc0` | `Adapters.TrajectoryConvert` → per-user chronological sessions (gate-clean) |
| 3 | `tokenize_catalog_bootstrap` | `Core.ResidualFSQ.encode_ids` over the 768-d embeds → 4-token IDs → catalog |
| 4 | `train_fuxi_gpu_bootstrap` | `mix recommender.pretrain` (real-data path) on a GPU host |
| 5 | `eval_hitk_vs_popularity` | `mix recommender.eval` — Hit@k / MRR vs a popularity floor |
| 6 | `serve_bootstrap_checkpoint` | `rfr recommend --checkpoint` |

## Phase 2 — Multimodal embedder (two independent tracks, then fuse)

**Track A · Qwen3-VL** (`weftspun/bumblebee@qwen3-vl-embedding`, frozen — port, not retrain):
7 `qwen3vl_featurizer` → 8 `qwen3vl_fusion_mrope_deepstack` → 9 `qwen3vl_registry` →
10 `qwen3vl_e2e_verify_golden` → 11 `qwen3vl_wire_embedder`

**Track B · Mesh VAE** (`weftspun/trellis2-mesh-vae-slang`, Slang + Lean 4):
12 `mesh_author_kernels_lean` → 13 `mesh_emit_slang_spirv` → 14 `mesh_verify_encoder_forward`

**Converge:** 15 `build_unified_embedder_768` (`weftspun/unified-modal-embedder`) — fuse per-modality
vectors, Matryoshka-truncate to 768-d.

> The planner emits a linear order (A before B), but the tracks are independent (different repos)
> and run in parallel. Annotate action durations to have taskweft's STN scheduler overlap them.

## Phase 3 — Upgrade the recommender onto the fused embedding

16 `retokenize_with_fused` → 17 `retrain_fuxi_fused` → 18 `reeval_fused`

## What needs a GPU / a human

- Steps 4, 17 (train / retrain) — GPU host (`XLA_TARGET=cuda12x`).
- Track A (7–11) — the ~10h remainder of the Qwen3-VL Elixir/Bumblebee port.
- Track B (12–14) — authoring the mesh-VAE encoder kernels in Lean + Slang.

Steps 2, 3, 5 and the pretrain real-data wiring are pure code (no GPU) and are the near-term work
in this repo.
