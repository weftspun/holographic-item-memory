defmodule Holo do
  @moduledoc """
  Holographic session memory over ResidualFSQ semantic IDs.

  * `Holo.HRR` — phase-vector algebra (bind / unbind / bundle / similarity),
    SHA-256-deterministic atoms.
  * `Holo.ResidualFSQ` — the semantic-ID codec: `levels = [8,8,8,8]`,
    `num_quantizers = 3` → 3 tokens per item, 4096 codes per token.
  * `Holo.SemanticID` — item vectors derived purely from semantic IDs, plus
    `asset_semantic_id.parquet` ingestion (the `multimodal-semantic-ids`
    contract).
  * `Holo.Memory` — the zero-shot next-item recommender: content bundles +
    an optional hetero-associative transition bank.

  ## Quick start

      mem =
        Holo.Memory.new(dim: 1024)
        |> Holo.Memory.add_items([{"sword", [17, 900, 3]}, {"shield", [17, 901, 44]}])

      {:ok, recs} = Holo.Memory.recommend(mem, ["sword"], top_k: 5)
  """
end
