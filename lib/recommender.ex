defmodule Recommender do
  @moduledoc """
  Generative next-item recommender over ResidualFSQ semantic IDs — hexagonal
  layout per `20260610-hexagonal-core-ports-adapters`.

  ## core/ (pure — no sockets, no devices, no frameworks)

  * `Recommender.Core.FSQ` / `Recommender.Core.ResidualFSQ` — the residual Finite Scalar
    Quantization codec: content embeddings → 4-token semantic IDs
    (`[t0,t1,t2,t3]`, each in `0..4095`), and the ID contract
    (`valid_id?/1`, `tokens_per_item/0`, `codebook_size/0`) every consumer
    shares. Certified in `formal/RecommenderModel.lean`.
  * `Recommender.Core.FuxiLinearInference*` / `Recommender.Core.{Decode,Trie,Training,Eval}`
    — the FuXi-Linear (linear-attention) generative recommender: forward pass,
    trie-constrained beam decode, training losses, metrics.

  ## ports/ (contracts the core is wired through; see `ports/sibling_repos.txt`)

  * `Recommender.Ports.ItemSource` / `Recommender.Ports.ItemSink` — read/write items and
    session transitions; `ItemSource.load_catalog/2` feeds `Recommender.Adapters.Serve`.
  * `Recommender.Ports.BlobSource` / `Recommender.Ports.BlobSink` — read/write
    content-addressed blobs.
  * `Recommender.Ports.{CheckpointSource,CheckpointSink,FsqParamsSource,EmbeddingSource}`
    — model checkpoint / params / embedding IO.

  ## adapters/ (concrete I/O at the edges)

  * `Recommender.Adapters.Serve` — wires the FuXi-Linear forward pass + constrained
    decode + catalog into `recommend/3`.
  * `Recommender.Adapters.CockroachStore` — embedded V-Sekai/cockroach single node
    (driven; item ports).
  * `Recommender.Adapters.VersityBlobStore` — embedded versity/versitygw S3 gateway
    + aria-storage content-defined chunking (driven; blob ports).
  * `Recommender.Adapters.FixtureStore` / `Recommender.Adapters.FixtureBlobStore` —
    in-memory fixture adapters (CI runs without live services).
  * `Recommender.Adapters.CLI` — the driving adapter: the standalone `rfr` binary.

  ## Quick start (library)

      # tokenize embeddings -> 4-token semantic IDs
      ids = Recommender.Core.ResidualFSQ.encode_ids(embeddings)

      # serve recommendations from a trained checkpoint over a catalog
      serve = Recommender.Adapters.Serve.new(defn_params, token_id_list)
      {:ok, item_indices} = Recommender.Adapters.Serve.recommend(serve, [0, 3], 5)
  """
end
