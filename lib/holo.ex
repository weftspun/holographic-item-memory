defmodule Holo do
  @moduledoc """
  Generative next-item recommender over ResidualFSQ semantic IDs — hexagonal
  layout per `20260610-hexagonal-core-ports-adapters`.

  ## core/ (pure — no sockets, no devices, no frameworks)

  * `Holo.Core.FSQ` / `Holo.Core.ResidualFSQ` — the residual Finite Scalar
    Quantization codec: content embeddings → 4-token semantic IDs
    (`[t0,t1,t2,t3]`, each in `0..4095`), and the ID contract
    (`valid_id?/1`, `tokens_per_item/0`, `codebook_size/0`) every consumer
    shares. Certified in `formal/HoloModel.lean`.
  * `Holo.Core.FuxiLinearInference*` / `Holo.Core.{Decode,Trie,Training,Eval}`
    — the FuXi-Linear (linear-attention) generative recommender: forward pass,
    trie-constrained beam decode, training losses, metrics.

  ## ports/ (contracts the core is wired through; see `ports/sibling_repos.txt`)

  * `Holo.Ports.ItemSource` / `Holo.Ports.ItemSink` — read/write items and
    session transitions; `ItemSource.load_catalog/2` feeds `Holo.Adapters.Serve`.
  * `Holo.Ports.BlobSource` / `Holo.Ports.BlobSink` — read/write
    content-addressed blobs.
  * `Holo.Ports.{CheckpointSource,CheckpointSink,FsqParamsSource,EmbeddingSource}`
    — model checkpoint / params / embedding IO.

  ## adapters/ (concrete I/O at the edges)

  * `Holo.Adapters.Serve` — wires the FuXi-Linear forward pass + constrained
    decode + catalog into `recommend/3`.
  * `Holo.Adapters.CockroachStore` — embedded V-Sekai/cockroach single node
    (driven; item ports).
  * `Holo.Adapters.VersityBlobStore` — embedded versity/versitygw S3 gateway
    + aria-storage content-defined chunking (driven; blob ports).
  * `Holo.Adapters.FixtureStore` / `Holo.Adapters.FixtureBlobStore` —
    in-memory fixture adapters (CI runs without live services).
  * `Holo.Adapters.CLI` — the driving adapter: the standalone `holo` binary.

  ## Quick start (library)

      # tokenize embeddings -> 4-token semantic IDs
      ids = Holo.Core.ResidualFSQ.encode_ids(embeddings)

      # serve recommendations from a trained checkpoint over a catalog
      serve = Holo.Adapters.Serve.new(defn_params, token_id_list)
      {:ok, item_indices} = Holo.Adapters.Serve.recommend(serve, [0, 3], 5)
  """
end
