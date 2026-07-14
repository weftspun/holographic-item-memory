defmodule Holo do
  @moduledoc """
  Holographic item memory over ResidualFSQ semantic IDs — hexagonal layout
  per `20260610-hexagonal-core-ports-adapters`.

  ## core/ (pure — no sockets, no devices, no frameworks)

  * `Holo.Core.HRR` — phase-vector algebra (bind / unbind / bundle /
    similarity), SHA-256-deterministic atoms.
  * `Holo.Core.Memory` — the zero-shot next-item recommender over semantic
    IDs (3 ResidualFSQ tokens in `0..4095`, the `multimodal-semantic-ids`
    contract).

  ## ports/ (contracts the core is wired through; see `ports/sibling_repos.txt`)

  * `Holo.Ports.ItemSource` / `Holo.Ports.ItemSink` — read/write items and
    session transitions.
  * `Holo.Ports.BlobSource` / `Holo.Ports.BlobSink` — read/write
    content-addressed blobs.

  ## adapters/ (concrete I/O at the edges)

  * `Holo.Adapters.CockroachStore` — embedded V-Sekai/cockroach single node
    (driven; item ports).
  * `Holo.Adapters.VersityBlobStore` — embedded versity/versitygw S3 gateway
    + aria-storage content-defined chunking (driven; blob ports).
  * `Holo.Adapters.FixtureStore` / `Holo.Adapters.FixtureBlobStore` —
    in-memory fixture adapters (CI runs without live services).
  * `Holo.Adapters.CLI` — the driving adapter: the standalone `holo` binary.

  ## Quick start (library, no services)

      mem =
        Holo.Core.Memory.new(dim: 4096)
        |> Holo.Core.Memory.add_items([{"sword", [17, 900, 3]}, {"shield", [17, 901, 44]}])

      {:ok, recs} = Holo.Core.Memory.recommend(mem, ["sword"], top_k: 5)
  """
end
