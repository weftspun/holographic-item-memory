# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Mix.Tasks.Recommender.Pretrain do
  @shortdoc "Train the FuXi-Linear recommender (synthetic smoke; wire real data for a baseline)"
  @moduledoc """
  Runs the AdamW training loop (`Recommender.Adapters.AxonTrainer.train/3`) from a
  fresh `Recommender.Core.FuxiLinearInference.init_random_params/1` and optionally
  writes the trained checkpoint (`Recommender.Adapters.NpyCheckpointSink`).

  By default it trains on **synthetic** data — random catalog IDs, sessions, and
  aux embeddings — which proves the loop, the JIT step, and checkpoint IO end to
  end. It is not a real model; the loss will fall on the synthetic set but means
  nothing. For a real baseline, replace the three synthetic inputs with:

    * `seqs` — per-user chronological item-index sessions
      (`Recommender.Adapters.TrajectoryConvert.sequences/2`, gate-clean corpora).
    * `token_id_list` — 4-token semantic IDs per item
      (`Recommender.Core.ResidualFSQ.encode_ids/2` over item embeddings).
    * `item_embeddings` — per-item aux features `{num_items, 4, 192}`
      (`Recommender.Adapters.BumblebeeEmbedding`).

  Real training wants a GPU: build XLA with `XLA_TARGET=cuda12x`.

  ## Known limitation — the retention unrolls per position

  `FuxiLinearInference` computes retention with an `Enum.reduce` over `seq_len`
  (an Elixir-level loop), so a batch of `L` token positions unrolls `L` steps into
  the compiled graph, per block. At the padded `@seq_token_capacity` (= 1024) the
  forward+backward graph is too large to compile on CPU (it OOMs). The `train/3`
  loop itself is verified at short sequences; running a real baseline first needs
  the retention (and the linear channels) rewritten as a rolled `Nx.while`/`while`
  so the graph size is independent of sequence length.

  ## Example

      mix recommender.pretrain --out data/ckpt --epochs 3 --lr 3.0e-4

  ## Options
    * `--out DIR`        write the trained checkpoint (npy export)
    * `--epochs N`       passes over the data (default 1)
    * `--lr F`           AdamW learning rate (default 3.0e-4)
    * `--batch-size N`   sessions per batch (default 8)
    * `--seed N`         PRNG seed (default 0)
    * `--items N` / `--sessions N` / `--session-len N`  synthetic sizes
  """
  use Mix.Task

  alias Recommender.Adapters.{AxonTrainer, NpyCheckpointSink}
  alias Recommender.Core.FuxiLinearInference

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          out: :string,
          epochs: :integer,
          lr: :float,
          batch_size: :integer,
          seed: :integer,
          items: :integer,
          sessions: :integer,
          session_len: :integer
        ]
      )

    {:ok, _} = Application.ensure_all_started(:exla)

    seed = opts[:seed] || 0
    n_items = opts[:items] || 40
    n_sessions = opts[:sessions] || 32
    slen = opts[:session_len] || 8

    Mix.shell().info("SYNTHETIC smoke data — not a real model (see `mix help recommender.pretrain`)")

    :rand.seed(:exsss, {seed + 1, seed + 2, seed + 3})
    seqs = for _ <- 1..n_sessions, do: for(_ <- 1..slen, do: :rand.uniform(n_items) - 1)
    token_id_list = for _ <- 1..n_items, do: for(_ <- 1..4, do: :rand.uniform(4096) - 1)

    {item_embeddings, _} =
      Nx.Random.normal(Nx.Random.key(seed), 0.0, 1.0, shape: {n_items, 4, 192}, type: {:f, 32})

    Mix.shell().info("init random params (seed #{seed}), #{n_items} items, #{n_sessions} sessions")
    params = FuxiLinearInference.init_random_params(seed: seed)

    batches =
      AxonTrainer.stream_batches(seqs, token_id_list, item_embeddings,
        batch_size: opts[:batch_size] || 8
      )

    epochs = opts[:epochs] || 1
    lr = opts[:lr] || 3.0e-4
    Mix.shell().info("training: #{epochs} epoch(s), lr #{lr} ...")
    {trained, losses} = AxonTrainer.train(params, batches, epochs: epochs, learning_rate: lr)

    shown = losses |> Enum.map(&fmt/1) |> Enum.join(" -> ")
    Mix.shell().info("loss: #{shown}")

    if out = opts[:out] do
      File.mkdir_p!(out)
      :ok = NpyCheckpointSink.write_export(trained, out)
      Mix.shell().info("checkpoint -> #{out} (#{map_size(trained)} tensors)")
    end
  end

  defp fmt(l) when is_float(l), do: Float.round(l, 4) |> Float.to_string()
  defp fmt(l), do: inspect(l)
end
