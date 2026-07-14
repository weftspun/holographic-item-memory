# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Mix.Tasks.Recommender.Bench do
  @shortdoc "Measure the durations behind the pretrain estimate (calibration probes)"
  @moduledoc """
  Operationalizes the `calibrate-pretrain-estimate` plan
  (`priv/plans/domains/calibrate-pretrain-estimate.jsonld`): prints the real per-unit
  costs so the pretrain estimate can be recomputed from measurements rather than guesses.

  Always measured (no data / no download):
    * **tokenize throughput** — `ResidualFSQ.encode_ids` items/s
    * **train-step time** — `AxonTrainer.train` steady s/step at `(batch, seq=1024)`

  Run it on the **GPU host** for the number that decides the training budget:

      XLA_TARGET=cuda12x mix recommender.bench           # real GPU s/step

  From `s/step` and the dataset size, training ≈ `ceil(sessions/batch) * epochs * s_per_step`.

  ## Optional probes
    * `--embed` — download MPNet and time embedding 200 texts → embed items/s (probe 3)
    * `--data DIR` — count sessions + catalog items via `TrajectoryConvert` (probe 2)

  ## Options
    * `--batch N` (default 8) · `--seq N` (default 1024) · `--n N` (tokenize sample, default 5000)
  """
  use Mix.Task

  alias Recommender.Core.{ResidualFSQ, FuxiLinearInference}
  alias Recommender.Adapters.AxonTrainer

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          batch: :integer,
          seq: :integer,
          n: :integer,
          embed: :boolean,
          data: :string,
          dataset: :string
        ]
      )

    {:ok, _} = Application.ensure_all_started(:exla)
    b = opts[:batch] || 8
    seq = opts[:seq] || 1024
    n = opts[:n] || 5000

    Mix.shell().info("backend: #{inspect(Nx.default_backend())}")

    # --- tokenize throughput ---
    {emb, _} = Nx.Random.normal(Nx.Random.key(1), shape: {n, 768}, type: {:f, 32})
    {t_us, _} = :timer.tc(fn -> ResidualFSQ.encode_ids(emb) end)
    tps = round(n / (t_us / 1_000_000))
    Mix.shell().info("tokenize: #{n} items in #{sec(t_us)}s -> #{tps} items/s")

    # --- train step (compile then steady) ---
    params = FuxiLinearInference.init_random_params(seed: 1)
    tok = Nx.iota({b, seq}, type: {:s, 32}) |> Nx.remainder(4096)
    lbl = Nx.iota({b, seq}, type: {:s, 32}) |> Nx.add(1) |> Nx.remainder(4096)
    {aux, _} = Nx.Random.normal(Nx.Random.key(2), shape: {b, seq, 192}, type: {:f, 32})
    mask = Nx.broadcast(1.0, {b, seq, 1})
    batch = {tok, lbl, aux, mask, nil}

    {c_us, _} = :timer.tc(fn -> AxonTrainer.train(params, [batch], epochs: 1) end)
    {s_us, _} = :timer.tc(fn -> AxonTrainer.train(params, [batch, batch, batch], epochs: 1) end)
    steady = (s_us - c_us) / 2 / 1_000_000
    Mix.shell().info("train (b=#{b}, seq=#{seq}): compile+1=#{sec(c_us)}s ; steady=#{Float.round(steady, 2)}s/step")
    Mix.shell().info("  -> training ~= ceil(sessions/#{b}) * epochs * #{Float.round(steady, 2)}s (plug in real sessions/epochs)")

    if opts[:embed], do: bench_embed()
    if d = opts[:data], do: bench_data(opts[:dataset] || "open-ecommerce", d)
  end

  defp bench_embed do
    Mix.shell().info("embed: loading MPNet (first run downloads ~420MB)...")
    texts = for i <- 0..199, into: %{}, do: {i, "sample item title number #{i} in a catalog"}

    {e_us, embeds} =
      :timer.tc(fn -> Recommender.Adapters.BumblebeeEmbedding.encode_item_text_dict(texts) end)

    dim = Nx.shape(embeds) |> elem(1)
    Mix.shell().info("embed: 200 items in #{sec(e_us)}s -> #{round(200 / (e_us / 1_000_000))} items/s, dim=#{dim}")
  end

  defp bench_data(dataset, dir) do
    seqs = Recommender.Adapters.TrajectoryConvert.sequences(dataset, dir)
    items = seqs |> List.flatten() |> Enum.uniq() |> length()
    Mix.shell().info("dataset (#{dataset} @ #{dir}): #{length(seqs)} sessions, #{items} distinct items")
  rescue
    e -> Mix.shell().info("dataset: could not read #{dataset} @ #{dir} — #{Exception.message(e)}")
  end

  defp sec(us), do: Float.round(us / 1_000_000, 2)
end
