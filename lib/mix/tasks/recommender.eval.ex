# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Mix.Tasks.Recommender.Eval do
  @shortdoc "Evaluate a FuXi-Linear checkpoint (Hit@k / MRR) vs a popularity floor"
  @moduledoc """
  Runs `Recommender.Core.Eval` over held-out sessions and prints Hit@k / MRR for the
  model and for a **most-popular** baseline (the floor a useful model must clear).

  By default it evaluates on **synthetic** data with a random-init engine — the
  metrics are ~chance, but it proves the eval harness (test cases → `Serve` →
  metrics → popularity comparison) end to end. Pass `--checkpoint` for a real eval.

  ## Example

      mix recommender.eval --checkpoint data/ckpt --top-k 10

  For a real run, replace the synthetic catalog/sessions with the stored catalog
  (`ItemSource.load_catalog/2`) and a chronological held-out split
  (`Core.Trajectories.Split`).

  ## Options
    * `--checkpoint DIR` — trained FuXi-Linear params (npy export); omit for smoke
    * `--top-k N` — cutoff k (default 10)
    * `--seed N` / `--items N` / `--sessions N` / `--session-len N` — synthetic sizes
  """
  use Mix.Task

  alias Recommender.Adapters.{NpyCheckpointSource, Serve}
  alias Recommender.Core.{Eval, FuxiLinearInferenceParams}

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          checkpoint: :string,
          top_k: :integer,
          seed: :integer,
          items: :integer,
          sessions: :integer,
          session_len: :integer,
          data: :string,
          dataset: :string,
          item_text: :string
        ]
      )

    {:ok, _} = Application.ensure_all_started(:exla)

    seed = opts[:seed] || 0
    k = opts[:top_k] || 10

    {sessions, token_id_list} = build_eval_inputs(opts, seed)

    serve =
      case opts[:checkpoint] do
        nil ->
          Mix.shell().info("no --checkpoint: random-init engine (metrics ~ chance)")
          Serve.from_init(token_id_list)

        dir ->
          dir
          |> NpyCheckpointSource.load_from_export()
          |> FuxiLinearInferenceParams.build_defn_params()
          |> Serve.new(token_id_list)
      end

    # held-out: context = all but the last item, target = the last item
    test_cases =
      for s <- sessions, length(s) >= 2 do
        %{context: Enum.slice(s, 0..-2//1), next_item: List.last(s)}
      end

    model = Eval.evaluate(serve, test_cases, top_k: k)

    # popularity floor: recommend the k globally most-frequent next items, ignoring context
    popular =
      test_cases
      |> Enum.map(& &1.next_item)
      |> Enum.frequencies()
      |> Enum.sort_by(fn {_i, c} -> -c end)
      |> Enum.map(&elem(&1, 0))
      |> Enum.take(k)

    pop = Eval.evaluate(serve, test_cases, top_k: k, recommend_fn: fn _ctx, _k -> {:ok, popular} end)

    Mix.shell().info("catalog=#{length(token_id_list)} cases=#{length(test_cases)} top_k=#{k}")
    Mix.shell().info("model       Hit@1=#{f(model.hit_at_1)} Hit@10=#{f(model.hit_at_10)} MRR=#{f(model.mrr)}")
    Mix.shell().info("popularity  Hit@1=#{f(pop.hit_at_1)} Hit@10=#{f(pop.hit_at_10)} MRR=#{f(pop.mrr)}")
    Mix.shell().info("chance Hit@1 ~ #{f(model.random_hit_at_1)}")
  end

  # Real held-out sessions + catalog IDs from a dataset, or synthetic.
  defp build_eval_inputs(%{data: dir} = opts, _seed) do
    dataset = opts[:dataset] || "open-ecommerce"
    csv = opts[:item_text] || Mix.raise("--data needs --item-text CSV (item_id + text columns)")
    Mix.shell().info("REAL data: #{dataset} @ #{dir}")
    {sessions_idx, token_id_list, _aux, _ids} = Recommender.Adapters.DatasetPrep.build(dataset, dir, csv)
    {sessions_idx, token_id_list}
  end

  defp build_eval_inputs(opts, seed) do
    n_items = opts[:items] || 40
    n_sessions = opts[:sessions] || 40
    slen = opts[:session_len] || 6

    Mix.shell().info(
      "SYNTHETIC smoke — random init (metrics ~ chance); pass --checkpoint (and --data) for a real eval"
    )

    :rand.seed(:exsss, {seed + 1, seed + 2, seed + 3})
    sessions = for _ <- 1..n_sessions, do: for(_ <- 1..slen, do: :rand.uniform(n_items) - 1)
    token_id_list = for _ <- 1..n_items, do: for(_ <- 1..4, do: :rand.uniform(4096) - 1)
    {sessions, token_id_list}
  end

  defp f(x), do: Float.round(x * 1.0, 4) |> Float.to_string()
end
