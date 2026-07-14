# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Core.PrepTest do
  use ExUnit.Case, async: true

  alias Recommender.Core.{Prep, ResidualFSQ, FuxiLinearInference}
  alias Recommender.Adapters.AxonTrainer

  defp emb(seed), do: Nx.Random.normal(Nx.Random.key(seed), shape: {768}, type: {:f, 32}) |> elem(0)

  test "prepare/2 builds sessions->indices, 4-token IDs, and {N,4,192} aux from 768-d embeds" do
    embeddings = %{"a" => emb(1), "b" => emb(2), "c" => emb(3)}
    sessions = [["a", "b", "c"], ["c", "a"]]

    {sessions_idx, token_id_list, item_aux, item_ids} = Prep.prepare(sessions, embeddings)

    assert item_ids == ["a", "b", "c"]
    assert sessions_idx == [[0, 1, 2], [2, 0]]
    assert length(token_id_list) == 3
    assert Enum.all?(token_id_list, &ResidualFSQ.valid_id?/1)
    assert Nx.shape(item_aux) == {3, 4, 192}
  end

  test "prepare/2 drops session ids with no embedding" do
    embeddings = %{"a" => emb(1), "b" => emb(2)}
    {sessions_idx, _ids, _aux, item_ids} = Prep.prepare([["a", "z", "b"]], embeddings)
    assert item_ids == ["a", "b"]
    assert sessions_idx == [[0, 1]]
  end

  # The whole real-data flow with the IO stubbed by synthetic 768-d embeddings:
  # Prep.prepare -> stream_batches -> train. Proves the data path end-to-end without
  # MPNet or CockroachDB. Heavy (full-model train); run with `--include training`.
  @tag :training
  @tag timeout: 600_000
  test "data flow: prepare -> stream_batches -> train stays finite and descends" do
    embeddings = for i <- 0..14, into: %{}, do: {"item#{i}", emb(i)}
    :rand.seed(:exsss, {1, 2, 3})
    sessions = for _ <- 1..8, do: for(_ <- 1..5, do: "item#{:rand.uniform(15) - 1}")

    {sidx, tids, aux, _ids} = Prep.prepare(sessions, embeddings)
    params = FuxiLinearInference.init_random_params(seed: 5)
    batches = AxonTrainer.stream_batches(sidx, tids, aux, batch_size: 4) |> Enum.to_list()

    {_p, losses} = AxonTrainer.train(params, batches, epochs: 3)
    assert Enum.all?(losses, &is_float/1)
    assert List.last(losses) <= hd(losses)
  end
end
