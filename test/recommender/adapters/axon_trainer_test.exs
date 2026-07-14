# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Adapters.AxonTrainerTest do
  use ExUnit.Case, async: true

  alias Recommender.Adapters.AxonTrainer
  alias Recommender.Core.FuxiLinearInference

  @tag timeout: 180_000
  test "loss over a batch is a finite non-negative scalar (FuXi forward + Training losses)" do
    params = FuxiLinearInference.init_full_params()

    seq = 8
    token_ids = Nx.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], type: {:s, 32})
    labels = Nx.tensor([[2, 3, 4, 5, 6, 7, 8, 9]], type: {:s, 32})
    aux = Nx.broadcast(0.0, {1, seq, 192})
    mask = Nx.broadcast(1.0, {1, seq, 1})

    loss = AxonTrainer.loss(params, token_ids, labels, aux, mask)

    assert Nx.shape(loss) == {}
    v = Nx.to_number(loss)
    assert v >= 0.0
    assert v == v  # not NaN
  end

  test "stream_batches builds the 5-tuple training batches" do
    seqs = [[0, 1, 2], [1, 2, 0]]
    token_id_list = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    # per-item aux: 3 items x 4 residual stages x 192
    item_embeddings = Nx.broadcast(0.5, {3, 4, 192})

    [batch | _] =
      AxonTrainer.stream_batches(seqs, token_id_list, item_embeddings, batch_size: 2)
      |> Enum.to_list()

    {token_seq, labels, aux, mask, _ts} = batch
    assert Nx.rank(token_seq) == 2
    assert Nx.rank(labels) == 2
    assert Nx.rank(aux) == 3
    assert Nx.rank(mask) == 3
  end
end
