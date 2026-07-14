# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Core.TrainingTest do
  use ExUnit.Case, async: true

  alias Holo.Core.Training

  @vocab 4097

  test "loss_shifted_ce is a finite non-negative scalar" do
    logits = Nx.broadcast(0.0, {2, 8, @vocab})
    labels = Nx.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]], type: {:s, 32})
    loss = Training.loss_shifted_ce(logits, labels)
    assert Nx.shape(loss) == {}
    assert Nx.to_number(loss) >= 0.0
  end

  test "loss_mtp_last_item scores the last 4 positions (one item)" do
    logits = Nx.broadcast(0.0, {1, 8, @vocab})
    labels = Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80]], type: {:s, 32})
    loss = Training.loss_mtp_last_item(logits, labels)
    assert Nx.shape(loss) == {}
    assert Nx.to_number(loss) >= 0.0
  end

  test "loss_mtp_last_item returns 0 when the sequence is shorter than one item" do
    logits = Nx.broadcast(0.0, {1, 3, @vocab})
    labels = Nx.tensor([[1, 2, 3]], type: {:s, 32})
    assert Nx.to_number(Training.loss_mtp_last_item(logits, labels)) == 0.0
  end

  test "encode_aux expands each item to 4 token positions" do
    # 3 items, each with 4 sub-embeddings of dim 192 (one per residual stage)
    item_embeddings = Nx.broadcast(1.0, {3, 4, 192})
    {embeds, mask} = Training.encode_aux([0, 1, -1], item_embeddings, 3)
    # 3 items * 4 tokens = 12 rows of 192
    assert Nx.shape(embeds) == {12, 192}
    assert Nx.shape(mask) == {12, 1}
    # the padded (-1) item is masked to 0 across its 4 token rows
    assert Nx.to_flat_list(mask) |> Enum.take(-4) == [0.0, 0.0, 0.0, 0.0]
  end
end
