# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Adapters.AxonTrainerTest do
  use ExUnit.Case, async: true

  alias Recommender.Adapters.AxonTrainer
  alias Recommender.Core.FuxiLinearInference

  # A single (batch=1, seq=8) example. aux defaults to random (non-degenerate).
  defp batch(opts \\ []) do
    seq = 8
    tok = Nx.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], type: {:s, 32})
    lbl = Nx.tensor([[2, 3, 4, 5, 6, 7, 8, 9]], type: {:s, 32})

    aux =
      case Keyword.get(opts, :aux, :random) do
        :zeros -> Nx.broadcast(0.0, {1, seq, 192})
        :random -> Nx.Random.normal(Nx.Random.key(7), 0.0, 1.0, shape: {1, seq, 192}) |> elem(0)
      end

    mask = Nx.broadcast(1.0, {1, seq, 1})
    {tok, lbl, aux, mask, nil}
  end

  @tag timeout: 180_000
  test "loss is a finite non-negative scalar (not the :nan atom)" do
    params = FuxiLinearInference.init_random_params(seed: 1)
    {tok, lbl, aux, mask, _} = batch()

    loss = AxonTrainer.loss(params, tok, lbl, aux, mask)
    assert Nx.shape(loss) == {}

    v = Nx.to_number(loss)
    # Nx.to_number returns the :nan atom for NaN, and `:nan >= 0.0` is true in
    # Elixir (atoms sort above numbers) — so assert it is actually a float.
    assert is_float(v)
    assert v >= 0.0
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

  # Heavy: full-model forward+backward JIT. Run with `mix test --include training`.
  @tag :training
  @tag timeout: 600_000
  test "train/3 drives the loss down when overfitting one batch" do
    params = FuxiLinearInference.init_random_params(seed: 2)
    {_trained, losses} = AxonTrainer.train(params, [batch()], epochs: 10, learning_rate: 1.0e-3)

    assert length(losses) == 10
    assert Enum.all?(losses, &is_float/1)
    assert List.last(losses) < hd(losses)
  end

  # Regression for the LayerNorm degenerate-input fix: constant (all-zero) aux has
  # zero variance; `sqrt(var + eps)` keeps the gradient finite where `sqrt(var)+eps`
  # would NaN. Training on it must stay finite.
  @tag :training
  @tag timeout: 600_000
  test "degenerate all-zero aux does not NaN the gradient" do
    params = FuxiLinearInference.init_random_params(seed: 3)
    {_trained, losses} = AxonTrainer.train(params, [batch(aux: :zeros)], epochs: 5)

    assert Enum.all?(losses, &is_float/1)
  end
end
