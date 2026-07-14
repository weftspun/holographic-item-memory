# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Core.FuxiLinearInferenceTest do
  use ExUnit.Case, async: true

  alias Recommender.Core.FuxiLinearInference
  alias Recommender.Core.FuxiLinearInferenceParams, as: Params
  alias Recommender.Core.FuxiLinearInferenceDefn, as: Defn

  @seq_len 6

  # The linear-attention forward JITs on the pure-Nx binary backend (no EXLA/Torchx
  # here), which is slow under concurrent load — allow headroom.
  @tag timeout: 180_000
  test "forward_last_item_logits emits (batch, 4, 4097) over the residual vocab" do
    params = FuxiLinearInference.init_full_params() |> Params.build_defn_params()

    # a batch of two sequences of `@seq_len` residual tokens (0..4095)
    token_ids = Nx.tensor([[1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60]], type: {:s, 32})
    aux = Nx.broadcast(0.0, {2, @seq_len, 192})
    mask = Nx.broadcast(1.0, {2, @seq_len, 1})

    logits = Defn.forward_last_item_logits(token_ids, aux, mask, params)

    assert Nx.shape(logits) == {2, 4, 4097}
    # finite logits, no NaN/Inf leaking through the linear-attention body
    assert Nx.all(Nx.is_infinity(logits) |> Nx.logical_not()) |> Nx.to_number() == 1
    assert Nx.all(Nx.is_nan(logits) |> Nx.logical_not()) |> Nx.to_number() == 1
  end
end
