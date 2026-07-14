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

  @tag timeout: 180_000
  test "retention_scan (rolled Nx.while) matches the unrolled fold" do
    key = Nx.Random.key(11)
    {q, k1} = Nx.Random.normal(key, shape: {2, @seq_len, 4, 32})
    {k, k2} = Nx.Random.normal(k1, shape: {2, @seq_len, 4, 32})
    {v, k3} = Nx.Random.normal(k2, shape: {2, @seq_len, 4, 32})
    {decay_bc, _} = Nx.Random.uniform(k3, 0.1, 0.99, shape: {1, 4, 1, 1})

    {rolled, _s} = FuxiLinearInference.retention_scan(q, k, v, decay_bc)
    ref = unrolled_retention(q, k, v, decay_bc)

    assert Nx.shape(rolled) == {2, @seq_len, 4, 32}
    max_diff = Nx.subtract(rolled, ref) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert max_diff < 1.0e-5
  end

  # Reference: the recurrence written as an explicit Elixir-unrolled fold (the pre-
  # rewrite formulation), used only to pin the rolled `while` to it.
  defp unrolled_retention(q, k, v, decay_bc) do
    {batch, seq_len, n_head, head_dim} = Nx.shape(q)
    zero_s = Nx.broadcast(0.0, {batch, n_head, head_dim, head_dim})

    {out_list, _} =
      Enum.reduce(0..(seq_len - 1), {[], zero_s}, fn t, {acc, s} ->
        k_t = Nx.slice_along_axis(k, t, 1, axis: 1) |> Nx.squeeze(axes: [1]) |> Nx.new_axis(-1)
        v_t = Nx.slice_along_axis(v, t, 1, axis: 1) |> Nx.squeeze(axes: [1]) |> Nx.new_axis(2)
        q_t = Nx.slice_along_axis(q, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        s_new = Nx.add(Nx.multiply(decay_bc, s), Nx.multiply(k_t, v_t))
        q_flat = Nx.reshape(q_t, {batch * n_head, 1, head_dim})
        s_flat = Nx.reshape(s_new, {batch * n_head, head_dim, head_dim})
        out_t = Nx.dot(q_flat, [2], [0], s_flat, [1], [0]) |> Nx.reshape({batch, n_head, head_dim})
        {[out_t | acc], s_new}
      end)

    Enum.reverse(out_list) |> Nx.stack(axis: 1)
  end
end
