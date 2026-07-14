# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Core.EvalTest do
  use ExUnit.Case, async: true

  alias Recommender.Core.Eval

  test "evaluate computes Hit@k and MRR from a supplied recommend_fn" do
    state = %{num_items: 10}

    test_cases = [
      %{"context" => [0], "next_item" => 1},
      %{"context" => [2], "next_item" => 9}
    ]

    # rank the target first for case 1, third for case 2
    recommend_fn = fn
      [0], _k -> {:ok, [1, 5, 6]}
      [2], _k -> {:ok, [7, 8, 9]}
    end

    m = Eval.evaluate(state, test_cases, recommend_fn: recommend_fn, top_k: 10)

    assert m.n == 2
    assert m.hit_at_1 == 0.5
    assert m.hit_at_10 == 1.0
    # MRR = (1/1 + 1/3) / 2
    assert_in_delta m.mrr, (1.0 + 1.0 / 3) / 2, 1.0e-9
    assert m.rejects_null
  end

  test "filter_to_catalog drops out-of-range cases" do
    cases = [
      %{"context" => [0, 1], "next_item" => 2},
      %{"context" => [0], "next_item" => 99},
      %{"context" => [50], "next_item" => 1}
    ]

    kept = Eval.filter_to_catalog(cases, 10)
    assert length(kept) == 1
    assert hd(kept)["next_item"] == 2
  end
end
