# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Adapters.ServeTest do
  use ExUnit.Case, async: true

  alias Holo.Adapters.Serve

  # 4 catalog items, distinct 4-token residual IDs.
  @catalog [
    [1, 2, 3, 4],
    [1, 2, 3, 5],
    [10, 11, 12, 13],
    [20, 21, 22, 23]
  ]

  @tag timeout: 120_000
  test "end-to-end recommend returns valid catalog item ids (init params, no checkpoint)" do
    serve = Serve.from_init(@catalog)

    assert {:ok, ids} = Serve.recommend(serve, [0], 2)
    assert length(ids) >= 1
    assert Enum.all?(ids, &(&1 in 0..3))
    assert Enum.uniq(ids) == ids
  end

  @tag timeout: 120_000
  test "recommend_batch maps over contexts" do
    serve = Serve.from_init(@catalog)
    results = Serve.recommend_batch(serve, [[0], [2]], 1)
    assert length(results) == 2
    assert Enum.all?(results, fn r -> match?({:ok, _}, r) end)
  end
end
