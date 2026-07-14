# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Core.ResidualFSQTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Holo.Core.ResidualFSQ
  alias Holo.Core.Memory

  # Deterministic pseudo-random embeddings (no RNG state): a fixed spread.
  defp embeddings(n, d) do
    for i <- 0..(n - 1) do
      for j <- 0..(d - 1), do: :math.sin(i * 0.7 + j * 0.13) + :math.cos(i * 0.31 - j * 0.05)
    end
    |> Nx.tensor(type: {:f, 32})
  end

  describe "encode/2" do
    test "emits {n, 4} stage indices in 0..4095" do
      tokens = ResidualFSQ.encode(embeddings(64, 768))
      assert Nx.shape(tokens) == {64, 4}
      vals = Nx.to_flat_list(tokens)
      assert Enum.all?(vals, &(&1 >= 0 and &1 < 4096))
    end

    test "is deterministic for the same input (reproducible tokenization)" do
      e = embeddings(32, 256)
      assert Nx.to_list(ResidualFSQ.encode(e)) == Nx.to_list(ResidualFSQ.encode(e))
    end

    test "respects a custom num_quantizers" do
      tokens = ResidualFSQ.encode(embeddings(8, 128), num_quantizers: 5)
      assert Nx.shape(tokens) == {8, 5}
    end

    test "honors supplied project_in params over the deterministic fallback" do
      e = embeddings(4, 16)
      kernel = Nx.broadcast(0.0, {16, Holo.Core.FSQ.num_dims()})
      params = %{"project_in" => %{"kernel" => kernel}}
      tokens = ResidualFSQ.encode(e, params: params)
      # a zero projection collapses every item onto the same stage index
      rows = Nx.to_list(tokens)
      assert Enum.uniq(rows) |> length() == 1
    end
  end

  describe "encode_ids/2 feeds Holo.Core.Memory" do
    test "every emitted ID satisfies Memory.valid_id?/1" do
      ids = ResidualFSQ.encode_ids(embeddings(50, 384))
      assert Enum.all?(ids, &Memory.valid_id?/1)
    end

    test "IDs load into a Memory as {item_id, tokens}" do
      ids = ResidualFSQ.encode_ids(embeddings(20, 128))
      items = ids |> Enum.with_index() |> Enum.map(fn {tokens, i} -> {i, tokens} end)
      mem = Memory.add_items(Memory.new(), items)
      assert mem != nil
    end
  end

  describe "item_key/1 mirrors HoloModel.itemKey_injective" do
    property "distinct valid semantic IDs never collide on the packed key" do
      tok = StreamData.integer(0..4095)
      id = StreamData.list_of(tok, length: 4)

      check all(a <- id, b <- id) do
        if a == b do
          assert ResidualFSQ.item_key(a) == ResidualFSQ.item_key(b)
        else
          assert ResidualFSQ.item_key(a) != ResidualFSQ.item_key(b)
        end
      end
    end

    test "packs base-4096 exactly like the Lean model" do
      assert ResidualFSQ.item_key([0, 0, 0, 0]) == 0
      assert ResidualFSQ.item_key([1, 0, 0, 0]) == 1
      assert ResidualFSQ.item_key([0, 1, 0, 0]) == 4096
      assert ResidualFSQ.item_key([0, 0, 1, 0]) == 4096 * 4096
      assert ResidualFSQ.item_key([0, 0, 0, 1]) == 4096 * 4096 * 4096

      assert ResidualFSQ.item_key([4095, 4095, 4095, 4095]) ==
               4095 + 4095 * 4096 + 4095 * 4096 * 4096 + 4095 * 4096 * 4096 * 4096
    end
  end
end
