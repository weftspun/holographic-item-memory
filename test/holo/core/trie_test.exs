# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Core.TrieTest do
  use ExUnit.Case, async: true

  alias Holo.Core.Trie

  @items [
    [1, 2, 3, 0],
    [1, 2, 3, 1],
    [1, 2, 4, 0],
    [7, 8, 9, 5]
  ]

  test "seq_len is 4 (residual contract)" do
    assert Trie.seq_len() == 4
  end

  test "build + lookup round-trips item ids" do
    trie = Trie.build(@items)
    assert Trie.lookup(trie, [1, 2, 3, 0]) == {:ok, 0}
    assert Trie.lookup(trie, [1, 2, 3, 1]) == {:ok, 1}
    assert Trie.lookup(trie, [7, 8, 9, 5]) == {:ok, 3}
    assert Trie.lookup(trie, [9, 9, 9, 9]) == :not_found
  end

  test "valid_next_tokens enumerates constrained continuations" do
    trie = Trie.build(@items)
    assert Enum.sort(Trie.valid_next_tokens(trie, [])) == [1, 7]
    assert Enum.sort(Trie.valid_next_tokens(trie, [1])) == [2]
    assert Enum.sort(Trie.valid_next_tokens(trie, [1, 2])) == [3, 4]
    assert Enum.sort(Trie.valid_next_tokens(trie, [1, 2, 3])) == [0, 1]
    assert Trie.valid_next_tokens(trie, [1, 9, 9]) == []
  end

  test "build_from_stream matches build" do
    stream = @items |> Enum.with_index() |> Enum.map(fn {toks, i} -> {i, toks} end)
    trie = Trie.build_from_stream(stream)
    assert Trie.lookup(trie, [1, 2, 4, 0]) == {:ok, 2}
  end

  test "to_tensors exports device-friendly transition tensors" do
    trie = Trie.build(@items)
    %{next_state: ns, item_at_leaf: leaf, num_states: n} = Trie.to_tensors(trie, 4096)
    assert n > 0
    assert Nx.shape(ns) == {n, 4096}
    assert Nx.shape(leaf) == {n, 4096}
    assert Nx.reduce_max(leaf) |> Nx.to_number() >= 0
  end
end
