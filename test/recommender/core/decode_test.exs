# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Core.DecodeTest do
  use ExUnit.Case, async: true

  alias Recommender.Core.Decode
  alias Recommender.Core.Trie

  @vocab 16
  # 3 catalog items, each a 4-token residual ID
  @items [[1, 2, 3, 10], [4, 5, 6, 11], [7, 8, 9, 12]]

  defp item_tokens, do: Nx.tensor(@items, type: {:s, 32})

  # logits (1, 4, vocab) that make the given item's tokens win at each position
  defp logits_favoring(item_tokens_list) do
    zero = List.duplicate(0.0, @vocab)

    rows =
      item_tokens_list
      |> Enum.with_index()
      |> Enum.map(fn {tok, _pos} -> List.replace_at(zero, tok, 10.0) end)

    Nx.tensor([rows], type: {:f, 32})
  end

  test "lookahead_top_k (MTP) scores the favored 4-token item first" do
    get_logits = fn _ctx -> logits_favoring([4, 5, 6, 11]) end

    assert {:ok, ids} =
             Decode.lookahead_top_k(item_tokens(), [0], 2, get_logits, Nx.BinaryBackend)

    assert hd(ids) == 1
  end

  test "beam_search_top_k_spmd returns valid catalog items under the trie constraint" do
    trie = Trie.build(@items)
    trie_tensors = Trie.to_tensors(trie, @vocab)
    get_logits = fn _ctx -> logits_favoring([7, 8, 9, 12]) end

    assert {:ok, ids} =
             Decode.beam_search_top_k_spmd(
               trie_tensors,
               item_tokens(),
               [0],
               2,
               get_logits,
               Nx.BinaryBackend,
               trie
             )

    assert Enum.all?(ids, &(&1 in [0, 1, 2]))
    assert 2 in ids
  end
end
