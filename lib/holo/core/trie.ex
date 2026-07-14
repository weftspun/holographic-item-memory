# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Core.Trie do
  @moduledoc """
  Catalog trie: maps 4-token residual-FSQ semantic IDs to `item_id` for
  constrained decode.

  Ported from RecGPT.Core.Trie over the 4-token residual IDs (`[t0, t1, t2, t3]`,
  each `0..4095`) this repo uses. Built from a `token_id_list` (list of 4-token
  lists, one per catalog item); supports full sequence → `item_id` lookup and
  valid-next-token enumeration at each prefix for constrained beam search.
  """

  @seq_len 4
  # Leaf states sit at depth `@seq_len - 1`; shallower states carry `next_state`.
  @leaf_depth @seq_len - 1

  @doc """
  Build a trie from `token_id_list`. Each element is a 4-token list
  (`0..vocab_size-1`). Returns an opaque nested map; use `lookup/2` and
  `valid_next_tokens/2`.
  """
  @spec build([[non_neg_integer()]]) :: map()
  def build(token_id_list) when is_list(token_id_list) do
    token_id_list
    |> Enum.with_index()
    |> Enum.reduce(%{}, fn {tokens, item_id}, acc -> add_item(acc, item_id, tokens) end)
  end

  @doc "Build a trie from a stream of `{item_id, [t0,t1,t2,t3]}` (constant memory)."
  @spec build_from_stream(Enumerable.t()) :: map()
  def build_from_stream(stream) do
    Enum.reduce(stream, %{}, fn {item_id, tokens}, acc -> add_item(acc, item_id, tokens) end)
  end

  @doc "Add one item (4-token list) to an existing trie."
  @spec add_item(map(), non_neg_integer(), [non_neg_integer(), ...]) :: map()
  def add_item(trie, item_id, [t0, t1, t2, t3])
      when is_integer(t0) and is_integer(t1) and is_integer(t2) and is_integer(t3) do
    put_path(trie, [t0, t1, t2, t3], item_id)
  end

  def add_item(trie, _item_id, _tokens), do: trie

  defp put_path(map, [k], v), do: Map.put(map, k, v)

  defp put_path(map, [k | rest], v) do
    child = Map.get(map, k) || %{}
    Map.put(map, k, put_path(child, rest, v))
  end

  @doc "Lookup `item_id` for a complete 4-token sequence: `{:ok, item_id}` or `:not_found`."
  @spec lookup(map(), [non_neg_integer(), ...]) :: {:ok, non_neg_integer()} | :not_found
  def lookup(trie, [t0, t1, t2, t3]) when is_map(trie) do
    case get_in(trie, [t0, t1, t2, t3]) do
      nil -> :not_found
      item_id when is_integer(item_id) -> {:ok, item_id}
      _ -> :not_found
    end
  end

  def lookup(_trie, _token_list), do: :not_found

  @doc """
  Valid next token IDs that extend `prefix` to some catalog item. `prefix` is 0..3
  tokens (`[]` for the first token, `[t0]` for the second, … `[t0,t1,t2]` for the fourth).
  """
  @spec valid_next_tokens(map(), [] | [non_neg_integer(), ...]) :: [non_neg_integer()]
  def valid_next_tokens(trie, []) when is_map(trie), do: Map.keys(trie)

  def valid_next_tokens(trie, [h | t]) when is_map(trie) do
    case Map.get(trie, h) do
      nil -> []
      next when is_map(next) -> valid_next_tokens(next, t)
      _ -> []
    end
  end

  def valid_next_tokens(_, prefix) when not is_list(prefix), do: []
  def valid_next_tokens(_, _), do: []

  @doc "Number of tokens per item (#{@seq_len})."
  @spec seq_len() :: pos_integer()
  def seq_len, do: @seq_len

  @doc """
  Export trie to device-friendly tensors for SPMD decode (no CPU trie lookups).

  Returns `%{next_state: tensor, item_at_leaf: tensor, num_states: n}`:
  - `next_state` `{num_states, vocab_size}` s32: next state id for `(state, token)` or -1.
  - `item_at_leaf` `{num_states, vocab_size}` s32: item_id when a depth-#{@leaf_depth}
    state's token completes a path, else -1.
  """
  @spec to_tensors(map(), pos_integer()) :: %{
          next_state: Nx.Tensor.t(),
          item_at_leaf: Nx.Tensor.t(),
          num_states: non_neg_integer()
        }
  def to_tensors(trie, vocab_size)
      when is_map(trie) and is_integer(vocab_size) and vocab_size > 0 do
    {num_states, next_updates, leaf_updates} = collect_trie_transitions(trie)
    next_state = build_transition_tensor(num_states, vocab_size, next_updates)
    item_at_leaf = build_transition_tensor(num_states, vocab_size, leaf_updates)
    %{next_state: next_state, item_at_leaf: item_at_leaf, num_states: num_states}
  end

  defp collect_trie_transitions(trie) do
    # BFS over prefix nodes. depth < @leaf_depth -> next_state; depth == @leaf_depth -> item_at_leaf.
    queue = [{0, trie, 0}]

    {_queue, next_id, next_updates, leaf_updates} =
      Enum.reduce_while(Stream.cycle([:ok]), {queue, 1, %{}, %{}}, fn _, acc ->
        {queue, next_id, next_up, leaf_up} = acc

        if queue == [] do
          {:halt, {[], next_id, next_up, leaf_up}}
        else
          [{state_id, node, depth} | rest] = queue
          node = node || %{}
          keys = Map.keys(node) |> Enum.sort()

          {queue2, next_id2, next_up2, leaf_up2} =
            Enum.reduce(keys, {rest, next_id, next_up, leaf_up}, fn t, {q, nid, nup, lup} ->
              value = node[t]

              cond do
                depth < @leaf_depth and is_map(value) ->
                  nup_new = Map.update(nup, state_id, %{t => nid}, fn m -> Map.put(m, t, nid) end)
                  {q ++ [{nid, value, depth + 1}], nid + 1, nup_new, lup}

                depth == @leaf_depth and is_integer(value) ->
                  lup_new =
                    Map.update(lup, state_id, %{t => value}, fn m -> Map.put(m, t, value) end)

                  {q, nid, nup, lup_new}

                true ->
                  {q, nid, nup, lup}
              end
            end)

          {:cont, {queue2, next_id2, next_up2, leaf_up2}}
        end
      end)

    {next_id, next_updates, leaf_updates}
  end

  defp build_transition_tensor(num_states, vocab_size, updates) do
    rows =
      for s <- 0..(num_states - 1) do
        row_map = Map.get(updates, s, %{})
        for t <- 0..(vocab_size - 1), do: Map.get(row_map, t, -1)
      end

    Nx.tensor(rows, type: {:s, 32})
  end
end
