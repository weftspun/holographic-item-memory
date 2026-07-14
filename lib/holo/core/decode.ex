# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Core.Decode do
  @moduledoc """
  Next-item prediction: beam search (default) or Multi-Token Prediction (MTP).

  - **Beam search**: Decodes 4 tokens (one RecGPT item) over 4 steps; the t-th token waits for
    the (t-1)-th (sequential dependency). Uses trie tensors and batch inference on device.
  - **Multi-Token Prediction (MTP)**: The model predicts K tokens at once (e.g. 4 for one item);
    acceleration is embedded in the model weights (no draft model or N-gram cache). One forward
    produces logits for the full 4-token window; we score every catalog item and take top-k.
    Can significantly reduce inference cost vs sequential decoding. Use `:mtp` or `:lookahead` strategy.

  STATIC-style minimal sync: top-k selection is done on GPU (Nx.top_k + gather);
  only the top-k item_ids (and scores) are transferred to host.
  """

  alias Holo.Core.Trie

  @neg_inf -1.0e9
  # Residual semantic IDs are 4 tokens (`[t0,t1,t2,t3]`); the leaf step is the last.
  @tokens_per_item 4
  @leaf_step @tokens_per_item - 1

  @doc """
  Multi-Token Prediction (MTP): one forward for the full 4-token window, then score all items.

  With fixed 4-token semantic IDs, MTP acts as a parallel path-classifier: one forward yields
  logits for t₀…t₃; we do product search (sum of logits at each item's 4 tokens) over valid
  catalog paths and take top-k—no recursive beam. Single-pass generation; top-k via constrained
  search. Same semantic IDs (FSQ) as beam search. Head: one shared LM head at last 4 positions
  (FuxiLinearInferenceDefn / InferenceDefn). Training currently uses shifted CE, not MTP loss;
  see docs/features/65_latency_flow.md § MTP theory alignment. Returns {:ok, [item_id]} or :not_found.
  """
  @spec lookahead_top_k(
          Nx.Tensor.t(),
          Nx.Tensor.t() | [non_neg_integer()],
          pos_integer(),
          (Nx.Tensor.t() -> Nx.Tensor.t()),
          term()
        ) :: {:ok, [non_neg_integer()]} | :not_found
  def lookahead_top_k(
        item_id_to_tokens,
        context_item_ids,
        top_k,
        get_logits_fn,
        backend
      )
      when top_k >= 1 and is_function(get_logits_fn, 1) do
    {context_tokens, _len} = build_context_tokens(item_id_to_tokens, context_item_ids, backend)
    num_items = Nx.axis_size(item_id_to_tokens, 0)

    logits_4 = get_logits_fn.(context_tokens)

    # logits_4: (1, 4, vocab_size) -> (4, vocab_size)
    logits_4 = logits_4 |> Nx.squeeze(axes: [0])

    # Score each item by sum of logits at its 4-token semantic ID. Tokens (num_items, 4) once.
    tokens = Nx.as_type(item_id_to_tokens, {:s, 64})

    score_list =
      for p <- 0..(@tokens_per_item - 1) do
        token_ids_p = Nx.slice_along_axis(tokens, p, 1, axis: 1) |> Nx.new_axis(-1)
        Nx.gather(logits_4[p], token_ids_p) |> Nx.reshape({:auto})
      end

    scores =
      Enum.reduce(Enum.drop(score_list, 1), hd(score_list), fn s, acc -> Nx.add(acc, s) end)

    k = min(top_k, num_items)
    {_top_scores, indices} = Nx.top_k(scores, k: k)
    indices = Nx.backend_copy(indices, Nx.BinaryBackend)
    item_ids = indices |> Nx.to_flat_list() |> Enum.map(&decode_item_id_to_int/1)

    list =
      item_ids
      |> Enum.uniq()
      |> Enum.take(top_k)

    case list do
      [] -> :not_found
      ids -> {:ok, ids}
    end
  end

  @doc """
  Direct all-item scoring (delegates to lookahead_top_k / MTP). Kept for backward compatibility.
  """
  @spec score_all_items_top_k(
          Nx.Tensor.t(),
          Nx.Tensor.t() | [non_neg_integer()],
          pos_integer(),
          (Nx.Tensor.t() -> Nx.Tensor.t()),
          term()
        ) :: {:ok, [non_neg_integer()]} | :not_found
  def score_all_items_top_k(
        item_id_to_tokens,
        context_item_ids,
        top_k,
        get_logits_fn,
        backend
      ),
      do: lookahead_top_k(item_id_to_tokens, context_item_ids, top_k, get_logits_fn, backend)

  @doc """
  SPMD-style beam search: one forward, trie and scoring on device, one sync at end.

  Single-forward decode: get_logits_fn runs one model forward, returns logits for the last 4
  positions (1, 4, vocab_size). Beam search runs over those precomputed logits.

  - `get_logits_fn`: (context_tokens) -> logits_4 with shape (1, 4, vocab_size)
  - `trie`: optional map trie from Trie.build/1; when given, used to resolve item_id from 4-token
    sequence when tensor item_at_leaf returns -1.

  Returns {:ok, [item_id]} or :not_found. Single sync after the 4 steps to get top-k item_ids.
  """
  @spec beam_search_top_k_spmd(
          %{next_state: Nx.Tensor.t(), item_at_leaf: Nx.Tensor.t()},
          Nx.Tensor.t(),
          Nx.Tensor.t() | [non_neg_integer()],
          pos_integer(),
          (Nx.Tensor.t() -> Nx.Tensor.t()),
          term(),
          map() | nil,
          keyword()
        ) :: {:ok, [non_neg_integer()]} | :not_found
  def beam_search_top_k_spmd(
        trie_tensors,
        item_id_to_tokens,
        context_item_ids,
        top_k,
        get_logits_fn,
        backend,
        trie \\ nil,
        opts \\ []
      )
      when is_map(trie_tensors) and top_k >= 1 and is_function(get_logits_fn, 1) do
    {context_tokens, _context_len} =
      build_context_tokens(item_id_to_tokens, context_item_ids, backend)

    {_num_states, vocab_size} = Nx.shape(trie_tensors.next_state)
    next_state = trie_tensors.next_state
    item_at_leaf = trie_tensors.item_at_leaf

    beam_width =
      case Keyword.get(opts, :beam_width_override) ||
             Application.get_env(:holographic_item_memory, :beam_width_override) do
        n when is_integer(n) and n >= 1 -> n
        _ -> max(4, min(top_k + 2, 20))
      end

    constants = Keyword.get(opts, :constants)

    root_state =
      if constants,
        do: constants.root_state,
        else: Nx.tensor([0], type: {:s, 32}) |> Nx.backend_copy(backend)

    {item_ids, beam_scores, prefix_tokens} =
      run_single_forward_beam(
        get_logits_fn,
        context_tokens,
        next_state,
        item_at_leaf,
        root_state,
        backend,
        constants,
        beam_width,
        vocab_size,
        opts
      )

    # Host-side top-k selection. The beam is at most `beam_width` states, so the
    # transfer is tiny; doing top_k + gather on host keeps the three beam tensors
    # and `sort_indices` on a single backend regardless of what backend the
    # forward pass ran on (EXLA logits vs. a `backend`-copied prefix, etc.).
    item_ids = Nx.backend_copy(item_ids, Nx.BinaryBackend)
    beam_scores = Nx.backend_copy(beam_scores, Nx.BinaryBackend)
    prefix_tokens = Nx.backend_copy(prefix_tokens, Nx.BinaryBackend)

    k = min(top_k, Nx.axis_size(item_ids, 0))
    {_top_scores, sort_indices} = Nx.top_k(beam_scores, k: k)
    sort_indices = Nx.new_axis(sort_indices, -1)
    item_ids_slice = Nx.gather(item_ids, sort_indices) |> Nx.reshape({:auto})
    scores_slice = Nx.gather(beam_scores, sort_indices) |> Nx.reshape({:auto})
    prefix_tokens_slice = Nx.gather(prefix_tokens, sort_indices)

    item_ids_list = item_ids_slice |> Nx.to_flat_list() |> Enum.map(&decode_item_id_to_int/1)
    scores_list = scores_slice |> Nx.to_flat_list()

    prefix_tokens_list =
      prefix_tokens_slice
      |> Nx.to_flat_list()
      |> Enum.chunk_every(@tokens_per_item)

    candidates =
      item_ids_list
      |> Enum.zip(scores_list)
      |> Enum.zip(prefix_tokens_list)
      |> Enum.map(fn {{iid, score}, tokens} ->
        iid_int = decode_item_id_to_int(iid)
        iid_resolved = if iid_int >= 0, do: iid_int, else: resolve_item_id(trie, tokens)
        {iid_resolved, score}
      end)
      |> Enum.filter(fn {iid, _} -> iid >= 0 end)
      |> Enum.sort_by(fn {_, s} -> s end, :desc)
      |> Enum.uniq_by(fn {iid, _} -> iid end)
      |> Enum.take(top_k)
      |> Enum.map(fn {iid, _} -> iid end)

    # Final coercion so response never contains Nx.Tensor (e.g. from any code path)
    list = Enum.map(candidates, &decode_item_id_to_int/1)

    case list do
      [] -> :not_found
      ids -> {:ok, ids}
    end
  end

  defp build_context_tokens(item_id_to_tokens, context_item_ids, backend) do
    if is_list(context_item_ids) and context_item_ids == [] do
      pad = Nx.tensor([[0]], type: {:s, 32}) |> Nx.backend_copy(backend)
      {pad, 1}
    else
      context_item_ids =
        if is_list(context_item_ids) do
          Nx.tensor(context_item_ids, type: {:s, 32}) |> Nx.backend_copy(backend)
        else
          context_item_ids
        end

      context_item_ids = Nx.new_axis(context_item_ids, -1)
      ctx = Nx.gather(item_id_to_tokens, context_item_ids, axes: [0]) |> Nx.reshape({:auto})
      len = Nx.size(ctx)
      {Nx.reshape(ctx, {1, len}), len}
    end
  end

  defp decode_item_id_to_int(x) when is_integer(x), do: x

  defp decode_item_id_to_int(%Nx.Tensor{} = t),
    do: t |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_number() |> round()

  defp decode_item_id_to_int(x) when is_number(x), do: round(x)

  defp run_single_forward_beam(
         get_logits_fn,
         context_tokens,
         next_state,
         item_at_leaf,
         root_state,
         backend,
         constants,
         beam_width,
         vocab_size,
         _opts
       ) do
    logits_4 = get_logits_fn.(context_tokens)

    logits_0 = logits_4 |> Nx.slice_along_axis(0, 1, axis: 1) |> Nx.squeeze(axes: [1])
    logits = Nx.reshape(logits_0, {:auto})
    valid = Nx.gather(next_state, root_state) |> Nx.reshape({:auto})
    valid_mask = Nx.greater_equal(valid, 0)

    neg_inf =
      if constants,
        do: constants.neg_inf,
        else: Nx.tensor(@neg_inf, type: Nx.type(logits)) |> Nx.backend_copy(backend)

    scores = Nx.select(valid_mask, logits, neg_inf)
    {top_scores, top_indices} = Nx.top_k(scores, k: beam_width)

    top_token_ids =
      Nx.reshape(top_indices, {:auto}) |> Nx.as_type({:s, 32}) |> Nx.backend_copy(backend)

    new_state_ids = gather_2d(next_state, root_state, top_token_ids, backend)
    new_state_ids = Nx.squeeze(new_state_ids, axes: [1])
    prefix_tokens = Nx.new_axis(top_token_ids, 1)
    beam_scores = Nx.as_type(top_scores, {:f, 32})

    {_state_ids, prefix_tokens, beam_scores, item_ids} =
      Enum.reduce(1..(@tokens_per_item - 1), {new_state_ids, prefix_tokens, beam_scores, nil}, fn step,
                                                                             {state_ids, prefixes,
                                                                              scores, _} ->
        logits_i = logits_4 |> Nx.slice_along_axis(step, 1, axis: 1) |> Nx.squeeze(axes: [1])
        logits_broadcast = Nx.broadcast(logits_i, {beam_width, vocab_size})

        spmd_step_from_logits(
          next_state,
          item_at_leaf,
          logits_broadcast,
          state_ids,
          prefixes,
          scores,
          step,
          beam_width,
          vocab_size,
          backend,
          constants
        )
      end)

    {item_ids, beam_scores, prefix_tokens}
  end

  defp spmd_step_from_logits(
         next_state,
         item_at_leaf,
         logits,
         state_ids,
         prefix_tokens,
         beam_scores,
         step,
         beam_width,
         vocab_size,
         backend,
         constants
       ) do
    state_ids_safe = Nx.max(state_ids, 0) |> Nx.backend_copy(backend)
    idx_2d = Nx.new_axis(state_ids_safe, -1)

    {valid_rows, transition_tensor} =
      if step == @leaf_step do
        {Nx.gather(item_at_leaf, idx_2d) |> Nx.reshape({beam_width, vocab_size}), item_at_leaf}
      else
        {Nx.gather(next_state, idx_2d) |> Nx.reshape({beam_width, vocab_size}), next_state}
      end

    valid_mask = Nx.greater_equal(valid_rows, 0)

    neg_inf =
      if constants,
        do: constants.neg_inf,
        else: Nx.tensor(@neg_inf, type: Nx.type(logits)) |> Nx.backend_copy(backend)

    scores_per_token = Nx.select(valid_mask, logits, neg_inf)
    beam_scores_broadcast = Nx.new_axis(beam_scores, 1)
    scores_per_token = Nx.add(scores_per_token, beam_scores_broadcast)
    flat = Nx.reshape(scores_per_token, {:auto})
    {top_scores, top_flat} = Nx.top_k(flat, k: beam_width)

    vocab_t =
      if constants,
        do: constants.vocab_t,
        else: Nx.tensor(vocab_size, type: {:s, 32}) |> Nx.backend_copy(backend)

    top_flat = Nx.as_type(top_flat, {:s, 32}) |> Nx.backend_copy(backend)
    batch_indices = Nx.quotient(top_flat, vocab_t)
    token_ids = Nx.remainder(top_flat, vocab_t)

    batch_indices_b = Nx.backend_copy(batch_indices, backend)
    state_at_top = Nx.gather(state_ids, Nx.new_axis(batch_indices_b, -1)) |> Nx.reshape({:auto})
    state_at_top_safe = Nx.max(state_at_top, 0) |> Nx.backend_copy(backend)
    token_ids_b = Nx.backend_copy(token_ids, backend)

    new_state_ids =
      if step == @leaf_step do
        gather_2d(item_at_leaf, state_at_top_safe, token_ids_b, backend) |> Nx.squeeze(axes: [1])
      else
        gather_2d(transition_tensor, state_at_top_safe, token_ids_b, backend)
        |> Nx.squeeze(axes: [1])
      end

    prefix_len = step

    old_prefixes =
      Nx.gather(prefix_tokens, Nx.new_axis(batch_indices_b, -1))
      |> Nx.reshape({beam_width, prefix_len})

    new_col = Nx.reshape(token_ids, {beam_width, 1})
    new_prefix_tokens = Nx.concatenate([old_prefixes, new_col], axis: 1)
    item_ids = if step == @leaf_step, do: new_state_ids, else: nil
    {new_state_ids, new_prefix_tokens, top_scores, item_ids}
  end

  defp resolve_item_id(nil, _tokens), do: -1

  defp resolve_item_id(trie, tokens) when length(tokens) == 4 do
    [t0, t1, t2, t3] = Enum.map(tokens, &round/1)

    case Trie.lookup(trie, [t0, t1, t2, t3]) do
      {:ok, id} -> id
      :not_found -> -1
    end
  end

  defp resolve_item_id(_trie, _), do: -1

  defp gather_2d(tensor, row_indices, col_indices, backend) do
    row_2d = Nx.new_axis(row_indices |> Nx.backend_copy(backend), -1)
    rows = Nx.gather(tensor, row_2d)
    rows = if Nx.rank(rows) == 1, do: Nx.reshape(rows, {1, :auto}), else: rows

    rows =
      if Nx.rank(rows) == 3,
        do: Nx.reshape(rows, {Nx.axis_size(rows, 0), Nx.axis_size(rows, 2)}),
        else: rows

    k = Nx.axis_size(col_indices, 0)
    {num_rows, vocab_size} = Nx.shape(rows)
    rows = if num_rows == 1 and k > 1, do: Nx.broadcast(rows, {k, vocab_size}), else: rows
    indices = Nx.reshape(col_indices |> Nx.backend_copy(backend), {k, 1})
    Nx.take_along_axis(rows, indices, axis: 1)
  end
end
