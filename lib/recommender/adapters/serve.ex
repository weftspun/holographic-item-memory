# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Adapters.Serve do
  @moduledoc """
  Next-item recommendation engine: wires the FuXi-Linear forward pass, the
  constrained trie decode, and the catalog into `recommend/3` + `recommend_batch/3`
  (the shape `Recommender.Core.Eval` consumes).

  Ported/retargeted from the archived elixir-sequential-recommendation (Adapters.Serve) to the residual 4-token contract.
  Runs on EXLA by default (`config/config.exs`); pass `backend: Nx.BinaryBackend`
  to `new/3` to force the pure-Nx path. `new/3` builds serve state from a
  defn-ready params map (`Recommender.Core.FuxiLinearInferenceParams.build_defn_params/2`)
  and a `token_id_list` (catalog of 4-token semantic IDs). `from_init/2` builds a
  smoke-test engine from random init params — no checkpoint required.

  For GPU, build XLA with `XLA_TARGET=cuda12x` and add a `:cuda` EXLA client, then
  pass `backend: EXLA.Backend` (or leave the config default).
  """

  alias Recommender.Core.{Decode, Trie, FuxiLinearInference, FuxiLinearInferenceParams}
  alias Recommender.Core.FuxiLinearInferenceDefn, as: Defn

  @vocab_size 4097
  @aux_dim 192

  defstruct [
    :defn_params,
    :trie,
    :trie_tensors,
    :token_id_list,
    :item_id_to_tokens,
    :num_items,
    :get_logits_fn,
    :backend
  ]

  @type t :: %__MODULE__{}

  @doc """
  Build serve state from defn-ready params and a `token_id_list` (list of 4-token
  IDs, one per catalog item, index = item_id).
  """
  @spec new(map(), [[non_neg_integer()]], keyword()) :: t()
  def new(defn_params, token_id_list, opts \\ []) do
    backend = Keyword.get(opts, :backend, EXLA.Backend)
    num_items = length(token_id_list)
    trie = Trie.build(token_id_list)
    trie_tensors = Trie.to_tensors(trie, @vocab_size)
    item_id_to_tokens = Nx.tensor(token_id_list, type: {:s, 32}) |> Nx.backend_copy(backend)

    %__MODULE__{
      defn_params: defn_params,
      trie: trie,
      trie_tensors: trie_tensors,
      token_id_list: token_id_list,
      item_id_to_tokens: item_id_to_tokens,
      num_items: num_items,
      get_logits_fn: build_get_logits_fn(defn_params),
      backend: backend
    }
  end

  @doc "Smoke-test engine over `token_id_list` using random init params (no checkpoint)."
  @spec from_init([[non_neg_integer()]], keyword()) :: t()
  def from_init(token_id_list, opts \\ []) do
    FuxiLinearInference.init_full_params()
    |> FuxiLinearInferenceParams.build_defn_params()
    |> new(token_id_list, opts)
  end

  @doc """
  Recommend the top-`k` next items for a session `context_item_ids` (item indices).
  Returns `{:ok, [item_id]}` or `:not_found`.
  """
  @spec recommend(t(), [non_neg_integer()], pos_integer()) ::
          {:ok, [non_neg_integer()]} | :not_found
  def recommend(%__MODULE__{} = s, context_item_ids, k) do
    Decode.beam_search_top_k_spmd(
      s.trie_tensors,
      s.item_id_to_tokens,
      context_item_ids,
      k,
      s.get_logits_fn,
      s.backend,
      s.trie
    )
  end

  @doc "Batch form of `recommend/3` (one result per context)."
  @spec recommend_batch(t(), [[non_neg_integer()]], pos_integer()) ::
          [{:ok, [non_neg_integer()]} | :not_found]
  def recommend_batch(%__MODULE__{} = s, contexts, k) do
    Enum.map(contexts, &recommend(s, &1, k))
  end

  defp build_get_logits_fn(defn_params) do
    fn context_tokens ->
      {_batch, seq_len} = Nx.shape(context_tokens)
      aux = Nx.broadcast(0.0, {1, seq_len, @aux_dim})
      mask = Nx.broadcast(1.0, {1, seq_len, 1})
      Defn.forward_last_item_logits(context_tokens, aux, mask, defn_params)
    end
  end
end
