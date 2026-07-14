# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Adapters.DatasetPrep do
  @moduledoc """
  Assemble the training/eval catalog from a raw dataset — the IO orchestration
  around `Recommender.Core.Prep`:

      TrajectoryConvert.sequences  (gate-clean sessions)
      + ItemText.read              (item_id -> text)
      + BumblebeeEmbedding         (text -> one 768-d embedding per item)
      -> Prep.prepare              ({sessions_idx, token_id_list, {N,4,192} aux, item_ids})

  Each leaf (`TrajectoryConvert`, `ItemText`, `Prep`) is unit-tested; this is the
  thin glue that needs the real dataset + MPNet at run time.
  """

  alias Recommender.Adapters.{TrajectoryConvert, ItemText, BumblebeeEmbedding}
  alias Recommender.Core.Prep

  @doc """
  `build(dataset, dir, item_text_csv, opts)` → `{sessions_idx, token_id_list,
  item_aux, item_ids}`. `opts` are forwarded to `ItemText.read/2` (column layout)
  and `Prep.prepare/3`.
  """
  @spec build(String.t(), String.t(), String.t(), keyword()) ::
          {[[non_neg_integer()]], [[non_neg_integer()]], Nx.Tensor.t(), [term()]}
  def build(dataset, dir, item_text_csv, opts \\ []) do
    sessions = TrajectoryConvert.sequences(dataset, dir)
    text_by_id = ItemText.read(item_text_csv, opts)

    ids =
      sessions
      |> List.flatten()
      |> Enum.uniq()
      |> Enum.filter(&Map.has_key?(text_by_id, &1))
      |> Enum.sort()

    texts = ids |> Enum.with_index() |> Map.new(fn {id, i} -> {i, text_by_id[id]} end)
    emb = BumblebeeEmbedding.encode_item_text_dict(texts)
    emb_by_id = ids |> Enum.with_index() |> Map.new(fn {id, i} -> {id, emb[i]} end)

    Prep.prepare(sessions, emb_by_id, opts)
  end
end
