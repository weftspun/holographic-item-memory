# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Core.Prep do
  @moduledoc """
  Turn raw sessions + per-item content embeddings into the tensors the training
  loop consumes — the bridge between ingestion (`TrajectoryConvert`, item text →
  embeddings) and `AxonTrainer.stream_batches/4`.

  One 768-d embedding per item feeds **both** the semantic ID
  (`ResidualFSQ.encode_ids`) and the aux (`Training.item_aux_embeddings`), so ID
  and aux share one space (the 4×192 = 768 alignment).
  """

  alias Recommender.Core.{ResidualFSQ, Training}

  @doc """
  `prepare(sessions_of_ids, embeddings_by_id, opts)` → `{sessions_idx, token_id_list,
  item_aux, item_ids}`.

    * `sessions_of_ids` — list of sessions, each a list of item ids (any term).
    * `embeddings_by_id` — `%{item_id => {768} f32 tensor}` (the frozen content embedding).

  The catalog order is the sorted unique ids in `embeddings_by_id`; `item_ids[i]`
  is model index `i`. `sessions_idx` remaps each session to those indices (ids with
  no embedding are dropped). `token_id_list` is `[[t0,t1,t2,t3], …]` (index = item)
  and `item_aux` is `{num_items, 4, 192}`.
  """
  @spec prepare([[term()]], %{term() => Nx.Tensor.t()}, keyword()) ::
          {[[non_neg_integer()]], [[non_neg_integer()]], Nx.Tensor.t(), [term()]}
  def prepare(sessions_of_ids, embeddings_by_id, opts \\ []) do
    item_ids = embeddings_by_id |> Map.keys() |> Enum.sort()
    index_of = item_ids |> Enum.with_index() |> Map.new()

    emb = Nx.stack(Enum.map(item_ids, &embeddings_by_id[&1]))

    token_id_list =
      emb
      |> ResidualFSQ.encode_ids(opts)

    item_aux = Training.item_aux_embeddings(emb)

    sessions_idx =
      Enum.map(sessions_of_ids, fn s ->
        s |> Enum.map(&Map.get(index_of, &1)) |> Enum.reject(&is_nil/1)
      end)

    {sessions_idx, token_id_list, item_aux, item_ids}
  end
end
