# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Ports.ItemSource do
  @moduledoc """
  Driven source port: read stored items and transition counts inbound.

  The FuXi-Linear recommender (`Holo.Adapters.Serve`) is fed from whatever
  implements this contract — the embedded CockroachDB adapter in production, the
  in-memory fixture adapter in CI. Per the hexagonal decision record
  (`20260610-hexagonal-core-ports-adapters`), `*_source` ports read data
  inbound; `state` is the adapter's opaque handle (a DB connection, an Agent
  pid, …).
  """

  @type state :: term()
  @type item_id :: term()
  @type semantic_id :: [non_neg_integer()]

  @doc "All stored `{item_id, semantic_id}` pairs (optionally limited)."
  @callback list_items(state(), limit :: pos_integer() | nil) :: [{item_id(), semantic_id()}]

  @doc "All observed `{prev, next, count}` transition aggregates."
  @callback list_transitions(state()) :: [{item_id(), item_id(), pos_integer()}]

  @doc """
  Load the catalog for `Holo.Adapters.Serve`: returns `{token_id_list, item_ids}`
  where `token_id_list` is the list of 4-token semantic IDs (index = model item
  index) and `item_ids` is the parallel list of the adapter's item ids, so the
  caller can map model indices back to real ids. Lives with the port so every
  adapter shares one wiring into the model.
  """
  @spec load_catalog(module(), state()) :: {[semantic_id()], [item_id()]}
  def load_catalog(source_mod, state) do
    {item_ids, token_id_list} =
      source_mod.list_items(state, nil)
      |> Enum.unzip()

    {token_id_list, item_ids}
  end
end
