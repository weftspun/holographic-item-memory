# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Ports.ItemSink do
  @moduledoc """
  Driven sink port: write items and session transitions outward.

  `*_sink` ports transmit core-side facts to the outside world (the embedded
  CockroachDB adapter in production, the in-memory fixture adapter in CI).
  `state` is the adapter's opaque handle.
  """

  @type state :: term()
  @type item_id :: term()
  @type semantic_id :: [non_neg_integer()]

  @doc "Store (or replace) an item's semantic ID."
  @callback upsert_item(state(), item_id(), semantic_id()) :: :ok

  @doc "Record one observed `prev -> next` session transition."
  @callback record_transition(state(), prev :: item_id(), next :: item_id()) :: :ok
end
