# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Ports.ItemSource do
  @moduledoc """
  Driven source port: read stored items and transition counts inbound.

  The core (`Holo.Core.Memory`) is rebuilt from whatever implements this
  contract — the embedded CockroachDB adapter in production, the in-memory
  fixture adapter in CI. Per the hexagonal decision record
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
  Rebuild a `Holo.Core.Memory` from any source adapter. Lives with the port
  so every adapter shares one wiring into the core.
  """
  @spec load_memory(module(), state(), keyword()) :: Holo.Core.Memory.t()
  def load_memory(source_mod, state, mem_opts \\ []) do
    mem =
      Holo.Core.Memory.new(mem_opts)
      |> Holo.Core.Memory.add_items(source_mod.list_items(state, nil))

    Enum.reduce(source_mod.list_transitions(state), mem, fn {prev, next, n}, acc ->
      Holo.Core.Memory.observe_transition(acc, prev, next, n)
    end)
  end
end
