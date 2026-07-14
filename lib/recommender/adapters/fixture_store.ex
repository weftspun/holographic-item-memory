# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Adapters.FixtureStore do
  @moduledoc """
  In-memory fixture adapter for `Recommender.Ports.ItemSource` / `Recommender.Ports.ItemSink`.

  Replaces the embedded CockroachDB in CI and library use, per the hexagonal
  decision record's "recorded-fixture adapters replace live hardware during
  CI testing". State is an `Agent` pid.
  """

  @behaviour Recommender.Ports.ItemSource
  @behaviour Recommender.Ports.ItemSink

  def start_link do
    Agent.start_link(fn -> %{items: %{}, transitions: %{}} end)
  end

  @impl Recommender.Ports.ItemSink
  def upsert_item(agent, item_id, [_, _, _, _] = semantic_id) do
    Agent.update(agent, &put_in(&1, [:items, item_id], semantic_id))
  end

  @impl Recommender.Ports.ItemSink
  def record_transition(agent, prev, next) do
    Agent.update(agent, fn state ->
      update_in(state, [:transitions], &Map.update(&1, {prev, next}, 1, fn n -> n + 1 end))
    end)
  end

  @impl Recommender.Ports.ItemSource
  def list_items(agent, limit) do
    items = agent |> Agent.get(& &1.items) |> Enum.sort()
    if limit, do: Enum.take(items, limit), else: items
  end

  @impl Recommender.Ports.ItemSource
  def list_transitions(agent) do
    agent
    |> Agent.get(& &1.transitions)
    |> Enum.map(fn {{prev, next}, n} -> {prev, next, n} end)
  end
end
