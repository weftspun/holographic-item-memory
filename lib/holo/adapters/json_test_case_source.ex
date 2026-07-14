# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Adapters.JsonTestCaseSource do
  @moduledoc """
  Adapter realizing `Holo.Ports.TestCaseSource` against a JSON file on disk.

  Accepts either the canonical `{"test_cases": [{"context": [...], "next_item": id}, ...]}` shape
  or a `{"sequences": [[id, ...], ...]}` shape (each sequence's last item becomes `next_item`).
  Returns cases in the shape `Holo.Core.Eval` consumes.
  """

  @behaviour Holo.Ports.TestCaseSource

  @doc "Loads test cases from a JSON file at `path`."
  @impl Holo.Ports.TestCaseSource
  @spec load(String.t()) :: {:ok, [map()]} | {:error, term()}
  def load(path) do
    if File.regular?(path) do
      raw = File.read!(path) |> Jason.decode!()

      cases =
        raw["test_cases"] ||
          (raw["sequences"] || [])
          |> Enum.map(fn seq ->
            seq = List.wrap(seq)

            if seq != [] do
              %{"context" => Enum.drop(seq, -1), "next_item" => List.last(seq)}
            else
              %{"context" => [], "next_item" => 0}
            end
          end)

      {:ok, List.wrap(cases)}
    else
      {:error, "file not found: #{path}"}
    end
  end
end
