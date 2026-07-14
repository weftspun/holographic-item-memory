# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Adapters.JsonTestCaseSource do
  @moduledoc """
  Adapter realizing `Recommender.Ports.TestCaseSource` against a JSON file on disk.

  Accepts either the canonical `{"test_cases": [{"context": [...], "next_item": id}, ...]}` shape
  or a `{"sequences": [[id, ...], ...]}` shape (each sequence's last item becomes `next_item`).
  Returns cases in the shape `Recommender.Core.Eval` consumes.
  """

  @behaviour Recommender.Ports.TestCaseSource

  @doc "Loads test cases from a JSON file at `path`."
  @impl Recommender.Ports.TestCaseSource
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
