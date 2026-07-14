# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Adapters.ItemText do
  @moduledoc """
  Read per-item text (title / description / category) from a CSV into
  `%{item_id => text}` — the input to `BumblebeeEmbedding` for the content
  embedding. Column layout is configurable so any catalog CSV works without
  guessing a fixed schema.
  """

  NimbleCSV.define(Recommender.Adapters.ItemText.CSV, separator: ",", escape: "\"")

  alias Recommender.Adapters.ItemText.CSV

  @doc """
  `read(path, opts)` → `%{item_id => text}`.

  ## Options
    * `:id_column` — 0-based item-id column (default 0)
    * `:text_columns` — 0-based columns joined (space) into the text (default `[1]`)
    * `:skip_headers` — skip the first row (default `true`)
  """
  @spec read(String.t(), keyword()) :: %{String.t() => String.t()}
  def read(path, opts \\ []) do
    id_col = Keyword.get(opts, :id_column, 0)
    text_cols = Keyword.get(opts, :text_columns, [1])
    skip = Keyword.get(opts, :skip_headers, true)

    path
    |> File.stream!()
    |> CSV.parse_stream(skip_headers: skip)
    |> Enum.reduce(%{}, fn row, acc ->
      id = Enum.at(row, id_col)
      text = text_cols |> Enum.map(&(Enum.at(row, &1) || "")) |> Enum.join(" ") |> String.trim()
      if id in [nil, ""], do: acc, else: Map.put(acc, id, text)
    end)
  end
end
