# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Adapters.TrajectoryConvert do
  @moduledoc """
  License-gated trajectory ingestion: raw (gate-clean) datasets → per-user
  chronological item sequences for training / eval.

  Every ingest path passes through `Holo.Core.Trajectories.LicenseGate`: a
  dataset is read only if its license is on the permissive allowlist **and** it
  is English/Western. Blocked corpora (copyleft / NonCommercial / no-redistribute
  / non-Western) raise `Holo.Core.Trajectories.LicenseError` before any bytes are
  read — there is no override. The chronological train/test split is delegated to
  `Holo.Core.Trajectories.Split`.

  Primary gate-clean corpus: **open-ecommerce** (Amazon purchases, CC0):
  `user = "Survey ResponseID"`, `time = "Order Date"`, `item = "ASIN"`.
  """

  alias Holo.Core.Trajectories.{LicenseGate, Split}

  NimbleCSV.define(Holo.Adapters.TrajectoryConvert.CSV, separator: ",", escape: "\"")
  alias Holo.Adapters.TrajectoryConvert.CSV

  # Column layout of open-ecommerce/amazon-purchases.csv
  @amazon_date_col 0
  @amazon_asin_col 5
  @amazon_user_col 7

  @doc """
  Gate `dataset_name` against the manifest, then read its per-user sequences from
  `root_dir`. Raises `LicenseError` for any non-ingestible dataset.

  Returns a list of `%{"user" => id, "sequence" => [item, ...], "timestamps" => [...]}`.
  """
  @spec sequences(String.t(), String.t()) :: [map()]
  def sequences(dataset_name, root_dir) do
    entry = find_entry!(dataset_name)
    # Hard gate: license allowlist. Raises for blocked datasets.
    LicenseGate.assert_allowed!(entry)

    unless entry["english"] == true do
      raise Holo.Core.Trajectories.LicenseError,
        message: "dataset #{inspect(dataset_name)} is not English/Western; refusing to ingest."
    end

    path = Path.join(root_dir, entry["path"] || dataset_name)
    read(dataset_name, path)
  end

  @doc """
  Gate + read + chronological leave-one-out split in one step.
  Returns `{:ok, train, test}`.
  """
  @spec split(String.t(), String.t()) :: {:ok, [map()], [map()]}
  def split(dataset_name, root_dir) do
    dataset_name
    |> sequences(root_dir)
    |> Split.split_train_test_chrono()
  end

  defp find_entry!(name) do
    LicenseGate.manifest()
    |> Map.fetch!("datasets")
    |> Enum.find(&(&1["name"] == name)) ||
      raise ArgumentError, "unknown dataset #{inspect(name)} (not in the license manifest)"
  end

  # --- per-dataset readers ---------------------------------------------------

  defp read("open-ecommerce", dir), do: read_amazon_csv(Path.join(dir, "amazon-purchases.csv"))

  defp read(name, _dir),
    do: raise(ArgumentError, "no trajectory reader implemented for #{inspect(name)}")

  @doc """
  Parse an Amazon `amazon-purchases.csv` into per-user chronological sequences:
  group by Survey ResponseID, order each user's purchases by Order Date, and emit
  the ASIN sequence. Exposed for direct testing.
  """
  @spec read_amazon_csv(String.t()) :: [map()]
  def read_amazon_csv(csv_path) do
    csv_path
    |> File.stream!()
    |> CSV.parse_stream()
    |> Enum.reduce(%{}, fn row, acc ->
      user = Enum.at(row, @amazon_user_col)
      asin = Enum.at(row, @amazon_asin_col)
      date = Enum.at(row, @amazon_date_col)

      if valid?(user) and valid?(asin) do
        Map.update(acc, user, [{date, asin}], &[{date, asin} | &1])
      else
        acc
      end
    end)
    |> Enum.map(fn {user, rows} ->
      sorted = Enum.sort_by(rows, fn {date, _} -> date end)
      %{
        "user" => user,
        "sequence" => Enum.map(sorted, &elem(&1, 1)),
        "timestamps" => Enum.map(sorted, &elem(&1, 0))
      }
    end)
  end

  defp valid?(s), do: is_binary(s) and s != ""
end
