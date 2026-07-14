# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Adapters.JsonTestCaseSourceTest do
  use ExUnit.Case, async: true

  alias Recommender.Adapters.JsonTestCaseSource, as: Src

  setup do
    path = Path.join(System.tmp_dir!(), "rfr_tc_#{System.unique_integer([:positive])}.json")
    on_exit(fn -> File.rm(path) end)
    {:ok, path: path}
  end

  test "loads canonical test_cases form", %{path: path} do
    File.write!(path, Jason.encode!(%{"test_cases" => [%{"context" => [1, 2], "next_item" => 3}]}))
    assert {:ok, [%{"context" => [1, 2], "next_item" => 3}]} = Src.load(path)
  end

  test "derives next_item from sequences form", %{path: path} do
    File.write!(path, Jason.encode!(%{"sequences" => [[1, 2, 3], [7, 8]]}))
    assert {:ok, cases} = Src.load(path)
    assert %{"context" => [1, 2], "next_item" => 3} in cases
    assert %{"context" => [7], "next_item" => 8} in cases
  end

  test "missing file returns an error", %{path: path} do
    assert {:error, _} = Src.load(path <> ".nope")
  end
end
