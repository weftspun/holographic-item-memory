# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Adapters.ItemTextTest do
  use ExUnit.Case, async: true

  alias Recommender.Adapters.ItemText

  setup do
    path = Path.join(System.tmp_dir!(), "rfr_items_#{System.unique_integer([:positive])}.csv")
    File.write!(path, "item_id,title,category\n1,\"Iron Sword\",weapon\n2,Round Shield,armor\n")
    on_exit(fn -> File.rm(path) end)
    {:ok, path: path}
  end

  test "reads item_id -> text with default columns", %{path: path} do
    assert ItemText.read(path) == %{"1" => "Iron Sword", "2" => "Round Shield"}
  end

  test "joins multiple text columns", %{path: path} do
    got = ItemText.read(path, text_columns: [1, 2])
    assert got == %{"1" => "Iron Sword weapon", "2" => "Round Shield armor"}
  end
end
