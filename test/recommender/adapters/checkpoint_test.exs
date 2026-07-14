# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Adapters.CheckpointTest do
  use ExUnit.Case, async: true

  alias Recommender.Adapters.NpyCheckpointSink, as: Sink
  alias Recommender.Adapters.NpyCheckpointSource, as: Source

  setup do
    dir = Path.join(System.tmp_dir!(), "rfr_ckpt_#{System.unique_integer([:positive])}")
    on_exit(fn -> File.rm_rf(dir) end)
    {:ok, dir: dir}
  end

  test "npy sink -> source round-trips a params map", %{dir: dir} do
    params = %{
      "wte" => Nx.iota({4, 3}, type: {:f, 32}),
      "pred_head.bias" => Nx.tensor([0.5, -1.0, 2.0], type: {:f, 32})
    }

    assert :ok == Sink.write_export(params, dir)
    assert File.regular?(Path.join(dir, "manifest.json"))

    loaded = Source.load_from_export(dir)

    assert Map.keys(loaded) |> Enum.sort() == ["pred_head.bias", "wte"]

    for {k, t} <- params do
      assert Nx.equal(loaded[k], t) |> Nx.all() |> Nx.to_number() == 1
    end
  end

  test "compute_sha256 is deterministic and detects tampering", %{dir: dir} do
    params = %{"a" => Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32})}
    :ok = Sink.write_export(params, dir)

    h1 = Source.get_sha256(dir)
    assert is_binary(h1) and byte_size(h1) == 64
    assert Source.get_sha256(dir) == h1

    # rewrite with different data -> different hash
    :ok = Sink.write_export(%{"a" => Nx.tensor([9.0, 9.0, 9.0], type: {:f, 32})}, dir)
    refute Source.get_sha256(dir) == h1
  end
end
