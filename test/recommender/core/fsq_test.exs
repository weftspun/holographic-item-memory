# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Core.FSQTest do
  use ExUnit.Case, async: true

  alias Recommender.Core.FSQ

  # Pure-Elixir mirror of RecommenderModel.stageIndex (basis [1,8,64,512]).
  defp stage_index([c0, c1, c2, c3]), do: c0 + c1 * 8 + c2 * 64 + c3 * 512

  @all_digits for d0 <- 0..7, d1 <- 0..7, d2 <- 0..7, d3 <- 0..7, do: [d0, d1, d2, d3]

  describe "constants match the residual contract / RecommenderModel.lean" do
    test "levels [8,8,8,8], basis [1,8,64,512], codebook 4096" do
      assert Nx.to_flat_list(FSQ.levels()) == [8.0, 8.0, 8.0, 8.0]
      assert Nx.to_flat_list(FSQ.basis()) == [1, 8, 64, 512]
      assert FSQ.codebook() == 4096
      assert FSQ.num_dims() == 4
    end
  end

  describe "bound/2" do
    test "keeps values within the representable per-dim range" do
      z = Nx.tensor([[10.0, -10.0, 0.0, 3.3], [-1.0, 1.0, 2.0, -2.0]])
      out = FSQ.bound(z)
      assert Nx.shape(out) == {2, 4}
      # levels 8 -> half_l ~= 3.5, offset 0.5 -> bounded in ~(-4, 3)
      assert Nx.all(Nx.less_equal(Nx.abs(out), 4.5)) |> Nx.to_number() == 1
    end
  end

  describe "codes_to_indices / stage_roundtrip / stage_injective (exhaustive)" do
    test "every one of the 4096 digit tuples packs to RecommenderModel.stageIndex" do
      digits = Nx.tensor(@all_digits, type: {:f, 32})
      idx = digits |> FSQ.scale_and_shift_inverse() |> FSQ.codes_to_indices()
      expected = Enum.map(@all_digits, &stage_index/1)
      assert Nx.to_flat_list(idx) == expected
    end

    test "stage_bound: all indices land in 0..4095" do
      digits = Nx.tensor(@all_digits, type: {:f, 32})
      idx = digits |> FSQ.scale_and_shift_inverse() |> FSQ.codes_to_indices()
      vals = Nx.to_flat_list(idx)
      assert Enum.all?(vals, &(&1 >= 0 and &1 < 4096))
    end

    test "stage_injective: the 4096 tuples map bijectively onto 0..4095" do
      digits = Nx.tensor(@all_digits, type: {:f, 32})
      idx = digits |> FSQ.scale_and_shift_inverse() |> FSQ.codes_to_indices()
      assert idx |> Nx.to_flat_list() |> Enum.sort() == Enum.to_list(0..4095)
    end

    test "stage_roundtrip: indices_to_digits recovers the original digits" do
      digits = Nx.tensor(@all_digits, type: {:f, 32})
      idx = digits |> FSQ.scale_and_shift_inverse() |> FSQ.codes_to_indices()
      back = FSQ.indices_to_digits(idx)
      assert Nx.to_flat_list(back) == List.flatten(@all_digits)
    end
  end

  describe "quantize/1" do
    test "returns normalized dequantized codes" do
      z = Nx.tensor([[0.3, -0.7, 1.5, -2.0]])
      out = FSQ.quantize(z)
      assert Nx.shape(out) == {1, 4}
      assert Nx.all(Nx.less_equal(Nx.abs(out), 1.1)) |> Nx.to_number() == 1
    end
  end
end
