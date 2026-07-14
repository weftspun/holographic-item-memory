# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Core.Trajectories.SplitTest do
  use ExUnit.Case, async: true

  alias Holo.Core.Trajectories.Split

  test "seq_from normalizes list and map forms" do
    assert Split.seq_from([1, 2, 3]) == [1, 2, 3]
    assert Split.seq_from(%{"sequence" => [4, 5]}) == [4, 5]
    assert Split.seq_from(:garbage) == []
  end

  test "leave-one-out chrono split holds out the final item per user" do
    seqs = [
      %{"sequence" => [10, 11, 12, 13], "timestamps" => [1, 2, 3, 4]},
      %{"sequence" => [20, 21]}
    ]

    {:ok, train, test} = Split.split_train_test_chrono(seqs)

    # long sequence: last item held out, rest is train
    assert %{"context" => [10, 11, 12], "next_item" => 13} in test
    assert %{"sequence" => [10, 11, 12], "timestamps" => [1, 2, 3]} in train
    # short sequence (<= 2) stays entirely in train, produces no test case
    assert %{"sequence" => [20, 21]} in train
    assert length(test) == 1
  end
end
