# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Core.Trajectories.Split do
  @moduledoc """
  Pure trajectory-splitting domain logic (no file or device I/O).

  Ported from the archived elixir-sequential-recommendation (Core.Trajectories.Split). The chronological train/test split
  can be property-tested against in-memory fixtures, independent of dataset
  parsing. The I/O adapter `Recommender.Adapters.TrajectoryConvert` reads raw
  (license-gated) datasets and delegates the split here.
  """

  @doc "Normalizes a sequence entry (bare list or `%{\"sequence\" => list}`) to its item list."
  @spec seq_from(term()) :: [term()]
  def seq_from(s) when is_list(s), do: s
  def seq_from(%{"sequence" => s}) when is_list(s), do: s
  def seq_from(_), do: []

  @doc """
  Chrono split: per-user temporal train/test split (leave-one-out).

  For each sequence sorted by timestamp:
  - Take the final item as the test target (leave-one-out)
  - Everything before it becomes the train sequence
  - The final item becomes a test context→next_item case

  If a sequence is too short (2 or fewer), the entire sequence goes to train.
  """
  @spec split_train_test_chrono([map()], float()) :: {:ok, [map()], [map()]}
  def split_train_test_chrono(sequences, _test_ratio \\ 0.2) do
    {train_acc, test_acc} =
      Enum.reduce(sequences, {[], []}, fn seq, {tr, te} ->
        seq_list = seq_from(seq)
        timestamps = Map.get(seq, "timestamps", [])
        n = length(seq_list)

        if n <= 2 do
          {[seq | tr], te}
        else
          train_len = n - 1
          train_ids = Enum.take(seq_list, train_len)
          next_item = List.last(seq_list)

          test_case = %{
            "context" => train_ids,
            "next_item" => next_item
          }

          train_seq =
            if timestamps != [] and length(timestamps) == n do
              tr_ts = Enum.take(timestamps, train_len)
              %{"sequence" => train_ids, "timestamps" => tr_ts}
            else
              %{"sequence" => train_ids}
            end

          {[train_seq | tr], [test_case | te]}
        end
      end)

    {:ok, Enum.reverse(train_acc), Enum.reverse(test_acc)}
  end
end
