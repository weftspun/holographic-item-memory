# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Core.Eval do
  @moduledoc """
  Next-item evaluation: Hit@k and MRR vs a standard JSON test set.

  Test format: `{"test_cases": [{"context": [id, ...], "next_item": id}, ...]}`.
  Item IDs are 0-based catalog indices.

  ## Metrics
  - **Hit@k**: fraction where ground-truth next item is in top-k recommendations.
  - **MRR**: mean reciprocal rank (1/rank, 0 if not in list).

  ## Random baseline
  For catalog size N, random Hit@1 ~ 1/N, MRR ~ 1/N.
  """

  @doc """
  Runs evaluation given serve state and test cases (list or stream).

  Returns a map with:
  - `:n` - number of test cases
  - `:hit_at_1`, `:hit_at_5`, `:hit_at_10` - Hit@k (0.0 .. 1.0)
  - `:mrr` - mean reciprocal rank
  - `:catalog_size`, `:random_hit_at_1`, `:rejects_null`
  """
  def evaluate(state, test_cases, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, 10) |> min(20)
    batch_size = Keyword.get(opts, :batch_size, 1) |> max(1)
    progress_interval_sec = Keyword.get(opts, :progress_interval_sec)
    progress_fn = Keyword.get(opts, :progress_fn) || default_progress_fn()
    total = Keyword.get(opts, :total) || if is_list(test_cases), do: length(test_cases), else: nil

    recommend_fn =
      Keyword.get(opts, :recommend_fn) ||
        fn ctx, k -> Recommender.Adapters.Serve.recommend(state, ctx, k) end

    recommend_batch_fn =
      Keyword.get(opts, :recommend_batch_fn) ||
        fn ctx_list, k -> Recommender.Adapters.Serve.recommend_batch(state, ctx_list, k) end

    start_sec = System.monotonic_time(:second)

    iter =
      if batch_size > 1 do
        Stream.chunk_every(test_cases, batch_size)
      else
        test_cases
      end

    result =
      Enum.reduce_while(iter, {0, 0, 0, 0.0, 0, start_sec}, fn element, acc ->
        {h1, h5, h10, rr_sum, n, last_progress_sec} = acc

        metrics_5 =
          if batch_size > 1 do
            update_acc_for_chunk(element, {h1, h5, h10, rr_sum, n}, recommend_batch_fn, top_k)
          else
            update_acc_for_tc(element, {h1, h5, h10, rr_sum, n}, recommend_fn, top_k)
          end

        {nh1, nh5, nh10, nrr_sum, nn} = metrics_5
        now_sec = System.monotonic_time(:second)

        maybe_report_progress =
          if progress_interval_sec && progress_interval_sec > 0 &&
               now_sec - last_progress_sec >= progress_interval_sec do
            elapsed_sec = now_sec - start_sec
            progress_opts = [elapsed_sec: elapsed_sec]
            arity = progress_fn |> :erlang.fun_info() |> Keyword.get(:arity)

            if arity == 4,
              do: progress_fn.(nn, total, metrics_5, progress_opts),
              else: progress_fn.(nn, total, metrics_5)

            now_sec
          else
            last_progress_sec
          end

        {:cont, {nh1, nh5, nh10, nrr_sum, nn, maybe_report_progress}}
      end)

    {h1, h5, h10, rr_sum, n} = final_metrics_tuple(result)
    build_metrics(h1, h5, h10, rr_sum, n, state.num_items)
  end

  @doc """
  Keeps only test cases whose context and next_item are in 0..(num_items - 1).
  """
  def filter_to_catalog(test_cases, num_items) when num_items > 0 do
    Enum.filter(test_cases, fn tc ->
      context = get_tc_context(tc)
      next_item = get_tc_next_item(tc)

      next_item != nil and next_item >= 0 and next_item < num_items and
        Enum.all?(context, fn id -> is_integer(id) and id >= 0 and id < num_items end)
    end)
  end

  def filter_to_catalog(test_cases, _), do: test_cases

  defp get_tc_context(tc) when is_map(tc) do
    (tc["context"] || tc[:context] || []) |> List.wrap()
  end

  defp get_tc_next_item(tc) when is_map(tc) do
    tc["next_item"] || tc[:next_item]
  end

  defp index_of(list, value) do
    Enum.find_index(list, &(&1 == value))
  end

  defp update_acc_for_tc(tc, acc, recommend_fn, top_k) do
    context = get_tc_context(tc)
    next_item = get_tc_next_item(tc)

    if context == [] or next_item == nil do
      acc
    else
      case recommend_fn.(context, top_k) do
        {:ok, preds} -> add_hit_metrics(acc, List.wrap(preds), next_item)
        _ -> acc
      end
    end
  end

  defp update_acc_for_chunk(chunk, acc, recommend_batch_fn, top_k) do
    if chunk == [] do
      acc
    else
      contexts = Enum.map(chunk, &get_tc_context/1)
      results = recommend_batch_fn.(contexts, top_k)

      Enum.reduce(Enum.zip(chunk, results), acc, fn {tc, result}, acc_ ->
        context = get_tc_context(tc)
        next_item = get_tc_next_item(tc)

        if context == [] or next_item == nil do
          acc_
        else
          preds =
            case result do
              {:ok, p} -> List.wrap(p)
              _ -> []
            end

          add_hit_metrics(acc_, preds, next_item)
        end
      end)
    end
  end

  defp add_hit_metrics({acc_h1, acc_h5, acc_h10, acc_rr, acc_n}, preds, next_item) do
    idx = index_of(preds, next_item)
    rr = if idx, do: 1.0 / (idx + 1), else: 0.0
    h1 = if idx != nil and idx < 1, do: 1, else: 0
    h5 = if idx != nil and idx < 5, do: 1, else: 0
    h10 = if idx != nil and idx < 10, do: 1, else: 0
    {acc_h1 + h1, acc_h5 + h5, acc_h10 + h10, acc_rr + rr, acc_n + 1}
  end

  defp final_metrics_tuple(result) do
    # Strip progress timestamp (6th element) if present
    case result do
      {h1, h5, h10, rr_sum, n, _last_sec} -> {h1, h5, h10, rr_sum, n}
      t when tuple_size(t) == 5 -> t
    end
  end

  defp default_progress_fn do
    fn done, total, {h1, _h5, _h10, rr_sum, n}, opts ->
      opts = opts || []
      hit1 = if n > 0, do: Float.round(h1 / n, 4), else: 0.0
      mrr = if n > 0, do: Float.round(rr_sum / n, 4), else: 0.0
      total_str = if total, do: "#{done}/#{total}", else: "#{done}"
      elapsed = Keyword.get(opts, :elapsed_sec, 0)

      rate_str =
        if elapsed > 0 and done > 0, do: "  #{Float.round(done / elapsed, 1)}/s", else: ""

      eta_str =
        if total && total > 0 && done > 0 && elapsed > 0 do
          remaining = total - done
          eta_sec = div(remaining * elapsed, done)

          if eta_sec >= 60,
            do: "  ETA ~ #{div(eta_sec, 60)}m #{rem(eta_sec, 60)}s",
            else: "  ETA ~ #{eta_sec}s"
        else
          ""
        end

      IO.puts("  eval progress: #{total_str}  Hit@1=#{hit1}  MRR=#{mrr}#{rate_str}#{eta_str}")
    end
  end

  defp build_metrics(h1, h5, h10, rr_sum, n, catalog_size) do
    n = max(n, 1)
    random_hit_at_1 = if catalog_size > 0, do: 1.0 / catalog_size, else: 0.0
    hit_at_1 = h1 / n

    %{
      n: n,
      hit_at_1: hit_at_1,
      hit_at_5: h5 / n,
      hit_at_10: h10 / n,
      mrr: rr_sum / n,
      catalog_size: catalog_size,
      random_hit_at_1: random_hit_at_1,
      rejects_null: hit_at_1 > random_hit_at_1
    }
  end
end
