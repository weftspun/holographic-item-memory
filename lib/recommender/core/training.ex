# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Core.Training do
  @moduledoc """
  Training: data batching and loss. `build_train_batch/5`, `encode_aux/3`,
  `loss_shifted_ce/2`, `loss_mtp_last_item/2`.

  Ported from the archived elixir-sequential-recommendation (Core.Training) and retargeted from grouped (4 tokens/item,
  vocab 15360) to **residual** (3 tokens/item, vocab 4096 + padding 4096). The
  per-item auxiliary content embeddings are `(num_items, @tokens_per_item, 192)`
  — one 192-d sub-embedding per residual stage. When timestamps are provided,
  builds `all_timestamps` `(batch, seq_len, 8)` for the FuXi Temporal Retention
  channel.
  """

  alias Recommender.Core.FSQ

  # Padding id sits just above the 4096 real codes (model vocab = 4097).
  @padding_id FSQ.codebook()
  @tokens_per_item 4
  @label_ignore -100
  @max_length 256
  @seq_token_capacity 2048
  @channel_t_heads 8

  def build_train_batch(seqs, token_id_list, item_embeddings, batch_indices, timestamps \\ nil) do
    batch_seqs = Enum.map(batch_indices, fn idx -> Enum.at(seqs, idx) end)

    batch_ts =
      if timestamps,
        do: Enum.map(batch_indices, fn idx -> Enum.at(timestamps, idx) end),
        else: nil

    {batch_seq, batch_labels, batch_aux_list, embed_mask_list, timestamp_rows} =
      Enum.reduce(
        Enum.zip(batch_seqs, batch_ts || List.duplicate(nil, length(batch_seqs))),
        {[], [], [], [], []},
        fn {seq, ts}, {seq_acc, label_acc, aux_acc, mask_acc, ts_acc} ->
          seq = if length(seq) > @max_length, do: Enum.take(seq, -@max_length), else: seq

          ts =
            if ts && length(ts) >= length(seq) do
              if length(seq) < length(ts), do: Enum.take(ts, -length(seq)), else: ts
            else
              nil
            end

          token_list =
            for item_id <- seq,
                do: Enum.at(token_id_list, item_id) || List.duplicate(0, @tokens_per_item)

          token_list = List.flatten(token_list)

          id_list =
            token_list ++ List.duplicate(@padding_id, @seq_token_capacity - length(token_list))

          label_list =
            token_list ++ List.duplicate(@label_ignore, @seq_token_capacity - length(token_list))

          aux_list = seq ++ List.duplicate(-1, @max_length - length(seq))
          {batch_aux, mask} = encode_aux(aux_list, item_embeddings, length(token_id_list))

          ts_row =
            if ts && ts != [] do
              t0 = Enum.min(ts)
              normalized = Enum.map(ts, fn t -> (t - t0) / 1.0 end)

              expanded =
                normalized
                |> Enum.flat_map(fn v -> List.duplicate(v, @tokens_per_item) end)
                |> then(fn list ->
                  list ++
                    List.duplicate(List.last(list) || 0.0, @seq_token_capacity - length(list))
                end)
                |> Enum.take(@seq_token_capacity)

              expanded
            else
              nil
            end

          {
            [id_list | seq_acc],
            [label_list | label_acc],
            [batch_aux | aux_acc],
            [mask | mask_acc],
            [ts_row | ts_acc]
          }
        end
      )

    batch_seq = Nx.tensor(batch_seq, type: {:s, 32})
    batch_labels = Nx.tensor(batch_labels, type: {:s, 32})
    batch_aux_embeds = Nx.stack(batch_aux_list)
    embed_mask = Nx.stack(embed_mask_list)

    all_timestamps =
      if Enum.all?(timestamp_rows, & &1) do
        ts_tensor = Nx.tensor(timestamp_rows, type: {:f, 32})
        {b, s} = Nx.shape(ts_tensor)
        Nx.new_axis(ts_tensor, -1) |> Nx.broadcast({b, s, @channel_t_heads})
      else
        nil
      end

    {batch_seq, batch_labels, batch_aux_embeds, embed_mask, all_timestamps}
  end

  def encode_aux(batch_ids, item_embeddings, num_items) do
    embed_mask = Enum.map(batch_ids, fn id -> if id >= 0, do: 1.0, else: 0.0 end)
    embeds_ids = Enum.map(batch_ids, fn id -> if id >= 0 and id < num_items, do: id, else: 0 end)
    n = length(embeds_ids)
    indices = Nx.tensor(embeds_ids, type: {:s, 32}, backend: Nx.BinaryBackend) |> Nx.new_axis(-1)
    batch_embeds = Nx.gather(item_embeddings, indices)
    batch_embeds = Nx.reshape(batch_embeds, {n * @tokens_per_item, 192})
    embed_mask = Enum.flat_map(embed_mask, fn m -> List.duplicate(m, @tokens_per_item) end)
    embed_mask = Nx.tensor(embed_mask, type: {:f, 32}) |> Nx.new_axis(-1)
    {batch_embeds, embed_mask}
  end

  def loss_shifted_ce(logits, labels) do
    {batch, seq_len, vocab} = Nx.shape(logits)
    logits_flat = Nx.reshape(logits, {batch * seq_len, vocab})
    labels_flat = Nx.reshape(labels, {batch * seq_len})
    valid = Nx.not_equal(labels_flat, @label_ignore)
    exp = Nx.exp(logits_flat)
    probs = Nx.divide(exp, Nx.sum(exp, axes: [1]) |> Nx.new_axis(-1))
    log_probs = Nx.log(Nx.add(probs, 1.0e-6))
    one_hot = Nx.equal(Nx.new_axis(labels_flat, 1), Nx.iota({vocab}))
    ce = Nx.negate(Nx.sum(Nx.multiply(log_probs, one_hot), axes: [1]))
    n_valid = Nx.sum(Nx.as_type(valid, {:f, 32})) |> Nx.add(1.0e-6)
    Nx.select(valid, ce, Nx.broadcast(0.0, Nx.shape(ce))) |> Nx.sum() |> Nx.divide(n_valid)
  end

  @doc """
  Multi-Token Prediction loss over the last #{@tokens_per_item} positions (the
  next item's residual `[t0,t1,t2]` tokens). Penalizes errors across all
  #{@tokens_per_item} tokens in the same training step. When `seq_len <
  #{@tokens_per_item}` returns 0.0.
  """
  def loss_mtp_last_item(logits, labels) do
    {_batch, seq_len, vocab} = Nx.shape(logits)

    if seq_len < @tokens_per_item do
      Nx.tensor(0.0, type: {:f, 32})
    else
      start_idx = seq_len - @tokens_per_item
      logits_last = Nx.slice_along_axis(logits, start_idx, @tokens_per_item, axis: 1)
      labels_last = Nx.slice_along_axis(labels, start_idx, @tokens_per_item, axis: 1)
      {batch, _n, _} = Nx.shape(logits_last)
      logits_flat = Nx.reshape(logits_last, {batch * @tokens_per_item, vocab})
      labels_flat = Nx.reshape(labels_last, {batch * @tokens_per_item})
      valid = Nx.not_equal(labels_flat, @label_ignore)
      exp = Nx.exp(logits_flat)
      probs = Nx.divide(exp, Nx.sum(exp, axes: [1]) |> Nx.new_axis(-1))
      log_probs = Nx.log(Nx.add(probs, 1.0e-6))
      one_hot = Nx.equal(Nx.new_axis(labels_flat, 1), Nx.iota({vocab}))
      ce = Nx.negate(Nx.sum(Nx.multiply(log_probs, one_hot), axes: [1]))
      n_valid = Nx.sum(Nx.as_type(valid, {:f, 32})) |> Nx.add(1.0e-6)
      Nx.select(valid, ce, Nx.broadcast(0.0, Nx.shape(ce))) |> Nx.sum() |> Nx.divide(n_valid)
    end
  end
end
