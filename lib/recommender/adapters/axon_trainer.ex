# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Adapters.AxonTrainer do
  @moduledoc """
  Training glue for the FuXi-Linear recommender.

  Rewired from the archived elixir-sequential-recommendation (Adapters.AxonTrainer), which drove the *non-FuXi*
  `Core.Inference` baseline — this repo is decoder-only FuXi-Linear, so the
  forward pass here is `Recommender.Core.FuxiLinearInference.forward_full_sequence/5`
  (full-sequence logits `(batch, seq, 4097)`) and the objective is
  `Recommender.Core.Training.loss_shifted_ce/2` plus a Multi-Token-Prediction term
  `loss_mtp_last_item/2` over the next item's 4 residual tokens.

  `logits/4` + `loss/6` are the pure loss pieces; `stream_batches/4` builds batches
  via `Recommender.Core.Training.build_train_batch/5`; `train/3` is the AdamW
  gradient-descent loop (jitted `value_and_grad` + optimizer update per batch).
  It runs under EXLA (the configured default backend + defn compiler); build XLA
  with `XLA_TARGET=cuda12x` for a GPU host.

  Start from `FuxiLinearInference.init_random_params/1` for training from scratch —
  `init_full_params/1` is a deterministic *smoke* init (huge logits) that does not
  descend.
  """

  alias Recommender.Core.{FuxiLinearInference, Training}

  @default_mtp_weight 0.5
  @default_learning_rate 3.0e-4

  @doc "Full-sequence training logits `(batch, seq, 4097)` from string-keyed params."
  @spec logits(map(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def logits(params, token_ids, aux, mask) do
    FuxiLinearInference.forward_full_sequence(token_ids, aux, mask, params)
  end

  @doc """
  Combined training loss: shifted cross-entropy + `mtp_weight` × MTP-last-item.
  Returns a scalar tensor. `mtp_weight` defaults to #{@default_mtp_weight}.
  """
  @spec loss(map(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          Nx.Tensor.t()
  def loss(params, token_ids, labels, aux, mask, opts \\ []) do
    w = Keyword.get(opts, :mtp_weight, @default_mtp_weight)
    lg = logits(params, token_ids, aux, mask)
    ce = Training.loss_shifted_ce(lg, labels)
    mtp = Training.loss_mtp_last_item(lg, labels)
    Nx.add(ce, Nx.multiply(w, mtp))
  end

  @doc """
  Stream training batches over `seqs`. Each element is the 5-tuple
  `Recommender.Core.Training.build_train_batch/5` returns.

  Options: `:batch_size` (default 8), `:timestamps` (per-seq timestamp lists).
  """
  @spec stream_batches([[non_neg_integer()]], [[non_neg_integer()]], Nx.Tensor.t(), keyword()) ::
          Enumerable.t()
  def stream_batches(seqs, token_id_list, item_embeddings, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 8)
    timestamps = Keyword.get(opts, :timestamps)
    n = length(seqs)

    0..(max(n, 1) - 1)
    |> Enum.chunk_every(batch_size)
    |> Stream.map(fn idxs ->
      Training.build_train_batch(seqs, token_id_list, item_embeddings, idxs, timestamps)
    end)
  end

  @doc """
  Train `params` on `batches` with AdamW; returns `{trained_params, losses}` where
  `losses` is one loss per batch per epoch (in order).

  `batches` is an enumerable of `{token_seq, labels, aux, mask, _timestamps}` tuples,
  e.g. from `stream_batches/4`. Each step is a single **jitted** `value_and_grad` +
  optimizer update, with the batch tensors passed as arguments (never closures), so
  it compiles once per distinct batch shape rather than every step.

  Options: `:learning_rate` (default #{@default_learning_rate}), `:epochs`
  (default 1), `:mtp_weight` (default #{@default_mtp_weight}).
  """
  @spec train(map(), Enumerable.t(), keyword()) :: {map(), [float()]}
  def train(params, batches, opts \\ []) do
    lr = Keyword.get(opts, :learning_rate, @default_learning_rate)
    w = Keyword.get(opts, :mtp_weight, @default_mtp_weight)
    epochs = Keyword.get(opts, :epochs, 1)

    {init_opt, update_opt} = Polaris.Optimizers.adamw(learning_rate: lr)

    step = fn p, tok, lbl, aux, mask, opt_state ->
      {loss, grads} =
        Nx.Defn.value_and_grad(p, fn pp -> loss(pp, tok, lbl, aux, mask, mtp_weight: w) end)

      {updates, opt_state} = update_opt.(grads, opt_state, p)
      {Polaris.Updates.apply_updates(p, updates), opt_state, loss}
    end

    batches = Enum.to_list(batches)

    {trained, _opt_state, losses} =
      Enum.reduce(1..epochs, {params, init_opt.(params), []}, fn _epoch, acc ->
        Enum.reduce(batches, acc, fn {tok, lbl, aux, mask, _ts}, {p, s, hist} ->
          {p, s, loss} = Nx.Defn.jit_apply(step, [p, tok, lbl, aux, mask, s])
          {p, s, [Nx.to_number(loss) | hist]}
        end)
      end)

    {trained, Enum.reverse(losses)}
  end
end
