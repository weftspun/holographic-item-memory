# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Core.FuxiLinearInferenceDefn do
  @moduledoc """
  Defn entry points for FuXi-Linear inference: `forward_last_item_logits/4`.

  Ported from the archived elixir-sequential-recommendation (Core.FuxiLinearInferenceDefn). The linear-attention body
  (retention + channel_t + channel_p + mFFN) is `seq_len`-generic and unchanged;
  the only retarget from the grouped reference is the item head — it emits logits
  for the last **3** positions (residual `[t0,t1,t2]`) instead of 4.

  Interface: `(batch_token_ids, batch_aux, embed_mask, params) -> logits
  (batch, 3, vocab_size)`. Used by single-forward decode when the checkpoint is
  FuXi. `@n_head`, `@channel_t_heads`, etc. are model dims (not tokens/item).
  """

  import Nx.Defn

  @n_embd 768
  @n_head 4
  @head_dim 32
  @value_dim 128
  @attn_dim 384
  @channel_t_heads 8
  @channel_p_dim 32
  # Tokens per item (residual FSQ num_quantizers) — the item head reads the last
  # @tokens_per_item positions.
  @tokens_per_item 4

  @doc """
  Single forward returning logits for the last #{@tokens_per_item} positions
  (one residual-FSQ item = `[t0, t1, t2]`).
  """
  defn forward_last_item_logits(batch_token_ids, batch_aux, embed_mask, params) do
    hidden = forward_hidden(batch_token_ids, batch_aux, embed_mask, params)
    {_batch, seq_len, _embd} = Nx.shape(hidden)
    last_start = seq_len - @tokens_per_item
    last_hidden = Nx.slice_along_axis(hidden, last_start, @tokens_per_item, axis: 1)
    {batch, _, embd} = Nx.shape(last_hidden)
    flat = Nx.reshape(last_hidden, {batch * @tokens_per_item, embd})
    flat_logits = apply_head(flat, params)
    vocab_size = elem(Nx.shape(flat_logits), 1)
    Nx.reshape(flat_logits, {batch, @tokens_per_item, vocab_size})
  end

  defnp forward_hidden(batch_token_ids, batch_aux, embed_mask, params) do
    wte = params[:wte]
    {batch, seq_len} = Nx.shape(batch_token_ids)
    flat_ids = Nx.reshape(batch_token_ids, {batch * seq_len})
    token_embeds = Nx.gather(wte, Nx.new_axis(flat_ids, -1))
    token_embeds = Nx.reshape(token_embeds, {batch, seq_len, @n_embd})
    aux_768 = apply_aux_encoder(batch_aux, embed_mask, params)
    x = Nx.add(token_embeds, aux_768)

    all_timestamps = position_timestamps(batch, seq_len)
    invalid_attn_mask = causal_mask(seq_len)

    h = block_0(x, seq_len, all_timestamps, invalid_attn_mask, params)
    h = block_1(h, seq_len, all_timestamps, invalid_attn_mask, params)
    h = block_2(h, seq_len, all_timestamps, invalid_attn_mask, params)
    h = block_3(h, seq_len, all_timestamps, invalid_attn_mask, params)

    apply_ln_f(h, params)
  end

  defnp apply_aux_encoder(aux_192, mask, params) do
    w = params[:ae_linear_weight]
    b = params[:ae_linear_bias]
    nw = params[:ae_norm_weight]
    nb = params[:ae_norm_bias]
    out = Nx.dot(aux_192, [2], w, [0])
    out = Nx.add(out, Nx.reshape(b, {1, 1, @n_embd}))
    out = layer_norm(out, nw, nb)
    Nx.multiply(out, mask)
  end

  defnp position_timestamps(batch, seq_len) do
    pos = Nx.iota({seq_len}, type: {:f, 32})
    Nx.broadcast(Nx.reshape(pos, {1, seq_len, 1}), {batch, seq_len, @channel_t_heads})
  end

  defnp causal_mask(seq_len) do
    row = Nx.iota({seq_len}, type: {:f, 32}) |> Nx.new_axis(-1)
    col = Nx.iota({seq_len}, type: {:f, 32}) |> Nx.new_axis(0)
    Nx.less_equal(col, row) |> Nx.as_type({:f, 32})
  end

  defnp apply_ln_f(hidden, params) do
    w = params[:ln_f_weight]
    b = params[:ln_f_bias]
    layer_norm(hidden, w, b)
  end

  defnp apply_head(hidden, params) do
    w = params[:pred_head_weight]
    b = params[:pred_head_bias]
    logits = Nx.dot(hidden, [1], w, [0])
    Nx.add(logits, b)
  end

  defnp layer_norm(x, weight, bias) do
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    var = Nx.variance(x, axes: [-1], keep_axes: true)
    x_norm = Nx.divide(Nx.subtract(x, mean), Nx.add(Nx.sqrt(var), 1.0e-6))
    Nx.add(Nx.multiply(x_norm, weight), bias)
  end

  defnp block_0(hidden, seq_len, all_timestamps, invalid_attn_mask, params) do
    block_impl(
      hidden,
      seq_len,
      all_timestamps,
      invalid_attn_mask,
      params[:block_0_ln_w],
      params[:block_0_ln_b],
      params[:block_0_uvqk],
      params[:block_0_ret_gamma],
      params[:block_0_ret_ln_w],
      params[:block_0_ret_ln_b],
      params[:block_0_ct_proj_v],
      params[:block_0_ct_gamma],
      params[:block_0_ct_alpha],
      params[:block_0_ct_beta],
      params[:block_0_cp_proj],
      params[:block_0_cp_emb],
      params[:block_0_cp_alpha],
      params[:block_0_cp_beta],
      params[:block_0_mffn_lin0],
      params[:block_0_mffn_lin1],
      params[:block_0_mffn_lin2],
      params[:block_0_mffn_lin3]
    )
  end

  defnp block_1(hidden, seq_len, all_timestamps, invalid_attn_mask, params) do
    block_impl(
      hidden,
      seq_len,
      all_timestamps,
      invalid_attn_mask,
      params[:block_1_ln_w],
      params[:block_1_ln_b],
      params[:block_1_uvqk],
      params[:block_1_ret_gamma],
      params[:block_1_ret_ln_w],
      params[:block_1_ret_ln_b],
      params[:block_1_ct_proj_v],
      params[:block_1_ct_gamma],
      params[:block_1_ct_alpha],
      params[:block_1_ct_beta],
      params[:block_1_cp_proj],
      params[:block_1_cp_emb],
      params[:block_1_cp_alpha],
      params[:block_1_cp_beta],
      params[:block_1_mffn_lin0],
      params[:block_1_mffn_lin1],
      params[:block_1_mffn_lin2],
      params[:block_1_mffn_lin3]
    )
  end

  defnp block_2(hidden, seq_len, all_timestamps, invalid_attn_mask, params) do
    block_impl(
      hidden,
      seq_len,
      all_timestamps,
      invalid_attn_mask,
      params[:block_2_ln_w],
      params[:block_2_ln_b],
      params[:block_2_uvqk],
      params[:block_2_ret_gamma],
      params[:block_2_ret_ln_w],
      params[:block_2_ret_ln_b],
      params[:block_2_ct_proj_v],
      params[:block_2_ct_gamma],
      params[:block_2_ct_alpha],
      params[:block_2_ct_beta],
      params[:block_2_cp_proj],
      params[:block_2_cp_emb],
      params[:block_2_cp_alpha],
      params[:block_2_cp_beta],
      params[:block_2_mffn_lin0],
      params[:block_2_mffn_lin1],
      params[:block_2_mffn_lin2],
      params[:block_2_mffn_lin3]
    )
  end

  defnp block_3(hidden, seq_len, all_timestamps, invalid_attn_mask, params) do
    block_impl(
      hidden,
      seq_len,
      all_timestamps,
      invalid_attn_mask,
      params[:block_3_ln_w],
      params[:block_3_ln_b],
      params[:block_3_uvqk],
      params[:block_3_ret_gamma],
      params[:block_3_ret_ln_w],
      params[:block_3_ret_ln_b],
      params[:block_3_ct_proj_v],
      params[:block_3_ct_gamma],
      params[:block_3_ct_alpha],
      params[:block_3_ct_beta],
      params[:block_3_cp_proj],
      params[:block_3_cp_emb],
      params[:block_3_cp_alpha],
      params[:block_3_cp_beta],
      params[:block_3_mffn_lin0],
      params[:block_3_mffn_lin1],
      params[:block_3_mffn_lin2],
      params[:block_3_mffn_lin3]
    )
  end

  defnp block_impl(
          hidden,
          seq_len,
          all_timestamps,
          invalid_attn_mask,
          ln_w,
          ln_b,
          uvqk,
          ret_gamma,
          ret_ln_w,
          ret_ln_b,
          ct_proj_v,
          ct_gamma,
          ct_alpha,
          ct_beta,
          cp_proj,
          cp_emb,
          cp_alpha,
          cp_beta,
          mffn_lin0,
          mffn_lin1,
          mffn_lin2,
          mffn_lin3
        ) do
    normed = layer_norm(hidden, ln_w, ln_b)
    mm = Nx.dot(normed, [2], uvqk, [0])
    mm = silu(mm)

    u = Nx.slice_along_axis(mm, 0, @attn_dim, axis: 2)
    q = Nx.slice_along_axis(mm, @attn_dim, @value_dim, axis: 2)
    k = Nx.slice_along_axis(mm, @attn_dim + @value_dim, @value_dim, axis: 2)
    v = Nx.slice_along_axis(mm, @attn_dim + 2 * @value_dim, @value_dim, axis: 2)

    ret_out =
      retention_forward(q, k, v, seq_len, invalid_attn_mask, ret_gamma, ret_ln_w, ret_ln_b)

    ct_out =
      channel_t_forward(
        normed,
        seq_len,
        all_timestamps,
        invalid_attn_mask,
        ct_proj_v,
        ct_gamma,
        ct_alpha,
        ct_beta
      )

    cp_out =
      channel_p_forward(normed, seq_len, invalid_attn_mask, cp_proj, cp_emb, cp_alpha, cp_beta)

    combined = Nx.concatenate([ret_out, ct_out, cp_out], axis: 2)
    attn_out = Nx.multiply(u, combined)
    mffn_out = mffn_forward(attn_out, hidden, mffn_lin0, mffn_lin1, mffn_lin2, mffn_lin3)
    Nx.add(hidden, mffn_out)
  end

  defnp(silu(x), do: Nx.multiply(x, Nx.sigmoid(x)))

  defnp retention_forward(q, k, v, seq_len, invalid_attn_mask, gamma_raw, ret_ln_w, ret_ln_b) do
    {batch, _n, _} = Nx.shape(q)
    q = Nx.reshape(q, {batch, seq_len, @n_head, @head_dim})
    k = Nx.reshape(k, {batch, seq_len, @n_head, @head_dim})
    v = Nx.reshape(v, {batch, seq_len, @n_head, @head_dim})

    gamma = Nx.log(Nx.add(1, Nx.exp(gamma_raw)))
    gamma = Nx.cumulative_sum(gamma, axis: 0)
    gamma = Nx.exp(Nx.negate(gamma))

    q_t = Nx.transpose(q, axes: [0, 2, 1, 3])
    k_t = Nx.transpose(k, axes: [0, 2, 3, 1])
    qk = Nx.dot(q_t, [3], [0, 1], k_t, [2], [0, 1])

    row = Nx.iota({seq_len}, type: {:f, 32}) |> Nx.new_axis(-1)
    col = Nx.iota({seq_len}, type: {:f, 32}) |> Nx.new_axis(0)
    diff = Nx.max(Nx.subtract(row, col), 0)
    gamma_bc = Nx.reshape(gamma, {@n_head, 1, 1})
    diff_bc = Nx.reshape(diff, {1, seq_len, seq_len})
    ts_attn = Nx.exp(Nx.negate(Nx.multiply(gamma_bc, diff_bc)))
    ts_attn = Nx.multiply(ts_attn, Nx.reshape(invalid_attn_mask, {1, seq_len, seq_len}))
    ts_attn = Nx.new_axis(ts_attn, 0)
    qk = Nx.multiply(qk, ts_attn)

    v_t = Nx.transpose(v, axes: [0, 2, 1, 3])
    out = Nx.dot(qk, [3], [0, 1], v_t, [2], [0, 1])
    out = Nx.transpose(out, axes: [0, 2, 1, 3])
    out = Nx.reshape(out, {batch, seq_len, @value_dim})
    layer_norm(out, ret_ln_w, ret_ln_b)
  end

  defnp channel_t_forward(
          normed_x,
          seq_len,
          all_timestamps,
          invalid_attn_mask,
          proj_v,
          gamma_t,
          alpha,
          beta
        ) do
    {batch, _n, _} = Nx.shape(normed_x)
    v = Nx.dot(normed_x, [2], proj_v, [0])

    idx = Nx.iota({@channel_t_heads}, type: {:s, 32})
    intervals = Nx.pow(2.0, Nx.as_type(idx, {:f, 32}))
    scale_factor = Nx.multiply(6.283185307179586, Nx.pow(0.5, Nx.as_type(idx, {:f, 32})))
    gamma = Nx.sigmoid(gamma_t)

    theta =
      Nx.multiply(
        Nx.remainder(all_timestamps, Nx.reshape(intervals, {1, 1, @channel_t_heads})),
        Nx.reshape(scale_factor, {1, 1, @channel_t_heads})
      )

    cos_t = Nx.cos(theta)
    sin_t = Nx.sin(theta)
    k = Nx.concatenate([cos_t, cos_t, sin_t, sin_t], axis: 2)

    q_sin =
      Nx.concatenate(
        [
          Nx.slice(sin_t, [0, 1, 0], [batch, seq_len - 1, @channel_t_heads]),
          Nx.negate(Nx.slice(cos_t, [0, 1, 0], [batch, seq_len - 1, @channel_t_heads]))
        ],
        axis: 2
      )

    q_cos =
      Nx.concatenate(
        [
          Nx.slice(cos_t, [0, 1, 0], [batch, seq_len - 1, @channel_t_heads]),
          Nx.slice(sin_t, [0, 1, 0], [batch, seq_len - 1, @channel_t_heads])
        ],
        axis: 2
      )

    q_part = Nx.concatenate([q_sin, q_cos], axis: 2)
    q_last_idx = max(0, seq_len - 2)
    q_last = Nx.slice(q_part, [0, q_last_idx, 0], [batch, 1, @channel_t_heads * 4])
    q = Nx.concatenate([q_part, q_last], axis: 1)

    interval_diff =
      Nx.clip(
        Nx.subtract(
          Nx.slice(all_timestamps, [0, 1, 0], [batch, seq_len - 1, @channel_t_heads]),
          Nx.slice(all_timestamps, [0, 0, 0], [batch, seq_len - 1, @channel_t_heads])
        ),
        0,
        1.0e9
      )

    interval_diff =
      Nx.concatenate([Nx.broadcast(0.0, {batch, 1, @channel_t_heads}), interval_diff], axis: 1)

    hinterval = Nx.multiply(interval_diff, Nx.reshape(scale_factor, {1, 1, @channel_t_heads}))
    log_decay_pos = Nx.multiply(hinterval, Nx.negate(Nx.log(gamma)))
    log_decay_pos = Nx.concatenate([log_decay_pos, log_decay_pos], axis: 2)

    decay_map = ext_decay_attn_map(log_decay_pos, seq_len)
    decay_map = Nx.multiply(decay_map, Nx.reshape(invalid_attn_mask, {1, seq_len, seq_len}))
    decay_map = Nx.reshape(decay_map, {batch, 16, 1, seq_len, seq_len})
    decay_map = Nx.broadcast(decay_map, {batch, 16, 2, seq_len, seq_len})
    decay_map = Nx.reshape(decay_map, {batch, 32, seq_len, seq_len})

    q_4d = Nx.reshape(q, {batch, seq_len, @channel_t_heads * 4, 1})
    k_4d = Nx.reshape(k, {batch, seq_len, @channel_t_heads * 4, 1})
    q_t = Nx.transpose(q_4d, axes: [0, 2, 1, 3])
    k_t = Nx.transpose(k_4d, axes: [0, 2, 3, 1])
    qk = Nx.dot(q_t, [3], [0, 1], k_t, [2], [0, 1])
    attn_maps = Nx.multiply(qk, decay_map)

    v_per_head = div(@value_dim, @channel_t_heads * 4)
    v_4d = Nx.reshape(v, {batch, seq_len, @channel_t_heads * 4, v_per_head})
    v_4d = Nx.transpose(v_4d, axes: [0, 2, 1, 3])
    out = Nx.dot(attn_maps, [3], [0, 1], v_4d, [2], [0, 1])
    out = Nx.transpose(out, axes: [0, 2, 1, 3])
    out = Nx.reshape(out, {batch, seq_len, @value_dim})

    Nx.add(Nx.multiply(out, alpha), Nx.multiply(v, beta))
  end

  defnp ext_decay_attn_map(log_decay, seq_len) do
    {batch, _n, n_h} = Nx.shape(log_decay)
    ext_log = Nx.concatenate([Nx.broadcast(0.0, {batch, 1, n_h}), log_decay], axis: 1)
    cumsum = Nx.cumulative_sum(ext_log, axis: 1)
    cumsum = Nx.transpose(cumsum, axes: [0, 2, 1])
    cs_j = Nx.slice(cumsum, [0, 0, 1], [batch, n_h, seq_len])
    cs_i = Nx.slice(cumsum, [0, 0, 0], [batch, n_h, seq_len])
    cs_j = Nx.reshape(cs_j, {batch, n_h, 1, seq_len})
    cs_i = Nx.reshape(cs_i, {batch, n_h, seq_len, 1})
    log_map = Nx.max(Nx.subtract(cs_j, cs_i), 0)
    Nx.exp(Nx.negate(log_map))
  end

  defnp channel_p_forward(normed_x, seq_len, invalid_attn_mask, proj_p, emb_full, alpha, beta) do
    {batch, _n, _} = Nx.shape(normed_x)
    v = Nx.dot(normed_x, [2], proj_p, [0])

    emb = Nx.slice(emb_full, [0, 0], [seq_len, @channel_p_dim])

    attn_w = Nx.dot(emb, [1], emb, [1])
    attn_w = Nx.divide(attn_w, 16)
    attn_w = Nx.multiply(attn_w, invalid_attn_mask)
    attn_w = Nx.reshape(attn_w, {1, seq_len, seq_len})
    attn_w = Nx.broadcast(attn_w, {batch, seq_len, seq_len})

    out = Nx.dot(attn_w, [2], [0], v, [1], [0])
    Nx.add(Nx.multiply(out, alpha), Nx.multiply(v, beta))
  end

  defnp mffn_forward(x, x0, lin0, lin1, lin2, lin3) do
    h = Nx.dot(x, [2], lin0, [0])
    h = Nx.add(h, x0)
    normed = rms_norm(h)
    x1 = Nx.multiply(silu(Nx.dot(normed, [2], lin1, [0])), Nx.dot(normed, [2], lin3, [0]))
    Nx.add(Nx.dot(x1, [2], lin2, [0]), h)
  end

  defnp rms_norm(x) do
    rms = Nx.sqrt(Nx.add(Nx.mean(Nx.pow(x, 2), axes: [-1], keep_axes: true), 1.0e-6))
    Nx.divide(x, rms)
  end
end
