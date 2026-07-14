defmodule Recommender.Core.FuxiLinearInferenceParams do
  @moduledoc """
  Builds defn-friendly param maps (atom keys) for Recommender.Core.FuxiLinearInferenceDefn.

  Takes string-keyed params from NpyCheckpointSource or FuxiLinearInference.init_full_params
  and produces atom-keyed maps suitable for Nx.Defn JIT.
  """

  @n_embd 768
  @vocab_size 4_097
  @n_blocks 4
  @value_dim 128
  @channel_t_heads 8
  @channel_p_dim 32

  @doc """
  Build full params for FuxiLinearInferenceDefn from checkpoint string-key map.

  - `params_map`: from NpyCheckpointSource or FuxiLinearInference.init_full_params
  - `dtype`: default `{:f, 32}`. Use `{:bf, 16}` for BF16.

  Returns atom-keyed map for forward_last_4_logits/4.
  """
  @spec build_defn_params(map(), tuple()) :: map()
  def build_defn_params(params_map, dtype \\ {:f, 32}) do
    base =
      %{}
      |> Map.put(:wte, get_wte(params_map) |> as_dtype(dtype))
      |> Map.put(:ae_linear_weight, get_ae_weight(params_map) |> as_dtype(dtype))
      |> Map.put(:ae_linear_bias, get_ae_bias(params_map) |> as_dtype(dtype))
      |> Map.put(:ae_norm_weight, get_ae_norm_weight(params_map) |> as_dtype(dtype))
      |> Map.put(:ae_norm_bias, get_ae_norm_bias(params_map) |> as_dtype(dtype))
      |> Map.put(:ln_f_weight, get_ln_f_weight(params_map) |> as_dtype(dtype))
      |> Map.put(:ln_f_bias, get_ln_f_bias(params_map) |> as_dtype(dtype))
      |> Map.put(:pred_head_weight, get_pred_head_weight(params_map) |> as_dtype(dtype))
      |> Map.put(:pred_head_bias, get_pred_head_bias(params_map) |> as_dtype(dtype))

    block_params =
      Enum.reduce(0..(@n_blocks - 1), %{}, fn i, acc ->
        Map.merge(acc, block_defn_params(params_map, i, dtype))
      end)

    Map.merge(base, block_params)
  end

  defp as_dtype(tensor, dtype) when is_tuple(dtype), do: Nx.as_type(tensor, dtype)
  defp as_dtype(tensor, _), do: tensor

  defp get_wte(params) do
    wte = params["wte"] || params["gpt2model.wte"] || params["gpt2model.wte.weight"]
    if is_nil(wte), do: raise("missing wte in FuXi params")
    {rows, _} = Nx.shape(wte)
    if rows >= @vocab_size, do: Nx.slice_along_axis(wte, 0, @vocab_size, axis: 0), else: wte
  end

  defp get_ae_weight(params) do
    w =
      params["ae.linear.weight"] || params["ae.weight"] || params["linear_layer.weight"] ||
        raise("missing ae.linear.weight in FuXi params")

    ensure_ae_shape(w, {192, @n_embd})
  end

  defp ensure_ae_shape(tensor, {rows, cols}) do
    {r, c} = Nx.shape(tensor)
    if {r, c} == {rows, cols}, do: tensor, else: Nx.transpose(tensor)
  end

  defp get_ae_bias(params) do
    params["ae.linear.bias"] || params["ae.bias"] ||
      Nx.broadcast(0.0, {@n_embd}) |> Nx.as_type({:f, 32})
  end

  defp get_ae_norm_weight(params) do
    params["ae.norm.weight"] || params["norm_aux.weight"] ||
      Nx.broadcast(1.0, {@n_embd}) |> Nx.as_type({:f, 32})
  end

  defp get_ae_norm_bias(params) do
    params["ae.norm.bias"] || params["norm_aux.bias"] ||
      Nx.broadcast(0.0, {@n_embd}) |> Nx.as_type({:f, 32})
  end

  defp get_ln_f_weight(params) do
    params["ln_f.weight"] || params["gpt2model.ln_f.weight"] ||
      Nx.broadcast(1.0, {@n_embd}) |> Nx.as_type({:f, 32})
  end

  defp get_ln_f_bias(params) do
    params["ln_f.bias"] || params["gpt2model.ln_f.bias"] ||
      Nx.broadcast(0.0, {@n_embd}) |> Nx.as_type({:f, 32})
  end

  defp get_pred_head_weight(params) do
    params["pred_head.weight"] || params["pred_head_weight"] ||
      raise("missing pred_head.weight in FuXi params")
  end

  defp get_pred_head_bias(params) do
    params["pred_head.bias"] || params["pred_head_bias"] ||
      Nx.broadcast(0.0, {@vocab_size}) |> Nx.as_type({:f, 32})
  end

  defp block_defn_params(params, i, dtype) do
    base = "fuxi.block.#{i}."

    %{
      :"block_#{i}_ln_w" => get_param(params, base <> "ln.weight") |> as_dtype(dtype),
      :"block_#{i}_ln_b" => get_param(params, base <> "ln.bias") |> as_dtype(dtype),
      :"block_#{i}_uvqk" => get_param(params, base <> "uvqk") |> as_dtype(dtype),
      :"block_#{i}_ret_gamma" =>
        (params[base <> "retention.gamma"] || ones({4}, dtype)) |> as_dtype(dtype),
      :"block_#{i}_ret_ln_w" =>
        get_param(params, base <> "retention.ln.weight") |> as_dtype(dtype),
      :"block_#{i}_ret_ln_b" => get_param(params, base <> "retention.ln.bias") |> as_dtype(dtype),
      :"block_#{i}_ct_proj_v" =>
        get_param(params, base <> "channel_t.proj_v.weight") |> as_dtype(dtype),
      :"block_#{i}_ct_gamma" =>
        (params[base <> "channel_t.gamma"] || zeros({@channel_t_heads}, dtype)) |> as_dtype(dtype),
      :"block_#{i}_ct_alpha" =>
        (params[base <> "channel_t.alpha"] || ones({1}, dtype)) |> as_dtype(dtype),
      :"block_#{i}_ct_beta" =>
        (params[base <> "channel_t.beta"] || ones({1}, dtype)) |> as_dtype(dtype),
      :"block_#{i}_cp_proj" =>
        (params[base <> "channel_p.proj_p.weight"] || zeros({@n_embd, @value_dim}, dtype))
        |> as_dtype(dtype),
      :"block_#{i}_cp_emb" =>
        (params[base <> "channel_p.emb"] || build_sinusoidal_emb(2048)) |> as_dtype(dtype),
      :"block_#{i}_cp_alpha" =>
        (params[base <> "channel_p.alpha"] || ones({1}, dtype)) |> as_dtype(dtype),
      :"block_#{i}_cp_beta" =>
        (params[base <> "channel_p.beta"] || ones({1}, dtype)) |> as_dtype(dtype),
      :"block_#{i}_mffn_lin0" => get_param(params, base <> "mffn.lin0.weight") |> as_dtype(dtype),
      :"block_#{i}_mffn_lin1" => get_param(params, base <> "mffn.lin1.weight") |> as_dtype(dtype),
      :"block_#{i}_mffn_lin2" => get_param(params, base <> "mffn.lin2.weight") |> as_dtype(dtype),
      :"block_#{i}_mffn_lin3" => get_param(params, base <> "mffn.lin3.weight") |> as_dtype(dtype)
    }
  end

  defp get_param(params, key) do
    t = params[key]
    if is_nil(t), do: raise("missing #{key} in FuXi params")
    t
  end

  defp build_sinusoidal_emb(max_len) do
    half = div(@channel_p_dim, 2)
    theta = Nx.pow(10_000, Nx.negate(Nx.divide(Nx.iota({half}), half)))
    pos = Nx.iota({max_len}, type: {:f, 32}) |> Nx.new_axis(-1)
    Nx.concatenate([Nx.sin(Nx.multiply(pos, theta)), Nx.cos(Nx.multiply(pos, theta))], axis: 1)
  end

  defp zeros(shape, dtype), do: Nx.broadcast(0.0, shape) |> Nx.as_type(dtype)
  defp ones(shape, dtype), do: Nx.broadcast(1.0, shape) |> Nx.as_type(dtype)
end
