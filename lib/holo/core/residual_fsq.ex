# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Core.ResidualFSQ do
  @moduledoc """
  Residual FSQ driver: content embeddings → 4-token semantic IDs `[t0, t1, t2, t3]`.

  This is the ADR's core change from the RecGPT reference, which used *grouped*
  FSQ (`FSQEncoder`: reshape a 768-d embedding into 4 disjoint 192-d groups,
  quantize each independently → 4 parallel, equal-weight tokens). Here the tokens
  are **coarse-to-fine residual** stages instead:

      r_0 = project_in(e)
      for s in 0..(num_quantizers-1):
        t_s   = codes_to_indices(quantize(r_s))   # index in 0..4095
        r_{s+1} = r_s - quantize(r_s)             # carry the residual to the next, finer stage

  Each `t_s` is a full 4096-code stage index (`Holo.Core.FSQ`, levels `[8,8,8,8]`),
  and the sequence `[t0, t1, t2, t3]` (`num_quantizers = 4`) is the semantic ID both
  `Holo.Core.Memory` (HRR) and the FuXi-Linear model consume. Because coarse mass
  lands in `t0`, similar items share ID prefixes — the property residual (not
  grouped) FSQ gives, and the one `Holo.Core.Memory.valid_id?/1` and
  `HoloModel.itemKey_injective` certify.

  `project_in` maps the embedding into the 4-dim quantization space. Supply real
  weights via `opts[:params]` (`%{"project_in" => %{"kernel" => k, "bias" => b}}`);
  with no params a deterministic seeded projection is used, so tokenization is
  reproducible without any trained checkpoint (ADR phase-2 substrate check).
  """

  alias Holo.Core.FSQ

  @default_num_quantizers 4
  @proj_seed 0x46_53_51_00

  @doc "Default number of residual stages (tokens per item): 4."
  def default_num_quantizers, do: @default_num_quantizers

  @doc """
  Encode a batch of embeddings `{n, d}` into a token tensor `{n, num_quantizers}`
  of `s32` stage indices in `0..4095`.

  ## Options
    * `:params` — FSQ projection params (`project_in` kernel/bias); default: a
      deterministic seeded projection derived from the embedding dim.
    * `:num_quantizers` — stages / tokens per item (default 3).
  """
  @spec encode(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def encode(embeddings, opts \\ []) do
    num_q = Keyword.get(opts, :num_quantizers, @default_num_quantizers)
    params = Keyword.get(opts, :params)

    r0 = project_in(embeddings, params)
    half_width = Nx.divide(FSQ.levels(), 2)

    {token_cols, _residual} =
      Enum.map_reduce(1..num_q, r0, fn _stage, r ->
        q = FSQ.quantize(r)
        idx = FSQ.codes_to_indices(q)
        # residual in the same (unnormalized) space as `r`: subtract the
        # dequantized lattice point (q rescaled by levels/2).
        recon = Nx.multiply(q, half_width)
        {idx, Nx.subtract(r, recon)}
      end)

    Nx.stack(token_cols, axis: 1)
  end

  @doc """
  Encode embeddings into a list of `[t0, t1, t2, t3]` integer token lists — the form
  `Holo.Core.Memory.add_items/2` consumes.
  """
  @spec encode_ids(Nx.Tensor.t(), keyword()) :: [[non_neg_integer()]]
  def encode_ids(embeddings, opts \\ []) do
    embeddings
    |> encode(opts)
    |> Nx.to_list()
  end

  @doc """
  Pack a semantic ID `[t0, t1, t2, t3]` into a flat key, base-4096 — the Elixir
  mirror of `HoloModel.itemKey t0 t1 t2 t3 = t0 + t1*4096 + t2*4096^2 + t3*4096^3`.
  Injective over valid IDs (`t_i in 0..4095`).
  """
  @spec item_key([non_neg_integer()]) :: non_neg_integer()
  def item_key([t0, t1, t2, t3]),
    do: t0 + t1 * 4096 + t2 * 4096 * 4096 + t3 * 4096 * 4096 * 4096

  # --- projection ------------------------------------------------------------

  defp project_in(embeddings, nil) do
    {_n, d} = Nx.shape(embeddings)
    key = Nx.Random.key(@proj_seed)
    {w, _} = Nx.Random.normal(key, shape: {d, FSQ.num_dims()}, type: {:f, 32})
    w = Nx.divide(w, :math.sqrt(d))
    Nx.dot(embeddings, w)
  end

  defp project_in(embeddings, %{"project_in" => %{"kernel" => kernel} = pin}) do
    out = Nx.dot(embeddings, kernel)
    if bias = pin["bias"], do: Nx.add(out, bias), else: out
  end
end
