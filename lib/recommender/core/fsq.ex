# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Core.FSQ do
  @moduledoc """
  Finite Scalar Quantization — **residual** variant, one stage.

  Port of the archived elixir-sequential-recommendation `Core.FSQ` stage math (`bound` / `round_ste` /
  `codes_to_indices`), retargeted from grouped/product FSQ (`levels [8,8,8,6,5]`,
  15 360 codes) to the **residual** contract this repo certifies in
  `formal/RecommenderModel.lean`:

      levels = [8, 8, 8, 8]   basis = [1, 8, 64, 512]   codes/stage = 4096

  A stage quantizes a 4-dimensional vector: each dim is scalar-quantized to 8
  levels, then packed mixed-radix into one index in `0..4095`. This packing is
  exactly `RecommenderModel.stageIndex c0 c1 c2 c3 = c0 + c1*8 + c2*64 + c3*512`, so
  `codes_to_indices` is index-compatible with the Lean `stage_bound` /
  `stage_roundtrip` / `stage_injective` theorems.

  Functions reduce over the **last axis**, so they accept any leading shape
  (`{n, 4}`, `{n, s, 4}`, …). `Recommender.Core.ResidualFSQ` iterates a stage over the
  quantization residual to emit the 3-token semantic ID `[t0, t1, t2]`.
  """

  @level_list [8, 8, 8, 8]
  @num_dims 4
  # cumulative product prefix of @level_list: [1, 8, 8*8, 8*8*8] — matches
  # RecommenderModel.stageIndex's [1, 8, 64, 512] basis.
  @basis [1, 8, 64, 512]
  @codebook 4096

  @doc "Number of scalar dims quantized per stage (length of `levels`)."
  def num_dims, do: @num_dims

  @doc "Codes per stage (product of `levels` = 8^4 = 4096)."
  def codebook, do: @codebook

  @doc "Per-dimension quantization levels `[8, 8, 8, 8]`."
  def levels, do: Nx.tensor(@level_list, type: {:f, 32})

  @doc "Mixed-radix basis `[1, 8, 64, 512]` (Lean `stageIndex` coefficients)."
  def basis, do: Nx.tensor(@basis, type: {:s, 32})

  @doc """
  Squash `z` into the representable per-dim range (the reference's `bound`, verbatim math).
  Even levels get a 0.5 offset so the 8 codes straddle zero symmetrically.
  """
  def bound(z, eps \\ 1.0e-3) do
    levels = levels()
    half_l = Nx.multiply(Nx.subtract(levels, 1), 1 - eps) |> Nx.divide(2)
    zero = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), levels)
    half = Nx.broadcast(Nx.tensor(0.5, type: {:f, 32}), levels)
    offset = Nx.select(Nx.equal(Nx.remainder(levels, 2), 0), half, zero)
    shift = Nx.divide(offset, half_l) |> Nx.tanh()
    Nx.subtract(Nx.multiply(Nx.tanh(Nx.add(z, shift)), half_l), offset)
  end

  @doc "Round with a straight-through estimator (identity forward = `round`)."
  def round_ste(z) do
    zhat = Nx.round(z)
    Nx.add(z, Nx.subtract(zhat, z))
  end

  @doc "Bound then round then normalize to ~`[-1, 1]` (dequantized codes)."
  def quantize(z) do
    bounded = bound(z)
    quantized = round_ste(bounded)
    half_width = Nx.divide(levels(), 2)
    Nx.divide(quantized, half_width)
  end

  @doc "Normalized codes → per-dim integer digit `0..(level-1)`."
  def scale_and_shift(codes) do
    half_width = Nx.divide(levels(), 2)
    Nx.add(Nx.multiply(codes, half_width), half_width)
  end

  @doc "Inverse of `scale_and_shift/1`: integer digits → normalized codes."
  def scale_and_shift_inverse(digits) do
    half_width = Nx.divide(levels(), 2)
    Nx.divide(Nx.subtract(digits, half_width), half_width)
  end

  @doc """
  Normalized codes `(..., 4)` → stage index `(...)` in `0..4095`.

  Equal to `RecommenderModel.stageIndex` on the recovered digits.
  """
  def codes_to_indices(codes) do
    zhat = scale_and_shift(codes)
    raw =
      zhat
      |> Nx.multiply(Nx.as_type(basis(), {:f, 32}))
      |> Nx.sum(axes: [-1])
      |> Nx.round()
      |> Nx.as_type({:s, 32})

    Nx.clip(raw, 0, @codebook - 1)
  end

  @doc """
  Stage index `(...)` → per-dim integer digits `(..., 4)`.

  Inverse of `codes_to_indices ∘ scale_and_shift_inverse`; mirrors
  `RecommenderModel.stage_roundtrip` (`idx % 8`, `idx / 8 % 8`, `idx / 64 % 8`,
  `idx / 512 % 8`).
  """
  def indices_to_digits(indices) do
    idx = Nx.new_axis(Nx.as_type(indices, {:s, 32}), -1)
    Nx.remainder(Nx.quotient(idx, basis()), Nx.as_type(levels(), {:s, 32}))
  end

  @doc """
  Normalize a checkpoint tensor map into the `project_in` params that
  `Recommender.Core.ResidualFSQ.encode/2` consumes (`%{"project_in" => %{"kernel" => k,
  "bias" => b}}`).

  Accepts the VAE/manifest/export key spellings for the residual projection
  (`quantizer.project_in.*`, `fsq.project_in.*`, `project_in/kernel`). A PyTorch
  `Linear(dim, num_dims)` stores its weight as `{num_dims, dim}`; `Nx.dot(e, k)`
  needs `{dim, num_dims}`, so a `{num_dims, dim}` kernel is transposed.
  """
  @spec load_params(map()) :: map()
  def load_params(tensor_map) when is_map(tensor_map) do
    kernel =
      tensor_map["project_in/kernel"] || tensor_map["quantizer.project_in.weight"] ||
        tensor_map["fsq.project_in.weight"]

    bias =
      tensor_map["project_in/bias"] || tensor_map["quantizer.project_in.bias"] ||
        tensor_map["fsq.project_in.bias"]

    kernel = maybe_orient_kernel(kernel)
    %{"project_in" => %{"kernel" => kernel, "bias" => bias}}
  end

  defp maybe_orient_kernel(nil), do: nil

  defp maybe_orient_kernel(w) do
    case Nx.shape(w) do
      {@num_dims, _dim} -> Nx.transpose(w)
      _ -> w
    end
  end
end
