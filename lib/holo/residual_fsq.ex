defmodule Holo.ResidualFSQ do
  @moduledoc """
  Residual Finite Scalar Quantization codec for semantic IDs.

  Matches the `multimodal-semantic-ids` contract
  (`20260713-multimodal-residual-fsq-semantic-ids.md`): one ResidualFSQ with
  `levels = [8, 8, 8, 8]` and `num_quantizers = 3` maps a fused, standardized
  multimodal vector to a per-asset semantic ID of **3 tokens**, each token an
  index in `0..4095` (`8⁴ = 4096` codes per stage).

  Two layers live here:

    * **Index codec** (pure integer arithmetic) — `codes_to_index/1` /
      `index_to_codes/1` convert between a stage's 4 quantization digits and
      its single token index via the mixed-radix basis `[1, 8, 64, 512]`.
      These are the facts certified in `formal/HoloModel.lean` (`omega`
      proofs: bound, roundtrip, injectivity).

    * **Quantizer** (`Nx`) — `encode/1` runs the residual loop over a 4-dim
      latent: each stage FSQ-quantizes the running residual and subtracts its
      reconstruction; `decode/1` sums the stage code vectors back into the
      reconstructed latent. The FSQ grid is fixed (no learned parameters);
      learned projections into the 4-dim latent live upstream in the Python
      encoder pipeline.
  """

  @levels [8, 8, 8, 8]
  @basis [1, 8, 64, 512]
  @num_quantizers 3
  @codebook_size 4096
  @dim length(@levels)
  @eps 1.0e-3

  def levels, do: @levels
  def basis, do: @basis
  def num_quantizers, do: @num_quantizers
  def codebook_size, do: @codebook_size
  def dim, do: @dim

  @doc "Tokens per item (= number of residual quantizer stages)."
  def tokens_per_item, do: @num_quantizers

  ## Index codec (integers; mirrored by the Lean model)

  @doc """
  Mixed-radix token index for one stage's digits `[c0, c1, c2, c3]`
  (each `0..7`): `c0 + 8·c1 + 64·c2 + 512·c3`, in `0..4095`.
  """
  @spec codes_to_index([non_neg_integer()]) :: non_neg_integer()
  def codes_to_index(digits) when length(digits) == @dim do
    Enum.zip(digits, @basis)
    |> Enum.map(fn {c, b} ->
      unless is_integer(c) and c >= 0 and c < 8, do: raise(ArgumentError, "digit out of range")
      c * b
    end)
    |> Enum.sum()
  end

  @doc "Inverse of `codes_to_index/1`: digit `i` is `(index / basis_i) % 8`."
  @spec index_to_codes(non_neg_integer()) :: [non_neg_integer()]
  def index_to_codes(index) when index >= 0 and index < @codebook_size do
    Enum.map(@basis, fn b -> index |> div(b) |> rem(8) end)
  end

  ## FSQ grid (Nx, fixed — no learned parameters)

  @doc """
  Quantize a 4-dim latent tensor onto the FSQ grid for one stage.

  Returns `{normalized_codes, digits}`: the centered code vector in
  `[-1, 1]` (steps of `1/4`) used for residual subtraction / reconstruction,
  and the non-centered digits (each `0..7`).
  """
  @spec quantize_stage(Nx.Tensor.t()) :: {Nx.Tensor.t(), [non_neg_integer()]}
  def quantize_stage(z) do
    levels = Nx.tensor(@levels, type: {:f, 64})

    half_l =
      Nx.multiply(Nx.subtract(levels, 1), Nx.tensor(1.0 - @eps, type: {:f, 64}))
      |> Nx.divide(2)

    # All levels are even (8), so the grid offset is 0.5 per dimension.
    offset = Nx.broadcast(Nx.tensor(0.5, type: {:f, 64}), levels)
    shift = Nx.divide(offset, half_l) |> Nx.tanh()

    bounded =
      z
      |> Nx.as_type({:f, 64})
      |> Nx.add(shift)
      |> Nx.tanh()
      |> Nx.multiply(half_l)
      |> Nx.subtract(offset)

    quantized = Nx.round(bounded)
    half_width = Nx.divide(levels, 2)
    normalized = Nx.divide(quantized, half_width)

    digits =
      quantized
      |> Nx.add(half_width)
      |> Nx.as_type({:s, 32})
      |> Nx.to_flat_list()

    {normalized, digits}
  end

  @doc """
  Encode a 4-dim latent into a semantic ID: `#{@num_quantizers}` token indices.

  Residual loop: each stage quantizes the running residual and subtracts its
  reconstruction. Deterministic — the same latent always yields the same ID.
  """
  @spec encode(Nx.Tensor.t() | [number()]) :: [non_neg_integer()]
  def encode(latent) do
    z = to_latent_tensor(latent)

    {_residual, tokens} =
      Enum.reduce(1..@num_quantizers, {z, []}, fn _stage, {residual, acc} ->
        {normalized, digits} = quantize_stage(residual)
        {Nx.subtract(residual, normalized), [codes_to_index(digits) | acc]}
      end)

    Enum.reverse(tokens)
  end

  @doc """
  Decode a semantic ID back to the reconstructed 4-dim latent: the sum of each
  stage's centered code vector.
  """
  @spec decode([non_neg_integer()]) :: Nx.Tensor.t()
  def decode(tokens) when length(tokens) == @num_quantizers do
    half_width = Nx.tensor(@levels, type: {:f, 64}) |> Nx.divide(2)

    tokens
    |> Enum.map(fn t ->
      t
      |> index_to_codes()
      |> Nx.tensor(type: {:f, 64})
      |> Nx.subtract(half_width)
      |> Nx.divide(half_width)
    end)
    |> Enum.reduce(&Nx.add/2)
  end

  @doc "True when `tokens` is a valid semantic ID (3 integers in `0..4095`)."
  @spec valid_id?(term()) :: boolean()
  def valid_id?(tokens) do
    is_list(tokens) and length(tokens) == @num_quantizers and
      Enum.all?(tokens, &(is_integer(&1) and &1 >= 0 and &1 < @codebook_size))
  end

  defp to_latent_tensor(%Nx.Tensor{} = t), do: Nx.as_type(t, {:f, 64})

  defp to_latent_tensor(list) when is_list(list) and length(list) == @dim,
    do: Nx.tensor(list, type: {:f, 64})
end
