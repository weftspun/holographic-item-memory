defmodule Holo.HRR do
  @moduledoc """
  Holographic Reduced Representations (HRR) with phase encoding.

  A vector symbolic architecture for encoding compositional structure into
  fixed-width distributed representations. Each concept is a vector of phase
  angles in `[0, 2π)`, held as an `Nx` f64 tensor. The algebra:

    * `bind/2`   — circular convolution (element-wise phase addition); associates
      two concepts into a composite that is quasi-orthogonal to both.
    * `unbind/2` — circular correlation (phase subtraction); retrieves a bound
      value: `unbind(bind(a, b), a) ≈ b` up to superposition noise.
    * `bundle/1` — superposition (circular mean of complex exponentials); merges
      vectors into one similar to each input. Holds `O(√dim)` items.
    * `similarity/2` — phase cosine similarity in `[-1, 1]`.

  Atoms are generated deterministically from SHA-256 counter blocks, so
  representations are identical across processes, machines, and language
  versions. This is a port of the Python `holographic.py` reference (Plate 1995;
  Gayler 2004); atom vectors match the reference bit-for-bit given the same
  word and dimension.
  """

  @two_pi 2.0 * :math.pi()
  @default_dim 1024
  @empty_atom "__hrr_empty__"

  @doc "Default vector dimensionality."
  def default_dim, do: @default_dim

  @doc """
  Deterministic phase vector for a word via SHA-256 counter blocks.

  Hashes `"word:i"` for `i = 0, 1, …`, interprets the concatenated digests as
  little-endian uint16 values, and scales them to `[0, 2π)`. Returns an f64
  tensor of shape `{dim}`.
  """
  @spec encode_atom(String.t(), pos_integer()) :: Nx.Tensor.t()
  def encode_atom(word, dim \\ @default_dim) when is_binary(word) and dim > 0 do
    # Each SHA-256 digest is 32 bytes = 16 uint16 values.
    blocks_needed = div(dim + 15, 16)

    bytes =
      for i <- 0..(blocks_needed - 1), into: <<>> do
        :crypto.hash(:sha256, "#{word}:#{i}")
      end

    values =
      for <<v::unsigned-little-16 <- bytes>>, do: v

    values
    |> Enum.take(dim)
    |> Nx.tensor(type: {:f, 64})
    |> Nx.multiply(f64(@two_pi / 65_536.0))
  end

  @doc "Circular convolution: element-wise phase addition (mod 2π)."
  @spec bind(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def bind(a, b), do: wrap(Nx.add(a, b))

  @doc "Circular correlation: element-wise phase subtraction (mod 2π)."
  @spec unbind(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def unbind(memory, key), do: wrap(Nx.subtract(memory, key))

  @doc """
  Superposition via circular mean: `angle(Σ e^{iv})` per component.

  Accepts a non-empty list of phase vectors.
  """
  @spec bundle([Nx.Tensor.t(), ...]) :: Nx.Tensor.t()
  def bundle([_ | _] = vectors) do
    {sin_sum, cos_sum} =
      Enum.reduce(vectors, {nil, nil}, fn v, {s, c} ->
        {add_or_init(s, Nx.sin(v)), add_or_init(c, Nx.cos(v))}
      end)

    wrap(Nx.atan2(sin_sum, cos_sum))
  end

  @doc "Phase cosine similarity: `mean(cos(a - b))`, in `[-1, 1]`."
  @spec similarity(Nx.Tensor.t(), Nx.Tensor.t()) :: float()
  def similarity(a, b) do
    Nx.subtract(a, b) |> Nx.cos() |> Nx.mean() |> Nx.to_number()
  end

  @doc """
  Bag-of-words text encoding: bundle of atom vectors for each token.

  Lowercases, splits on whitespace, strips leading/trailing punctuation.
  Empty text encodes to the reserved `#{inspect(@empty_atom)}` atom.
  """
  @spec encode_text(String.t(), pos_integer()) :: Nx.Tensor.t()
  def encode_text(text, dim \\ @default_dim) when is_binary(text) do
    tokens =
      text
      |> String.downcase()
      |> String.split()
      |> Enum.map(&strip_punctuation/1)
      |> Enum.reject(&(&1 == ""))

    case tokens do
      [] -> encode_atom(@empty_atom, dim)
      tokens -> tokens |> Enum.map(&encode_atom(&1, dim)) |> bundle()
    end
  end

  @doc "Serialize a phase vector to binary (f64, native byte order)."
  @spec to_binary(Nx.Tensor.t()) :: binary()
  def to_binary(phases), do: phases |> Nx.as_type({:f, 64}) |> Nx.to_binary()

  @doc "Deserialize a binary produced by `to_binary/1`."
  @spec from_binary(binary()) :: Nx.Tensor.t()
  def from_binary(data) when is_binary(data), do: Nx.from_binary(data, {:f, 64})

  @doc """
  Signal-to-noise ratio estimate for holographic storage: `√(dim / n_items)`.

  Retrieval errors become likely once SNR falls below ~2.0
  (i.e. `n_items > dim / 4`). Returns `:infinity` when the store is empty.
  """
  @spec snr_estimate(pos_integer(), non_neg_integer()) :: float() | :infinity
  def snr_estimate(_dim, n_items) when n_items <= 0, do: :infinity

  def snr_estimate(dim, n_items), do: :math.sqrt(dim / n_items)

  # Normalize any real-valued angle tensor into [0, 2π). Nx.remainder keeps the
  # sign of the dividend, so wrap twice to cover negative inputs. Scalars go in
  # as f64 tensors — bare floats would be truncated to f32 by Nx's auto-cast.
  defp wrap(t) do
    two_pi = f64(@two_pi)

    t
    |> Nx.remainder(two_pi)
    |> Nx.add(two_pi)
    |> Nx.remainder(two_pi)
  end

  defp f64(x), do: Nx.tensor(x, type: {:f, 64})

  defp add_or_init(nil, t), do: t
  defp add_or_init(acc, t), do: Nx.add(acc, t)

  defp strip_punctuation(token) do
    String.replace(token, ~r/^[.,!?;:"'()\[\]{}]+|[.,!?;:"'()\[\]{}]+$/, "")
  end
end
