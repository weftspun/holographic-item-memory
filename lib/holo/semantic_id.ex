defmodule Holo.SemanticID do
  @moduledoc """
  Semantic IDs as holographic item vectors, plus `asset_semantic_id` ingestion.

  An item's semantic ID is the `Holo.ResidualFSQ` token tuple `[t0, t1, t2]`
  produced upstream by the `multimodal-semantic-ids` pipeline (per-modality
  FOSS encoders → fixed-slot concat → one ResidualFSQ). Here each token becomes
  a deterministic HRR atom bound to its stage role, and the item vector is the
  bundle of the three bound tokens:

      item = bundle(bind(atom("sid:q0:" <> t0), ROLE_Q0),
                    bind(atom("sid:q1:" <> t1), ROLE_Q1),
                    bind(atom("sid:q2:" <> t2), ROLE_Q2))

  Because atoms are SHA-256-deterministic, any process can reconstruct any
  item's vector from its ID alone — no training, no stored embeddings. Items
  sharing coarse (early-stage) tokens land near each other, which is exactly
  the residual-FSQ coarse-to-fine structure. This is what makes recall
  zero-shot: a brand-new asset is recommendable the moment it has an ID.
  """

  alias Holo.HRR
  alias Holo.ResidualFSQ

  @doc """
  Deterministic HRR vector for a semantic ID (3 ResidualFSQ tokens).
  """
  @spec item_vector([non_neg_integer()], pos_integer()) :: Nx.Tensor.t()
  def item_vector(tokens, dim \\ HRR.default_dim()) do
    unless ResidualFSQ.valid_id?(tokens) do
      raise ArgumentError,
            "expected #{ResidualFSQ.tokens_per_item()} tokens in 0..#{ResidualFSQ.codebook_size() - 1}, got: #{inspect(tokens)}"
    end

    tokens
    |> Enum.with_index()
    |> Enum.map(fn {token, stage} ->
      HRR.bind(
        HRR.encode_atom("sid:q#{stage}:#{token}", dim),
        HRR.encode_atom("__holo_role_q#{stage}__", dim)
      )
    end)
    |> HRR.bundle()
  end

  @doc """
  Flatten a semantic ID to a single integer key:
  `t0 + 4096·t1 + 4096²·t2`. Injective on valid IDs (certified in
  `formal/HoloModel.lean`).
  """
  @spec flat_key([non_neg_integer()]) :: non_neg_integer()
  def flat_key([t0, t1, t2] = tokens) do
    true = ResidualFSQ.valid_id?(tokens)
    c = ResidualFSQ.codebook_size()
    t0 + c * t1 + c * c * t2
  end

  @doc "Inverse of `flat_key/1`."
  @spec from_flat_key(non_neg_integer()) :: [non_neg_integer()]
  def from_flat_key(key) when is_integer(key) and key >= 0 do
    c = ResidualFSQ.codebook_size()
    [rem(key, c), key |> div(c) |> rem(c), key |> div(c * c) |> rem(c)]
  end

  @doc """
  Load `{asset_id, [t0, t1, t2]}` pairs from an `asset_semantic_id` parquet
  file (the artifact `multimodal-semantic-ids/scripts/semantic_ids.py` writes).

  Accepts either a list column holding the token tuple (`semantic_id`) or
  three scalar columns (`sid_0`/`sid_1`/`sid_2` or `code_0`/`code_1`/`code_2`).
  The id column may be `asset_id`, `item_id`, or `id`.
  """
  @spec load_parquet(Path.t()) :: {:ok, [{term(), [non_neg_integer()]}]} | {:error, term()}
  def load_parquet(path) do
    with {:ok, df} <- Explorer.DataFrame.from_parquet(path) do
      names = Explorer.DataFrame.names(df)

      id_col = Enum.find(["asset_id", "item_id", "id"], &(&1 in names))

      cond do
        id_col == nil ->
          {:error, "no asset_id/item_id/id column in #{path} (columns: #{inspect(names)})"}

        "semantic_id" in names ->
          ids = df[id_col] |> Explorer.Series.to_list()
          tokens = df["semantic_id"] |> Explorer.Series.to_list()
          {:ok, zip_valid(ids, tokens)}

        Enum.all?(sid_cols(names), &(&1 != nil)) ->
          [c0, c1, c2] = sid_cols(names)
          ids = df[id_col] |> Explorer.Series.to_list()

          tokens =
            Enum.zip([
              Explorer.Series.to_list(df[c0]),
              Explorer.Series.to_list(df[c1]),
              Explorer.Series.to_list(df[c2])
            ])
            |> Enum.map(fn {a, b, c} -> [a, b, c] end)

          {:ok, zip_valid(ids, tokens)}

        true ->
          {:error,
           "no semantic_id list column or sid_0..2/code_0..2 columns in #{path} " <>
             "(columns: #{inspect(names)})"}
      end
    end
  end

  defp sid_cols(names) do
    for i <- 0..2 do
      Enum.find(["sid_#{i}", "code_#{i}"], &(&1 in names))
    end
  end

  defp zip_valid(ids, token_lists) do
    Enum.zip(ids, token_lists)
    |> Enum.filter(fn {_id, tokens} -> ResidualFSQ.valid_id?(tokens) end)
  end
end
