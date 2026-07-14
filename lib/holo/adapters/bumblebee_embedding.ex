# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Adapters.BumblebeeEmbedding do
  @moduledoc """
  Text → 768-d embeddings for RecGPT-style training (MPNet, sentence-transformers/all-mpnet-base-v2).

  Uses Bumblebee with mean pooling, L2 normalization, and fixed sequence length 384 to align with
  sentence-transformers (SentenceTransformer.encode). Run `mix holo.compare_embeddings` to check
  cosine similarity vs the dataset item_text_embeddings.npy.

  ## Usage

      serving = Holo.Adapters.BumblebeeEmbedding.serving()
      result = Nx.Serving.run(serving, "Some market title Yes")
      result = Nx.Serving.run(serving, ["Text A", "Text B"])
      embeddings = Holo.Adapters.BumblebeeEmbedding.encode_item_text_dict(%{0 => "Title A", 1 => "Title B"})
  """

  @behaviour Holo.Ports.EmbeddingSource

  @model_id "sentence-transformers/all-mpnet-base-v2"
  @embed_batch_size 100
  # Match sentence-transformers default for all-mpnet-base-v2 (truncate/pad to 384 tokens).
  @max_sequence_length 384

  @doc """
  Builds the same string upstream RecGPT-style uses for encoding: Python's str(dict).replace('{','').replace('}','').
  Uses only the \"title\" key so the result matches item_text_dict.pkl value shape (e.g. \"'title': 'Game Name'\").
  Use when you need embeddings to match the dataset's item_text_embeddings.npy.
  """
  @spec recgpt_item_text(map()) :: String.t()
  def recgpt_item_text(item) when is_map(item) do
    title =
      Map.get(item, "embedding_text") || Map.get(item, :embedding_text) ||
        Map.get(item, "title") || Map.get(item, :title) || Map.get(item, "text") ||
        Map.get(item, "raw") || ""

    pairs = [{"title", to_string(title)}]
    Enum.map_join(pairs, ", ", fn {k, v} -> "'#{k}': '#{escape_single(v)}'" end)
  end

  defp escape_single(s), do: String.replace(s, "'", "\\'")

  @doc "Loads the MPNet model and tokenizer, returns a text embedding serving. Cached in application env :recgpt."
  def serving do
    case Application.get_env(:holographic_item_memory, :embedding_serving) do
      nil ->
        serving = load_serving!()
        Application.put_env(:holographic_item_memory, :embedding_serving, serving)
        serving

      cached ->
        cached
    end
  end

  defp load_serving! do
    IO.puts("Downloading model #{@model_id} (first run may take several minutes)...")

    # Load as :base for hidden_state/pooled_state (sentence-transformers has LM head; we use encoder only).
    {:ok, model_info} =
      Bumblebee.load_model({:hf, @model_id}, spec_overrides: [architecture: :base])

    IO.puts("Model loaded. Downloading tokenizer...")
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, @model_id})
    IO.puts("Tokenizer loaded. Building serving...")

    serving =
      Bumblebee.Text.text_embedding(model_info, tokenizer,
        output_pool: :mean_pooling,
        output_attribute: :hidden_state,
        embedding_processor: :l2_norm,
        compile: [batch_size: @embed_batch_size, sequence_length: @max_sequence_length]
      )

    IO.puts("Embedding serving ready.")
    serving
  end

  defp encode_texts(texts) when is_list(texts) do
    serv = serving()
    results = Nx.Serving.run(serv, texts)
    tensors = Enum.map(results, fn %{embedding: t} -> t end)
    Nx.stack(tensors)
  end

  @doc "Encodes item_text_dict (map of item_index => text) to Nx tensor {num_items, 768}. Indices 0..num_items-1, sorted. Processes in batches of #{@embed_batch_size} to limit memory."
  @impl Holo.Ports.EmbeddingSource
  def encode_item_text_dict(item_text_dict) when is_map(item_text_dict) do
    indices = item_text_dict |> Map.keys() |> Enum.sort()
    texts = Enum.map(indices, &Map.fetch!(item_text_dict, &1))
    encode_texts_batched(texts, @embed_batch_size)
  end

  defp encode_texts_batched(texts, batch_size) when length(texts) <= batch_size do
    encode_texts(texts)
  end

  defp encode_texts_batched(texts, batch_size) do
    texts
    |> Enum.chunk_every(batch_size)
    |> Enum.map(&encode_texts/1)
    |> Nx.concatenate(axis: 0)
  end
end
