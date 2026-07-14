# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Ports.EmbeddingSource do
  @moduledoc """
  Driven port (`*_source`): turns item text into dense embeddings for the training/fixture pipeline.

  Driving adapters (fixture build, pretraining) need `{num_items, dim}` embeddings but must not bind
  to a specific model runtime. Adapters realize this port against a concrete encoder (MPNet via
  Bumblebee, a recorded fixture for CI, a remote service).

  Implemented by: `Recommender.Adapters.BumblebeeEmbedding`.
  """

  @doc """
  Encodes `item_text_dict` (map of `item_index => text`) to an `Nx` tensor of shape
  `{num_items, dim}`, rows ordered by sorted item index.
  """
  @callback encode_item_text_dict(item_text_dict :: %{optional(term()) => String.t()}) ::
              Nx.Tensor.t()
end
