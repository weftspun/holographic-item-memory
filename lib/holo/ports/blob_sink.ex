# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Ports.BlobSink do
  @moduledoc """
  Driven sink port: write content-addressed blobs outward.

  Implemented by the versitygw/aria-storage adapter in production and the
  in-memory fixture adapter in CI.
  """

  @type state :: term()

  @doc "Chunk `file` and store it as `name` (content-defined dedup)."
  @callback put(state(), name :: String.t(), file :: Path.t()) ::
              {:ok,
               %{
                 chunks: non_neg_integer(),
                 new_chunks: non_neg_integer(),
                 bytes: non_neg_integer()
               }}
              | {:error, term()}
end
