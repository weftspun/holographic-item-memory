# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Ports.BlobSource do
  @moduledoc """
  Driven source port: read content-addressed blobs inbound.

  Implemented by the versitygw/aria-storage adapter in production and the
  in-memory fixture adapter in CI. `state` is the adapter's opaque handle
  (S3 config, Agent pid, …).
  """

  @type state :: term()

  @doc "Reassemble blob `name` into `out_path`, verifying integrity."
  @callback get(state(), name :: String.t(), out_path :: Path.t()) ::
              {:ok, %{bytes: non_neg_integer(), chunks: non_neg_integer()}} | {:error, term()}

  @doc "List stored blob names."
  @callback list(state()) :: [String.t()]
end
