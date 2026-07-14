# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Ports.CheckpointSource do
  @moduledoc """
  Driven port (`*_source`): loads model params into the core for inference/serving.

  The core inference stack (`Holo.Core.Inference`) consumes a `%{key => Nx.Tensor}` params map
  but must not know its on-disk representation. Adapters realize this port against a concrete format
  (npy export dir, `.pt`, a recorded fixture).

  Implemented by: `Holo.Adapters.NpyCheckpointSource`.
  """

  @doc "Loads a checkpoint identified by `export_dir`, returning `%{key => Nx.Tensor}`."
  @callback load_from_export(export_dir :: String.t()) :: %{optional(String.t()) => Nx.Tensor.t()}
end
