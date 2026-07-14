# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Ports.CheckpointSink do
  @moduledoc """
  Driven port (`*_sink`): writes the core's trained params outward to durable storage.

  Training (`Holo.Adapters.AxonTrainer`) produces a `%{key => Nx.Tensor}` params map; this port
  is the seam for persisting it without the trainer knowing the on-disk format. One params map can
  fan out to several sinks (npy export dir, an archive, a remote store).

  Implemented by: `Holo.Adapters.NpyCheckpointSink`.
  """

  @doc "Writes `params` (`%{key => Nx.Tensor}`) to `export_dir`."
  @callback write_export(
              params :: %{optional(String.t()) => Nx.Tensor.t()},
              export_dir :: String.t()
            ) ::
              :ok
end
