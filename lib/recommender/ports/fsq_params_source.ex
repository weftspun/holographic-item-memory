# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Ports.FsqParamsSource do
  @moduledoc """
  Driven port (`*_source`): supplies FSQ projection params to the core.

  The core (`Recommender.Core.FSQ`) needs `project_in`/`project_out` weights to encode item
  embeddings into tokens, but must not know where they come from. Adapters realize this port
  against concrete backing stores (a VAE `.pt` file, a fixture, etc.) and return params in the
  shape `Recommender.Core.FSQ.load_params/1` produces.

  Implemented by: `Recommender.Adapters.PtFsqParamsSource`.
  """

  @doc "Load FSQ params from `ref` (e.g. a filesystem path). Returns the `load_params/1` map shape."
  @callback load(ref :: term()) :: map()
end
