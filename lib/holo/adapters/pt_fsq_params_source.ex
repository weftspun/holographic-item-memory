# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Adapters.PtFsqParamsSource do
  @moduledoc """
  Adapter realizing `Holo.Ports.FsqParamsSource` against a VAE `.pt` checkpoint on disk.

  Reads the file with `Holo.Adapters.PtCheckpointSource` (zip-based `.pt` → tensor map) and
  normalizes it via the pure core `Holo.Core.FSQ.load_params/1`. Use this when building a fixture
  so `token_id_list` matches the Python RecGPT-style pipeline (same FSQ codebook as eval/predict).

  The VAE state_dict uses keys like `quantizer.project_in.weight`, e.g. `vae_len4_fsq88865_ep90.pt`.
  """

  @behaviour Holo.Ports.FsqParamsSource

  alias Holo.Adapters.PtCheckpointSource
  alias Holo.Core.FSQ

  @doc """
  Load FSQ params from a VAE checkpoint `.pt` file path. Returns the same shape as
  `Holo.Core.FSQ.load_params/1`.
  """
  @impl Holo.Ports.FsqParamsSource
  @spec load(String.t()) :: map()
  def load(vae_pt_path) when is_binary(vae_pt_path) do
    vae_pt_path = Path.expand(vae_pt_path)
    unless File.regular?(vae_pt_path), do: raise(File.Error, path: vae_pt_path, reason: :enoent)

    vae_pt_path
    |> PtCheckpointSource.load!()
    |> FSQ.load_params()
  end
end
