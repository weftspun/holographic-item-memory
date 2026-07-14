# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Adapters.NpyCheckpointSink do
  @moduledoc """
  Write RecGPT checkpoint to an export directory (manifest.json + .npy files)
  compatible with `Holo.Adapters.NpyCheckpointSource.load_from_export/1`.

  Uses the `:npy` package to save each tensor. Use this to:
  - Save trained params from `Holo.Adapters.AxonTrainer.run/3` to disk.
  - Re-export an existing export dir (load then write to another path).
  - Produce the same format that the former Python script `inspect_recgpt_checkpoint.py --export` produced.
  """

  @behaviour Holo.Ports.CheckpointSink

  @doc """
  Writes a params map (key => Nx.Tensor) to an export directory.

  - Creates `export_dir` if it does not exist.
  - Writes one .npy file per key (filename is a safe version of the key).
  - Writes `manifest.json`: map of key => %{"file" => filename, "shape" => [dims]}.

  Keys are preserved in the manifest so `NpyCheckpointSource.load_from_export/1` returns
  the same key => tensor map.
  """
  @impl Holo.Ports.CheckpointSink
  def write_export(params, export_dir) when is_map(params) and is_binary(export_dir) do
    File.mkdir_p!(export_dir)

    manifest =
      Enum.reduce(params, %{}, fn {key, tensor}, acc ->
        unless is_struct(tensor, Nx.Tensor) do
          raise ArgumentError, "value for key #{inspect(key)} is not an Nx.Tensor"
        end

        fname = "#{key}.npy"
        path = Path.join(export_dir, fname)
        :ok = Npy.save(tensor, path)
        shape = Nx.shape(tensor) |> Tuple.to_list()
        Map.put(acc, key, %{"file" => fname, "shape" => shape})
      end)

    manifest_path = Path.join(export_dir, "manifest.json")
    File.write!(manifest_path, Jason.encode!(manifest, pretty: true))
    :ok
  end
end
