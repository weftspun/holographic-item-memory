# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Mix.Tasks.Recommender.ExportCkpt do
  @shortdoc "Export a PyTorch .pt checkpoint to manifest.json + .npy"
  @moduledoc """
  Converts a PyTorch `.pt` file (zip format, PyTorch 1.6+) to an export directory
  (`manifest.json` + one `.npy` per tensor) that
  `Recommender.Adapters.NpyCheckpointSource.load_from_export/1` can load. Pure Elixir
  (Unpickler + Unzip) — no Python.

  ## Example

      mix recommender.export_ckpt --from-pt data/model.pt --out data/ckpt_export

  ## Options
    * `--from-pt` — path to a PyTorch `.pt` checkpoint (required)
    * `--out` — output export directory (required)
  """
  use Mix.Task

  alias Recommender.Adapters.{PtCheckpointSource, NpyCheckpointSink}

  @impl true
  def run(args) do
    {opts, _, _} = OptionParser.parse(args, switches: [from_pt: :string, out: :string])

    out_dir = opts[:out] || Mix.raise("--out DIR is required")
    pt_path = opts[:from_pt] || Mix.raise("--from-pt PATH is required")

    unless File.regular?(pt_path), do: Mix.raise("PyTorch checkpoint not found: #{pt_path}")

    pt_path = Path.expand(pt_path)
    out_dir = Path.expand(out_dir)
    File.mkdir_p!(out_dir)

    Application.ensure_all_started(:nx)
    Mix.shell().info("Loading .pt from #{pt_path}...")
    params = PtCheckpointSource.load!(pt_path)
    Mix.shell().info("Writing export to #{out_dir} (#{map_size(params)} tensors)...")
    :ok = NpyCheckpointSink.write_export(params, out_dir)
    Mix.shell().info("Done.")
  end
end
