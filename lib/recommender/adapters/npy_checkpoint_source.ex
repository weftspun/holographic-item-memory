# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Adapters.NpyCheckpointSource do
  @moduledoc """
  Load model checkpoint from an export directory (manifest.json + .npy files).

  Export dir can be produced by:
  - `Recommender.Adapters.NpyCheckpointSink.write_export/2` or `mix recommender.export_ckpt --from-export DIR --out DIR`
  - `mix recommender.export_ckpt --from-pt path.pt --out DIR` (requires Python + torch)
  - Formerly: `scripts/inspect_checkpoint.py --export DIR`

  The export dir must contain:
  - manifest.json: map of state_dict key -> %{"file" => "key.npy", "shape" => [dims]}
  - One .npy file per tensor.

  Returns a map of key => Nx.Tensor. The inference model (Recommender.Core.Inference) expects
  keys such as: wte (FSQ embed table), ae.* (aux encoder), gpt2model.* (GPT-2), pred_head.* (head).

  Tensors are created with Nx.BinaryBackend so loading works regardless
  of the default Nx backend; callers can then transfer params to EXLA (e.g. in Serve).

  ## Checkpoint integrity
  When `config :residual_fsq_recommender, :ckpt_expected_sha256` or `RFR_CKPT_SHA256` is set,
  load_from_export/1 verifies the checkpoint hash before loading. Compute the hash with
  `mix recommender.ckpt_sha256 --ckpt path`.
  """

  @behaviour Recommender.Ports.CheckpointSource

  @doc """
  Returns checkpoint SHA256 hex string, or nil if manifest missing.
  Does not load tensors.
  """
  def get_sha256(export_dir) when is_binary(export_dir) do
    manifest_path = Path.join(export_dir, "manifest.json")

    if File.regular?(manifest_path) do
      manifest = File.read!(manifest_path) |> Jason.decode!()
      compute_sha256(export_dir, manifest)
    else
      nil
    end
  end

  @doc """
  Compute deterministic SHA256 of checkpoint (manifest + all .npy files in sorted order).
  Used for integrity verification. Returns lowercase hex string.
  """
  def compute_sha256(export_dir, manifest) when is_binary(export_dir) and is_map(manifest) do
    # Sort by key for determinism; hash each file and concatenate binary hashes
    combined =
      manifest
      |> Enum.sort_by(&elem(&1, 0))
      |> Enum.reduce(<<>>, fn {_key, meta}, acc ->
        path = Path.join(export_dir, meta["file"])

        blob =
          try do
            File.read!(path)
          rescue
            e in [File.Error] ->
              reraise "Failed to load #{path}: #{Exception.message(e)}", __STACKTRACE__
          end

        acc <> :crypto.hash(:sha256, blob)
      end)

    :crypto.hash(:sha256, combined) |> Base.encode16(case: :lower)
  end

  @doc """
  Load checkpoint from an export directory. Returns %{key => Nx.Tensor}.
  Uses BinaryBackend; transfer to EXLA in Serve if desired.
  Verifies SHA256 when config :residual_fsq_recommender, :ckpt_expected_sha256 or RFR_CKPT_SHA256 is set.
  """
  @impl Recommender.Ports.CheckpointSource
  def load_from_export(export_dir) when is_binary(export_dir) do
    do_load_from_export(export_dir)
  end

  defp do_load_from_export(export_dir) do
    manifest_path = Path.join(export_dir, "manifest.json")

    if not File.regular?(manifest_path) do
      raise File.Error, path: manifest_path, reason: :enoent
    end

    manifest = File.read!(manifest_path) |> Jason.decode!()

    expected = expected_ckpt_sha256()

    if expected && expected != "" do
      actual = compute_sha256(export_dir, manifest)

      if actual != expected do
        raise "Checkpoint SHA256 mismatch: expected #{expected}, got #{actual}. " <>
                "Run mix recommender.ckpt_sha256 --ckpt #{export_dir} to get the correct hash."
      end
    end

    Enum.reduce(manifest, %{}, fn {key, meta}, acc ->
      fname = meta["file"]
      path = Path.join(export_dir, fname)

      case Npy.load(path, :npy) do
        {:ok, npy} ->
          tensor = npy_to_tensor_binary_backend(npy)
          Map.put(acc, key, tensor)

        {:error, reason} ->
          raise "Failed to load #{path}: #{inspect(reason)}"
      end
    end)
  end

  # Build Nx tensor from %Npy{} using BinaryBackend only. Same descr→type
  # mapping as Npy.npy2tensor/1; caller can backend_transfer to EXLA.
  defp npy_to_tensor_binary_backend(%Npy{descr: descr, shape: shape, data: data}) do
    type = npy_descr_to_nx_type(descr)
    prev = Nx.default_backend()
    Nx.default_backend(Nx.BinaryBackend)

    try do
      data
      |> Nx.from_binary(type)
      |> Nx.reshape(shape)
    after
      Nx.default_backend(prev)
    end
  end

  defp expected_ckpt_sha256 do
    System.get_env("RFR_CKPT_SHA256") || Application.get_env(:residual_fsq_recommender, :ckpt_expected_sha256)
  end

  defp npy_descr_to_nx_type(descr) do
    case descr do
      "<i1" -> {:s, 8}
      "<i2" -> {:s, 16}
      "<i4" -> {:s, 32}
      "<i8" -> {:s, 64}
      "<u1" -> {:u, 8}
      "<u2" -> {:u, 16}
      "<u4" -> {:u, 32}
      "<u8" -> {:u, 64}
      "<f4" -> {:f, 32}
      "<f8" -> {:f, 64}
      "<f2" -> {:bf, 16}
      # big-endian variants
      ">i1" -> {:s, 8}
      ">i2" -> {:s, 16}
      ">i4" -> {:s, 32}
      ">i8" -> {:s, 64}
      ">u1" -> {:u, 8}
      ">u2" -> {:u, 16}
      ">u4" -> {:u, 32}
      ">u8" -> {:u, 64}
      ">f4" -> {:f, 32}
      ">f8" -> {:f, 64}
      ">f2" -> {:bf, 16}
      other -> raise "Unsupported npy descr: #{inspect(other)}"
    end
  end
end
