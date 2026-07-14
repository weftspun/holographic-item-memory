# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Adapters.FixtureBlobStore do
  @moduledoc """
  In-memory fixture adapter for `Holo.Ports.BlobSource` / `Holo.Ports.BlobSink`.

  Uses the same aria-storage content-defined chunker as the versitygw adapter
  but keeps chunks and manifests in an `Agent`, so the chunk/manifest logic is
  exercised in CI without a gateway process.
  """

  @behaviour Holo.Ports.BlobSource
  @behaviour Holo.Ports.BlobSink

  def start_link do
    Agent.start_link(fn -> %{chunks: %{}, manifests: %{}} end)
  end

  @impl Holo.Ports.BlobSink
  def put(agent, name, file) do
    data = File.read!(file)
    sha256 = :crypto.hash(:sha256, data) |> Base.encode16(case: :lower)
    {:ok, chunks} = AriaStorage.Chunks.create_chunks(file, compression: :none)

    new =
      Enum.count(chunks, fn chunk ->
        id = Base.encode16(chunk.id, case: :lower)

        Agent.get_and_update(agent, fn state ->
          if Map.has_key?(state.chunks, id) do
            {false, state}
          else
            {true, put_in(state, [:chunks, id], chunk.data)}
          end
        end)
      end)

    manifest = %{
      "name" => name,
      "size" => byte_size(data),
      "sha256" => sha256,
      "chunks" => Enum.map(chunks, &%{"id" => Base.encode16(&1.id, case: :lower)})
    }

    Agent.update(agent, &put_in(&1, [:manifests, name], manifest))
    {:ok, %{chunks: length(chunks), new_chunks: new, bytes: byte_size(data)}}
  end

  @impl Holo.Ports.BlobSource
  def get(agent, name, out_path) do
    case Agent.get(agent, & &1.manifests[name]) do
      nil ->
        {:error, "no such blob: #{name}"}

      manifest ->
        data =
          manifest["chunks"]
          |> Enum.map(fn %{"id" => id} -> Agent.get(agent, & &1.chunks[id]) end)
          |> IO.iodata_to_binary()

        actual = :crypto.hash(:sha256, data) |> Base.encode16(case: :lower)

        if actual == manifest["sha256"] do
          File.write!(out_path, data)
          {:ok, %{bytes: byte_size(data), chunks: length(manifest["chunks"])}}
        else
          {:error, "checksum mismatch for #{name}"}
        end
    end
  end

  @impl Holo.Ports.BlobSource
  def list(agent) do
    agent |> Agent.get(& &1.manifests) |> Map.keys() |> Enum.sort()
  end
end
