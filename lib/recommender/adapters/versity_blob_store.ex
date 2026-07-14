# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Adapters.VersityBlobStore do
  @moduledoc """
  Content-addressed blob storage: aria-storage chunking into an embedded
  Versity S3 gateway.

  The **host lifecycle** — provisioning the single `versitygw` binary, starting/
  stopping an embedded gateway over a posix directory, and building the S3
  endpoint config — is delegated to
  [`VersitygwLocal`](https://github.com/weftspun/versitygw-local). This module
  keeps only the app-specific parts: content-defined (BuzHash) chunking via
  [`V-Sekai-fire/aria-storage`](https://github.com/V-Sekai-fire/aria-storage),
  storing each chunk once under `chunks/<sha512-256>` with a per-blob JSON
  manifest under `blobs/<name>.json`, so shared content is deduplicated.

  `with_gateway/2` delegates to `VersitygwLocal.with_gateway/2`, ensures the
  bucket exists, then runs the caller's function with the S3 config.
  """

  @behaviour Recommender.Ports.BlobSink
  @behaviour Recommender.Ports.BlobSource

  @bucket "holo"
  @access_key "holo"
  @secret_key "holo-secret"

  @type opts :: %{
          optional(:data_dir) => String.t(),
          optional(:s3_port) => pos_integer()
        }

  @doc """
  Run `fun.(s3_config)` against the gateway, starting (and stopping) the
  embedded versitygw when nothing is listening. Ensures the bucket first.
  """
  @spec with_gateway(opts(), (keyword() -> result)) :: result | {:error, term()} when result: var
  def with_gateway(opts \\ %{}, fun) do
    VersitygwLocal.with_gateway(local_opts(opts), fn config ->
      ensure_bucket(config)
      fun.(config)
    end)
  end

  @doc """
  Chunk `file` (content-defined, aria-storage BuzHash) and store it as `name`.

  Returns `{:ok, %{chunks: n, new_chunks: m, bytes: size}}`.
  """
  @impl Recommender.Ports.BlobSink
  def put(config, name, file) do
    unless File.regular?(file), do: throw({:error, "no such file: #{file}"})

    data = File.read!(file)
    sha256 = :crypto.hash(:sha256, data) |> Base.encode16(case: :lower)

    {:ok, chunks} = AriaStorage.Chunks.create_chunks(file, compression: :none)

    new =
      Enum.count(chunks, fn chunk ->
        key = "chunks/#{chunk_id_hex(chunk)}"

        if object_exists?(config, key) do
          false
        else
          s3!(ExAws.S3.put_object(@bucket, key, chunk_data(chunk)), config)
          true
        end
      end)

    manifest = %{
      "name" => name,
      "size" => byte_size(data),
      "sha256" => sha256,
      "chunks" =>
        Enum.map(chunks, fn chunk ->
          %{"id" => chunk_id_hex(chunk), "size" => byte_size(chunk_data(chunk))}
        end)
    }

    s3!(ExAws.S3.put_object(@bucket, "blobs/#{name}.json", Jason.encode!(manifest)), config)
    {:ok, %{chunks: length(chunks), new_chunks: new, bytes: byte_size(data)}}
  catch
    :throw, {:error, _} = err -> err
  end

  @doc """
  Reassemble blob `name` into `out_path`, verifying the whole-file SHA-256.
  """
  @impl Recommender.Ports.BlobSource
  def get(config, name, out_path) do
    manifest =
      case s3(ExAws.S3.get_object(@bucket, "blobs/#{name}.json"), config) do
        {:ok, %{body: body}} -> Jason.decode!(body)
        {:error, _} -> throw({:error, "no such blob: #{name}"})
      end

    data =
      manifest["chunks"]
      |> Enum.map(fn %{"id" => id} ->
        %{body: body} = s3!(ExAws.S3.get_object(@bucket, "chunks/#{id}"), config)
        body
      end)
      |> IO.iodata_to_binary()

    actual = :crypto.hash(:sha256, data) |> Base.encode16(case: :lower)

    if actual != manifest["sha256"] do
      throw({:error, "checksum mismatch for #{name}: #{actual} != #{manifest["sha256"]}"})
    end

    File.write!(out_path, data)
    {:ok, %{bytes: byte_size(data), chunks: length(manifest["chunks"])}}
  catch
    :throw, {:error, _} = err -> err
  end

  @doc "List stored blob manifests."
  @impl Recommender.Ports.BlobSource
  def list(config) do
    %{body: %{contents: contents}} =
      s3!(ExAws.S3.list_objects(@bucket, prefix: "blobs/"), config)

    for %{key: "blobs/" <> rest} <- contents, String.ends_with?(rest, ".json") do
      String.trim_trailing(rest, ".json")
    end
  end

  ## Bucket + option mapping

  # Map the CLI's opts map into the keyword opts VersitygwLocal expects, pointing
  # binary resolution at this app's bundled priv/ and env var, and the gateway
  # root at <data-dir>/blobs with this store's fixed credentials.
  defp local_opts(opts) do
    root = Path.join(opts[:data_dir] || Recommender.Adapters.CockroachStore.default_data_dir(), "blobs")

    [
      port: opts[:s3_port],
      root: root,
      access_key: @access_key,
      secret_key: @secret_key,
      priv_app: :residual_fsq_recommender,
      bin_env: "RFR_VERSITYGW_BIN"
    ]
    |> Enum.reject(fn {_k, v} -> is_nil(v) end)
  end

  defp ensure_bucket(config) do
    case s3(ExAws.S3.head_bucket(@bucket), config) do
      {:ok, _} -> :ok
      {:error, _} -> s3!(ExAws.S3.put_bucket(@bucket, "us-east-1"), config)
    end

    :ok
  end

  defp object_exists?(config, key) do
    match?({:ok, _}, s3(ExAws.S3.head_object(@bucket, key), config))
  end

  defp s3(op, config), do: ExAws.request(op, config)

  defp s3!(op, config) do
    case ExAws.request(op, config) do
      {:ok, result} -> result
      {:error, reason} -> throw({:error, "S3 request failed: #{inspect(reason)}"})
    end
  end

  defp chunk_id_hex(chunk) do
    id = Map.get(chunk, :id) || Map.get(chunk, :chunk_id) || raise "chunk without id"

    if is_binary(id) and String.valid?(id) and String.match?(id, ~r/^[0-9a-f]+$/i),
      do: id,
      else: Base.encode16(id, case: :lower)
  end

  defp chunk_data(chunk) do
    Map.get(chunk, :data) || Map.get(chunk, :compressed_data) || raise "chunk without data"
  end
end
