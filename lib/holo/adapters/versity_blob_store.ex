# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Adapters.VersityBlobStore do
  @moduledoc """
  Content-addressed blob storage: aria-storage chunking into an embedded
  Versity S3 gateway.

  The standalone binary ships the single-binary `versitygw` from
  [`versity/versitygw`](https://github.com/versity/versitygw) in
  `priv/versitygw/`, serving the S3 API over a plain directory
  (`<data-dir>/blobs`). Files are chunked with
  [`V-Sekai-fire/aria-storage`](https://github.com/V-Sekai-fire/aria-storage)'s
  content-defined (BuzHash) chunker; each chunk is stored once under
  `chunks/<sha512-256>` and a per-blob JSON manifest under
  `blobs/<name>.json` lists the chunk sequence — so shared content between
  blobs (asset revisions, palette swaps) is deduplicated.

  `with_gateway/2` mirrors `Holo.Adapters.CockroachStore.with_db/2`: reuse a running gateway if
  one answers on the port, otherwise spawn the bundled binary for the duration
  of the command.
  """

  @behaviour Holo.Ports.BlobSink
  @behaviour Holo.Ports.BlobSource

  require Logger

  @default_port 7070
  @bucket "holo"
  @access_key "holo"
  @secret_key "holo-secret"
  @ready_timeout_ms 30_000
  @poll_interval_ms 250

  @type opts :: %{
          optional(:data_dir) => String.t(),
          optional(:s3_port) => pos_integer()
        }

  @doc """
  Run `fun.(s3_config)` against the gateway, starting (and stopping) the
  embedded versitygw when nothing is listening.
  """
  @spec with_gateway(opts(), (map() -> result)) :: result | {:error, term()} when result: var
  def with_gateway(opts \\ %{}, fun) do
    port = opts[:s3_port] || @default_port

    {erl_port, os_pid} =
      if listening?(port), do: {nil, nil}, else: spawn_gateway(opts, port)

    try do
      case await_ready(port, @ready_timeout_ms) do
        :ok ->
          config = s3_config(port)
          ensure_bucket(config)
          fun.(config)

        {:error, reason} ->
          {:error, "S3 gateway did not become ready: #{inspect(reason)}"}
      end
    after
      stop_gateway(erl_port, os_pid)
    end
  end

  @doc """
  Chunk `file` (content-defined, aria-storage BuzHash) and store it as `name`.

  Returns `{:ok, %{chunks: n, new_chunks: m, bytes: size}}`.
  """
  @impl Holo.Ports.BlobSink
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
  @impl Holo.Ports.BlobSource
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
  @impl Holo.Ports.BlobSource
  def list(config) do
    %{body: %{contents: contents}} =
      s3!(ExAws.S3.list_objects(@bucket, prefix: "blobs/"), config)

    for %{key: "blobs/" <> rest} <- contents, String.ends_with?(rest, ".json") do
      String.trim_trailing(rest, ".json")
    end
  end

  @doc """
  Path to the versitygw binary: `HOLO_VERSITYGW_BIN`, then the bundled
  `priv/versitygw/`, then `$PATH`.
  """
  def versitygw_bin do
    exe = if match?({:win32, _}, :os.type()), do: "versitygw.exe", else: "versitygw"

    bundled =
      case :code.priv_dir(:holographic_item_memory) do
        {:error, _} -> nil
        priv -> Path.join([to_string(priv), "versitygw", exe])
      end

    cond do
      bin = System.get_env("HOLO_VERSITYGW_BIN") -> {:ok, bin}
      bundled && File.exists?(bundled) -> {:ok, bundled}
      bin = System.find_executable("versitygw") -> {:ok, bin}
      true -> {:error, "no versitygw binary: not bundled in priv/versitygw, not on PATH"}
    end
  end

  ## Gateway lifecycle

  defp spawn_gateway(opts, port) do
    case versitygw_bin() do
      {:ok, bin} ->
        root =
          Path.join(opts[:data_dir] || Holo.Adapters.CockroachStore.default_data_dir(), "blobs")

        File.mkdir_p!(root)
        Logger.debug("holo: starting embedded versitygw on port #{port}")

        erl_port =
          Port.open({:spawn_executable, bin}, [
            :binary,
            :exit_status,
            :stderr_to_stdout,
            args: ["--port", ":#{port}", "posix", root],
            env: [
              {~c"ROOT_ACCESS_KEY", String.to_charlist(@access_key)},
              {~c"ROOT_SECRET_KEY", String.to_charlist(@secret_key)}
            ]
          ])

        os_pid =
          case Port.info(erl_port, :os_pid) do
            {:os_pid, pid} -> pid
            _ -> nil
          end

        {erl_port, os_pid}

      {:error, reason} ->
        raise reason
    end
  end

  defp stop_gateway(nil, _), do: :ok

  defp stop_gateway(erl_port, os_pid) do
    if os_pid do
      case :os.type() do
        {:win32, _} -> System.cmd("taskkill", ["/PID", to_string(os_pid), "/T", "/F"])
        _ -> System.cmd("kill", [to_string(os_pid)])
      end
    end

    if is_port(erl_port) and Port.info(erl_port) != nil, do: Port.close(erl_port)
    :ok
  catch
    _, _ -> :ok
  end

  defp listening?(port) do
    case :gen_tcp.connect(~c"localhost", port, [:binary, active: false], 500) do
      {:ok, sock} ->
        :gen_tcp.close(sock)
        true

      {:error, _} ->
        false
    end
  end

  defp await_ready(port, timeout_ms) do
    deadline = System.monotonic_time(:millisecond) + timeout_ms
    do_await(port, deadline)
  end

  defp do_await(port, deadline) do
    cond do
      listening?(port) ->
        :ok

      System.monotonic_time(:millisecond) > deadline ->
        {:error, :timeout}

      true ->
        Process.sleep(@poll_interval_ms)
        do_await(port, deadline)
    end
  end

  ## S3 plumbing (ex_aws against the local gateway)

  defp s3_config(port) do
    [
      access_key_id: @access_key,
      secret_access_key: @secret_key,
      scheme: "http://",
      host: "localhost",
      port: port,
      region: "us-east-1"
    ]
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
