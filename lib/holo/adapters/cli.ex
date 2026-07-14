# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Adapters.CLI do
  @moduledoc """
  Driving adapter: single entrypoint for the standalone `holo` binary.

  ## Subcommands

      holo add <item_id> <t0> <t1> <t2> <t3>       store an item's semantic ID
      holo add --json                          bulk add from stdin:
                                               [{"item_id": "...", "semantic_id": [t0,t1,t2,t3]}, ...]
      holo observe <id> <id> [...]             record a session's transitions
      holo recommend <id> [...] [--top-k N]    next-item recall for a session (JSON);
                     --checkpoint <dir>        FuXi-Linear model over the stored catalog
      holo items [--limit N]                   list stored items (JSON)
      holo blob put <name> <file>              chunk + store a file (aria-storage -> S3)
      holo blob get <name> <out>               reassemble a stored blob
      holo blob list                           list stored blobs (JSON)
      holo db start                            run the embedded CockroachDB in the
                                               foreground (other holo commands reuse it)
      holo version                             print version
      holo help                                print this usage

  ## Global options

      --data-dir DIR    data directory (default ~/.holographic-item-memory)
      --port N          embedded cockroach SQL port (default 26257)
      --s3-port N       embedded versitygw S3 port (default 7070)
      --db-url URL      connect to an external cockroach/postgres instead of
                        starting the embedded node
      --checkpoint DIR  npy export dir of trained FuXi-Linear params (recommend)

  ## Structure

  `run/1` is pure with respect to process control — it returns a tagged
  result and never writes to a device or halts, so it is unit-testable.
  `main/1` is the release entrypoint: it resolves argv (via the `__BURRITO`
  marker when packaged), calls `run/1`, prints, and halts. Persistence goes
  through the driven ports: `Holo.Ports.ItemSource` / `ItemSink` (embedded
  CockroachDB adapter) and `Holo.Ports.BlobSource` / `BlobSink` (embedded
  versitygw + aria-storage adapter).
  """

  alias Holo.Adapters.CockroachStore, as: Store
  alias Holo.Adapters.VersityBlobStore, as: Blob
  alias Holo.Adapters.{NpyCheckpointSource, Serve}
  alias Holo.Core.{FuxiLinearInferenceParams, ResidualFSQ}
  alias Holo.Ports.ItemSource

  @version Mix.Project.config()[:version]

  @typedoc "Outcome of `run/1` — never performs IO or halts the VM."
  @type outcome :: {:ok, iodata()} | {:error, iodata(), non_neg_integer()}

  @doc """
  Burrito/release entrypoint. Resolves argv, dispatches, prints, and halts.
  """
  @spec main([String.t()] | nil) :: no_return()
  def main(argv \\ nil) do
    argv = argv || resolve_argv()
    emit(run(argv))
  rescue
    e ->
      IO.puts(:stderr, "holo: #{Exception.message(e)}")
      halt(1)
  catch
    kind, reason ->
      IO.puts(:stderr, "holo: #{Exception.format(kind, reason, __STACKTRACE__)}")
      halt(1)
  end

  defp emit({:ok, output}) do
    IO.puts(output)
    halt(0)
  end

  defp emit({:error, message, code}) do
    IO.puts(:stderr, message)
    halt(code)
  end

  @doc """
  Dispatch `argv` to a subcommand and return its outcome without halting.
  """
  @spec run([String.t()]) :: outcome()
  def run(argv) do
    {opts, args} = parse_opts(argv)

    case args do
      ["help"] -> {:ok, usage()}
      [] -> {:ok, usage()}
      ["version"] -> {:ok, "holo #{@version}"}
      ["add" | rest] -> cmd_add(rest, opts)
      ["observe" | rest] -> cmd_observe(rest, opts)
      ["recommend" | rest] -> cmd_recommend(rest, opts)
      ["items" | rest] -> cmd_items(rest, opts)
      ["blob" | rest] -> cmd_blob(rest, opts)
      ["db", "start"] -> cmd_db_start(opts)
      [other | _] -> {:error, "holo: unknown subcommand #{inspect(other)}\n\n#{usage()}", 2}
    end
  end

  @doc "Parse global options; returns `{opts_map, remaining_args}`."
  @spec parse_opts([String.t()]) :: {map(), [String.t()]}
  def parse_opts(argv) do
    {parsed, args, _invalid} =
      OptionParser.parse(argv,
        strict: [
          data_dir: :string,
          port: :integer,
          s3_port: :integer,
          db_url: :string,
          checkpoint: :string,
          top_k: :integer,
          limit: :integer,
          json: :boolean
        ]
      )

    {Map.new(parsed), args}
  end

  ## Subcommands

  defp cmd_add([], %{json: true} = opts) do
    with {:ok, entries} <- parse_json_items(IO.read(:stdio, :eof)) do
      Store.with_db(store_opts(opts), fn conn ->
        Enum.each(entries, fn {id, tokens} -> Store.upsert_item(conn, id, tokens) end)
        {:ok, "added #{length(entries)} items"}
      end)
      |> normalize()
    end
  end

  defp cmd_add([item_id, t0, t1, t2, t3], opts) do
    with {:ok, tokens} <- parse_tokens([t0, t1, t2, t3]) do
      Store.with_db(store_opts(opts), fn conn ->
        :ok = Store.upsert_item(conn, item_id, tokens)
        {:ok, "added #{item_id} #{inspect(tokens)}"}
      end)
      |> normalize()
    end
  end

  defp cmd_add(_, _opts),
    do:
      {:error, "usage: holo add <item_id> <t0> <t1> <t2> <t3>  |  holo add --json < items.json", 2}

  defp cmd_observe(ids, opts) when length(ids) >= 2 do
    Store.with_db(store_opts(opts), fn conn ->
      pairs = Enum.chunk_every(ids, 2, 1, :discard)
      Enum.each(pairs, fn [a, b] -> Store.record_transition(conn, a, b) end)
      {:ok, "recorded #{length(pairs)} transitions"}
    end)
    |> normalize()
  end

  defp cmd_observe(_, _opts),
    do: {:error, "usage: holo observe <item_id> <item_id> [...]  (at least 2 ids)", 2}

  defp cmd_recommend([], _opts),
    do:
      {:error, "usage: holo recommend <item_id> [...] [--top-k N] --checkpoint <export_dir>", 2}

  defp cmd_recommend(session, %{checkpoint: ckpt_dir} = opts) do
    Store.with_db(store_opts(opts), fn conn ->
      {token_id_list, item_ids} = ItemSource.load_catalog(Store, conn)

      cond do
        token_id_list == [] ->
          {:error, "holo: no items stored", 1}

        true ->
          index_of = item_ids |> Enum.with_index() |> Map.new()

          case map_session(session, index_of) do
            {:missing, missing} ->
              {:error, "holo: unknown item id(s): #{Enum.join(missing, ", ")}", 1}

            {:ok, context} ->
              serve =
                ckpt_dir
                |> NpyCheckpointSource.load_from_export()
                |> FuxiLinearInferenceParams.build_defn_params()
                |> Serve.new(token_id_list)

              case Serve.recommend(serve, context, opts[:top_k] || 5) do
                {:ok, indices} ->
                  ids = Enum.map(indices, &Enum.at(item_ids, &1))
                  {:ok, Jason.encode!(Enum.map(ids, &%{item_id: &1}))}

                :not_found ->
                  {:ok, Jason.encode!([])}
              end
          end
      end
    end)
    |> normalize()
  end

  defp cmd_recommend(_session, _opts),
    do:
      {:error,
       "holo: recommend requires --checkpoint <export_dir> (a trained FuXi-Linear " <>
         "checkpoint; the training-free HRR recommender was removed)", 2}

  # Map a session of item ids to model indices; {:missing, ids} if any are unknown.
  defp map_session(session, index_of) do
    {found, missing} =
      Enum.reduce(session, {[], []}, fn id, {f, m} ->
        case Map.fetch(index_of, id) do
          {:ok, idx} -> {[idx | f], m}
          :error -> {f, [id | m]}
        end
      end)

    case missing do
      [] -> {:ok, Enum.reverse(found)}
      _ -> {:missing, Enum.reverse(missing)}
    end
  end

  defp cmd_items(_rest, opts) do
    Store.with_db(store_opts(opts), fn conn ->
      items =
        conn
        |> Store.list_items(opts[:limit])
        |> Enum.map(fn {id, tokens} -> %{item_id: id, semantic_id: tokens} end)

      {:ok, Jason.encode!(items)}
    end)
    |> normalize()
  end

  defp cmd_blob(["put", name, file], opts) do
    Blob.with_gateway(blob_opts(opts), fn config ->
      case Blob.put(config, name, file) do
        {:ok, stats} -> {:ok, Jason.encode!(Map.put(stats, :name, name))}
        {:error, reason} -> {:error, "holo: #{reason}", 1}
      end
    end)
    |> normalize()
  end

  defp cmd_blob(["get", name, out_path], opts) do
    Blob.with_gateway(blob_opts(opts), fn config ->
      case Blob.get(config, name, out_path) do
        {:ok, stats} -> {:ok, Jason.encode!(Map.merge(stats, %{name: name, out: out_path}))}
        {:error, reason} -> {:error, "holo: #{reason}", 1}
      end
    end)
    |> normalize()
  end

  defp cmd_blob(["list"], opts) do
    Blob.with_gateway(blob_opts(opts), fn config ->
      {:ok, Jason.encode!(Blob.list(config))}
    end)
    |> normalize()
  end

  defp cmd_blob(_, _opts),
    do: {:error, "usage: holo blob put <name> <file> | blob get <name> <out> | blob list", 2}

  defp cmd_db_start(opts) do
    case Store.run_foreground(store_opts(opts)) do
      {:ok, out} -> {:ok, out}
      {:error, msg, code} -> {:error, msg, code}
      {:error, msg} -> {:error, msg, 1}
    end
  end

  ## Helpers

  defp store_opts(opts), do: Map.take(opts, [:data_dir, :port, :db_url])
  defp blob_opts(opts), do: Map.take(opts, [:data_dir, :s3_port])

  defp normalize({:ok, _} = ok), do: ok
  defp normalize({:error, msg, code}), do: {:error, msg, code}
  defp normalize({:error, reason}), do: {:error, "holo: #{inspect(reason)}", 1}

  @doc false
  def parse_tokens(strings) do
    tokens =
      Enum.map(strings, fn s ->
        case Integer.parse(s) do
          {n, ""} -> n
          _ -> :error
        end
      end)

    if :error in tokens or not ResidualFSQ.valid_id?(tokens) do
      {:error,
       "holo: semantic ID must be #{ResidualFSQ.tokens_per_item()} integers in " <>
         "0..#{ResidualFSQ.codebook_size() - 1}, got: #{inspect(strings)}", 2}
    else
      {:ok, tokens}
    end
  end

  @doc false
  def parse_json_items(body) do
    case Jason.decode(body) do
      {:ok, list} when is_list(list) ->
        entries =
          Enum.map(list, fn
            %{"item_id" => id, "semantic_id" => tokens} -> {id, tokens}
            _ -> :error
          end)

        if :error in entries or
             Enum.any?(entries, fn {_id, tokens} -> not ResidualFSQ.valid_id?(tokens) end) do
          {:error,
           ~s(holo: --json expects [{"item_id": "...", "semantic_id": [t0,t1,t2,t3]}, ...] ) <>
             "with tokens in 0..#{ResidualFSQ.codebook_size() - 1}", 2}
        else
          {:ok, entries}
        end

      _ ->
        {:error, "holo: stdin is not a JSON list", 2}
    end
  end

  defp usage do
    """
    holo #{@version} — holographic item memory over ResidualFSQ semantic IDs

    usage:
      holo add <item_id> <t0> <t1> <t2> <t3>       store an item's semantic ID
      holo add --json                          bulk add from stdin
      holo observe <id> <id> [...]             record a session's transitions
      holo recommend <id> [...] [--top-k N]    next-item recall (JSON);
                     --checkpoint <dir>        FuXi-Linear model over the catalog
      holo items [--limit N]                   list stored items (JSON)
      holo blob put <name> <file>              chunk + store a file (dedup)
      holo blob get <name> <out>               reassemble a stored blob
      holo blob list                           list stored blobs (JSON)
      holo db start                            run the embedded CockroachDB in foreground
      holo version | help

    global options:
      --data-dir DIR   data directory (default ~/.holographic-item-memory)
      --port N         embedded SQL port (default 26257)
      --s3-port N      embedded S3 gateway port (default 7070)
      --db-url URL     use an external cockroach/postgres
      --checkpoint DIR npy export of trained FuXi-Linear params (recommend)
    """
    |> String.trim_trailing()
  end

  defp resolve_argv do
    if System.get_env("__BURRITO") != nil do
      :init.get_plain_arguments() |> Enum.map(&to_string/1)
    else
      System.argv()
    end
  end

  # Wrapped so tests can exercise run/1 without halting; real halts the VM.
  defp halt(code), do: System.halt(code)
end
