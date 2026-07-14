# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Adapters.CockroachStore do
  @moduledoc """
  Embedded CockroachDB persistence for the `holo` CLI.

  The standalone binary ships the single-binary `cockroach` from
  [`V-Sekai/cockroach`](https://github.com/V-Sekai/cockroach) (22.1 LTS) in
  `priv/cockroach/`. `with_db/2` runs one command's work against it:

    1. If nothing answers on the SQL port, spawn
       `cockroach start-single-node --insecure` on the data dir as a child
       OS process (foreground, held by an Erlang port).
    2. Poll `Postgrex` until the node accepts connections, ensure the schema.
    3. Run the caller's function with the connection.
    4. If this process started the node, terminate it (`kill` / `taskkill`).

  If a node is already listening (e.g. `holo db start` in another terminal,
  or an external cluster via `--db-url`), it is reused and left running.

  Schema — items are semantic IDs, transitions the hetero-associative counts:

      items(item_id STRING PRIMARY KEY, t0 INT, t1 INT, t2 INT, t3 INT)
      transitions(prev STRING, next STRING, n INT, PRIMARY KEY (prev, next))
  """

  require Logger

  @default_port 26_257
  @ready_timeout_ms 60_000
  @poll_interval_ms 250

  @type opts :: %{
          optional(:data_dir) => String.t(),
          optional(:port) => pos_integer(),
          optional(:db_url) => String.t() | nil
        }

  @doc "Default data directory (override with `--data-dir` or `HOLO_DATA_DIR`)."
  def default_data_dir do
    System.get_env("HOLO_DATA_DIR") ||
      Path.join(System.user_home!(), ".holographic-item-memory")
  end

  @doc """
  Run `fun.(conn)` against the store, starting (and stopping) an embedded
  single-node cockroach when nothing is listening. Returns `fun`'s result,
  or `{:error, reason}`.
  """
  @spec with_db(opts(), (pid() -> result)) :: result | {:error, term()} when result: var
  def with_db(opts \\ %{}, fun) do
    conn_opts = conn_opts(opts)
    port = Keyword.fetch!(conn_opts, :port)

    {started_port, os_pid} =
      if is_nil(opts[:db_url]) and not listening?(port) do
        spawn_cockroach(opts, port)
      else
        {nil, nil}
      end

    try do
      case await_conn(conn_opts, @ready_timeout_ms) do
        {:ok, conn} ->
          try do
            :ok = ensure_schema(conn)
            fun.(conn)
          after
            GenServer.stop(conn, :normal, 5_000)
          end

        {:error, reason} ->
          {:error, "database did not become ready: #{inspect(reason)}"}
      end
    after
      stop_cockroach(started_port, os_pid)
    end
  end

  @doc """
  Run the embedded cockroach in the foreground (for `holo db start`),
  streaming its output. Blocks until the node exits.
  """
  @spec run_foreground(opts()) :: {:ok, iodata()} | {:error, iodata(), pos_integer()}
  def run_foreground(opts \\ %{}) do
    with {:ok, bin} <- cockroach_bin() do
      args = start_args(opts, opts[:port] || @default_port)
      {_out, status} = System.cmd(bin, args, into: IO.stream(:stdio, :line))

      if status == 0,
        do: {:ok, ""},
        else: {:error, "cockroach exited with status #{status}", status}
    end
  end

  ## Item / transition persistence (Holo.Ports.ItemSink / ItemSource)

  @behaviour Holo.Ports.ItemSink
  @behaviour Holo.Ports.ItemSource

  @impl Holo.Ports.ItemSink
  def upsert_item(conn, item_id, [t0, t1, t2, t3]) do
    Postgrex.query!(
      conn,
      "UPSERT INTO items (item_id, t0, t1, t2, t3) VALUES ($1, $2, $3, $4, $5)",
      [item_id, t0, t1, t2, t3]
    )

    :ok
  end

  @impl Holo.Ports.ItemSink
  def record_transition(conn, prev, next) do
    Postgrex.query!(
      conn,
      """
      INSERT INTO transitions (prev, next, n) VALUES ($1, $2, 1)
      ON CONFLICT (prev, next) DO UPDATE SET n = transitions.n + 1
      """,
      [prev, next]
    )

    :ok
  end

  @impl Holo.Ports.ItemSource
  def list_items(conn, limit) do
    sql = "SELECT item_id, t0, t1, t2, t3 FROM items ORDER BY item_id" <> limit_clause(limit)

    for [id, t0, t1, t2, t3] <- Postgrex.query!(conn, sql, []).rows do
      {id, [t0, t1, t2, t3]}
    end
  end

  @impl Holo.Ports.ItemSource
  def list_transitions(conn) do
    for [prev, next, n] <-
          Postgrex.query!(conn, "SELECT prev, next, n FROM transitions", []).rows do
      {prev, next, n}
    end
  end

  ## Cockroach binary + lifecycle

  @doc """
  Path to the cockroach binary: `HOLO_COCKROACH_BIN`, then the bundled
  `priv/cockroach/`, then `$PATH`.
  """
  @spec cockroach_bin() :: {:ok, String.t()} | {:error, String.t()}
  def cockroach_bin do
    exe = if match?({:win32, _}, :os.type()), do: "cockroach.exe", else: "cockroach"

    bundled =
      case :code.priv_dir(:holographic_item_memory) do
        {:error, _} -> nil
        priv -> Path.join([to_string(priv), "cockroach", exe])
      end

    cond do
      bin = System.get_env("HOLO_COCKROACH_BIN") -> {:ok, bin}
      bundled && File.exists?(bundled) -> {:ok, bundled}
      bin = System.find_executable("cockroach") -> {:ok, bin}
      true -> {:error, "no cockroach binary: not bundled in priv/cockroach, not on PATH"}
    end
  end

  defp conn_opts(opts) do
    case opts[:db_url] do
      nil ->
        [
          hostname: "localhost",
          port: opts[:port] || @default_port,
          username: "root",
          database: "defaultdb",
          ssl: false
        ]

      url ->
        uri = URI.parse(url)
        [username, password] = String.split(uri.userinfo || "root", ":") |> pad2()

        [
          hostname: uri.host || "localhost",
          port: uri.port || @default_port,
          username: username,
          password: password,
          database: String.trim_leading(uri.path || "/defaultdb", "/"),
          ssl: uri.query != nil and String.contains?(uri.query, "sslmode=require")
        ]
        |> Enum.reject(fn {_k, v} -> is_nil(v) end)
    end
  end

  defp pad2([a]), do: [a, nil]
  defp pad2([a, b | _]), do: [a, b]

  defp listening?(port) do
    case :gen_tcp.connect(~c"localhost", port, [:binary, active: false], 500) do
      {:ok, sock} ->
        :gen_tcp.close(sock)
        true

      {:error, _} ->
        false
    end
  end

  defp spawn_cockroach(opts, port) do
    case cockroach_bin() do
      {:ok, bin} ->
        args = start_args(opts, port)
        Logger.debug("holo: starting embedded cockroach on port #{port}")

        erl_port =
          Port.open({:spawn_executable, bin}, [
            :binary,
            :exit_status,
            :stderr_to_stdout,
            args: args
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

  defp start_args(opts, port) do
    data_dir = opts[:data_dir] || default_data_dir()
    File.mkdir_p!(data_dir)

    [
      "start-single-node",
      "--insecure",
      "--store=path=#{data_dir}",
      "--listen-addr=localhost:#{port}",
      "--http-addr=localhost:0"
    ]
  end

  defp stop_cockroach(nil, _), do: :ok

  defp stop_cockroach(erl_port, os_pid) do
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

  defp await_conn(conn_opts, timeout_ms) do
    deadline = System.monotonic_time(:millisecond) + timeout_ms
    do_await(conn_opts, deadline)
  end

  defp do_await(conn_opts, deadline) do
    case try_connect(conn_opts) do
      {:ok, conn} ->
        {:ok, conn}

      {:error, reason} ->
        if System.monotonic_time(:millisecond) > deadline do
          {:error, reason}
        else
          Process.sleep(@poll_interval_ms)
          do_await(conn_opts, deadline)
        end
    end
  end

  defp try_connect(conn_opts) do
    # A quiet one-shot probe: start Postgrex and issue SELECT 1.
    case Postgrex.start_link(conn_opts ++ [backoff_type: :stop, sync_connect: false]) do
      {:ok, conn} ->
        case safe_query(conn, "SELECT 1") do
          {:ok, _} ->
            {:ok, conn}

          {:error, reason} ->
            GenServer.stop(conn, :normal, 1_000)
            {:error, reason}
        end

      {:error, reason} ->
        {:error, reason}
    end
  catch
    :exit, reason -> {:error, reason}
  end

  defp safe_query(conn, sql) do
    Postgrex.query(conn, sql, [])
  catch
    :exit, reason -> {:error, reason}
  end

  defp ensure_schema(conn) do
    Postgrex.query!(
      conn,
      """
      CREATE TABLE IF NOT EXISTS items (
        item_id STRING PRIMARY KEY,
        t0 INT NOT NULL, t1 INT NOT NULL, t2 INT NOT NULL, t3 INT NOT NULL
      )
      """,
      []
    )

    Postgrex.query!(
      conn,
      """
      CREATE TABLE IF NOT EXISTS transitions (
        prev STRING NOT NULL, next STRING NOT NULL, n INT NOT NULL DEFAULT 1,
        PRIMARY KEY (prev, next)
      )
      """,
      []
    )

    :ok
  end

  defp limit_clause(nil), do: ""
  defp limit_clause(n) when is_integer(n) and n > 0, do: " LIMIT #{n}"
end
