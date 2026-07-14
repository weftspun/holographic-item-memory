# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Adapters.CockroachStore do
  @moduledoc """
  Embedded CockroachDB persistence for the `rfr` CLI.

  The **host lifecycle** — provisioning the single `cockroach` binary, starting/
  stopping an embedded single-node instance, and handing over a `Postgrex`
  connection — is delegated to [`CockroachLocal`](https://github.com/weftspun/cockroach-local).
  This module keeps only the app-specific parts: the schema and the item/
  transition queries behind `Recommender.Ports.ItemSink` / `Recommender.Ports.ItemSource`.

  `with_db/2` runs one command's work against the store: it delegates to
  `CockroachLocal.with_db/2` (which reuses a running node or spawns one), ensures
  the schema, then runs the caller's function with the connection.

  Schema — items are semantic IDs, transitions the hetero-associative counts:

      items(item_id STRING PRIMARY KEY, t0 INT, t1 INT, t2 INT, t3 INT)
      transitions(prev STRING, next STRING, n INT, PRIMARY KEY (prev, next))
  """

  @type opts :: %{
          optional(:data_dir) => String.t(),
          optional(:port) => pos_integer(),
          optional(:db_url) => String.t() | nil
        }

  @doc "Default data directory (override with `--data-dir` or `RFR_DATA_DIR`)."
  def default_data_dir do
    System.get_env("RFR_DATA_DIR") ||
      Path.join(System.user_home!(), ".residual-fsq-recommender")
  end

  @doc """
  Run `fun.(conn)` against the store, starting (and stopping) an embedded
  single-node cockroach when nothing is listening. Ensures the schema first.
  Returns `fun`'s result, or `{:error, reason}`.
  """
  @spec with_db(opts(), (pid() -> result)) :: result | {:error, term()} when result: var
  def with_db(opts \\ %{}, fun) do
    CockroachLocal.with_db(local_opts(opts), fn conn ->
      :ok = ensure_schema(conn)
      fun.(conn)
    end)
  end

  @doc """
  Run the embedded cockroach in the foreground (for `rfr db start`),
  streaming its output. Blocks until the node exits.
  """
  @spec run_foreground(opts()) :: {:ok, iodata()} | {:error, iodata(), pos_integer()}
  def run_foreground(opts \\ %{}) do
    CockroachLocal.run_foreground(local_opts(opts))
  end

  ## Item / transition persistence (Recommender.Ports.ItemSink / ItemSource)

  @behaviour Recommender.Ports.ItemSink
  @behaviour Recommender.Ports.ItemSource

  @impl Recommender.Ports.ItemSink
  def upsert_item(conn, item_id, [t0, t1, t2, t3]) do
    Postgrex.query!(
      conn,
      "UPSERT INTO items (item_id, t0, t1, t2, t3) VALUES ($1, $2, $3, $4, $5)",
      [item_id, t0, t1, t2, t3]
    )

    :ok
  end

  @impl Recommender.Ports.ItemSink
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

  @impl Recommender.Ports.ItemSource
  def list_items(conn, limit) do
    sql = "SELECT item_id, t0, t1, t2, t3 FROM items ORDER BY item_id" <> limit_clause(limit)

    for [id, t0, t1, t2, t3] <- Postgrex.query!(conn, sql, []).rows do
      {id, [t0, t1, t2, t3]}
    end
  end

  @impl Recommender.Ports.ItemSource
  def list_transitions(conn) do
    for [prev, next, n] <-
          Postgrex.query!(conn, "SELECT prev, next, n FROM transitions", []).rows do
      {prev, next, n}
    end
  end

  ## Schema + option mapping

  # Map the CLI's opts map into the keyword opts CockroachLocal expects, pointing
  # binary resolution at this app's bundled priv/ and env var.
  defp local_opts(opts) do
    [
      data_dir: opts[:data_dir] || default_data_dir(),
      port: opts[:port],
      db_url: opts[:db_url],
      priv_app: :residual_fsq_recommender,
      bin_env: "RFR_COCKROACH_BIN"
    ]
    |> Enum.reject(fn {_k, v} -> is_nil(v) end)
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
