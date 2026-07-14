# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Release do
  @moduledoc """
  Release-time helpers for the standalone `rfr` binary.

  `wrap/1` is the final `mix release` step. It packages the assembled release
  into a self-contained per-triplet binary with Burrito, but only when a zig
  cross-compiler is available (Burrito's backend) or the build explicitly
  opts in via `RFR_BURRITO=1`. Otherwise it returns the release untouched,
  so `mix release rfr` still succeeds on a machine without the toolchain;
  the shippable binaries are produced in CI where zig is installed.
  """

  require Logger

  @doc "Burrito wrap step, guarded on toolchain availability."
  def wrap(release) do
    if wrap?() do
      Burrito.wrap(release)
    else
      Logger.info(
        "rfr: skipping Burrito wrap (no zig toolchain / RFR_BURRITO unset) — " <>
          "assembled release only"
      )

      release
    end
  end

  defp wrap? do
    System.get_env("RFR_BURRITO") == "1" or System.find_executable("zig") != nil
  end
end

defmodule Recommender.Release.CockroachStep do
  @moduledoc """
  Burrito patch-phase step: bundle the per-target embedded service binaries.

  Delegates provisioning to `CockroachLocal.Provision.install/3` and
  `VersitygwLocal.Provision.install/3`, which download the matching
  [`V-Sekai/cockroach`](https://github.com/V-Sekai/cockroach) (22.1 LTS SQL
  store) and [`versity/versitygw`](https://github.com/versity/versitygw)
  (S3 gateway) single binaries, extract, and land them in the payload's
  `lib/residual_fsq_recommender-*/priv/<tool>/` so `CockroachLocal.bin/1` /
  `VersitygwLocal.bin/1` find them at runtime via `:code.priv_dir/1`.

  Only the target being built is bundled — each triplet's binary carries one
  cockroach + one versitygw.
  """

  @behaviour Burrito.Builder.Step

  require Logger

  @impl Burrito.Builder.Step
  def execute(%Burrito.Builder.Context{} = context) do
    target = {context.target.os, context.target.cpu}
    dest_root = priv_dir!(context.work_dir)

    {:ok, crdb} = CockroachLocal.Provision.install(target, dest_root)
    {:ok, vgw} = VersitygwLocal.Provision.install(target, dest_root)
    Logger.info("rfr: bundled #{Path.basename(crdb)} + #{Path.basename(vgw)} -> #{dest_root}")

    context
  end

  defp priv_dir!(work_dir) do
    case Path.wildcard(Path.join(work_dir, "lib/residual_fsq_recommender-*/priv")) do
      [priv | _] ->
        priv

      [] ->
        # priv/ may not exist yet if the app ships none — create it in the
        # versioned app dir.
        case Path.wildcard(Path.join(work_dir, "lib/residual_fsq_recommender-*")) do
          [app_dir | _] ->
            priv = Path.join(app_dir, "priv")
            File.mkdir_p!(priv)
            priv

          [] ->
            raise "residual_fsq_recommender app dir not found under #{work_dir}/lib"
        end
    end
  end
end
