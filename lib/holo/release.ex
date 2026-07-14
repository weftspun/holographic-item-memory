# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Release do
  @moduledoc """
  Release-time helpers for the standalone `holo` binary.

  `wrap/1` is the final `mix release` step. It packages the assembled release
  into a self-contained per-triplet binary with Burrito, but only when a zig
  cross-compiler is available (Burrito's backend) or the build explicitly
  opts in via `HOLO_BURRITO=1`. Otherwise it returns the release untouched,
  so `mix release holo` still succeeds on a machine without the toolchain;
  the shippable binaries are produced in CI where zig is installed.
  """

  require Logger

  @doc "Burrito wrap step, guarded on toolchain availability."
  def wrap(release) do
    if wrap?() do
      Burrito.wrap(release)
    else
      Logger.info(
        "holo: skipping Burrito wrap (no zig toolchain / HOLO_BURRITO unset) — " <>
          "assembled release only"
      )

      release
    end
  end

  defp wrap? do
    System.get_env("HOLO_BURRITO") == "1" or System.find_executable("zig") != nil
  end
end

defmodule Holo.Release.CockroachStep do
  @moduledoc """
  Burrito patch-phase step: bundle the per-target embedded service binaries.

  Downloads the matching single-binary release assets for
  [`V-Sekai/cockroach`](https://github.com/V-Sekai/cockroach) (22.1 LTS SQL
  store) and [`versity/versitygw`](https://github.com/versity/versitygw)
  (S3 gateway for the aria-storage blob layer), extracts the executables, and
  lands them in the payload's `lib/holographic_item_memory-*/priv/<tool>/` so
  `Holo.Adapters.CockroachStore.cockroach_bin/0` / `Holo.Adapters.VersityBlobStore.versitygw_bin/0` find them at
  runtime via `:code.priv_dir/1`.

  Only the target being built is bundled — each triplet's binary carries one
  cockroach + one versitygw. Downloads are cached under
  `~/.cache/holo_cockroach/`.
  """

  @behaviour Burrito.Builder.Step

  require Logger

  @crdb_tag "v22.1.64b21683521d9a8735ad"
  @crdb_base "https://github.com/V-Sekai/cockroach/releases/download/#{@crdb_tag}"
  @vgw_tag "v1.6.0"
  @vgw_base "https://github.com/versity/versitygw/releases/download/#{@vgw_tag}"

  @assets %{
    {:linux, :x86_64} => %{
      "cockroach" => "#{@crdb_base}/cockroach-#{@crdb_tag}.linux-amd64.tgz",
      "versitygw" => "#{@vgw_base}/versitygw_#{@vgw_tag}_Linux_x86_64.tar.gz"
    },
    {:darwin, :aarch64} => %{
      "cockroach" => "#{@crdb_base}/cockroach-#{@crdb_tag}.darwin-arm64.tgz",
      "versitygw" => "#{@vgw_base}/versitygw_#{@vgw_tag}_Darwin_arm64.tar.gz"
    },
    {:windows, :x86_64} => %{
      "cockroach" => "#{@crdb_base}/cockroach-v22.1.22.windows-6.2-amd64.zip",
      "versitygw" => "#{@vgw_base}/versitygw_#{@vgw_tag}_Windows_x86_64.zip"
    }
  }

  @impl Burrito.Builder.Step
  def execute(%Burrito.Builder.Context{} = context) do
    os = context.target.os
    cpu = context.target.cpu

    case Map.fetch(@assets, {os, cpu}) do
      {:ok, tools} ->
        dest_root = priv_dir!(context.work_dir)

        for {tool, url} <- tools do
          exe = if os == :windows, do: "#{tool}.exe", else: tool
          bin = fetch_tool(url, exe)
          File.mkdir_p!(Path.join(dest_root, tool))
          dest = Path.join([dest_root, tool, exe])
          File.cp!(bin, dest)
          File.chmod!(dest, 0o755)
          Logger.info("holo: bundled #{Path.basename(url)} -> #{dest}")
        end

        context

      :error ->
        raise "no embedded-service assets for target #{os}/#{cpu}"
    end
  end

  defp priv_dir!(work_dir) do
    case Path.wildcard(Path.join(work_dir, "lib/holographic_item_memory-*/priv")) do
      [priv | _] ->
        priv

      [] ->
        # priv/ may not exist yet if the app ships none — create it in the
        # versioned app dir.
        case Path.wildcard(Path.join(work_dir, "lib/holographic_item_memory-*")) do
          [app_dir | _] ->
            priv = Path.join(app_dir, "priv")
            File.mkdir_p!(priv)
            priv

          [] ->
            raise "holographic_item_memory app dir not found under #{work_dir}/lib"
        end
    end
  end

  defp fetch_tool(url, exe) do
    asset = Path.basename(url)
    cache = Path.join([System.user_home!(), ".cache", "holo_cockroach", asset])
    extracted = Path.join(Path.dirname(cache), "#{asset}.extracted")
    File.mkdir_p!(Path.dirname(cache))

    unless File.exists?(cache) do
      Logger.info("holo: downloading #{url}")
      {_, 0} = System.cmd("curl", ["-fsSL", "--retry", "3", "-o", cache, url])
    end

    unless File.dir?(extracted) do
      tmp = extracted <> ".tmp"
      File.rm_rf!(tmp)
      File.mkdir_p!(tmp)

      # Extract with Erlang built-ins — GNU tar (e.g. Git Bash on Windows)
      # cannot read .zip, so no external tool is portable enough.
      if String.ends_with?(asset, ".zip") do
        {:ok, _} = :zip.extract(String.to_charlist(cache), cwd: String.to_charlist(tmp))
      else
        :ok =
          :erl_tar.extract(String.to_charlist(cache), [
            :compressed,
            {:cwd, String.to_charlist(tmp)}
          ])
      end

      File.rename!(tmp, extracted)
    end

    case Path.wildcard(Path.join(extracted, "**/#{exe}")) do
      [bin | _] -> bin
      [] -> raise "no #{exe} inside #{asset}"
    end
  end
end
