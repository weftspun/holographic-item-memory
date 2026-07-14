# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Core.Trajectories.LicenseError do
  @moduledoc "Raised when a trajectory dataset fails the permissive-license gate."
  defexception [:message]
end

defmodule Recommender.Core.Trajectories.LicenseGate do
  @moduledoc """
  Hard license gate for trajectory training corpora.

  Only datasets under a **permissive** license may be ingested for tokenization
  or training. The allowlist is:

      CC0-1.0  ·  MIT  ·  Apache-2.0  ·  CC-BY-4.0

  These permit commercial use and redistribution without copyleft (ShareAlike)
  or NonCommercial restrictions. CC-BY-4.0 is included on the attribution tier
  (its notice requirement is analogous to Apache-2.0's NOTICE). Everything else
  — copyleft (`*-SA-*`), NonCommercial (`*-NC-*`), and custom no-redistribute
  licenses like GroupLens/MovieLens — is **blocked**.

  This is a hard gate with **no override**: `assert_allowed!/1` raises for any
  dataset not on the allowlist, so a non-permissive corpus can never silently
  enter the training pipeline. The dataset↔license mapping lives in
  `priv/trajectories/licenses.json`.
  """

  alias Recommender.Core.Trajectories.LicenseError

  @allowlist ~w(CC0-1.0 MIT Apache-2.0 CC-BY-4.0)

  @manifest_path Application.app_dir(:residual_fsq_recommender, "priv/trajectories/licenses.json")

  @doc "The permissive SPDX allowlist enforced by the gate."
  @spec allowlist() :: [String.t()]
  def allowlist, do: @allowlist

  @doc "True iff `spdx` is a permissive, gate-clean license identifier."
  @spec allowed?(String.t()) :: boolean()
  def allowed?(spdx) when is_binary(spdx), do: spdx in @allowlist

  @doc """
  Assert that a license or dataset entry passes the gate; return `:ok` or raise
  `Recommender.Core.Trajectories.LicenseError`.

  Accepts a bare SPDX string, or a dataset map with a `"spdx"`/`:spdx` key (as
  loaded from the manifest).
  """
  @spec assert_allowed!(String.t() | map()) :: :ok
  def assert_allowed!(spdx) when is_binary(spdx) do
    if allowed?(spdx) do
      :ok
    else
      raise LicenseError,
        message:
          "license #{inspect(spdx)} is not on the permissive allowlist " <>
            "#{inspect(@allowlist)}; refusing to ingest. Non-permissive datasets " <>
            "(copyleft/NonCommercial/no-redistribute) are blocked with no override."
    end
  end

  def assert_allowed!(%{} = entry) do
    spdx = entry["spdx"] || entry[:spdx]
    name = entry["name"] || entry[:name] || "(unnamed)"

    unless is_binary(spdx) do
      raise LicenseError, message: "dataset #{inspect(name)} has no spdx license recorded"
    end

    if allowed?(spdx) do
      :ok
    else
      raise LicenseError,
        message:
          "dataset #{inspect(name)} is licensed #{inspect(spdx)}, which is not on the " <>
            "permissive allowlist #{inspect(@allowlist)}; refusing to ingest."
    end
  end

  @doc "Gate verdict for one manifest entry: `{spdx, allowed?, reason}` folded into a map."
  @spec verdict(map()) :: %{name: String.t(), spdx: String.t() | nil, allowed: boolean()}
  def verdict(%{} = entry) do
    spdx = entry["spdx"] || entry[:spdx]

    %{
      name: entry["name"] || entry[:name],
      spdx: spdx,
      allowed: is_binary(spdx) and allowed?(spdx)
    }
  end

  @doc "Load and decode the dataset license manifest (`priv/trajectories/licenses.json`)."
  @spec manifest() :: map()
  def manifest do
    @manifest_path
    |> File.read!()
    |> Jason.decode!()
  end

  @doc "All manifest dataset entries with their gate verdict attached."
  @spec scan() :: [map()]
  def scan do
    manifest()
    |> Map.fetch!("datasets")
    |> Enum.map(fn entry -> Map.put(entry, "allowed", verdict(entry).allowed) end)
  end

  @doc """
  Names of the datasets that are **ingestible**: license-clean (on the permissive
  allowlist) AND English/Western locale (`"english": true` in the manifest).

  The license gate is the hard, no-override rule; the English/Western filter is an
  additional project requirement — non-Western corpora (Alibaba/Taobao/AL-GR,
  Yandex Yambda, VK-LSVD) are excluded even when their license is permissive.
  """
  @spec allowed_datasets() :: [String.t()]
  def allowed_datasets do
    scan()
    |> Enum.filter(fn e -> e["allowed"] and e["english"] == true end)
    |> Enum.map(& &1["name"])
  end

  @doc "Names of datasets that pass the license gate, ignoring the locale filter."
  @spec license_clean_datasets() :: [String.t()]
  def license_clean_datasets do
    scan()
    |> Enum.filter(& &1["allowed"])
    |> Enum.map(& &1["name"])
  end
end
