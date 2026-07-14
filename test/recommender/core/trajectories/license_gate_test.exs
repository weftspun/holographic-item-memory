# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Core.Trajectories.LicenseGateTest do
  use ExUnit.Case, async: true

  alias Recommender.Core.Trajectories.LicenseGate
  alias Recommender.Core.Trajectories.LicenseError

  describe "allowlist / allowed?" do
    test "permissive licenses pass" do
      for spdx <- ~w(CC0-1.0 MIT Apache-2.0 CC-BY-4.0) do
        assert LicenseGate.allowed?(spdx), "#{spdx} should be allowed"
      end
    end

    test "copyleft, NonCommercial, and custom licenses are blocked" do
      for spdx <- ~w(CC-BY-SA-4.0 CC-BY-NC-4.0 LicenseRef-GroupLens GPL-3.0-only) do
        refute LicenseGate.allowed?(spdx), "#{spdx} should be blocked"
      end
    end
  end

  describe "assert_allowed!/1" do
    test "returns :ok for a permissive spdx string" do
      assert :ok == LicenseGate.assert_allowed!("CC0-1.0")
    end

    test "raises for a blocked spdx string" do
      assert_raise LicenseError, ~r/not on the permissive allowlist/, fn ->
        LicenseGate.assert_allowed!("CC-BY-SA-4.0")
      end
    end

    test "accepts a dataset map and gates on its spdx" do
      assert :ok == LicenseGate.assert_allowed!(%{"name" => "x", "spdx" => "CC-BY-4.0"})

      assert_raise LicenseError, ~r/CC-BY-NC-4.0/, fn ->
        LicenseGate.assert_allowed!(%{"name" => "merrec", "spdx" => "CC-BY-NC-4.0"})
      end
    end

    test "raises when a dataset entry has no license recorded" do
      assert_raise LicenseError, ~r/no spdx license/, fn ->
        LicenseGate.assert_allowed!(%{"name" => "mystery"})
      end
    end
  end

  describe "manifest scan" do
    test "manifest allowlist matches the module allowlist" do
      assert Enum.sort(LicenseGate.manifest()["allowlist"]) == Enum.sort(LicenseGate.allowlist())
    end

    test "ingestible = license-clean AND English/Western" do
      allowed = MapSet.new(LicenseGate.allowed_datasets())

      # English/Western + permissive license
      assert "open-ecommerce" in allowed
      assert "otto" in allowed
      # license-clean (Apache-2.0) but non-Western -> excluded by locale filter
      refute "yambda" in allowed
      refute "al-gr" in allowed
      refute "taobao-mm" in allowed
      # license-blocked
      refute "KuaiRand-Pure" in allowed
      refute "merrec" in allowed
      refute "movielens-20m" in allowed
    end

    test "license gate is locale-agnostic: non-Western permissive corpora are license-clean" do
      clean = MapSet.new(LicenseGate.license_clean_datasets())
      # Apache-2.0 datasets pass the LICENSE gate even though the locale filter excludes them
      assert "yambda" in clean
      assert "al-gr" in clean
      refute "merrec" in clean
    end

    test "every allowed dataset in the manifest carries a permissive spdx" do
      for entry <- LicenseGate.scan(), entry["allowed"] do
        assert LicenseGate.allowed?(entry["spdx"])
      end
    end
  end
end
