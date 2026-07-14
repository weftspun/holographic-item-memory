# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Adapters.TrajectoryConvertTest do
  use ExUnit.Case, async: true

  alias Holo.Adapters.TrajectoryConvert, as: TC
  alias Holo.Core.Trajectories.LicenseError

  test "gate refuses a license-blocked dataset before reading" do
    assert_raise LicenseError, ~r/allowlist/, fn -> TC.sequences("merrec", "/nonexistent") end
  end

  test "gate refuses a non-Western (locale-blocked) dataset" do
    # yambda is Apache-2.0 (license-clean) but non-Western -> blocked by locale
    assert_raise LicenseError, ~r/English|Western/, fn -> TC.sequences("yambda", "/nonexistent") end
  end

  test "unknown dataset raises" do
    assert_raise ArgumentError, ~r/unknown dataset/, fn -> TC.sequences("nope", "/x") end
  end

  test "reads Amazon CC0 into per-user chronological ASIN sequences" do
    path = Path.join(System.tmp_dir!(), "amazon_#{System.unique_integer([:positive])}.csv")

    on_exit(fn -> File.rm(path) end)

    File.write!(path, """
    Order Date,Purchase Price Per Unit,Quantity,Shipping Address State,Title,ASIN/ISBN (Product Code),Category,Survey ResponseID
    2018-12-04,7.98,1.0,NJ,Card,B0143RTB1E,FLASH,R_user1
    2018-12-01,3.50,1.0,NJ,Cable,B01MA1MJ6H,CABLE,R_user1
    2019-01-15,9.99,2.0,CA,Book,B078JZTFN3,BOOK,R_user2
    """)

    seqs = TC.read_amazon_csv(path)

    u1 = Enum.find(seqs, &(&1["user"] == "R_user1"))
    u2 = Enum.find(seqs, &(&1["user"] == "R_user2"))

    # user1's two purchases sorted chronologically (Dec 1 before Dec 4)
    assert u1["sequence"] == ["B01MA1MJ6H", "B0143RTB1E"]
    assert u2["sequence"] == ["B078JZTFN3"]
  end
end
