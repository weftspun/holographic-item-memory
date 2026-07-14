defmodule Holo.Core.MemoryTest do
  use ExUnit.Case, async: true

  alias Holo.Core.HRR
  alias Holo.Core.Memory

  # Deterministic pseudo-random catalog: 40 items with distinct semantic IDs.
  defp catalog do
    for i <- 0..39 do
      {"item-#{i}", [rem(i * 97 + 13, 4096), rem(i * 389 + 7, 4096), rem(i * 811 + 3, 4096)]}
    end
  end

  defp loaded_memory(dim \\ 1024) do
    Memory.new(dim: dim) |> Memory.add_items(catalog())
  end

  describe "semantic IDs" do
    test "contract constants match the multimodal-semantic-ids decision record" do
      assert Memory.tokens_per_item() == 3
      assert Memory.codebook_size() == 4096
    end

    test "valid_id?" do
      assert Memory.valid_id?([0, 4095, 17])
      refute Memory.valid_id?([0, 4096, 17])
      refute Memory.valid_id?([0, 17])
      refute Memory.valid_id?([0, 1, -1])
      refute Memory.valid_id?(nil)
    end

    test "item_vector is deterministic" do
      a = Memory.item_vector([17, 900, 3], 256)
      b = Memory.item_vector([17, 900, 3], 256)
      assert Nx.equal(a, b) |> Nx.all() |> Nx.to_number() == 1
    end

    test "item_vector rejects invalid ids" do
      assert_raise ArgumentError, fn -> Memory.item_vector([1, 2], 64) end
      assert_raise ArgumentError, fn -> Memory.item_vector([1, 2, 4096], 64) end
    end

    test "items sharing semantic-id tokens are more similar than disjoint items" do
      dim = 1024
      base = Memory.item_vector([17, 900, 3], dim)
      near = Memory.item_vector([17, 900, 44], dim)
      far = Memory.item_vector([2000, 31, 999], dim)

      assert HRR.similarity(base, near) > HRR.similarity(base, far)
      assert HRR.similarity(base, near) > 0.2
      assert abs(HRR.similarity(base, far)) < 0.15
    end
  end

  test "add_items / size" do
    mem = loaded_memory()
    assert Memory.size(mem) == 40
  end

  test "recommend with no known session items errors" do
    assert {:error, _} = Memory.recommend(loaded_memory(), ["nope"])
  end

  test "content recall ranks an id-sharing item highly (zero-shot, no training)" do
    mem =
      loaded_memory()
      # twin shares the two coarse tokens of item-0 ([13, 7, 3] vs [13, 7, 999])
      |> Memory.add_item("twin-of-0", [13, 7, 999])

    {:ok, recs} = Memory.recommend(mem, ["item-0"], top_k: 3)
    top_ids = Enum.map(recs, &elem(&1, 0))
    assert "twin-of-0" in top_ids
  end

  test "transition recall: an observed a -> b transition ranks b first" do
    mem = loaded_memory()

    session = ["item-3", "item-11", "item-27"]
    mem = Memory.observe(mem, session)
    assert mem.n_transitions == 2

    {:ok, recs} = Memory.recommend(mem, ["item-11"], top_k: 1)
    assert [{"item-27", _score}] = recs
  end

  test "several transitions can be recalled from one bank" do
    mem = loaded_memory()

    transitions = [
      {"item-1", "item-20"},
      {"item-5", "item-33"},
      {"item-9", "item-14"}
    ]

    mem =
      Enum.reduce(transitions, mem, fn {a, b}, acc ->
        Memory.observe_transition(acc, a, b)
      end)

    for {a, b} <- transitions do
      {:ok, recs} = Memory.recommend(mem, [a], top_k: 3)
      top_ids = Enum.map(recs, &elem(&1, 0))
      assert b in top_ids, "expected #{b} in top-3 after #{a}, got #{inspect(top_ids)}"
    end
  end

  test "zero-shot: an item added after observations is immediately recommendable" do
    mem = loaded_memory()
    mem = Memory.observe(mem, ["item-2", "item-8"])

    # New item appears post-hoc with an ID sharing item-8's coarse tokens
    [t0, t1, _t2] = catalog() |> Enum.find(fn {id, _} -> id == "item-8" end) |> elem(1)
    mem = Memory.add_item(mem, "new-arrival", [t0, t1, 42])

    {:ok, recs} = Memory.recommend(mem, ["item-8"], top_k: 5)
    top_ids = Enum.map(recs, &elem(&1, 0))
    assert "new-arrival" in top_ids
  end

  test "seen items are excluded by default, includable on request" do
    mem = loaded_memory()
    {:ok, recs} = Memory.recommend(mem, ["item-0", "item-1"], top_k: 40)
    ids = Enum.map(recs, &elem(&1, 0))
    refute "item-0" in ids
    refute "item-1" in ids

    {:ok, recs2} = Memory.recommend(mem, ["item-0"], top_k: 40, exclude_seen: false)
    assert "item-0" in Enum.map(recs2, &elem(&1, 0))
  end

  test "snr_estimate reflects transition count" do
    mem = loaded_memory()
    assert Memory.snr_estimate(mem) == :infinity
    mem = Memory.observe(mem, ["item-0", "item-1", "item-2"])
    assert_in_delta Memory.snr_estimate(mem), :math.sqrt(1024 / 2), 1.0e-9
  end

  test "observe skips unknown items" do
    mem = loaded_memory()
    mem = Memory.observe(mem, ["item-0", "ghost", "item-1"])
    # ghost breaks both pairs
    assert mem.n_transitions == 0
  end
end
