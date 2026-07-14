defmodule Holo.SemanticIDTest do
  use ExUnit.Case, async: true

  alias Holo.HRR
  alias Holo.SemanticID

  test "item_vector is deterministic" do
    a = SemanticID.item_vector([17, 900, 3], 256)
    b = SemanticID.item_vector([17, 900, 3], 256)
    assert Nx.equal(a, b) |> Nx.all() |> Nx.to_number() == 1
  end

  test "item_vector rejects invalid ids" do
    assert_raise ArgumentError, fn -> SemanticID.item_vector([1, 2], 64) end
    assert_raise ArgumentError, fn -> SemanticID.item_vector([1, 2, 4096], 64) end
  end

  test "items sharing semantic-id tokens are more similar than disjoint items" do
    dim = 1024
    base = SemanticID.item_vector([17, 900, 3], dim)
    near = SemanticID.item_vector([17, 900, 44], dim)
    far = SemanticID.item_vector([2000, 31, 999], dim)

    assert HRR.similarity(base, near) > HRR.similarity(base, far)
    assert HRR.similarity(base, near) > 0.2
    assert abs(HRR.similarity(base, far)) < 0.15
  end

  test "flat_key round-trips and orders stages" do
    id = [5, 17, 4095]
    key = SemanticID.flat_key(id)
    assert key == 5 + 4096 * 17 + 4096 * 4096 * 4095
    assert SemanticID.from_flat_key(key) == id
  end

  test "load_parquet reads asset_semantic_id with scalar sid columns" do
    path =
      Path.join(
        System.tmp_dir!(),
        "holo_asset_semantic_id_#{System.unique_integer([:positive])}.parquet"
      )

    df =
      Explorer.DataFrame.new(%{
        "asset_id" => ["a", "b", "c"],
        "sid_0" => [1, 2, 5000],
        "sid_1" => [10, 20, 1],
        "sid_2" => [100, 200, 2]
      })

    :ok = Explorer.DataFrame.to_parquet(df, path)

    assert {:ok, pairs} = SemanticID.load_parquet(path)
    # the 5000 row is invalid (>= 4096) and filtered out
    assert pairs == [{"a", [1, 10, 100]}, {"b", [2, 20, 200]}]
  after
    :ok
  end

  test "load_parquet reads a semantic_id list column" do
    path =
      Path.join(System.tmp_dir!(), "holo_sid_list_#{System.unique_integer([:positive])}.parquet")

    df =
      Explorer.DataFrame.new(%{
        "item_id" => [7, 8],
        "semantic_id" => [[1, 2, 3], [4, 5, 6]]
      })

    :ok = Explorer.DataFrame.to_parquet(df, path)

    assert {:ok, pairs} = SemanticID.load_parquet(path)
    assert pairs == [{7, [1, 2, 3]}, {8, [4, 5, 6]}]
  end

  test "load_parquet errors on missing columns" do
    path = Path.join(System.tmp_dir!(), "holo_bad_#{System.unique_integer([:positive])}.parquet")
    df = Explorer.DataFrame.new(%{"whatever" => [1]})
    :ok = Explorer.DataFrame.to_parquet(df, path)
    assert {:error, _} = SemanticID.load_parquet(path)
  end
end
