defmodule Holo.Core.HRRTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Holo.Core.HRR

  @golden Path.join(__DIR__, "../../fixtures/hrr_golden.json")
          |> File.read!()
          |> Jason.decode!()

  @two_pi 2.0 * :math.pi()

  defp assert_close(tensor, expected_list, tol \\ 1.0e-9) do
    got = Nx.to_flat_list(tensor)
    assert length(got) == length(expected_list)

    Enum.zip(got, expected_list)
    |> Enum.with_index()
    |> Enum.each(fn {{g, e}, i} ->
      assert abs(g - e) < tol, "component #{i}: got #{g}, expected #{e}"
    end)
  end

  describe "golden parity with the Python reference" do
    test "encode_atom matches bit-for-bit semantics" do
      dim = @golden["dim"]
      assert_close(HRR.encode_atom("alice", dim), @golden["atom_alice"], 1.0e-12)
      assert_close(HRR.encode_atom("bob", dim), @golden["atom_bob"], 1.0e-12)
      assert_close(HRR.encode_atom("__holo_role_q0__", dim), @golden["atom_role_q0"], 1.0e-12)
    end

    test "encode_atom at dim 1024" do
      atom = HRR.encode_atom("alice", 1024)
      assert Nx.shape(atom) == {1024}
      assert_close(Nx.slice(atom, [0], [4]), @golden["atom_alice_1024_head"], 1.0e-12)
      mean = atom |> Nx.mean() |> Nx.to_number()
      assert abs(mean - @golden["atom_alice_1024_mean"]) < 1.0e-10
    end

    test "bind and unbind match" do
      dim = @golden["dim"]
      a = HRR.encode_atom("alice", dim)
      b = HRR.encode_atom("bob", dim)
      assert_close(HRR.bind(a, b), @golden["bind_alice_bob"])
      assert_close(HRR.unbind(HRR.bind(a, b), b), @golden["unbind_bind_alice_bob__bob"])
    end

    test "bundle matches" do
      dim = @golden["dim"]

      bundled =
        HRR.bundle([
          HRR.encode_atom("alice", dim),
          HRR.encode_atom("bob", dim),
          HRR.encode_atom("__holo_role_q0__", dim)
        ])

      assert_close(bundled, @golden["bundle_alice_bob_role"])
    end

    test "similarity matches" do
      dim = @golden["dim"]
      a = HRR.encode_atom("alice", dim)
      b = HRR.encode_atom("bob", dim)
      assert abs(HRR.similarity(a, a) - @golden["similarity_alice_alice"]) < 1.0e-12
      assert abs(HRR.similarity(a, b) - @golden["similarity_alice_bob"]) < 1.0e-12
    end

    test "encode_text matches (tokenization + bundling)" do
      dim = @golden["dim"]
      assert_close(HRR.encode_text("Hello, world! (hello)", dim), @golden["encode_text_hello"])
      assert_close(HRR.encode_text("  ", dim), @golden["encode_text_empty"])
    end
  end

  describe "algebra" do
    test "unbind recovers the bound value exactly" do
      a = HRR.encode_atom("key", 256)
      b = HRR.encode_atom("value", 256)
      recovered = HRR.unbind(HRR.bind(a, b), a)
      assert HRR.similarity(recovered, b) > 0.999999
    end

    test "bound composite is quasi-orthogonal to both inputs" do
      a = HRR.encode_atom("key", 1024)
      b = HRR.encode_atom("value", 1024)
      c = HRR.bind(a, b)
      assert abs(HRR.similarity(c, a)) < 0.2
      assert abs(HRR.similarity(c, b)) < 0.2
    end

    test "bundle is similar to each input, unrelated atoms are not" do
      dim = 1024
      atoms = for w <- ~w(red green blue), do: HRR.encode_atom(w, dim)
      bundled = HRR.bundle(atoms)

      for atom <- atoms do
        assert HRR.similarity(bundled, atom) > 0.4
      end

      stranger = HRR.encode_atom("orange", dim)
      assert abs(HRR.similarity(bundled, stranger)) < 0.2
    end

    property "bind then unbind is the identity on the phase grid" do
      check all(
              word_a <- StreamData.string(:alphanumeric, min_length: 1, max_length: 12),
              word_b <- StreamData.string(:alphanumeric, min_length: 1, max_length: 12),
              max_runs: 25
            ) do
        a = HRR.encode_atom(word_a, 64)
        b = HRR.encode_atom(word_b, 64)
        assert HRR.similarity(HRR.unbind(HRR.bind(a, b), b), a) > 0.999999
      end
    end

    property "all phases stay in [0, 2π)" do
      check all(
              word <- StreamData.string(:alphanumeric, min_length: 1, max_length: 12),
              max_runs: 25
            ) do
        a = HRR.encode_atom(word, 64)
        b = HRR.encode_atom(word <> "-x", 64)

        for t <- [a, HRR.bind(a, b), HRR.unbind(a, b), HRR.bundle([a, b])] do
          assert Nx.all(Nx.greater_equal(t, 0.0)) |> Nx.to_number() == 1
          assert Nx.all(Nx.less(t, @two_pi)) |> Nx.to_number() == 1
        end
      end
    end
  end

  test "serialization round-trips" do
    a = HRR.encode_atom("persist-me", 128)

    assert a |> HRR.to_binary() |> HRR.from_binary() |> Nx.equal(a) |> Nx.all() |> Nx.to_number() ==
             1
  end

  test "snr_estimate" do
    assert HRR.snr_estimate(1024, 0) == :infinity
    assert_in_delta HRR.snr_estimate(1024, 64), 4.0, 1.0e-12
  end
end
