defmodule Holo.ResidualFSQTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Holo.ResidualFSQ

  test "contract constants match the multimodal-semantic-ids decision record" do
    assert ResidualFSQ.levels() == [8, 8, 8, 8]
    assert ResidualFSQ.num_quantizers() == 3
    assert ResidualFSQ.codebook_size() == 4096
    assert ResidualFSQ.basis() == [1, 8, 64, 512]
    assert ResidualFSQ.tokens_per_item() == 3
  end

  property "codes_to_index / index_to_codes round-trip on all digits" do
    check all(digits <- StreamData.list_of(StreamData.integer(0..7), length: 4)) do
      index = ResidualFSQ.codes_to_index(digits)
      assert index >= 0 and index < 4096
      assert ResidualFSQ.index_to_codes(index) == digits
    end
  end

  property "codes_to_index is injective" do
    check all(
            a <- StreamData.list_of(StreamData.integer(0..7), length: 4),
            b <- StreamData.list_of(StreamData.integer(0..7), length: 4)
          ) do
      if ResidualFSQ.codes_to_index(a) == ResidualFSQ.codes_to_index(b) do
        assert a == b
      end
    end
  end

  test "encode is deterministic and in range" do
    latent = [0.31, -0.7, 1.2, -0.05]
    id = ResidualFSQ.encode(latent)
    assert id == ResidualFSQ.encode(latent)
    assert ResidualFSQ.valid_id?(id)
  end

  test "residual stages reduce reconstruction error" do
    latent = Nx.tensor([0.63, -0.41, 0.9, -1.3], type: {:f, 64})
    [t0, t1, t2] = ResidualFSQ.encode(latent)

    # Reconstruction with more stages should not be worse than stage 1 alone.
    err = fn tokens_padded ->
      ResidualFSQ.decode(tokens_padded)
      |> Nx.subtract(latent)
      |> Nx.abs()
      |> Nx.sum()
      |> Nx.to_number()
    end

    # Pad "fewer stages" with the neutral digit index (digits [4,4,4,4] -> code 0.0)
    zero = ResidualFSQ.codes_to_index([4, 4, 4, 4])
    assert err.([t0, t1, t2]) <= err.([t0, zero, zero]) + 1.0e-9
    assert err.([t0, t1, zero]) <= err.([t0, zero, zero]) + 1.0e-9
  end

  test "decode of the neutral id is the zero vector" do
    zero = ResidualFSQ.codes_to_index([4, 4, 4, 4])

    assert ResidualFSQ.decode([zero, zero, zero])
           |> Nx.abs()
           |> Nx.sum()
           |> Nx.to_number() == 0.0
  end

  test "valid_id?" do
    assert ResidualFSQ.valid_id?([0, 4095, 17])
    refute ResidualFSQ.valid_id?([0, 4096, 17])
    refute ResidualFSQ.valid_id?([0, 17])
    refute ResidualFSQ.valid_id?([0, 1, -1])
    refute ResidualFSQ.valid_id?(nil)
  end
end
