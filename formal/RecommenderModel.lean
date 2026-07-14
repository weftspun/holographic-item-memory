/-! # Formal model of the residual-FSQ semantic-ID codec

Certifies the ID *format* the whole recommender rests on, at the **real** scale.
`codes_to_index` maps one stage's 4 FSQ digits `(c0,c1,c2,c3)` with `levels = [8,8,8,8]`
to a token index via the basis `[1,8,64,512]`: `index = c0 + 8·c1 + 64·c2 + 512·c3`,
vocabulary `4096` per stage; an item's ID is 4 stage tokens, packable base-4096. There is
deliberately no Elixir mirror of this codec — these theorems certify the contract itself.

Proved by `omega` (symbolic — no enumeration — so the facts hold at the real
4096-code / 4096⁴-key scale): stage bound, digit round-trip, stage injectivity, and
item-key injectivity (no two distinct semantic IDs collide).
-/

namespace RecommenderModel

/-! ## ResidualFSQ index codec (`lib/recommender/core/residual_fsq.ex`) -/

/-- Codes per stage `= 8^4` (`Recommender.Core.ResidualFSQ.codebook_size`). -/
def codebook : Nat := 4096

/-- Residual quantizer stages = tokens per item (`Recommender.Core.ResidualFSQ.tokens_per_item`). -/
def numQuantizers : Nat := 4

/-- Mixed-radix token index for one stage's digits, basis `[1,8,64,512]`
    (`ResidualFSQ` stage indexing). -/
def stageIndex (c0 c1 c2 c3 : Nat) : Nat :=
  c0 + c1 * 8 + c2 * 64 + c3 * 512

/-- Every valid digit tuple lands inside the stage codebook (`< 4096`). -/
theorem stage_bound (c0 c1 c2 c3 : Nat)
    (h0 : c0 < 8) (h1 : c1 < 8) (h2 : c2 < 8) (h3 : c3 < 8) :
    stageIndex c0 c1 c2 c3 < codebook := by
  unfold stageIndex codebook; omega

/-- `index_to_codes ∘ codes_to_index = id`: digit `i` is `(idx / basis_i) % 8`. -/
theorem stage_roundtrip (c0 c1 c2 c3 : Nat)
    (h0 : c0 < 8) (h1 : c1 < 8) (h2 : c2 < 8) (h3 : c3 < 8) :
    stageIndex c0 c1 c2 c3 % 8 = c0 ∧
    stageIndex c0 c1 c2 c3 / 8 % 8 = c1 ∧
    stageIndex c0 c1 c2 c3 / 64 % 8 = c2 ∧
    stageIndex c0 c1 c2 c3 / 512 % 8 = c3 := by
  unfold stageIndex; omega

/-- The stage codec is injective — distinct digit tuples never share a token. -/
theorem stage_injective (c0 c1 c2 c3 d0 d1 d2 d3 : Nat)
    (hc0 : c0 < 8) (hc1 : c1 < 8) (hc2 : c2 < 8) (_hc3 : c3 < 8)
    (hd0 : d0 < 8) (hd1 : d1 < 8) (hd2 : d2 < 8) (_hd3 : d3 < 8)
    (h : stageIndex c0 c1 c2 c3 = stageIndex d0 d1 d2 d3) :
    c0 = d0 ∧ c1 = d1 ∧ c2 = d2 ∧ c3 = d3 := by
  unfold stageIndex at h; omega

/-- Flat item key: 4 stage tokens packed base-4096. -/
def itemKey (t0 t1 t2 t3 : Nat) : Nat :=
  t0 + t1 * 4096 + t2 * 4096 * 4096 + t3 * 4096 * 4096 * 4096

/-- Semantic-ID uniqueness: distinct valid IDs never collide on the flat key. -/
theorem itemKey_injective (t0 t1 t2 t3 u0 u1 u2 u3 : Nat)
    (ht0 : t0 < 4096) (ht1 : t1 < 4096) (ht2 : t2 < 4096) (_ht3 : t3 < 4096)
    (hu0 : u0 < 4096) (hu1 : u1 < 4096) (hu2 : u2 < 4096) (_hu3 : u3 < 4096)
    (h : itemKey t0 t1 t2 t3 = itemKey u0 u1 u2 u3) :
    t0 = u0 ∧ t1 = u1 ∧ t2 = u2 ∧ t3 = u3 := by
  unfold itemKey at h; omega

end RecommenderModel
