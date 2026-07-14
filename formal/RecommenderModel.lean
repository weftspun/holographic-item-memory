import PlausibleWitnessDag

/-! # Formal model of the residual-FSQ recommender core

Two things the recommender rests on, certified at the **real** scale.

## ResidualFSQ semantic-ID codec
`codes_to_index` maps one stage's 4 FSQ digits `(c0,c1,c2,c3)` with `levels = [8,8,8,8]`
to a token index via the basis `[1,8,64,512]`: `index = c0 + 8·c1 + 64·c2 + 512·c3`,
vocabulary `4096` per stage; an item's ID is 4 stage tokens, packable base-4096. There is
deliberately no Elixir mirror of this codec — these theorems certify the contract itself.
Proved by `omega` (symbolic — so the facts hold at the real 4096-code / 4096⁴-key scale):
stage bound, digit round-trip, stage injectivity, and item-key injectivity.

## Linear-attention retention recurrence
`Recommender.Core.FuxiLinearInference.retention_scan` was rewritten from an Elixir-unrolled
fold to a rolled `Nx.while` (so the compiled graph is O(1) in sequence length).
`retention_rolled_eq_unrolled` proves the rolled tail-recursion computes exactly the same
`foldl` as the unrolled version — the rewrite changes only *how* the loop runs, not *what* it
computes — and `plausible-witness-dag` certifies it as a budgeted witness (a shallow rung
budget-hits, a deeper rung resolves).
-/

namespace RecommenderModel

open PlausibleWitnessDag

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

/-! ## Linear-attention retention: rolled `Nx.while` = unrolled fold
    (`lib/recommender/core/fuxi_linear_inference.ex`, `retention_scan`) -/

/-- Per-component state grid; the recurrence is modelled over `Nat` to stay exact. -/
def retGrid : Nat := 65536

/-- One recurrence step: `S' = decay·S + kv` (per component, mod the grid). -/
def retStep (decay s kv : Nat) : Nat := (decay * s + kv) % retGrid

/-- Unrolled retention: `List.foldl` over the `kv` inputs — the pre-rewrite Elixir
    `Enum.reduce`. -/
def retUnrolled (decay : Nat) (kvs : List Nat) : Nat :=
  kvs.foldl (retStep decay) 0

/-- Rolled retention: explicit tail recursion carrying the state — the `Nx.while`. -/
def retRolled (decay : Nat) : List Nat → Nat → Nat
  | [], s => s
  | kv :: rest, s => retRolled decay rest (retStep decay s kv)

/-- The rewrite is faithful: rolling the loop computes exactly the unrolled fold. -/
theorem retention_rolled_eq_unrolled (decay : Nat) (kvs : List Nat) :
    retRolled decay kvs 0 = retUnrolled decay kvs := by
  unfold retUnrolled
  suffices h : ∀ s, retRolled decay kvs s = kvs.foldl (retStep decay) s from h 0
  intro s
  induction kvs generalizing s with
  | nil => rfl
  | cons x xs ih =>
      simp only [retRolled, List.foldl_cons]
      exact ih (retStep decay s x)

/-! ### Budgeted witness (plausible-witness-dag) -/

/-- A concrete sample input sequence; equivalence is certified up to `retTargetLen`. -/
def retSample (n : Nat) : List Nat := (List.range n).map (fun i => (i * 37 + 11) % retGrid)
def retDecay : Nat := 3
def retTargetLen : Nat := 8

/-- Equivalence holds at length `n` (decidable; always true by the theorem). -/
def retEquivAt (n : Nat) : Bool :=
  decide (retRolled retDecay (retSample n) 0 = retUnrolled retDecay (retSample n))

/-- Deterministic walk: within a `steps` budget, verify the equivalence for every length
    up to the target; resolves once the budget reaches the target length. -/
def retWalk (steps : Nat) : Option Nat :=
  if steps ≥ retTargetLen ∧ (List.range (retTargetLen + 1)).all retEquivAt then
    some retTargetLen
  else
    none

/-- Two-rung ladder: L0's budget (4) is below the target length (8) so it budget-hits;
    L1 (8) resolves. -/
def retLevels : Array Level := #[
  { idx := 0, walkSteps := 4, finBound := 256, numInst := 200 },
  { idx := 1, walkSteps := 8, finBound := 256, numInst := 200 }]

def retCandidate (lvl : Level) (c : Nat) : Bool :=
  c == retTargetLen && (retWalk lvl.walkSteps).isSome

def retReadback (steps : Nat) : Readback Nat :=
  match retWalk steps with
  | some n => { value := n, found := true, witnessIdx := n, budgetHit := false }
  | none => { value := 0, found := false, witnessIdx := 0, budgetHit := (retWalk retTargetLen).isSome }

/-- Resolve the retention-equivalence witness through the generic ladder. -/
def runRetentionSample : IO (Nat × Nat × TraceEntry) :=
  resolve "retention rolled = unrolled up to length 8" retCandidate retReadback retLevels

/-- Executable proof: L0 budget-hits, L1 certifies rolled = unrolled up to length 8. -/
def runSmokeTest : IO Unit := do
  let (n, lvl, trace) ← runRetentionSample
  IO.println s!"resolved level: L{lvl}"
  IO.println s!"retention rolled = unrolled certified up to length {n}"
  IO.println s!"trace: {repr trace}"
  if lvl != 1 || n != retTargetLen then
    throw <| IO.userError "retention equivalence sample did not resolve at L1"
  IO.println "codec (stage_*, itemKey_injective) + retention (rolled = unrolled) certified"

end RecommenderModel

/-- Lake executables need a top-level `main`. -/
def main (_args : List String) : IO Unit :=
  RecommenderModel.runSmokeTest
