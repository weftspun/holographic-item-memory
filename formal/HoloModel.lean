import PlausibleWitnessDag

/-! # Formal model of the holographic-semantic-memory core

Models what recall must get exactly right, at the **real** scale:

* **ResidualFSQ semantic-ID codec (upstream contract)** — the `multimodal-semantic-ids` Python
  pipeline that produces the IDs this library consumes (`Holo.Memory` validates 3 tokens in
  `0..4095`). `codes_to_index` maps one stage's 4 FSQ digits `(c0,c1,c2,c3)` with
  `levels = [8,8,8,8]` to a token index via the basis `[1,8,64,512]`:
  `index = c0 + 8·c1 + 64·c2 + 512·c3`, vocabulary `4096` per stage; an item's ID is 3 stage
  tokens, packable base-4096. There is deliberately no Elixir mirror of this codec — these
  theorems certify the ID *format* the whole system rests on. Proved by `omega` (symbolic — no
  enumeration — so the facts hold at the real 4096-code / 4096³-key scale): bound, digit
  round-trip, stage injectivity, and item-key injectivity (no two distinct semantic IDs collide).

* **HRR phase algebra on the generation grid** — `lib/holo/hrr.ex`. Atoms are SHA-256 uint16 values
  scaled by `2π/65536`, so the algebra lives on the grid `ℤ/65536` per component: `bind` is addition,
  `unbind` subtraction (mod 65536). `unbind (bind a b) b = a` is exact on the grid — the float
  implementation only adds representation noise. Proved by `omega`.

* **Cleanup (nearest-item) transition recall** — `lib/holo/memory.ex`. A stored `a → b` transition is
  `bind(vec a, vec b)`; recall unbinds `vec a` and scans the catalog for the match. Certified by a
  `fire/plausible-witness-dag` witness: a budgeted catalog scan recovers the stored successor; a
  shallow ladder rung (budget below the successor's catalog position) budget-hits, a deeper rung
  resolves.
-/

namespace HoloModel

open PlausibleWitnessDag

/-! ## ResidualFSQ index codec (upstream `multimodal-semantic-ids` contract) -/

/-- Codes per stage `= 8^4` (`Holo.Memory.codebook_size`). -/
def codebook : Nat := 4096

/-- Residual quantizer stages = tokens per item (`Holo.Memory.tokens_per_item`). -/
def numQuantizers : Nat := 4

/-- Mixed-radix token index for one stage's digits, basis `[1,8,64,512]`
    (upstream `ResidualFSQ` stage indexing). -/
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

/-! ## HRR phase algebra on the uint16 grid (`lib/holo/hrr.ex`) -/

/-- Atom phases are `uint16 · 2π/65536`, so one component lives on `ℤ/65536`. -/
def grid : Nat := 65536

/-- `bind` = element-wise phase addition (mod the grid). -/
def bindG (a b : Nat) : Nat := (a + b) % grid

/-- `unbind` = element-wise phase subtraction (mod the grid). -/
def unbindG (m k : Nat) : Nat := (m + grid - k % grid) % grid

/-- Retrieval is exact on the grid: `unbind (bind a b) b = a`. The float
    implementation of `Holo.HRR` only adds f64 representation noise on top. -/
theorem unbind_bind (a b : Nat) (ha : a < grid) (hb : b < grid) :
    unbindG (bindG a b) b = a := by
  unfold unbindG bindG grid at *
  rw [Nat.mod_eq_of_lt hb]
  omega

/-- Binding is commutative, matching circular convolution. -/
theorem bind_comm (a b : Nat) : bindG a b = bindG b a := by
  unfold bindG
  rw [Nat.add_comm]

/-! ## Cleanup transition recall as a plausible-witness-dag witness -/

/-- Component count of the toy phase vectors (real vectors are 4096-dim; the
    algebraic facts are per-component, so 4 components lose no generality). -/
def vecDim : Nat := 4

/-- A tiny catalog: item id ↦ grid phase vector (distinct, "SHA-like" values). -/
def catalog : List (List Nat) :=
  [[11, 60000, 123, 4096], [7, 7, 7, 7], [500, 1, 65535, 2], [40000, 31, 900, 12345]]

/-- Stored trace for a transition `a → b`: component-wise `bind`. -/
def trace (va vb : List Nat) : List Nat := List.zipWith bindG va vb

/-- The stored transition in the sample: item 0 → item 3. -/
def storedPrev : List Nat := catalog[0]!
def storedNext : List Nat := catalog[3]!

/-- Budgeted cleanup: unbind the previous item's vector from the trace, then scan
    the first `steps` catalog entries for an exact match. Mirrors
    `Holo.Memory.recommend`'s cleanup against the item map. -/
def recallWalk (steps : Nat) : Option Nat := Id.run do
  let unbound := List.zipWith unbindG (trace storedPrev storedNext) storedPrev
  let mut found : Option Nat := none
  for slot in List.range catalog.length do
    if found.isNone && slot < steps then
      if catalog[slot]! == unbound then
        found := some slot
  pure found

/-- Two-rung ladder: L0's scan budget (2) is below the successor's catalog
    position (3), so it budget-hits; L1 (4 = full catalog) resolves. -/
def holoLevels : Array Level := #[
  { idx := 0, walkSteps := 2, finBound := 256, numInst := 200 },
  { idx := 1, walkSteps := 4, finBound := 256, numInst := 200 }]

/-- Candidate: a witness iff it names the successor and the budgeted recall reaches it. -/
def candidate (lvl : Level) (c : Nat) : Bool :=
  recallWalk lvl.walkSteps == some c

/-- Deterministic read-back: the recalled successor id, else a budget-hit flag. -/
def readback (steps : Nat) : Readback Nat :=
  match recallWalk steps with
  | some idx => { value := idx, found := true, witnessIdx := idx, budgetHit := false }
  | none => { value := 0, found := false, witnessIdx := 0,
              budgetHit := (recallWalk catalog.length).isSome }

/-- Resolve the transition recall through the generic ladder. -/
def runSample : IO (Nat × Nat × TraceEntry) :=
  resolve "holographic transition recall of item 0 → item 3" candidate readback holoLevels

/-- Executable smoke test: L0 budget-hits, L1 recalls exactly the stored successor. -/
def runSmokeTest : IO Unit := do
  let (next, lvl, traceEntry) ← runSample
  IO.println s!"resolved level: L{lvl}"
  IO.println s!"recalled successor: item {next}"
  IO.println s!"trace: {repr traceEntry}"
  if lvl != 1 || next != 3 then
    throw <| IO.userError "holographic recall sample did not resolve item 3 at L1"
  IO.println "codec + phase algebra certified (stage_*, itemKey_injective, unbind_bind by omega)"

end HoloModel

/-- Lake executables need a top-level `main`. -/
def main (_args : List String) : IO Unit :=
  HoloModel.runSmokeTest
