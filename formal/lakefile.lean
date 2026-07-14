import Lake
open Lake DSL

-- SPDX-License-Identifier: MIT
-- Formal model of the holographic-semantic-memory core that ships in this repo:
--   * ResidualFSQ semantic-ID index codec (lib/holo/residual_fsq.ex:
--     codes_to_index / index_to_codes; flat item key in lib/holo/semantic_id.ex)
--   * HRR phase algebra on the uint16 grid (lib/holo/hrr.ex: bind / unbind)
--   * cleanup (nearest-item) recall as a budgeted witness walk (lib/holo/memory.ex)
-- Built on fire/plausible-witness-dag.

package «holo-memory-model» where

require «plausible-witness-dag» from git
  "https://github.com/fire/plausible-witness-dag" @ "main"

@[default_target] lean_lib HoloModel where

lean_exe «holo-memory-sample» where
  root := `HoloModel
