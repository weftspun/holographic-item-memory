import Lake
open Lake DSL

-- SPDX-License-Identifier: MIT
-- Formal model of the holographic-semantic-memory core:
--   * ResidualFSQ semantic-ID index codec (upstream multimodal-semantic-ids
--     contract; Recommender.Memory validates/consumes the 3-token IDs)
--   * HRR phase algebra on the uint16 grid (lib/recommender/hrr.ex: bind / unbind)
--   * cleanup (nearest-item) recall as a budgeted witness walk (lib/recommender/memory.ex)
-- Built on fire/plausible-witness-dag.

package «recommender-model» where

require «plausible-witness-dag» from git
  "https://github.com/fire/plausible-witness-dag" @ "main"

@[default_target] lean_lib RecommenderModel where

lean_exe «recommender-sample» where
  root := `RecommenderModel
