import Lake
open Lake DSL

-- SPDX-License-Identifier: MIT
-- Formal model of the residual-FSQ recommender core:
--   * semantic-ID codec — stage index bijection + injective item key
--     (lib/recommender/core/residual_fsq.ex), proved by `omega`;
--   * retention recurrence — rolled Nx.while = unrolled fold
--     (lib/recommender/core/fuxi_linear_inference.ex retention_scan),
--     proved by induction and certified as a fire/plausible-witness-dag witness.

package «recommender-model» where

require «plausible-witness-dag» from git
  "https://github.com/fire/plausible-witness-dag" @ "main"

@[default_target] lean_lib RecommenderModel where

lean_exe «recommender-model-sample» where
  root := `RecommenderModel
