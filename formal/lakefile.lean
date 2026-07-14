import Lake
open Lake DSL

-- SPDX-License-Identifier: MIT
-- Formal model of the residual-FSQ semantic-ID codec: stage index bijection +
-- injective item key (lib/recommender/core/residual_fsq.ex), proved by `omega`
-- at the real 4096-code / 4096⁴-key scale.

package «recommender-model» where

@[default_target] lean_lib RecommenderModel where
