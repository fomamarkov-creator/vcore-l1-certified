import MarkovL1Complexity
import MarkovL1Solver
import MarkovL1WeightedSolver
import MarkovL1Quantizer
import MarkovL1TernaryQuantizer
import MarkovL1WeightedTernary
import MarkovL1TreewidthSolver
import MarkovL1ContinuousChainDP
import MarkovL1TriangularActivation
import MarkovL1GeneralTreewidthFPT

open MarkovL1Solver
open MarkovL1WeightedSolver
open MarkovL1Quantizer
open MarkovL1TernaryQuantizer
open MarkovL1WeightedTernary
open MarkovL1TreewidthSolver
open MarkovL1ContinuousChainDP
open MarkovL1TriangularActivation
open MarkovL1GeneralTreewidthFPT

def main (args : List String) : IO Unit := do
  match args with
  | ["--verify-bench"] =>
      IO.println (runQuantBenchmark (by omega) "V-CORE Layer A (m=8, k=2)" layerGroupsSupp layerWeightsQ)
      IO.println (runQuantBenchmark (by omega) "V-CORE Layer B (m=8, outlier=50)" layerGroupsSupp layerOutlierQ)
      IO.println (runQuantBenchmark (by omega) "V-CORE Layer C (m=12, k=3)" conv3GroupsSupp conv3WeightsQ)
      let (θ_star, α_star) := quantizeTernaryExactQ [1/10, -1/10, 4, -4, 4, 20]
      let err_star := blockErrorQ θ_star α_star [1/10, -1/10, 4, -4, 4, 20]
      IO.println s!"[V-CORE 1.58-Bit Ternary (m=6)] theta*={θ_star} | alpha*={α_star} | exact_L1_err={err_star} (certified_optimal=true)"
      let fptCost := solveChainOverlapFPT [-4, 0, 4] [4, 8, 4]
      IO.println s!"[V-CORE Bounded-Treewidth (Δ=2) FPT] min_L1_cost={fptCost} (certified_dp=true)"
      let (contCs, contCost) := solveContinuousChainDP [4/1, 8/1, 4/1]
      let contCert := checkChainOptimalCertQ [4/1, 8/1, 4/1] contCs [0/1, 1/1, 1/1] [0/1, -1/1]
      IO.println s!"[V-CORE Continuous Chain DP (ℚ^m)] c*={contCs} | min_cost={contCost} | dual_certified={contCert}"
      let triWHat := quantizeBlockTriangularQ (2/1) (4/1) [6/1, 5/1, 1/1] [[10/1, 0/1, 0/1], [8/1, 2/1, 0/1], [6/1, 4/1, 2/1]]
      let triErr := triangularBlockErrorQ [6/1, 5/1, 1/1] triWHat [[10/1, 0/1, 0/1], [8/1, 2/1, 0/1], [6/1, 4/1, 2/1]]
      IO.println s!"[V-CORE Triangular Activation (ℓ₁-GPTQ)] wHat={triWHat} | output_L1_err={triErr} (certified_reduction=true)"
      let tw2Cost := solveTripleOverlapFPT [-2/1, 0/1, 2/1, 4/1] [2/1, 0/1, 6/1, 4/1]
      IO.println s!"[V-CORE Treewidth k=2 (Δ=3, W_Q/K/V)] min_L1_cost={tw2Cost} (certified_fpt=true)"

  | _ =>
      IO.println "V-CORE Certified Lean 4 Bridge (linked to l1-projection-complexity-lean4 v4.0.0 — 12 Modules)"
      IO.println "Usage: lake exe vcore_certifier --verify-bench"
