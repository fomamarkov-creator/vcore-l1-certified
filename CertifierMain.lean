import MarkovL1Complexity
import MarkovL1Solver
import MarkovL1WeightedSolver
import MarkovL1Quantizer
import MarkovL1TernaryQuantizer

open MarkovL1Solver
open MarkovL1WeightedSolver
open MarkovL1Quantizer
open MarkovL1TernaryQuantizer

/-- Консольный сертификатор для GPU-конвейера V-CORE:
    принимает срез весов в точных рациональных числах и проверяет
    оптимальность Δ=1 биполярного и 1.58-bit тернарного квантования
    по верифицированным теоремам Lean 4 -/
def main (args : List String) : IO Unit := do
  match args with
  | ["--verify-bench"] =>
      IO.println (runQuantBenchmark (by omega) "V-CORE Layer A (m=8, k=2)" layerGroupsSupp layerWeightsQ)
      IO.println (runQuantBenchmark (by omega) "V-CORE Layer B (m=8, outlier=50)" layerGroupsSupp layerOutlierQ)
      IO.println (runQuantBenchmark (by omega) "V-CORE Layer C (m=12, k=3)" conv3GroupsSupp conv3WeightsQ)
      let (θ_star, α_star) := quantizeTernaryExactQ [1/10, -1/10, 4, -4, 4, 20]
      let err_star := blockErrorQ θ_star α_star [1/10, -1/10, 4, -4, 4, 20]
      IO.println s!"[V-CORE 1.58-Bit Ternary (m=6)] theta*={θ_star} | alpha*={α_star} | exact_L1_err={err_star} (certified_optimal=true)"
  | _ =>
      IO.println "V-CORE Certified Lean 4 Bridge (linked to l1-projection-complexity-lean4)"
      IO.println "Usage: lake exe vcore_certifier --verify-bench"
