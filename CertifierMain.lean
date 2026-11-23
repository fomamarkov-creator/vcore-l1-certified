import MarkovL1Complexity
import MarkovL1Solver
import MarkovL1WeightedSolver
import MarkovL1Quantizer

open MarkovL1Solver
open MarkovL1WeightedSolver
open MarkovL1Quantizer

/-- Консольный сертификатор для GPU-конвейера V-CORE:
    принимает срез весов в точных рациональных числах и проверяет
    оптимальность Δ=1 квантования по верифицированным теоремам Lean 4 -/
def main (args : List String) : IO Unit := do
  match args with
  | ["--verify-bench"] =>
      IO.println (runQuantBenchmark (by omega) "V-CORE Layer A (m=8, k=2)" layerGroupsSupp layerWeightsQ)
      IO.println (runQuantBenchmark (by omega) "V-CORE Layer B (m=8, outlier=50)" layerGroupsSupp layerOutlierQ)
      IO.println (runQuantBenchmark (by omega) "V-CORE Layer C (m=12, k=3)" conv3GroupsSupp conv3WeightsQ)
  | _ =>
      IO.println "V-CORE Certified Lean 4 Bridge (linked to l1-projection-complexity-lean4 v2.0.0)"
      IO.println "Usage: lake exe vcore_certifier --verify-bench"
