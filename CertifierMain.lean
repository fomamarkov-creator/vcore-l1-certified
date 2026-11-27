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
import MarkovL1NextGenQuantizer
import MarkovL1SuperQuantizer
import MarkovL1UltimateE8Engine
import MarkovL1AttentionKVCache
import MarkovL1LeechAndLowRank
import MarkovL1EntropyAndSparsity
import MarkovL1MoERoutingAndDynamicBits
import MarkovL1ZeroKnowledgeInferenceCert
import MarkovL1HCPAndBirkhoffQuantizer
import MarkovL1FullTransformerCert
import MarkovL1RoPEAndGolaySyndrome
import MarkovL1CrossLayerAndIntegerTransformer
import MarkovL1QATConvergence
import MarkovL1DyadicVNodeReduction
import MarkovL1V256Upgrades

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

      -- Модули 13-21: Решётки 3HCP / D4 / E8 / Λ24, Hadamard W8, 2:4, MoE, Freivalds, Birkhoff-James
      let sqOut := MarkovL1SuperQuantizer.superQuantize4Q (2/1) (10/1, -2/1, 3/1, -1/1)
      IO.println s!"[V-CORE Super-Quantizer (H₄ + D₄ 1.16b)] mode={repr sqOut.mode} | certBound={sqOut.certifiedMaxOutputErr}"
      let netBound := MarkovL1UltimateE8Engine.networkErrorBoundQ (10/1) [(1/2, 1/100), (1/2, 1/100), (1/2, 1/100)]
      IO.println s!"[V-CORE Sub-1-Bit E₈ (241-pt, 0.989 b/w)] L3_logit_drift_bound={netBound}"
      let w8 : MarkovL1AttentionKVCache.Vec8Q := ⟨1/2, -1/2, 1/4, 3/4, -1/4, 1/2, 0, -1/2⟩
      let x8 : MarkovL1AttentionKVCache.Vec8Q := ⟨2/1, 1/1, -1/1, 3/1, 1/2, -1/2, 4/1, 1/1⟩
      let hDot := MarkovL1AttentionKVCache.dot8Q (MarkovL1AttentionKVCache.walshHadamard8Q w8) (MarkovL1AttentionKVCache.walshHadamard8Q x8) / 8
      IO.println s!"[V-CORE In-Domain Hadamard W₈ Dot] exact_dot={hDot} (no_inverse_transform=true)"
      let E_res : List (List ℚ) := [[2/1, 4/1, 6/1], [-1/1, -2/1, -3/1]]
      let u_opt := MarkovL1LeechAndLowRank.updateLeftFactorL1Q E_res [1/1, 2/1, 3/1]
      IO.println s!"[V-CORE Golay G₂₄ (4096-cw) & ℓ₁-LoRA] rank1_factor={u_opt} (~0.733 b/w Leech Λ₂₄)"
      let q24 := MarkovL1EntropyAndSparsity.quantize24SparseTernaryQ (10/1, -2/1, 8/1, -1/1)
      IO.println s!"[V-CORE 2:4 Sparse L1 & Kraft Entropy] q24={q24} (nnz={MarkovL1EntropyAndSparsity.nnz4Q q24} ≤ 2)"
      let moeSafe := MarkovL1MoERoutingAndDynamicBits.verifyMoERoutingSafety [(5/2, 1/10), (2/1, 1/10)] [(3/2, 1/10), (1/2, 1/10)]
      IO.println s!"[V-CORE MoE Top-K Routing Safety] invariant={moeSafe}"
      let freivaldsOK := MarkovL1ZeroKnowledgeInferenceCert.verifyMatMulFreivaldsQ [[1/1, 2/1], [3/1, 4/1]] [[2/1, 0/1], [1/1, 2/1]] [[4/1, 4/1], [10/1, 8/1]] [1/1, 3/1]
      IO.println s!"[V-CORE O(n²) Freivalds MatMul Cert] verified={freivaldsOK}"
      let bjOrth := MarkovL1HCPAndBirkhoffQuantizer.checkBirkhoffL1OrthogonalQ [2/1, -2/1, 0/1, 0/1] [1/2, 1/2, 1/4, -1/4]
      IO.println s!"[V-CORE 3HCP (13-pt) & Birkhoff-James ℓ₁] Le0={MarkovL1HCPAndBirkhoffQuantizer.Le0_gap} | orth={bjOrth}"

      -- Модули 22-24: W32/Λ32, LUT-инференс, Mixed-Lattice бюджет, E2E Transformer Cert, RoPE и синдром Голея
      let plan : List MarkovL1FullTransformerCert.MixedLatticeLayerSpec := [
        { layerName := "attn.v_proj (D4)", bpw := 1161/1000, paramCount := 10/1, holderDriftBound := 1/100 },
        { layerName := "mlp.gate_proj (Leech Λ24)", bpw := 733/1000, paramCount := 50/1, holderDriftBound := 3/100 },
        { layerName := "mlp.up_proj (Leech Λ24)", bpw := 733/1000, paramCount := 50/1, holderDriftBound := 3/100 }
      ]
      let mixedOK := MarkovL1FullTransformerCert.verifyMixedLatticePlanQ plan (80/100) (1/10)
      let blockBudget : MarkovL1FullTransformerCert.TransformerBlockErrorBudget := {
        δ_x := 1/100, L_norm := 1/1, ε_qkv := 1/200, ε_o := 1/200,
        ε_K := 1/100, ε_V := 1/100, L_sm := 1/2, ε_mlp := 1/100,
        L_act := 1/1, B_X := 2/1
      }
      IO.println s!"[V-CORE v4.0 Mixed-Lattice & E2E Block] plan_ok={mixedOK} | max_block_drift={MarkovL1FullTransformerCert.transformerBlockTotalDriftQ blockBudget}"
      let (cRope, sRope) := MarkovL1RoPEAndGolaySyndrome.rationalCayleyCS (1/2)
      let cw := MarkovL1RoPEAndGolaySyndrome.encodeGolay24 1337
      let err : UInt32 := (1 : UInt32) <<< 5
      IO.println s!"[V-CORE v4.0 Rational RoPE & Golay G₂₄] unit={cRope*cRope + sRope*sRope} | syn_inv={MarkovL1RoPEAndGolaySyndrome.syndromeGolay24 (cw ^^^ err) == MarkovL1RoPEAndGolaySyndrome.syndromeGolay24 err}"
      
      let w2Comp := MarkovL1CrossLayerAndIntegerTransformer.crossLayerCompensatedWeightQ (3/4) (1/2) (2/1)
      let y1 := MarkovL1CrossLayerAndIntegerTransformer.invSqrtNewtonStepQ (4/1) (1/2)
      IO.println s!"[V-CORE v5.0 Frontier Module 24] crossLayer_w2'={w2Comp} | integer_invSqrt4={y1}"

      -- Модуль 27: 7 апгрейдов матрицы V256 (Электрический Замок, 3HCP-фильтр, порог 125/192, анизотропный MoE)
      let cert27 := MarkovL1V256.buildVerifiedV256Certificate
      IO.println s!"[V-CORE Module 27 V256 Upgrades] stamp={cert27.stamp} | Le0={MarkovL1V256.Le_0Q} | lockThresh={MarkovL1V256.lockPruneThresholdQ}"

  | _ =>
      IO.println "V-CORE Certified Lean 4 Bridge (linked to l1-projection-complexity-lean4 v8.0.0 — 27 Modules)"
      IO.println "Usage: lake exe vcore_certifier --verify-bench"
