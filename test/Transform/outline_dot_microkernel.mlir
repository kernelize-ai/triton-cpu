// RUN: triton-opt %s -outline-dot-microkernel | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [16, 16], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK-LABEL: tt.func private @matmul_kernel_dot_microkernel
  // CHECK-SAME:    !tt.ptr<f32> {tt.divisibility = 64 : i32}
  // CHECK-SAME:    !tt.ptr<f32> {tt.divisibility = 64 : i32}
  // CHECK-SAME:    -> tensor<64x64xf32, #blocked>
  // Constant operands of the generic are rematerialized in-body, not passed as
  // args (if any had leaked into the signature the CHECK-SAME chain above would
  // have matched a different arg list; the call below pins the operand count).
  // CHECK:         arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
  // CHECK:         arith.constant 32 : i32
  // CHECK:         arith.constant 64 : i32
  // The computation now lives in the private func.
  // CHECK:         ttc.generic
  // CHECK:         tt.dot
  // CHECK:         ttc.yield
  // CHECK:         tt.return

  // CHECK-LABEL: tt.func public @matmul_kernel
  // CHECK:         tt.call @matmul_kernel_dot_microkernel(%arg0, %arg1, %arg3)
  // CHECK-NOT:     tt.dot
  // CHECK:         tt.return
  tt.func public @matmul_kernel(
      %a_ptr: !tt.ptr<f32> {tt.divisibility = 64 : i32},
      %b_ptr: !tt.ptr<f32> {tt.divisibility = 64 : i32},
      %c_ptr: !tt.ptr<f32> {tt.divisibility = 64 : i32},
      %K: i32 {tt.divisibility = 16 : i32}) attributes {ttc.persistent_kernel} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32

    // Anchor: reduction generic, loop-carried accumulator, tt.dot, result stored.
    %0 = ttc.generic init(%cst : tensor<64x64xf32, #blocked>)
         ins(%a_ptr, %b_ptr, %c32_i32 : !tt.ptr<f32>, !tt.ptr<f32>, i32)
         blocks[%K, %c64_i32, %c64_i32 : i32, i32, i32]
         attributes {reductionDims = array<i32: 0>, tileShape = array<i32: 32, 16, 16>} body {
    ^bb0(%i0: i32, %i1: i32, %i2: i32, %acc: tensor<16x16xf32, #blocked>,
         %ap: !tt.ptr<f32>, %bp: !tt.ptr<f32>, %kc: i32):
      %aptrs = tt.splat %ap : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>, #blocked1>
      %a = tt.load %aptrs : tensor<16x32x!tt.ptr<f32>, #blocked1>
      %ad = ttg.convert_layout %a : tensor<16x32xf32, #blocked1> -> tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %bptrs = tt.splat %bp : !tt.ptr<f32> -> tensor<32x16x!tt.ptr<f32>, #blocked1>
      %b = tt.load %bptrs : tensor<32x16x!tt.ptr<f32>, #blocked1>
      %bd = ttg.convert_layout %b : tensor<32x16xf32, #blocked1> -> tensor<32x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %d = tt.dot %ad, %bd, %acc : tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf32, #blocked>
      ttc.yield %d : tensor<16x16xf32, #blocked>
    } -> tensor<64x64xf32, #blocked>

    // Epilogue store generic: makes the dot result "exit to a store", which is
    // the clause that distinguishes an outlinable GEMM tail from a mid-fusion dot.
    ttc.generic ins(%0, %c_ptr : tensor<64x64xf32, #blocked>, !tt.ptr<f32>)
         blocks[%c64_i32, %c64_i32 : i32, i32]
         attributes {reductionDims = array<i32>, tileShape = array<i32: 16, 16>} body {
    ^bb0(%i0: i32, %i1: i32, %tile: tensor<16x16xf32, #blocked>, %cp: !tt.ptr<f32>):
      %cptrs = tt.splat %cp : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked>
      tt.store %cptrs, %tile : tensor<16x16x!tt.ptr<f32>, #blocked>
      ttc.yield
    }
    tt.return
  }
}
