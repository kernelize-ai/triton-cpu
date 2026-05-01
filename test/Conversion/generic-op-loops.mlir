// RUN: triton-opt %s -split-input-file --convert-triton-cpu-to-llvm | FileCheck %s

// COM: Verify that emitNestedLoops emits exactly one LLVM loop level per block
// COM: dimension, rather than a single flat loop with div/mod index arithmetic.
// COM: Each test scales a block-shaped tensor by a scalar and checks that the
// COM: number of llvm.cond_br ops in the output matches the block rank.

// -----

// COM: Rank-1 block [16] with tile [4] → 4 iterations, 1 loop level.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK-LABEL: @scale_1d
  // CHECK-COUNT-1: llvm.cond_br
  // CHECK-NOT: llvm.cond_br
  tt.func public @scale_1d(%scale: f32) {
    %c16_i32 = arith.constant 16 : i32

    %0 = ttc.generic (%scale) blocks [%c16_i32 : i32] attributes {tileShape = array<i32: 4>} body {
      ^bb0(%offset: i32, %s: f32):
        %cst = arith.constant dense<1.0> : tensor<4xf32, #blocked>
        %splat = tt.splat %s : f32 -> tensor<4xf32, #blocked>
        %result = arith.mulf %cst, %splat : tensor<4xf32, #blocked>
        ttc.yield %result : tensor<4xf32, #blocked>
    } combiners {} : (f32) -> tensor<16xf32, #blocked>
    tt.return
  }
}

// -----

// COM: Rank-2 block [4, 8] with tile [1, 4] → 4×2 iterations, 2 loop levels.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK-LABEL: @scale_2d
  // CHECK-COUNT-2: llvm.cond_br
  // CHECK-NOT: llvm.cond_br
  tt.func public @scale_2d(%scale: f32) {
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = ttc.generic (%scale) blocks [%c4_i32, %c8_i32 : i32, i32] attributes {tileShape = array<i32: 1, 4>} body {
      ^bb0(%dim0: i32, %dim1: i32, %s: f32):
        %cst = arith.constant dense<1.0> : tensor<1x4xf32, #blocked>
        %splat = tt.splat %s : f32 -> tensor<1x4xf32, #blocked>
        %result = arith.mulf %cst, %splat : tensor<1x4xf32, #blocked>
        ttc.yield %result : tensor<1x4xf32, #blocked>
    } combiners {} : (f32) -> tensor<4x8xf32, #blocked>
    tt.return
  }
}

// -----

// COM: Rank-3 block [4, 4, 8] with tile [1, 1, 4] → 4×4×2 iterations, 3 loop levels.

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [1, 1, 1], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK-LABEL: @scale_3d
  // CHECK-COUNT-3: llvm.cond_br
  // CHECK-NOT: llvm.cond_br
  tt.func public @scale_3d(%scale: f32) {
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = ttc.generic (%scale) blocks [%c4_i32, %c4_i32, %c8_i32 : i32, i32, i32] attributes {tileShape = array<i32: 1, 1, 4>} body {
      ^bb0(%dim0: i32, %dim1: i32, %dim2: i32, %s: f32):
        %cst = arith.constant dense<1.0> : tensor<1x1x4xf32, #blocked>
        %splat = tt.splat %s : f32 -> tensor<1x1x4xf32, #blocked>
        %result = arith.mulf %cst, %splat : tensor<1x1x4xf32, #blocked>
        ttc.yield %result : tensor<1x1x4xf32, #blocked>
    } combiners {} : (f32) -> tensor<4x4x8xf32, #blocked>
    tt.return
  }
}

// -----

// COM: Rank-4 block [2, 2, 4, 8] with tile [1, 1, 1, 4] → 2×2×4×2 iterations, 4 loop levels.

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1, 4], threadsPerWarp = [1, 1, 1, 1], warpsPerCTA = [1, 1, 1, 1], order = [3, 2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK-LABEL: @scale_4d
  // CHECK-COUNT-4: llvm.cond_br
  // CHECK-NOT: llvm.cond_br
  tt.func public @scale_4d(%scale: f32) {
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = ttc.generic (%scale) blocks [%c2_i32, %c2_i32, %c4_i32, %c8_i32 : i32, i32, i32, i32] attributes {tileShape = array<i32: 1, 1, 1, 4>} body {
      ^bb0(%dim0: i32, %dim1: i32, %dim2: i32, %dim3: i32, %s: f32):
        %cst = arith.constant dense<1.0> : tensor<1x1x1x4xf32, #blocked>
        %splat = tt.splat %s : f32 -> tensor<1x1x1x4xf32, #blocked>
        %result = arith.mulf %cst, %splat : tensor<1x1x1x4xf32, #blocked>
        ttc.yield %result : tensor<1x1x1x4xf32, #blocked>
    } combiners {} : (f32) -> tensor<2x2x4x8xf32, #blocked>
    tt.return
  }
}
