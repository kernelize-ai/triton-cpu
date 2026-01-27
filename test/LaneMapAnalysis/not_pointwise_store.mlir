// RUN: triton-opt %s -split-input-file --test-lane-map-analysis 2>&1 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [32], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = ttc.block_end
    %1 = ttc.block_start
    %offsets = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
    %mask = tt.splat %arg3 : i32 -> tensor<32xi32, #blocked>
    %x = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
    %y = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
    scf.for %arg4 = %1 to %0 step %c1_i32  : i32 {
      %3 = ttc.current_block %arg4 : i32
      %block_start = arith.muli %3, %c32_i32 : i32
      %offsets_0 = tt.splat %block_start : i32 -> tensor<32xi32, #blocked>
      %offsets_1 = arith.addi %offsets_0, %offsets : tensor<32xi32, #blocked>
      %mask_2 = arith.cmpi slt, %offsets_1, %mask : tensor<32xi32, #blocked>
      %x_3 = tt.addptr %x, %offsets_1 : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
      %x_4 = tt.load %x_3, %mask_2 : tensor<32x!tt.ptr<f32>, #blocked>
      // COM: This tt.trans is a no-op but (and arguably makes no sense) but the presence of a transpose in the chain is a signal that we cannot assume an affine lane map
      %output = tt.trans %x_4 {order=array<i32: 0>} : tensor<32xf32, #blocked> -> tensor<32xf32, #blocked>
      %4 = tt.addptr %2, %offsets_1 : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
      tt.store %4, %output, %mask_2 : tensor<32x!tt.ptr<f32>, #blocked>
    }
    tt.return
  }
}

// CHECK: STORE NOT_POINTWISE
