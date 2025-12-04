// RUN: triton-opt %s -split-input-file --make-persistent-kernel | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK tt.func public @add_kernel
  // CHECK-DAG: %[[BLOCK_START:.*]] = ttc.block_start
  // CHECK-DAG: %[[BLOCK_END:.*]] = ttc.block_end
  // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : i32
  // CHECK: scf.for %[[BID:.*]] = %[[BLOCK_START]] to %[[BLOCK_END]] step %[[c1]]
  // CHECK: tt.call @add_kernel.impl({{.*}}, %[[BID]])
  // COM: We don't want noinline on the wrapped function - check to make sure it has no attributes in this example
  // CHECK: tt.func private @add_kernel.impl({{.*}}) {
  tt.func public @add_kernel(%x_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %output_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %n_elements: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %c16_i32 = arith.constant 16 : i32
    %pid = tt.get_program_id x : i32
    %block_start = arith.muli %pid, %c16_i32 : i32
    %offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked>
    %offsets_0 = tt.splat %block_start : i32 -> tensor<16xi32, #blocked>
    %offsets_1 = arith.addi %offsets_0, %offsets : tensor<16xi32, #blocked>
    %mask = tt.splat %n_elements : i32 -> tensor<16xi32, #blocked>
    %mask_2 = arith.cmpi slt, %offsets_1, %mask : tensor<16xi32, #blocked>
    %x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked>
    %x_3 = tt.addptr %x, %offsets_1 : tensor<16x!tt.ptr<f32>, #blocked>, tensor<16xi32, #blocked>
    %x_4 = tt.load %x_3, %mask_2 : tensor<16x!tt.ptr<f32>, #blocked>
    %y = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked>
    %y_5 = tt.addptr %y, %offsets_1 : tensor<16x!tt.ptr<f32>, #blocked>, tensor<16xi32, #blocked>
    %y_6 = tt.load %y_5, %mask_2 : tensor<16x!tt.ptr<f32>, #blocked>
    %output = arith.addf %x_4, %y_6 : tensor<16xf32, #blocked>
    %0 = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked>
    %1 = tt.addptr %0, %offsets_1 : tensor<16x!tt.ptr<f32>, #blocked>, tensor<16xi32, #blocked>
    tt.store %1, %output, %mask_2 : tensor<16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
