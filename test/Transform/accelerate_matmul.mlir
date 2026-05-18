// RUN: triton-opt %s -split-input-file --tritoncpu-accelerate-matmul="canonicalize-k-loop=true" | FileCheck %s

// K-loop with two pointer iter args and an accumulator is rewritten to carry
// only the accumulator; pointer arithmetic is inlined per iteration as
// addptr(init, splat(iv) * step).
//
// CHECK-LABEL: tt.func public @test_canonicalize_k_loop
// CHECK:         scf.for [[IV:%[^ ]+]] = {{.*}} iter_args([[ACC:%[^ ]+]] = {{.*}}) -> (tensor<4x16xf32
// CHECK:           tt.splat [[IV]]
// CHECK:           arith.muli
// CHECK:           tt.addptr
// CHECK:           tt.splat [[IV]]
// CHECK:           arith.muli
// CHECK:           tt.addptr
// CHECK:           tt.dot {{.*}}[[ACC]]
// CHECK:           scf.yield

#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  tt.func public @test_canonicalize_k_loop(
      %a_init: tensor<4x8x!tt.ptr<f16>, #blocked1>,
      %b_init: tensor<8x16x!tt.ptr<f16>, #blocked1>,
      %a_step: tensor<4x8xi32, #blocked1>,
      %b_step: tensor<8x16xi32, #blocked1>,
      %k_iters: i32,
      %out: tensor<4x16x!tt.ptr<f32>, #blocked2>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %acc_init = arith.constant dense<0.0> : tensor<4x16xf32, #blocked2>
    %result:3 = scf.for %k = %c0 to %k_iters step %c1
        iter_args(%a_ptr = %a_init, %b_ptr = %b_init, %acc = %acc_init)
        -> (tensor<4x8x!tt.ptr<f16>, #blocked1>,
            tensor<8x16x!tt.ptr<f16>, #blocked1>,
            tensor<4x16xf32, #blocked2>) : i32 {
      %a_loaded = tt.load %a_ptr : tensor<4x8x!tt.ptr<f16>, #blocked1>
      %b_loaded = tt.load %b_ptr : tensor<8x16x!tt.ptr<f16>, #blocked1>
      %a_cvt = ttg.convert_layout %a_loaded : tensor<4x8xf16, #blocked1> -> tensor<4x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
      %b_cvt = ttg.convert_layout %b_loaded : tensor<8x16xf16, #blocked1> -> tensor<8x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
      %d = tt.dot %a_cvt, %b_cvt, %acc : tensor<4x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<8x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<4x16xf32, #blocked2>
      %a_next = tt.addptr %a_ptr, %a_step : tensor<4x8x!tt.ptr<f16>, #blocked1>, tensor<4x8xi32, #blocked1>
      %b_next = tt.addptr %b_ptr, %b_step : tensor<8x16x!tt.ptr<f16>, #blocked1>, tensor<8x16xi32, #blocked1>
      scf.yield %a_next, %b_next, %d : tensor<4x8x!tt.ptr<f16>, #blocked1>, tensor<8x16x!tt.ptr<f16>, #blocked1>, tensor<4x16xf32, #blocked2>
    }
    tt.store %out, %result#2 : tensor<4x16x!tt.ptr<f32>, #blocked2>
    tt.return
  }
}
