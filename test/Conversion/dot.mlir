// RUN: triton-opt %s -split-input-file --convert-triton-cpu-to-llvm | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  tt.func public @kernel(
    %a: tensor<2x4xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %b: tensor<4x2xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %c: tensor<2x2xf32, #blocked>) -> tensor<2x2xf32, #blocked> attributes {noinline = false} {
    %d = tt.dot %a, %b, %c :
        tensor<2x4xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> *
        tensor<4x2xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> ->
        tensor<2x2xf32, #blocked>
    // COM: We should see a bunch of repetitions of this pattern:
    // CHECK: [[A:%.*]] = llvm.fpext {{%.*}} : f16 to f32
    // CHECK: [[B:%.*]] = llvm.fpext {{%.*}} : f16 to f32
    // CHECK: [[MUL:%.*]] = llvm.fmul [[A]], [[B]] : f32
    // CHECK: {{%.*}} = llvm.fadd {{%.*}}, [[MUL]] : f32
    tt.return %d : tensor<2x2xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  tt.func public @kernel(
    %a: tensor<2x4xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %b: tensor<4x2xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %c: tensor<2x2xf32, #blocked>) -> tensor<2x2xf32, #blocked> attributes {noinline = false} {
    %d = tt.dot %a, %b, %c :
        tensor<2x4xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> *
        tensor<4x2xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> ->
        tensor<2x2xf32, #blocked>
    // COM: We should see a bunch of repetitions of this pattern:
    // CHECK: [[A:%.*]] = llvm.fpext {{%.*}} : bf16 to f32
    // CHECK: [[B:%.*]] = llvm.fpext {{%.*}} : bf16 to f32
    // CHECK: [[MUL:%.*]] = llvm.fmul [[A]], [[B]] : f32
    // CHECK: {{%.*}} = llvm.fadd {{%.*}}, [[MUL]] : f32
    tt.return %d : tensor<2x2xf32, #blocked>
  }
}
