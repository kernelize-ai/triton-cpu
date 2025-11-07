// RUN: triton-opt %s -split-input-file --allocate-shared-memory-cpu --convert-triton-cpu-to-llvm | FileCheck %s -check-prefix=MASKED-OP
// RUN: triton-opt %s -split-input-file --allocate-shared-memory-cpu --convert-triton-cpu-to-llvm -convert-masked-ops-to-llvm | FileCheck %s -check-prefix=LLVM

// COM: Lowers tt.load and tt.store to TritonCPU masked_load/masked_store ops. Then re-runs the test suite lowering the masked ops to LLVM intrinsics.

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  tt.func public @kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 8 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #blocked>
    %2 = tt.addptr %1, %0 : tensor<64x!tt.ptr<i32>, #blocked>, tensor<64xi32, #blocked>
    %3 = tt.load %2 : tensor<64x!tt.ptr<i32>, #blocked>
    // MASKED-OP: [[MASK:%.*]] = llvm.mlir.constant(dense<true> : vector<2xi1>) : vector<2xi1>
    // MASKED-OP: [[OTHER:%.*]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
    // MASKED-OP: ttc.masked_load {{.*}}, [[MASK]], [[OTHER]] : {{.*}} -> vector<2xi32>
    // MASKED-OP-COUNT-31: ttc.masked_load
    // MASKED-OP-NOT: ttc.masked_load

    // COM: Prevent the masked load op from being optimized out
    tt.print " x: " {hex = false, isSigned = array<i32: 0>} : %3 : tensor<64xi32, #blocked>
    // LLVM: [[OTHER:%.*]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
    // LLVM: [[MASK:%.*]] = llvm.mlir.constant(dense<true> : vector<2xi1>) : vector<2xi1>
    // LLVM-COUNT-32: llvm.intr.masked.load {{.*}}, [[MASK]], [[OTHER]] {alignment = 8 : i32} : {{.*}} -> vector<2xi32>
    // LLVM-NOT: llvm.intr.masked.load
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
    // LLVM-LABEL: masked_load_store
    tt.func @masked_load_store(%x_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %n_elements: i32 {tt.divisibility = 8 : i32}) {
        %offsets = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked>
        %n_elem_tensor = tt.splat %n_elements : i32 -> tensor<8xi32, #blocked>
        %mask = arith.cmpi slt, %offsets, %n_elem_tensor : tensor<8xi32, #blocked>
        %x_ptr_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>, #blocked>
        %x_tensor_of_ptr = tt.addptr %x_ptr_splat, %offsets : tensor<8x!tt.ptr<f32>, #blocked>, tensor<8xi32, #blocked>
        // MASKED-OP-COUNT-4: ttc.masked_load {{.*}} : {{.*}} -> vector<2xf32>
        // MASKED-OP-NOT: ttc.masked_load

        // LLVM-COUNT-4: llvm.intr.masked.load {{.*}} {alignment = 8 : i32} : {{.*}} -> vector<2xf32>
        %x_tensor = tt.load %x_tensor_of_ptr, %mask : tensor<8x!tt.ptr<f32>, #blocked>

        %y_ptr_splat = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>, #blocked>
        %y_tensor_of_ptr = tt.addptr %y_ptr_splat, %offsets : tensor<8x!tt.ptr<f32>, #blocked>, tensor<8xi32, #blocked>

        // MASKED-OP-COUNT-4: ttc.masked_store {{.*}}
        // MASKED-OP-NOT: ttc.masked_store

        // LLVM-COUNT-4: llvm.intr.masked.store {{.*}} {alignment = 8 : i32} : vector<2xf32>, vector<2xi1> into !llvm.ptr<1>
        tt.store %y_tensor_of_ptr, %x_tensor, %mask : tensor<8x!tt.ptr<f32>, #blocked>
        tt.return
    }
}

// -----

#blocked4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [1], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 1 : i32} {
  // MASKED-OP-LABEL: reduce_xor_max
  tt.func @reduce_xor_max(%arg0: tensor<8xf32, #blocked4>) {
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.maxnumf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<8xf32, #blocked4>) -> f32
    // MASKED-OP: ttc.masked_load {{.*}} -> f32
    // MASKED-OP: ttc.masked_store {{.*}} : (!llvm.ptr, f32, i1) -> ()
    // MASKED-OP: @_cpu_barrier

    // LLVM-NOT: llvm.intr.masked
    // LLVM: llvm.call @_cpu_barrier
    // LLVM: llvm.load {{.*}} {alignment = 4 : i64} : !llvm.ptr -> f32
    tt.return
  }
}
