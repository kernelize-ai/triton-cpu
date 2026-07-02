// RUN: triton-opt %s -split-input-file --convert-triton-npu-to-ttkernel -canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.padded_shared<[1:+1] {order = [1, 0], shape = [32, 32]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK: func.func public @matmul_kernel__reader()
  tt.func public @matmul_kernel__reader(%a_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %b_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %c_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %M: i32 {tt.divisibility = 8 : i32}, %N: i32 {tt.divisibility = 8 : i32}, %K: i32 {tt.divisibility = 8 : i32}, %stride_am: i32 {tt.divisibility = 8 : i32}, %stride_bk: i32 {tt.divisibility = 8 : i32}, %stride_cm: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %c31_i32 = arith.constant 31 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #blocked1>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #blocked2>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c32_i32 = arith.constant 32 : i32
    // CHECK-DAG: %[[c0_i32:.*]] = arith.constant 0 : i32
    // CHECK-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
    // CHECK-DAG: %[[c2_i32:.*]] = arith.constant 2 : i32
    // CHECK-DAG: %[[c3_i32:.*]] = arith.constant 3 : i32
    // CHECK-DAG: %[[c64_i32:.*]] = arith.constant 64 : i32
    // CHECK-DAG: %[[c1_i8:.*]] = arith.constant 1 : i8

    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c3:.*]] = arith.constant 3 : index
    // CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
    // CHECK-DAG: %[[c5:.*]] = arith.constant 5 : index
    // CHECK-DAG: %[[c6:.*]] = arith.constant 6 : index
    // CHECK-DAG: %[[c7:.*]] = arith.constant 7 : index

    // One chained TensorAccessorArgs per tensor input (A, B, output), built at
    // func entry and tagged by its buffer's common-arg index. The first sits at
    // cta_base = 3 (number of CB ports); the rest chain via prev_args.
    // CHECK-DAG: %[[A_ARGS:.*]] = ttkernel.TensorAccessorArgs(%[[c3_i32]], %[[c0_i32]]) {tt.accessor_base_arg_index = 0 : i32}
    // CHECK-DAG: %[[A_PTR:.*]] = ttkernel.get_common_arg_val(%[[c0]])
    // CHECK-DAG: %[[B_ARGS:.*]] = ttkernel.TensorAccessorArgs(prev = %[[A_ARGS]]) {tt.accessor_base_arg_index = 1 : i32}
    // CHECK-DAG: %[[B_PTR:.*]] = ttkernel.get_common_arg_val(%[[c1]])
    // CHECK-DAG: ttkernel.TensorAccessorArgs(prev = %[[B_ARGS]]) {tt.accessor_base_arg_index = 2 : i32}
    // CHECK-DAG: %[[M_SIZE:.*]] = ttkernel.get_common_arg_val(%[[c3]])
    // CHECK-DAG: %[[N_SIZE:.*]] = ttkernel.get_common_arg_val(%[[c4]])
    // CHECK-DAG: %[[K_SIZE:.*]] = ttkernel.get_common_arg_val(%[[c5]])
    // CHECK-DAG: %[[A_BLOCK_STRIDE_M:.*]] = ttkernel.get_common_arg_val(%[[c6]])
    // CHECK-DAG: %[[B_BLOCK_STRIDE_K:.*]] = ttkernel.get_common_arg_val(%[[c7]])
    // CHECK-DAG: %[[BLOCK_INDEX:.*]] = ttkernel.get_arg_val(%[[c0]])

    // Per-buffer TensorAccessor = accessor args + base address + page size.
    // CHECK-DAG: %[[B_CB:.*]] = ttkernel.get_compile_time_arg_val(1)
    // CHECK-DAG: %[[B_TILE_SIZE:.*]] = ttkernel.get_tile_size(%[[B_CB]])
    // CHECK-DAG: %[[B_ACCESSOR:.*]] = ttkernel.TensorAccessor(%[[B_ARGS]], %[[B_PTR]], %[[B_TILE_SIZE]])

    // CHECK-DAG: %[[A_CB:.*]] = ttkernel.get_compile_time_arg_val(0)
    // CHECK-DAG: %[[A_TILE_SIZE:.*]] = ttkernel.get_tile_size(%[[A_CB]])
    // CHECK-DAG: %[[A_ACCESSOR:.*]] = ttkernel.TensorAccessor(%[[A_ARGS]], %[[A_PTR]], %[[A_TILE_SIZE]])

    %cst_4 = arith.constant dense<32> : tensor<32x32xi32, #blocked2>
    %b = ttg.local_alloc {alloc_idx = 1 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %a = ttg.local_alloc {alloc_idx = 0 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %pid = tt.get_program_id x : i32
    %num_pid_m = arith.addi %M, %c31_i32 : i32
    %num_pid_m_5 = arith.divsi %num_pid_m, %c32_i32 : i32
    %num_pid_n = arith.addi %N, %c31_i32 : i32
    %num_pid_n_6 = arith.divsi %num_pid_n, %c32_i32 : i32
    %group_id = arith.divsi %pid, %num_pid_n_6 : i32
    %group_size_m = arith.subi %num_pid_m_5, %group_id : i32
    %group_size_m_7 = arith.minsi %group_size_m, %c1_i32 : i32
    %pid_m = arith.remsi %pid, %num_pid_n_6 : i32
    %pid_m_8 = arith.remsi %pid_m, %group_size_m_7 : i32
    %pid_m_9 = arith.addi %group_id, %pid_m_8 : i32
    %pid_n = arith.divsi %pid_m, %group_size_m_7 : i32
    %0 = arith.cmpi sge, %pid_m_9, %c0_i32 : i32
    llvm.intr.assume %0 : i1
    %1 = arith.cmpi sge, %pid_n, %c0_i32 : i32
    llvm.intr.assume %1 : i1
    %2 = arith.cmpi sgt, %stride_am, %c0_i32 : i32
    llvm.intr.assume %2 : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    %3 = arith.cmpi sgt, %stride_bk, %c0_i32 : i32
    llvm.intr.assume %3 : i1
    %4 = arith.cmpi sgt, %stride_cm, %c0_i32 : i32
    llvm.intr.assume %4 : i1
    llvm.intr.assume %true : i1
    %offs_am = arith.muli %pid_m_9, %c32_i32 : i32
    // CHECK-DAG: %[[c31_i32:.*]] = arith.constant 31 : i32
    // CHECK-DAG: %[[c32_i32:.*]] = arith.constant 32 : i32
    // CHECK-DAG: %[[M_PLUS_31:.*]] = arith.addi %[[M_SIZE]], %[[c31_i32]] : i32
    // CHECK-DAG: %[[M_TILES_END:.*]] = arith.divsi %[[M_PLUS_31]], %[[c32_i32]] : i32
    // CHECK-DAG: %[[N_PLUS_31:.*]] = arith.addi %[[N_SIZE]], %[[c31_i32]] : i32
    // CHECK-DAG: %[[N_TILES_END:.*]] = arith.divsi %[[N_PLUS_31]], %[[c32_i32]] : i32

    // CHECK-DAG: %[[BLOCK_INDEX_DIV_N:.*]] = arith.divsi %[[BLOCK_INDEX]], %[[N_TILES_END]] : i32
    // CHECK-DAG: %[[M_TILES_REMAINING:.*]] = arith.subi %[[M_TILES_END]], %[[BLOCK_INDEX_DIV_N]] : i32
    // CHECK-DAG: %[[GROUP_SIZE_M:.*]] = arith.minsi %[[M_TILES_REMAINING]], %[[c1_i32]] : i32
    // CHECK-DAG: %[[BLOCK_INDEX_MOD_N:.*]] = arith.remsi %[[BLOCK_INDEX]], %[[N_TILES_END]] : i32
    // CHECK-DAG: %[[GROUP_OFFSET:.*]] = arith.remsi %[[BLOCK_INDEX_MOD_N]], %[[GROUP_SIZE_M]] : i32
    // CHECK-DAG: %[[PID_M:.*]] = arith.addi %[[BLOCK_INDEX_DIV_N]], %[[GROUP_OFFSET]] : i32
    // CHECK-DAG: %[[PID_N:.*]] = arith.divsi %[[BLOCK_INDEX_MOD_N]], %[[GROUP_SIZE_M]] : i32

    // COM: Convert pid_m / pid_n into element indices (× 32, then mod M/N)
    // CHECK-DAG: %[[M_START:.*]] = arith.muli %[[PID_M]], %[[c32_i32]] : i32
    // CHECK-DAG: %[[M_INDEX:.*]] = arith.remsi %[[M_START]], %[[M_SIZE]] : i32
    // CHECK-DAG: %[[N_START:.*]] = arith.muli %[[PID_N]], %[[c32_i32]] : i32
    // CHECK-DAG: %[[N_INDEX:.*]] = arith.remsi %[[N_START]], %[[N_SIZE]] : i32

    // COM: Row/col offsets for A and B (uses A_BLOCK_STRIDE_M and f16 element size 2)
    // CHECK-DAG: %[[A_ROW_OFFSET_ELTS:.*]] = arith.muli %[[M_INDEX]], %[[A_BLOCK_STRIDE_M]] : i32
    // CHECK-DAG: %[[A_ROW_OFFSET_BYTES:.*]] = arith.muli %[[A_ROW_OFFSET_ELTS]], %[[c2_i32]] : i32
    // CHECK-DAG: %[[B_COL_OFFSET_BYTES:.*]] = arith.muli %[[N_INDEX]], %[[c2_i32]] : i32

    // COM: K tiling: ceil(K / 32)
    // CHECK-DAG: %[[K_PLUS_31:.*]] = arith.addi %[[K_SIZE]], %[[c31_i32]] : i32
    // CHECK-DAG: %[[K_TILES_END:.*]] = arith.divsi %[[K_PLUS_31]], %[[c32_i32]] : i32

    %offs_am_10 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_am_11 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_am_12 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %offs_am_13 = tt.splat %offs_am : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_am_14 = arith.addi %offs_am_13, %offs_am_10 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_am_15 = tt.splat %M : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_am_16 = arith.remsi %offs_am_14, %offs_am_15 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_bn = arith.muli %pid_n, %c32_i32 : i32
    %offs_bn_17 = tt.splat %offs_bn : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_bn_18 = arith.addi %offs_bn_17, %offs_am_11 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_bn_19 = tt.splat %N : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_bn_20 = arith.remsi %offs_bn_18, %offs_bn_19 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %a_ptrs = tt.expand_dims %offs_am_16 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi32, #blocked2>
    %a_ptrs_21 = tt.splat %stride_am : i32 -> tensor<32x1xi32, #blocked2>
    %a_ptrs_22 = arith.muli %a_ptrs, %a_ptrs_21 : tensor<32x1xi32, #blocked2>
    %a_ptrs_23 = tt.expand_dims %offs_am_12 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x32xi32, #blocked2>
    %a_ptrs_24 = tt.broadcast %a_ptrs_22 : tensor<32x1xi32, #blocked2> -> tensor<32x32xi32, #blocked2>
    %a_ptrs_25 = tt.broadcast %a_ptrs_23 : tensor<1x32xi32, #blocked2> -> tensor<32x32xi32, #blocked2>
    %a_ptrs_26 = arith.addi %a_ptrs_24, %a_ptrs_25 : tensor<32x32xi32, #blocked2>
    %a_ptrs_27 = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked2>
    %a_ptrs_28 = tt.addptr %a_ptrs_27, %a_ptrs_26 : tensor<32x32x!tt.ptr<f16>, #blocked2>, tensor<32x32xi32, #blocked2>
    %b_ptrs = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %b_ptrs_29 = tt.expand_dims %b_ptrs {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
    %b_ptrs_30 = tt.splat %stride_bk : i32 -> tensor<32x1xi32, #blocked1>

    %b_ptrs_31 = arith.muli %b_ptrs_29, %b_ptrs_30 : tensor<32x1xi32, #blocked1>
    %b_ptrs_32 = tt.expand_dims %offs_bn_20 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %b_ptrs_33 = tt.broadcast %b_ptrs_31 : tensor<32x1xi32, #blocked1> -> tensor<32x32xi32, #blocked1>
    %b_ptrs_34 = tt.broadcast %b_ptrs_32 : tensor<1x32xi32, #blocked1> -> tensor<32x32xi32, #blocked1>
    %b_ptrs_35 = arith.addi %b_ptrs_33, %b_ptrs_34 : tensor<32x32xi32, #blocked1>
    %b_ptrs_36 = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked1>
    %b_ptrs_37 = tt.addptr %b_ptrs_36, %b_ptrs_35 : tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32xi32, #blocked1>
    %5 = arith.addi %K, %c31_i32 : i32
    %6 = arith.divsi %5, %c32_i32 : i32
    // CHECK: scf.for {{.*}} iter_args(%[[A_LOOP_OFFSET:.*]] = %[[A_ROW_OFFSET_BYTES]], %[[B_LOOP_OFFSET:.*]] = %[[B_COL_OFFSET_BYTES]]) -> (i32, i32) : i32
    %accumulator:3 = scf.for %accumulator_38 = %c0_i32 to %6 step %c1_i32 iter_args(%accumulator_39 = %cst_1, %a_ptrs_40 = %a_ptrs_28, %b_ptrs_41 = %b_ptrs_37) -> (tensor<32x32xf32, #blocked>, tensor<32x32x!tt.ptr<f16>, #blocked2>, tensor<32x32x!tt.ptr<f16>, #blocked1>)  : i32 {
      // CHECK: %[[A_TILE_INDEX:.*]] = arith.divui %[[A_LOOP_OFFSET]], %[[A_TILE_SIZE]]
      // CHECK: ttkernel.cb_reserve_back(%[[A_CB]], %[[c1_i32]])
      // CHECK: %[[A_CB_WRITE_PTR:.*]] = ttkernel.get_write_ptr(%[[A_CB]])
      // CHECK: ttkernel.noc_async_read_tile(%[[A_TILE_INDEX]], %[[A_ACCESSOR]], %[[A_CB_WRITE_PTR]], %[[c1_i8]])
      // CHECK: ttkernel.noc_async_read_barrier(%[[c1_i8]])
      // CHECK: ttkernel.cb_push_back(%[[A_CB]], %[[c1_i32]])

      %a_42 = arith.muli %accumulator_38, %c32_i32 : i32
      %a_43 = arith.subi %K, %a_42 : i32
      %a_44 = tt.splat %a_43 : i32 -> tensor<1x32xi32, #blocked2>
      %a_45 = arith.cmpi slt, %a_ptrs_23, %a_44 : tensor<1x32xi32, #blocked2>
      %a_46 = tt.broadcast %a_45 : tensor<1x32xi1, #blocked2> -> tensor<32x32xi1, #blocked2>
      %a_47 = tt.load %a_ptrs_40, %a_46, %cst_3 : tensor<32x32x!tt.ptr<f16>, #blocked2>
      ttg.local_store %a_47, %a : tensor<32x32xf16, #blocked2> -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
      // CHECK: %[[B_TILE_INDEX:.*]] = arith.divui %[[B_LOOP_OFFSET]], %[[B_TILE_SIZE]]
      // CHECK: ttkernel.cb_reserve_back(%[[B_CB]], %[[c1_i32]])
      // CHECK: %[[B_CB_WRITE_PTR:.*]] = ttkernel.get_write_ptr(%[[B_CB]])
      // CHECK: ttkernel.noc_async_read_tile(%[[B_TILE_INDEX]], %[[B_ACCESSOR]], %[[B_CB_WRITE_PTR]], %[[c1_i8]])
      // CHECK: ttkernel.noc_async_read_barrier(%[[c1_i8]])
      // CHECK: ttkernel.cb_push_back(%[[B_CB]], %[[c1_i32]])

      %b_48 = tt.splat %a_43 : i32 -> tensor<32x1xi32, #blocked1>
      %b_49 = arith.cmpi slt, %b_ptrs_29, %b_48 : tensor<32x1xi32, #blocked1>
      %b_50 = tt.broadcast %b_49 : tensor<32x1xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
      %b_51 = tt.load %b_ptrs_41, %b_50, %cst_2 : tensor<32x32x!tt.ptr<f16>, #blocked1>
      ttg.local_store %b_51, %b : tensor<32x32xf16, #blocked1> -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
      %accumulator_52 = tt.dot %cst_0, %cst, %accumulator_39 : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x32xf32, #blocked>
      %a_ptrs_53 = tt.addptr %a_ptrs_40, %cst_4 : tensor<32x32x!tt.ptr<f16>, #blocked2>, tensor<32x32xi32, #blocked2>
      %b_ptrs_54 = arith.muli %stride_bk, %c32_i32 : i32
      %b_ptrs_55 = tt.splat %b_ptrs_54 : i32 -> tensor<32x32xi32, #blocked1>
      %b_ptrs_56 = tt.addptr %b_ptrs_41, %b_ptrs_55 : tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32xi32, #blocked1>
      // CHECK: %[[A_NEXT_VALUE:.*]] = arith.addi %[[A_LOOP_OFFSET]], %[[c64_i32]]
      // CHECK: %[[B_LOOP_INCREMENT:.*]] = arith.muli %[[B_BLOCK_STRIDE_K]], %[[c64_i32]]
      // CHECK: %[[B_NEXT_VALUE:.*]] = arith.addi %[[B_LOOP_OFFSET]], %[[B_LOOP_INCREMENT]]
      // CHECK: scf.yield %[[A_NEXT_VALUE]], %[[B_NEXT_VALUE]]
      scf.yield %accumulator_52, %a_ptrs_53, %b_ptrs_56 : tensor<32x32xf32, #blocked>, tensor<32x32x!tt.ptr<f16>, #blocked2>, tensor<32x32x!tt.ptr<f16>, #blocked1>
    }
    // CHECK: return
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.padded_shared<[1:+1] {order = [1, 0], shape = [32, 32]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
    // CHECK: func.func public @matmul_kernel__compute()
    tt.func public @matmul_kernel__compute(%a_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %b_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %c_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %M: i32 {tt.divisibility = 8 : i32}, %N: i32 {tt.divisibility = 8 : i32}, %K: i32 {tt.divisibility = 8 : i32}, %stride_am: i32 {tt.divisibility = 8 : i32}, %stride_bk: i32 {tt.divisibility = 8 : i32}, %stride_cm: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
        %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
        %c31_i32 = arith.constant 31 : i32
        %c1_i32 = arith.constant 1 : i32
        %c0_i32 = arith.constant 0 : i32
        %true = arith.constant true
        %c32_i32 = arith.constant 32 : i32
        // CHECK-DAG: %[[c0_i32:.*]] = arith.constant 0 : i32
        // CHECK-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
        // CHECK-DAG: %[[c31_i32:.*]] = arith.constant 31 : i32
        // CHECK-DAG: %[[c32_i32:.*]] = arith.constant 32 : i32

        // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
        // CHECK-DAG: %[[c5:.*]] = arith.constant 5 : index

        // CHECK-DAG: %[[K_SIZE:.*]] = ttkernel.get_common_arg_val(%[[c5]])

        // CHECK-DAG: %[[A_CB:.*]] = ttkernel.get_compile_time_arg_val(0)
        // CHECK-DAG: %[[B_CB:.*]] = ttkernel.get_compile_time_arg_val(1)
        // CHECK-DAG: %[[C_CB:.*]] = ttkernel.get_compile_time_arg_val(2)

        // CHECK: ttkernel.mm_init(%[[A_CB]], %[[B_CB]], %[[C_CB]], %[[c0_i32]])
        %0 = ttg.local_alloc {alloc_idx = 2 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
        %b = ttg.local_alloc {alloc_idx = 1 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
        %a = ttg.local_alloc {alloc_idx = 0 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
        %pid = tt.get_program_id x : i32
        %num_pid_m = arith.addi %M, %c31_i32 : i32
        %num_pid_m_0 = arith.divsi %num_pid_m, %c32_i32 : i32
        %num_pid_n = arith.addi %N, %c31_i32 : i32
        %num_pid_n_1 = arith.divsi %num_pid_n, %c32_i32 : i32
        %group_id = arith.divsi %pid, %num_pid_n_1 : i32
        %group_size_m = arith.subi %num_pid_m_0, %group_id : i32
        %group_size_m_2 = arith.minsi %group_size_m, %c1_i32 : i32
        %pid_m = arith.remsi %pid, %num_pid_n_1 : i32
        %pid_m_3 = arith.remsi %pid_m, %group_size_m_2 : i32
        %pid_m_4 = arith.addi %group_id, %pid_m_3 : i32
        %pid_n = arith.divsi %pid_m, %group_size_m_2 : i32
        %1 = arith.cmpi sge, %pid_m_4, %c0_i32 : i32
        llvm.intr.assume %1 : i1
        %2 = arith.cmpi sge, %pid_n, %c0_i32 : i32
        llvm.intr.assume %2 : i1
        %3 = arith.cmpi sgt, %stride_am, %c0_i32 : i32
        llvm.intr.assume %3 : i1
        llvm.intr.assume %true : i1
        llvm.intr.assume %true : i1
        %4 = arith.cmpi sgt, %stride_bk, %c0_i32 : i32
        llvm.intr.assume %4 : i1
        %5 = arith.cmpi sgt, %stride_cm, %c0_i32 : i32
        llvm.intr.assume %5 : i1
        llvm.intr.assume %true : i1
        %6 = arith.addi %K, %c31_i32 : i32
        %7 = arith.divsi %6, %c32_i32 : i32
        // CHECK: ttkernel.tile_regs_acquire()
        // CHECK: ttkernel.mm_init_short(%[[A_CB]], %[[B_CB]], %[[c0_i32]])
        // CHECK: %[[K_PLUS_31:.*]] = arith.addi %[[K_SIZE]], %[[c31_i32]]
        // CHECK: %[[K_TILES_END:.*]] = arith.divsi %[[K_PLUS_31]], %[[c32_i32]]
        // CHECK: scf.for %[[ARG0:.*]] = %[[c0_i32]] to %[[K_TILES_END]] step %[[c1_i32]] : i32 {
        %accumulator = scf.for %accumulator_5 = %c0_i32 to %7 step %c1_i32 iter_args(%arg10 = %cst) -> (tensor<32x32xf32, #blocked>)  : i32 {
          // CHECK-DAG: ttkernel.cb_wait_front(%[[A_CB]], %[[c1_i32]])
          // CHECK-DAG: ttkernel.cb_wait_front(%[[B_CB]], %[[c1_i32]])
          // CHECK: ttkernel.matmul_tiles(%[[A_CB]], %[[B_CB]], %[[c0_i32]], %[[c0_i32]], %[[c0]])
        %a_6 = ttg.local_load %a : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
        %b_7 = ttg.local_load %b : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
        %accumulator_8 = tt.dot %a_6, %b_7, %arg10 {triton_tenstorrent.alloc_offset = 0 : i32, triton_tenstorrent.alloc_size = 1 : i32} : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x32xf32, #blocked>
        scf.yield %accumulator_8 : tensor<32x32xf32, #blocked>
        }
        %c = arith.truncf %accumulator {triton_tenstorrent.alloc_offset = 0 : i32, triton_tenstorrent.alloc_size = 1 : i32} : tensor<32x32xf32, #blocked> to tensor<32x32xf16, #blocked>
        ttg.local_store %c, %0 : tensor<32x32xf16, #blocked> -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
        // CHECK-DAG: ttkernel.cb_pop_front(%[[A_CB]], %[[c1_i32]])
        // CHECK-DAG: ttkernel.cb_pop_front(%[[B_CB]], %[[c1_i32]])
        // CHECK: }
        // CHECK: ttkernel.cb_reserve_back(%[[C_CB]], %[[c1_i32]])
        // CHECK: ttkernel.tile_regs_commit()
        // CHECK: ttkernel.tile_regs_wait()
        // CHECK: ttkernel.pack_tile(%[[c0_i32]], %[[C_CB]], %[[c0_i32]], true)
        // CHECK: ttkernel.tile_regs_release()
        // CHECK: ttkernel.cb_push_back(%[[C_CB]], %[[c1_i32]])
        // CHECK: return
        tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.padded_shared<[1:+1] {order = [1, 0], shape = [32, 32]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
// CHECK: func.func public @matmul_kernel__writer()
tt.func public @matmul_kernel__writer(%a_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %b_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %c_ptr: !tt.ptr<f16> {tt.divisibility = 8 : i32}, %M: i32 {tt.divisibility = 8 : i32}, %N: i32 {tt.divisibility = 8 : i32}, %K: i32 {tt.divisibility = 8 : i32}, %stride_am: i32 {tt.divisibility = 8 : i32}, %stride_bk: i32 {tt.divisibility = 8 : i32}, %stride_cm: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %c31_i32 = arith.constant 31 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c32_i32 = arith.constant 32 : i32
    // CHECK-DAG: %[[c0_i32:.*]] = arith.constant 0 : i32
    // CHECK-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
    // CHECK-DAG: %[[c3_i32:.*]] = arith.constant 3 : i32
    // CHECK-DAG: %[[c31_i32:.*]] = arith.constant 31 : i32
    // CHECK-DAG: %[[c32_i32:.*]] = arith.constant 32 : i32
    // CHECK-DAG: %[[c64_i32:.*]] = arith.constant 64 : i32
    // CHECK-DAG: %[[c0_i8:.*]] = arith.constant 0 : i8


    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[c3:.*]] = arith.constant 3 : index
    // CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index

    // Three chained TensorAccessorArgs (A, B, output as anchors); only the
    // output (tag 2) accessor is built, since the writer only stores C.
    // CHECK-DAG: %[[A_ARGS:.*]] = ttkernel.TensorAccessorArgs(%[[c3_i32]], %[[c0_i32]]) {tt.accessor_base_arg_index = 0 : i32}
    // CHECK-DAG: %[[B_ARGS:.*]] = ttkernel.TensorAccessorArgs(prev = %[[A_ARGS]]) {tt.accessor_base_arg_index = 1 : i32}
    // CHECK-DAG: %[[C_ARGS:.*]] = ttkernel.TensorAccessorArgs(prev = %[[B_ARGS]]) {tt.accessor_base_arg_index = 2 : i32}
    // CHECK-DAG: %[[C_PTR:.*]] = ttkernel.get_common_arg_val(%[[c2]])
    // CHECK-DAG: %[[M_SIZE:.*]] = ttkernel.get_common_arg_val(%[[c3]])
    // CHECK-DAG: %[[N_SIZE:.*]] = ttkernel.get_common_arg_val(%[[c4]])

    %0 = ttg.local_alloc {alloc_idx = 2 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    // CHECK-DAG: %[[C_CB:.*]] = ttkernel.get_compile_time_arg_val(2)
    // CHECK: %[[TILE_SIZE:.*]] = ttkernel.get_tile_size(%[[C_CB]])
    // CHECK: %[[C_ACCESSOR:.*]] = ttkernel.TensorAccessor(%[[C_ARGS]], %[[C_PTR]], %[[TILE_SIZE]])
    %pid = tt.get_program_id x : i32
    // CHECK: %[[PID:.*]] = ttkernel.get_arg_val(%[[c0]])
    %num_pid_m = arith.addi %M, %c31_i32 : i32
    %num_pid_m_0 = arith.divsi %num_pid_m, %c32_i32 : i32
    %num_pid_n = arith.addi %N, %c31_i32 : i32
    %num_pid_n_1 = arith.divsi %num_pid_n, %c32_i32 : i32
    %group_id = arith.divsi %pid, %num_pid_n_1 : i32
    %group_size_m = arith.subi %num_pid_m_0, %group_id : i32
    %group_size_m_2 = arith.minsi %group_size_m, %c1_i32 : i32
    %pid_m = arith.remsi %pid, %num_pid_n_1 : i32
    %pid_m_3 = arith.remsi %pid_m, %group_size_m_2 : i32
    %pid_m_4 = arith.addi %group_id, %pid_m_3 : i32
    %pid_n = arith.divsi %pid_m, %group_size_m_2 : i32
    %1 = arith.cmpi sge, %pid_m_4, %c0_i32 : i32
    // CHECK: %[[M_PADDED:.*]] = arith.addi %[[M_SIZE]], %[[c31_i32]]
    // CHECK: %[[M_TILES_END:.*]] = arith.divsi %[[M_PADDED]], %[[c32_i32]]
    // CHECK: %[[N_PADDED:.*]] = arith.addi %[[N_SIZE]], %[[c31_i32]]
    // CHECK: %[[N_TILES_END:.*]] = arith.divsi %[[N_PADDED]], %[[c32_i32]]
    llvm.intr.assume %1 : i1
    %2 = arith.cmpi sge, %pid_n, %c0_i32 : i32
    llvm.intr.assume %2 : i1
    %3 = arith.cmpi sgt, %stride_am, %c0_i32 : i32
    llvm.intr.assume %3 : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    %4 = arith.cmpi sgt, %stride_bk, %c0_i32 : i32
    llvm.intr.assume %4 : i1
    %5 = arith.cmpi sgt, %stride_cm, %c0_i32 : i32
    llvm.intr.assume %5 : i1
    llvm.intr.assume %true : i1
    // CHECK: %[[GROUP_ID_M:.*]] = arith.divsi %[[PID]], %[[N_TILES_END]] : i32
    // CHECK: %[[REMAINING_M_TILES:.*]] = arith.subi %[[M_TILES_END]], %[[GROUP_ID_M]] : i32
    // CHECK: %[[GROUP_HEIGHT_TILES:.*]] = arith.minsi %[[REMAINING_M_TILES]], %[[c1_i32]] : i32
    // CHECK: %[[GROUP_TILE_REMAINDER:.*]] = arith.remsi %[[PID]], %[[N_TILES_END]] : i32
    // CHECK: %[[GROUP_ROW_INDEX:.*]] = arith.divsi %[[GROUP_TILE_REMAINDER]], %[[GROUP_HEIGHT_TILES]] : i32
    // CHECK: %[[ROW_OFFSET_ELEMS:.*]] = arith.muli %[[GROUP_ROW_INDEX]], %[[c64_i32]] : i32

    %offs_am = arith.muli %pid_m_4, %c32_i32 : i32
    %offs_am_5 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_am_6 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %offs_am_7 = tt.splat %offs_am : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_am_8 = arith.addi %offs_am_7, %offs_am_5 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %offs_bn = arith.muli %pid_n, %c32_i32 : i32
    %offs_bn_9 = tt.splat %offs_bn : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %offs_bn_10 = arith.addi %offs_bn_9, %offs_am_6 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %c_ptrs = tt.expand_dims %offs_am_8 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi32, #blocked2>
    %c_ptrs_11 = tt.splat %stride_cm : i32 -> tensor<32x1xi32, #blocked2>
    %c_ptrs_12 = arith.muli %c_ptrs_11, %c_ptrs : tensor<32x1xi32, #blocked2>
    %c_ptrs_13 = tt.splat %c_ptr : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked2>
    %c_ptrs_14 = tt.addptr %c_ptrs_13, %c_ptrs_12 : tensor<32x1x!tt.ptr<f16>, #blocked2>, tensor<32x1xi32, #blocked2>
    %c_ptrs_15 = tt.expand_dims %offs_bn_10 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x32xi32, #blocked2>
    %c_ptrs_16 = tt.broadcast %c_ptrs_14 : tensor<32x1x!tt.ptr<f16>, #blocked2> -> tensor<32x32x!tt.ptr<f16>, #blocked2>
    %c_ptrs_17 = tt.broadcast %c_ptrs_15 : tensor<1x32xi32, #blocked2> -> tensor<32x32xi32, #blocked2>
    %c_ptrs_18 = tt.addptr %c_ptrs_16, %c_ptrs_17 : tensor<32x32x!tt.ptr<f16>, #blocked2>, tensor<32x32xi32, #blocked2>
    %c_mask = tt.splat %M : i32 -> tensor<32x1xi32, #blocked2>
    %c_mask_19 = arith.cmpi slt, %c_ptrs, %c_mask : tensor<32x1xi32, #blocked2>
    %c_mask_20 = tt.splat %N : i32 -> tensor<1x32xi32, #blocked2>
    %c_mask_21 = arith.cmpi slt, %c_ptrs_15, %c_mask_20 : tensor<1x32xi32, #blocked2>
    %c_mask_22 = tt.broadcast %c_mask_19 : tensor<32x1xi1, #blocked2> -> tensor<32x32xi1, #blocked2>
    %c_mask_23 = tt.broadcast %c_mask_21 : tensor<1x32xi1, #blocked2> -> tensor<32x32xi1, #blocked2>
    %c_mask_24 = arith.andi %c_mask_22, %c_mask_23 : tensor<32x32xi1, #blocked2>
    %6 = ttg.local_load %0 : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #blocked2>
    // CHECK: ttkernel.cb_wait_front(%[[C_CB]], %[[c1_i32]])
    // CHECK: %[[NOC_TILE_INDEX:.*]] = arith.divui %[[ROW_OFFSET_ELEMS]], %[[TILE_SIZE]]
    // CHECK: %[[READ_PTR:.*]] = ttkernel.get_read_ptr(%[[C_CB]])
    // CHECK: ttkernel.noc_async_write_tile(%[[NOC_TILE_INDEX]], %[[C_ACCESSOR]], %[[READ_PTR]], %[[c0_i8]])
    // CHECK: ttkernel.noc_async_write_barrier(%[[c0_i8]])
    // CHECK: ttkernel.cb_pop_front(%[[C_CB]], %[[c1_i32]])
    tt.store %c_ptrs_18, %6, %c_mask_24 : tensor<32x32x!tt.ptr<f16>, #blocked2>
    // CHECK: return
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.padded_shared<[1:+1] {order = [1, 0], shape = [32, 32]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK: func.func public @matmul_kernel_tma__compute()
  tt.func public @matmul_kernel_tma__compute(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i1, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.ptr<f16>, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i1, %arg16: i32, %arg17: i32, %arg18: i64, %arg19: i64, %arg20: !tt.ptr<f16>, %arg21: i32, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i1, %arg26: i32, %arg27: i32, %arg28: i64, %arg29: i64, %arg30: i32 {tt.divisibility = 8 : i32}, %arg31: i32 {tt.divisibility = 8 : i32}, %arg32: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %c31_i32 = arith.constant 31 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %0 = ttg.local_alloc {alloc_idx = 2 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc {alloc_idx = 1 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %a = ttg.local_alloc {alloc_idx = 0 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %1 = ttc.block_end
    %2 = ttc.block_start
    // CHECK-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
    // CHECK-DAG: %[[c0_index:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1_index:.*]] = arith.constant 1 : index
    // CHECK: ttkernel.mm_init
    // CHECK-DAG: %[[START:.*]] = ttkernel.get_arg_val(%[[c0_index]])
    // CHECK-DAG: %[[END:.*]] = ttkernel.get_arg_val(%[[c1_index]])

    %k_tiles = arith.addi %arg32, %c31_i32 : i32
    %k_tiles_0 = arith.divsi %k_tiles, %c32_i32 : i32
    // CHECK: scf.for %[[arg0:.*]] = %[[START]] to %[[END]] step %[[c1_i32]] : i32 {
    // CHECK: ttkernel.tile_regs_acquire()
    // CHECK: ttkernel.mm_init_short
    scf.for %arg33 = %2 to %1 step %c1_i32  : i32 {
      %3 = ttc.current_block %arg33 : i32
      %accumulator = scf.for %accumulator_1 = %c0_i32 to %k_tiles_0 step %c1_i32 iter_args(%arg35 = %cst) -> (tensor<32x32xf32, #blocked>)  : i32 {
      %a_2 = ttg.local_load %a : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %b_3 = ttg.local_load %b : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %accumulator_4 = tt.dot %a_2, %b_3, %arg35 {triton_tenstorrent.alloc_offset = 2 : i32, triton_tenstorrent.alloc_size = 1 : i32} : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x32xf32, #blocked>
      scf.yield %accumulator_4 : tensor<32x32xf32, #blocked>
      }
      %c = arith.truncf %accumulator {triton_tenstorrent.alloc_offset = 2 : i32, triton_tenstorrent.alloc_size = 1 : i32} : tensor<32x32xf32, #blocked> to tensor<32x32xf16, #blocked>
      ttg.local_store %c, %0 : tensor<32x32xf16, #blocked> -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    }
    // CHECK: }
    // CHECK: ttkernel.tile_regs_commit()
    // CHECK: ttkernel.tile_regs_wait()
    // CHECK: ttkernel.pack_tile
    // CHECK: ttkernel.tile_regs_release()
    // CHECK: return
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.padded_shared<[1:+1] {order = [1, 0], shape = [32, 32]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK: func.func public @matmul_kernel_tma__writer()
  tt.func public @matmul_kernel_tma__writer(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i1, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.ptr<f16>, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i1, %arg16: i32, %arg17: i32, %arg18: i64, %arg19: i64, %arg20: !tt.ptr<f16>, %arg21: i32, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i1, %arg26: i32, %arg27: i32, %arg28: i64, %arg29: i64, %arg30: i32 {tt.divisibility = 8 : i32}, %arg31: i32 {tt.divisibility = 8 : i32}, %arg32: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<1x32xi32, #blocked1>
    %cst_0 = arith.constant dense<32> : tensor<32x1xi32, #blocked1>
    %c31_i32 = arith.constant 31 : i32
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %0 = ttg.local_alloc {alloc_idx = 2 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %1 = ttc.block_end
    %2 = ttc.block_start
    %num_pid_m = arith.addi %arg30, %c31_i32 : i32
    %num_pid_m_1 = arith.divsi %num_pid_m, %c32_i32 : i32
    %num_pid_n = arith.addi %arg31, %c31_i32 : i32
    %num_pid_n_2 = arith.divsi %num_pid_n, %c32_i32 : i32
    %num_pid_in_group = arith.muli %num_pid_n_2, %c8_i32 : i32
    %a = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %a_3 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %3 = arith.ceildivsi %arg22, %c32_i32 : i32
    %4 = tt.splat %arg20 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked1>

    // CHECK-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
    // CHECK-DAG: %[[c0_index:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1_index:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[START:.*]] = ttkernel.get_arg_val(%[[c0_index]])
    // CHECK-DAG: %[[END:.*]] = ttkernel.get_arg_val(%[[c1_index]])

    // CHECK: scf.for %[[arg0:.*]] = %[[START]] to %[[END]] step %[[c1_i32]] : i32 {
    scf.for %arg33 = %2 to %1 step %c1_i32  : i32 {
      %5 = ttc.current_block %arg33 : i32
      %group_id = arith.divsi %5, %num_pid_in_group : i32
      // CHECK: %[[GROUP_ID:.*]] = arith.divsi %[[arg0]]
      %first_pid_m = arith.muli %group_id, %c8_i32 : i32
      %group_size_m = arith.subi %num_pid_m_1, %first_pid_m : i32
      %group_size_m_4 = arith.minsi %group_size_m, %c8_i32 : i32
      %pid_m = arith.remsi %5, %group_size_m_4 : i32
      %pid_m_5 = arith.addi %first_pid_m, %pid_m : i32
      %pid_n = arith.remsi %5, %num_pid_in_group : i32
      %pid_n_6 = arith.divsi %pid_n, %group_size_m_4 : i32
      %offs_am = arith.muli %pid_m_5, %c32_i32 : i32
      %offs_bn = arith.muli %pid_n_6, %c32_i32 : i32
      %a_7 = tt.splat %offs_am : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %a_8 = arith.addi %a_7, %a : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %a_9 = tt.expand_dims %a_8 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
      %a_10 = arith.divsi %offs_am, %c32_i32 : i32
      %a_11 = arith.remui %a_9, %cst_0 : tensor<32x1xi32, #blocked1>
      %a_12 = arith.muli %a_11, %cst_0 : tensor<32x1xi32, #blocked1>
      %a_13 = tt.broadcast %a_12 : tensor<32x1xi32, #blocked1> -> tensor<32x32xi32, #blocked1>
      %b = tt.splat %offs_bn : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %b_14 = arith.addi %b, %a_3 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %b_15 = tt.expand_dims %b_14 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
      %b_16 = arith.divsi %offs_bn, %c32_i32 : i32
      %b_17 = arith.remui %b_15, %cst : tensor<1x32xi32, #blocked1>
      %b_18 = tt.broadcast %b_17 : tensor<1x32xi32, #blocked1> -> tensor<32x32xi32, #blocked1>
      %6 = arith.muli %a_10, %3 : i32
      %7 = arith.addi %6, %b_16 : i32
      %8 = arith.muli %7, %c1024_i32 : i32
      %9 = arith.addi %a_13, %b_18 : tensor<32x32xi32, #blocked1>
      %10 = tt.splat %8 : i32 -> tensor<32x32xi32, #blocked1>
      %11 = arith.addi %10, %9 : tensor<32x32xi32, #blocked1>
      %12 = tt.addptr %4, %11 : tensor<32x32x!tt.ptr<f16>, #blocked1>, tensor<32x32xi32, #blocked1>
      %13 = ttg.local_load %0 : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #blocked1>
      tt.store %12, %13 : tensor<32x32x!tt.ptr<f16>, #blocked1>
    // CHECK: }
    // CHECK: return
    }
    tt.return
  }
}

// -----

// COM: Tensor Descriptor loads with tiled dot layout op A (reads a 32x64 block using 32x32 tiles)
#shared = #ttg.padded_shared<[1:+1] {order = [1, 0], shape = [64, 64]}>
#shared1 = #ttg.padded_shared<[1:+1] {order = [1, 0], shape = [32, 64]}>
#smem = #ttg.shared_memory
#tiled = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [1, 2], order = [1, 0], tileShape = [32, 32]}>
#tiled1 = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [2, 2], order = [1, 0], tileShape = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 1 : i32} {
  // CHECK: func public @load_tiled_dot_op_A__reader
  tt.func public @load_tiled_dot_op_A__reader(%arg0: !tt.tensordesc<tensor<32x64xf16>>, %offs_am: i32, %offs_k: i32) {
    // CHECK-DAG: %[[c0_i32:.*]] = arith.constant 0 : i32
    // CHECK-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
    // CHECK-DAG: %[[c2_i32:.*]] = arith.constant 2 : i32
    // CHECK-DAG: %[[c32_i32:.*]] = arith.constant 32 : i32
    // CHECK-DAG: %[[c2048_i32:.*]] = arith.constant 2048 : i32
    // CHECK-DAG: %[[c1_i8:.*]] = arith.constant 1 : i8
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[c6:.*]] = arith.constant 6 : index
    // CHECK-DAG: %[[c7:.*]] = arith.constant 7 : index
    // COM: Single tensor input (A) -> one TensorAccessorArgs at cta_base = 1 (one CB port).
    // CHECK: %[[ARGS:.*]] = ttkernel.TensorAccessorArgs(%[[c1_i32]], %[[c0_i32]]) {tt.accessor_base_arg_index = 0 : i32}
    // CHECK: %[[PTR:.*]] = ttkernel.get_common_arg_val(%[[c0]]) : (index) -> i32
    // CHECK: %[[DESC_SHAPE0:.*]] = ttkernel.get_common_arg_val(%[[c2]]) : (index) -> i32
    // CHECK: %[[SHAPE0:.*]] = ttkernel.get_common_arg_val(%[[c6]]) : (index) -> i32
    // CHECK: %[[SHAPE1:.*]] = ttkernel.get_common_arg_val(%[[c7]]) : (index) -> i32
    %a = ttg.local_alloc {alloc_idx = 0 : i32} : () -> !ttg.memdesc<32x64xf16, #shared1, #smem, mutable>
    // CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, f16>>
    // CHECK: %[[TILESIZE:.*]] = ttkernel.get_tile_size(%[[CB]]) : (!ttkernel.cb<2, !ttcore.tile<32x32, f16>>) -> i32
    // CHECK: ttkernel.cb_reserve_back(%[[CB]], %[[c2_i32]]) : (!ttkernel.cb<2, !ttcore.tile<32x32, f16>>, i32) -> ()
    // CHECK: %[[ACCESSOR:.*]] = ttkernel.TensorAccessor(%[[ARGS]], %[[PTR]], %[[TILESIZE]])
    // CHECK: %[[CB_WRITE_PTR:.*]] = ttkernel.get_write_ptr(%[[CB]]) : (!ttkernel.cb<2, !ttcore.tile<32x32, f16>>) -> i32

    // CHECK-DAG: %[[TILES_PER_DIM0:.*]] = arith.ceildivsi %[[DESC_SHAPE0]], %[[c32_i32]] : i32

    %a_6 = tt.descriptor_load %arg0[%offs_am, %offs_k] : !tt.tensordesc<tensor<32x64xf16>> -> tensor<32x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #tiled}>>

    // CHECK-DAG: %[[X_TILE_ID:.*]] = arith.divsi %[[SHAPE0]], %[[c32_i32]] : i32
    // CHECK-DAG: %[[Y_TILE_ID:.*]] = arith.divsi %[[SHAPE1]], %[[c32_i32]] : i32

    // COM: 1. Calculate Base DRAM Address (Row * Stride + Col)
    // CHECK: %[[ROW_OFFSET:.*]] = arith.muli %[[X_TILE_ID]], %[[TILES_PER_DIM0]] : i32
    // CHECK: %[[BASE_DRAM_ID:.*]] = arith.addi %[[ROW_OFFSET]], %[[Y_TILE_ID]] : i32

    // COM: 2. Tile 0 Issue (Offset 0, 0)
    // CHECK: ttkernel.noc_async_read_tile(%[[BASE_DRAM_ID]], %[[ACCESSOR]], %[[CB_WRITE_PTR]], %[[c1_i8]])

    // COM: 3. Tile 1 Issue (Offset 0, 1)
    // CHECK: %[[L1_ADDR_1:.*]] = arith.addi %[[CB_WRITE_PTR]], %[[c2048_i32]] : i32
    // CHECK: %[[DRAM_ID_1:.*]] = arith.addi %[[BASE_DRAM_ID]], %[[c1_i32]] : i32
    // CHECK: ttkernel.noc_async_read_tile(%[[DRAM_ID_1]], %[[ACCESSOR]], %[[L1_ADDR_1]], %[[c1_i8]])

    // CHECK: ttkernel.noc_async_read_barrier(%[[c1_i8]]) : (i8) -> ()
    ttg.local_store %a_6, %a : tensor<32x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #tiled}>> -> !ttg.memdesc<32x64xf16, #shared1, #smem, mutable>
    // CHECK: ttkernel.cb_push_back(%[[CB]], %[[c2_i32]]) : (!ttkernel.cb<2, !ttcore.tile<32x32, f16>>, i32) -> ()
    // CHECK: return
    tt.return
  }
}

// -----

// COM: Tensor Descriptor loads with tiled dot layout op B (reads a 64x64 block using 32x32 tiles)
#shared = #ttg.padded_shared<[1:+1] {order = [1, 0], shape = [64, 64]}>
#shared1 = #ttg.padded_shared<[1:+1] {order = [1, 0], shape = [32, 64]}>
#smem = #ttg.shared_memory
#tiled = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [1, 2], order = [1, 0], tileShape = [32, 32]}>
#tiled1 = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [2, 2], order = [1, 0], tileShape = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 1 : i32} {
  // CHECK: func public @load_tiled_dot_op_B__reader
  tt.func public @load_tiled_dot_op_B__reader(%arg5: !tt.tensordesc<tensor<64x64xf16>>, %offs_bn: i32, %offs_k: i32) {
    // CHECK-DAG: %[[c0_i32:.*]] = arith.constant 0 : i32
    // CHECK-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
    // CHECK-DAG: %[[c4_i32:.*]] = arith.constant 4 : i32
    // CHECK-DAG: %[[c32_i32:.*]] = arith.constant 32 : i32
    // CHECK-DAG: %[[c2048_i32:.*]] = arith.constant 2048 : i32
    // CHECK-DAG: %[[c4096_i32:.*]] = arith.constant 4096 : i32
    // CHECK-DAG: %[[c6144_i32:.*]] = arith.constant 6144 : i32
    // CHECK-DAG: %[[c1_i8:.*]] = arith.constant 1 : i8
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[c6:.*]] = arith.constant 6 : index
    // CHECK-DAG: %[[c7:.*]] = arith.constant 7 : index
    // COM: Single tensor input (B) -> one TensorAccessorArgs at cta_base = 1 (one CB port).
    // CHECK: %[[ARGS:.*]] = ttkernel.TensorAccessorArgs(%[[c1_i32]], %[[c0_i32]]) {tt.accessor_base_arg_index = 0 : i32}
    // CHECK: %[[PTR:.*]] = ttkernel.get_common_arg_val(%[[c0]]) : (index) -> i32
    // CHECK: %[[DESC_SHAPE0:.*]] = ttkernel.get_common_arg_val(%[[c2]]) : (index) -> i32
    // CHECK: %[[SHAPE0:.*]] = ttkernel.get_common_arg_val(%[[c6]]) : (index) -> i32
    // CHECK: %[[SHAPE1:.*]] = ttkernel.get_common_arg_val(%[[c7]]) : (index) -> i32
    // CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, f16>>
    // CHECK: %[[TILESIZE:.*]] = ttkernel.get_tile_size(%[[CB]]) : (!ttkernel.cb<4, !ttcore.tile<32x32, f16>>) -> i32

    %b = ttg.local_alloc {alloc_idx = 1 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: ttkernel.cb_reserve_back(%[[CB]], %[[c4_i32]])
    // CHECK: %[[ACCESSOR:.*]] = ttkernel.TensorAccessor(%[[ARGS]], %[[PTR]], %[[TILESIZE]])
    // CHECK: %[[CB_ADDR:.*]] = ttkernel.get_write_ptr(%[[CB]])

    // CHECK-DAG: %[[TILES_PER_DIM0:.*]] = arith.ceildivsi %[[DESC_SHAPE0]], %[[c32_i32]] : i32

    // CHECK-DAG: %[[X_TILE_ID:.*]] = arith.divsi %[[SHAPE0]], %[[c32_i32]] : i32
    // CHECK-DAG: %[[Y_TILE_ID:.*]] = arith.divsi %[[SHAPE1]], %[[c32_i32]] : i32
    %b_7 = tt.descriptor_load %arg5[%offs_k, %offs_bn] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 1, parent = #tiled1}>>

    // COM: Calculate Base DRAM Address (Row * Stride + Col)
    // CHECK: %[[ROW_OFFSET:.*]] = arith.muli %[[Y_TILE_ID]], %[[TILES_PER_DIM0]] : i32
    // CHECK: %[[BASE_DRAM_ID:.*]] = arith.addi %[[ROW_OFFSET]], %[[X_TILE_ID]] : i32

    // COM: Tile 0 (Row 0, Col 0) -> L1 Offset 0
    // CHECK: ttkernel.noc_async_read_tile(%[[BASE_DRAM_ID]], %[[ACCESSOR]], %[[CB_ADDR]], %[[c1_i8]])

    // COM: Tile 1 (Row 0, Col 1) -> L1 Offset 2 (4096 bytes)
    // CHECK: %[[L1_ADDR_1:.*]] = arith.addi %[[CB_ADDR]], %[[c4096_i32]] : i32
    // CHECK: %[[DRAM_ID_1:.*]] = arith.addi %[[BASE_DRAM_ID]], %[[c1_i32]] : i32
    // CHECK: ttkernel.noc_async_read_tile(%[[DRAM_ID_1]], %[[ACCESSOR]], %[[L1_ADDR_1]], %[[c1_i8]])

    // COM: Tile 2 (Row 1, Col 0) -> L1 Offset 1 (2048 bytes)
    // CHECK: %[[L1_ADDR_2:.*]] = arith.addi %[[CB_ADDR]], %[[c2048_i32]] : i32
    // CHECK: %[[DRAM_ID_2:.*]] = arith.addi %[[BASE_DRAM_ID]], %[[TILES_PER_DIM0]] : i32
    // CHECK: ttkernel.noc_async_read_tile(%[[DRAM_ID_2]], %[[ACCESSOR]], %[[L1_ADDR_2]], %[[c1_i8]])

    // COM: Tile 3 (Row 1, Col 1) -> L1 Offset 3 (6144 bytes)
    // CHECK: %[[L1_ADDR_3:.*]] = arith.addi %[[CB_ADDR]], %[[c6144_i32]] : i32
    // COM: Note: DRAM calc here is (Base + 1) + Stride
    // CHECK: %[[DRAM_ID_3_PART:.*]] = arith.addi %[[BASE_DRAM_ID]], %[[c1_i32]] : i32
    // CHECK: %[[DRAM_ID_3:.*]] = arith.addi %[[DRAM_ID_3_PART]], %[[TILES_PER_DIM0]] : i32
    // CHECK: ttkernel.noc_async_read_tile(%[[DRAM_ID_3]], %[[ACCESSOR]], %[[L1_ADDR_3]], %[[c1_i8]])

    // CHECK: ttkernel.noc_async_read_barrier(%[[c1_i8]]) : (i8) -> ()
    ttg.local_store %b_7, %b : tensor<64x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 1, parent = #tiled1}>> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: return
    tt.return
  }
}

// -----

// COM: descriptor store

#shared = #ttg.padded_shared<[1:+1] {order = [1, 0], shape = [64, 64]}>
#shared1 = #ttg.padded_shared<[1:+1] {order = [1, 0], shape = [32, 64]}>
#smem = #ttg.shared_memory
#tiled = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [1, 2], order = [1, 0], tileShape = [32, 32]}>
#tiled1 = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [2, 2], order = [1, 0], tileShape = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 1 : i32} {
  // CHECK: func public @store_tiled_dot_op__writer
  tt.func public @store_tiled_dot_op__writer(%arg10: !tt.tensordesc<tensor<32x64xf16>>, %offs_am: i32, %offs_bn: i32) {
    // CHECK-DAG: %[[c0_i32:.*]] = arith.constant 0 : i32
    // CHECK-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
    // CHECK-DAG: %[[c2_i32:.*]] = arith.constant 2 : i32
    // CHECK-DAG: %[[c32_i32:.*]] = arith.constant 32 : i32
    // CHECK-DAG: %[[c2048_i32:.*]] = arith.constant 2048 : i32
    // CHECK-DAG: %[[c0_i8:.*]] = arith.constant 0 : i8
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[c6:.*]] = arith.constant 6 : index
    // CHECK-DAG: %[[c7:.*]] = arith.constant 7 : index
    // COM: Single tensor input (output) -> one TensorAccessorArgs at cta_base = 1 (one CB port).
    // CHECK: %[[ARGS:.*]] = ttkernel.TensorAccessorArgs(%[[c1_i32]], %[[c0_i32]]) {tt.accessor_base_arg_index = 0 : i32}
    // CHECK: %[[PTR:.*]] = ttkernel.get_common_arg_val(%[[c0]]) : (index) -> i32
    // CHECK: %[[DESC_SHAPE0:.*]] = ttkernel.get_common_arg_val(%[[c2]]) : (index) -> i32
    // CHECK: %[[SHAPE0:.*]] = ttkernel.get_common_arg_val(%[[c6]]) : (index) -> i32
    // CHECK: %[[SHAPE1:.*]] = ttkernel.get_common_arg_val(%[[c7]]) : (index) -> i32
    %0 = ttg.local_alloc {alloc_idx = 2 : i32} : () -> !ttg.memdesc<32x64xf16, #shared1, #smem, mutable>
    // CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, f16>>
    // CHECK: %[[TILESIZE:.*]] = ttkernel.get_tile_size(%[[CB]]) : (!ttkernel.cb<2, !ttcore.tile<32x32, f16>>) -> i32
    // CHECK: %[[ACCESSOR:.*]] = ttkernel.TensorAccessor(%[[ARGS]], %[[PTR]], %[[TILESIZE]])
    // CHECK: ttkernel.cb_wait_front(%[[CB]], %[[c2_i32]]) : (!ttkernel.cb<2, !ttcore.tile<32x32, f16>>, i32) -> ()


    // CHECK: %[[CB_READ_PTR:.*]] = ttkernel.get_read_ptr(%[[CB]]) : (!ttkernel.cb<2, !ttcore.tile<32x32, f16>>) -> i32

    // CHECK-DAG: %[[TILES_PER_DIM0:.*]] = arith.ceildivsi %[[DESC_SHAPE0]], %[[c32_i32]] : i32
    // CHECK-DAG: %[[ROW_TILE_ID:.*]] = arith.divsi %[[SHAPE0]], %[[c32_i32]] : i32
    // CHECK-DAG: %[[COL_TILE_ID:.*]] = arith.divsi %[[SHAPE1]], %[[c32_i32]] : i32
    %4 = ttg.local_load %0 : !ttg.memdesc<32x64xf16, #shared1, #smem, mutable> -> tensor<32x64xf16, #tiled>
    // COM: Calculate Base DRAM Address (Row * Stride + Col)
    // CHECK: %[[ROW_OFFSET:.*]] = arith.muli %[[ROW_TILE_ID]], %[[TILES_PER_DIM0]] : i32
    // CHECK: %[[BASE_DRAM_ID:.*]] = arith.addi %[[ROW_OFFSET]], %[[COL_TILE_ID]] : i32

    // COM: Tile 0 (Offset 0)
    // CHECK: ttkernel.noc_async_write_tile(%[[BASE_DRAM_ID]], %[[ACCESSOR]], %[[CB_READ_PTR]], %[[c0_i8]])

    // COM: Tile 1 (Offset 1)
    // CHECK: %[[L1_ADDR_1:.*]] = arith.addi %[[CB_READ_PTR]], %[[c2048_i32]] : i32
    // CHECK: %[[DRAM_ID_1:.*]] = arith.addi %[[BASE_DRAM_ID]], %[[c1_i32]] : i32
    // CHECK: ttkernel.noc_async_write_tile(%[[DRAM_ID_1]], %[[ACCESSOR]], %[[L1_ADDR_1]], %[[c0_i8]])

    // CHECK: ttkernel.noc_async_write_barrier(%[[c0_i8]]) : (i8) -> ()
    tt.descriptor_store %arg10[%offs_am, %offs_bn], %4 : !tt.tensordesc<tensor<32x64xf16>>, tensor<32x64xf16, #tiled>
    // CHECK: ttkernel.cb_pop_front(%[[CB]], %[[c2_i32]]) : (!ttkernel.cb<2, !ttcore.tile<32x32, f16>>, i32) -> ()
    // CHECK: return
    tt.return
  }
}

// -----

// COM: multicast
#shared = #ttg.padded_shared<[1:+1] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
#tiled = #triton_tenstorrent.tiled_encoding<{tilesPerCore = [2, 2], order = [1, 0], tileShape = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 1 : i32} {
 // CHECK: func.func public @matmul_kernel_tma__writer
 tt.func public @matmul_kernel_tma__writer(%arg0: !tt.tensordesc<tensor<64x64xf16>>, %offs_am: i32, %k_tiles_2: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1_i32 = arith.constant 1 : i32
  // CHECK-DAG: %[[c0_i8:.*]] = arith.constant 0 : i8
  // CHECK-DAG: %[[c0_index:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[c3_index:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[c4_index:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[MCAST_SZ:.*]] = arith.constant 8192 : i32

  %a = ttg.local_alloc {alloc_idx = 0 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  // CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0)

  scf.for %accumulator = %c0_i32 to %k_tiles_2 step %c1_i32  : i32 {
    // CHECK-DAG: %[[LX_Index:.*]] = ttkernel.my_logical_x_
    // CHECK-DAG: %[[LY_Index:.*]] = ttkernel.my_logical_y_
    // CHECK-DAG: %[[MCAST_START_COORD:.*]] = ttkernel.get_arg_val(%[[c3_index]])
    // CHECK-DAG: %[[MCAST_END_COORD:.*]] = ttkernel.get_arg_val(%[[c4_index]])

    // CHECK-DAG: %[[LX:.*]] = arith.index_cast %[[LX_Index]] : index to i32
    // CHECK-DAG: %[[LY:.*]] = arith.index_cast %[[LY_Index]] : index to i32

    // CHECK-DAG: %[[START_X_SHIFT:.*]] = arith.shrui %[[MCAST_START_COORD]]
    // CHECK-DAG: %[[START_X:.*]] = arith.andi %[[START_X_SHIFT]]
    // CHECK-DAG: %[[START_Y:.*]] = arith.andi %[[MCAST_START_COORD]]

    // CHECK-DAG: %[[END_X_SHIFT:.*]] = arith.shrui %[[MCAST_END_COORD]]
    // CHECK-DAG: %[[END_X:.*]] = arith.andi %[[END_X_SHIFT]]
    // CHECK-DAG: %[[END_Y:.*]] = arith.andi %[[MCAST_END_COORD]]

    // CHECK-DAG: %[[IS_SENDER_X:.*]] = arith.cmpi eq, %[[START_X]], %[[LX]]
    // CHECK-DAG: %[[IS_SENDER_Y:.*]] = arith.cmpi eq, %[[START_Y]], %[[LY]]
    // CHECK: %[[IS_SENDER:.*]] = arith.andi %[[IS_SENDER_X]], %[[IS_SENDER_Y]]

    // CHECK-DAG: %[[SEM_SEND_ID:.*]] = ttkernel.get_compile_time_arg_val(1)
    // CHECK-DAG: %[[SEM_RECV_ID:.*]] = ttkernel.get_compile_time_arg_val(2)
    // CHECK-DAG: %[[SEM_SEND:.*]] = ttkernel.get_semaphore(%[[SEM_SEND_ID]])
    // CHECK-DAG: %[[SEM_RECV:.*]] = ttkernel.get_semaphore(%[[SEM_RECV_ID]])

    %offs_k = arith.muli %accumulator, %c64_i32 : i32
    %a_6 = triton_tenstorrent.multicast %a {
      // CHECK: scf.if %[[IS_SENDER]] {
      // CHECK: ttkernel.cb_reserve_back(%[[CB]], %{{.*}}) {{.*}}
      // COM: build the output TensorAccessor and read the 4 tiles from DRAM into sender SRAM
      // CHECK: %[[ACCESSOR:.*]] = ttkernel.TensorAccessor(
      // CHECK-COUNT-4: ttkernel.noc_async_read_tile(%{{.*}}, %[[ACCESSOR]], %{{.*}})
      // CHECK: ttkernel.noc_async_read_barrier

      // COM: wait for receivers to be ready
      // CHECK: %[[SEND_L1:.*]] = ttkernel.reinterpret_cast(%[[SEM_SEND]]) : (!ttkernel.local_semaphore) -> !ttkernel.l1_addr_ptr
      // CHECK: ttkernel.experimental.semaphore_wait(%[[SEND_L1]], %{{.*}})

      // COM: multicast the tiles to all receivers (start_xy/end_xy are inline operands)
      // CHECK: ttkernel.noc_async_write_multicast(%{{.*}}, %[[MCAST_SZ]], %{{.*}})
      // CHECK: ttkernel.noc_async_write_barrier

      // COM: signal multicast data is available
      // CHECK: %[[RECV_L1:.*]] = ttkernel.reinterpret_cast(%[[SEM_RECV]]) : (!ttkernel.local_semaphore) -> !ttkernel.l1_addr_ptr
      // CHECK: ttkernel.noc_semaphore_set(%[[RECV_L1]], %{{.*}})
      // CHECK: %[[MCAST_ADDR:.*]] = ttkernel.get_noc_multicast_addr
      // CHECK: ttkernel.noc_semaphore_set_multicast(%[[SEM_RECV]], %[[MCAST_ADDR]], %{{.*}})

      // CHECK: } else {
      // CHECK-DAG: %[[NY:.*]] = ttkernel.experimental.convert_logical_y_to_translated(%[[START_Y]])
      // CHECK-DAG: %[[NX:.*]] = ttkernel.experimental.convert_logical_x_to_translated(%[[START_X]])

      // COM: signal readiness to receive
      // CHECK: %[[SEND_NOC:.*]] = ttkernel.get_noc_addr(%[[NX]], %[[NY]], %[[SEM_SEND]], %[[c0_i8]])
      // CHECK: ttkernel.noc_semaphore_inc(%[[SEND_NOC]], %{{.*}})

      // COM: wait for the sender to indicate data is available
      // CHECK: %[[RECV_L1B:.*]] = ttkernel.reinterpret_cast(%[[SEM_RECV]]) : (!ttkernel.local_semaphore) -> !ttkernel.l1_addr_ptr
      // CHECK: ttkernel.experimental.semaphore_wait(%[[RECV_L1B]], %{{.*}})
      // CHECK: ttkernel.noc_semaphore_set(%[[RECV_L1B]], %{{.*}})
      // CHECK: }

      // COM: both sender and receiver push back
      // CHECK: ttkernel.cb_push_back(%[[CB]], %{{.*}})

      %a_7 = tt.descriptor_load %arg0[%offs_am, %offs_k] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #tiled}>>
      triton_tenstorrent.yield %a_7, %a : tensor<64x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #tiled}>>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    } : (!ttg.memdesc<64x64xf16, #shared, #smem, mutable>) -> tensor<64x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #tiled}>>
    ttg.local_store %a_6, %a : tensor<64x64xf16, #triton_tenstorrent.tiled_dot_op<{opIdx = 0, parent = #tiled}>> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  }
  tt.return
 }
}
