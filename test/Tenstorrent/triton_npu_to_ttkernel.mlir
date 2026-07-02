// RUN: triton-opt %s -split-input-file --convert-triton-npu-to-ttkernel -canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.padded_shared<[1:+1] {order = [0], shape = [1024]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK: func.func public @add_kernel__compute() attributes {tt.function_type = "kernel", tt.num_common_args = 4 : i32, tt.num_per_core_args = 0 : i32, ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 0>, <arg_type = buffer_address, operand_index = 1>, <arg_type = buffer_address, operand_index = 2>] ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<compute>} {
  tt.func public @add_kernel__compute(%x_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %output_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %n_elements: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {

    // CHECK-DAG: %[[c0_i32:.*]] = arith.constant 0 : i32
    // CHECK-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
    // CHECK-DAG: %[[c2_i32:.*]] = arith.constant 2 : i32

    // CHECK-DAG: %[[c0_index:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1_index:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c2_index:.*]] = arith.constant 2 : index

    // CHECK-DAG: %[[X:.*]] = ttkernel.get_compile_time_arg_val(0)
    // CHECK-DAG: %[[Y:.*]] = ttkernel.get_compile_time_arg_val(1)
    // CHECK-DAG: %[[OUTPUT:.*]] = ttkernel.get_compile_time_arg_val(2)
    // CHECK: ttkernel.init_sfpu(%[[X]], %[[OUTPUT]])
    // CHECK: ttkernel.tile_regs_acquire
    %0 = ttg.local_alloc {alloc_idx = 2 : i32} : () -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %y = ttg.local_alloc {alloc_idx = 1 : i32} : () -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %x = ttg.local_alloc {alloc_idx = 0 : i32} : () -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %x_0 = ttg.local_load %x {triton_tenstorrent.alloc_offset = 0 : i32, triton_tenstorrent.alloc_size = 1 : i32} : !ttg.memdesc<1024xf32, #shared, #smem, mutable> -> tensor<1024xf32, #blocked>
    %y_1 = ttg.local_load %y {triton_tenstorrent.alloc_offset = 1 : i32, triton_tenstorrent.alloc_size = 1 : i32} : !ttg.memdesc<1024xf32, #shared, #smem, mutable> -> tensor<1024xf32, #blocked>
    // CHECK: ttkernel.cb_wait_front(%[[X]], %[[c1_i32]])

    // CHECK: ttkernel.copy_tile_init(%[[X]])
    // CHECK: ttkernel.copy_tile(%[[X]], %[[c0_index]], %[[c0_index]])
    // CHECK: ttkernel.cb_pop_front(%[[X]], %[[c1_i32]])

    // CHECK-DAG: ttkernel.cb_wait_front(%[[Y]], %[[c1_i32]])
    // CHECK: ttkernel.copy_tile_init(%[[Y]])
    // CHECK: ttkernel.copy_tile(%[[Y]], %[[c0_index]], %[[c1_index]])
    // CHECK: ttkernel.cb_pop_front(%[[Y]], %[[c1_i32]])

    // CHECK: ttkernel.add_binary_tile_init()
    // CHECK: ttkernel.add_binary_tile(%[[c0_index]], %[[c1_index]], %[[c2_index]])
    %output = triton_tenstorrent.binary_compute["arith.addf"] %x_0, %y_1 {triton_tenstorrent.alloc_offset = 2 : i32, triton_tenstorrent.alloc_size = 1 : i32} : (tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>

    // CHECK-DAG: ttkernel.cb_reserve_back(%[[OUTPUT]], %[[c1_i32]])
    // CHECK: ttkernel.tile_regs_commit()
    // CHECK: ttkernel.tile_regs_wait()
    // CHECK: ttkernel.pack_tile(%[[c2_i32]], %[[OUTPUT]], %[[c0_i32]], true)
    // CHECK: ttkernel.tile_regs_release()
    // CHECK: ttkernel.cb_push_back(%[[OUTPUT]], %[[c1_i32]])
    ttg.local_store %output, %0 : tensor<1024xf32, #blocked> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    // CHECK: return
    tt.return
  }
}


// -----

#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.padded_shared<[1:+1] {order = [0], shape = [1024]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK: func.func public @add_kernel__reader()
  tt.func public @add_kernel__reader(%x_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %output_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %n_elements: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32

    // CHECK-DAG: %[[c4096_i32:.*]] = arith.constant 4096 : i32
    // CHECK-DAG: %[[c1_i8:.*]] = arith.constant 1 : i8
    // CHECK-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
    // CHECK-DAG: %[[c0_index:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1_index:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c3_i32:.*]] = arith.constant 3 : i32
    // CHECK-DAG: %[[c0_i32:.*]] = arith.constant 0 : i32

    // One chained TensorAccessorArgs per tensor input (x, y, output), built at
    // func entry and tagged by its buffer's common-arg index. The first sits at
    // cta_base = 3 (number of CB ports); the rest chain via prev_args.
    // CHECK-DAG: %[[X_ARGS:.*]] = ttkernel.TensorAccessorArgs(%[[c3_i32]], %[[c0_i32]]) {tt.accessor_base_arg_index = 0 : i32}
    // CHECK-DAG: %[[X_PTR:.*]] = ttkernel.get_common_arg_val(%[[c0_index]])
    // CHECK-DAG: %[[Y_ARGS:.*]] = ttkernel.TensorAccessorArgs(prev = %[[X_ARGS]]) {tt.accessor_base_arg_index = 1 : i32}
    // CHECK-DAG: %[[Y_PTR:.*]] = ttkernel.get_common_arg_val(%[[c1_index]])
    // CHECK-DAG: ttkernel.TensorAccessorArgs(prev = %[[Y_ARGS]]) {tt.accessor_base_arg_index = 2 : i32}
    %y = ttg.local_alloc {alloc_idx = 1 : i32} : () -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %x = ttg.local_alloc {alloc_idx = 0 : i32} : () -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>

    // Per-buffer TensorAccessor = accessor args + base address + page size.
    // CHECK-DAG: %[[Y:.*]] = ttkernel.get_compile_time_arg_val(1)
    // CHECK-DAG: %[[Y_TILE_SIZE:.*]] = ttkernel.get_tile_size(%[[Y]])
    // CHECK-DAG: %[[Y_ACCESSOR:.*]] = ttkernel.TensorAccessor(%[[Y_ARGS]], %[[Y_PTR]], %[[Y_TILE_SIZE]])
    // CHECK-DAG: %[[X:.*]] = ttkernel.get_compile_time_arg_val(0)
    // CHECK-DAG: %[[X_TILE_SIZE:.*]] = ttkernel.get_tile_size(%[[X]])
    // CHECK-DAG: %[[X_ACCESSOR:.*]] = ttkernel.TensorAccessor(%[[X_ARGS]], %[[X_PTR]], %[[X_TILE_SIZE]])
    // CHECK: %[[PID:.*]] = ttkernel.get_arg_val(%[[c0_index]])
    %pid = tt.get_program_id x : i32

    // CHECK-DAG: %[[TILE_INDEX_X_BYTES:.*]] = arith.muli %[[PID]], %[[c4096_i32]] : i32
    // CHECK-DAG: %[[TILE_INDEX_Y_BYTES:.*]] = arith.muli %[[PID]], %[[c4096_i32]] : i32
    // CHECK: %[[TILE_INDEX_X:.*]] = arith.divui %[[TILE_INDEX_X_BYTES]], %[[X_TILE_SIZE]]
    %block_start = arith.muli %pid, %c1024_i32 : i32
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %offsets_0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %offsets_1 = tt.splat %block_start : i32 -> tensor<1024xi32, #blocked>
    %offsets_2 = tt.splat %block_start : i32 -> tensor<1024xi32, #blocked>
    %offsets_3 = arith.addi %offsets_1, %offsets : tensor<1024xi32, #blocked>
    %offsets_4 = arith.addi %offsets_2, %offsets_0 : tensor<1024xi32, #blocked>
    %x_8 = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_9 = tt.addptr %x_8, %offsets_3 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: ttkernel.cb_reserve_back(%[[X]], %[[c1_i32]])
    // CHECK: %[[X_WRITE_PTR:.*]] = ttkernel.get_write_ptr(%[[X]])
    // CHECK: ttkernel.noc_async_read_tile(%[[TILE_INDEX_X]], %[[X_ACCESSOR]], %[[X_WRITE_PTR]], %[[c1_i8]])
    // CHECK: ttkernel.noc_async_read_barrier(%[[c1_i8]])
    // CHECK: ttkernel.cb_push_back(%[[X]], %[[c1_i32]])
    %x_10 = tt.load %x_9 : tensor<1024x!tt.ptr<f32>, #blocked>
    ttg.local_store %x_10, %x : tensor<1024xf32, #blocked> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %y_11 = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %y_12 = tt.addptr %y_11, %offsets_4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: %[[TILE_INDEX_Y:.*]] = arith.divui %[[TILE_INDEX_Y_BYTES]], %[[Y_TILE_SIZE]]
    // CHECK: ttkernel.cb_reserve_back(%[[Y]], %[[c1_i32]])
    // CHECK: %[[Y_WRITE_PTR:.*]] = ttkernel.get_write_ptr(%[[Y]])
    // CHECK: ttkernel.noc_async_read_tile(%[[TILE_INDEX_Y]], %[[Y_ACCESSOR]], %[[Y_WRITE_PTR]], %[[c1_i8]])
    // CHECK: ttkernel.noc_async_read_barrier(%[[c1_i8]])
    // CHECK: ttkernel.cb_push_back(%[[Y]], %[[c1_i32]])
    %y_13 = tt.load %y_12 : tensor<1024x!tt.ptr<f32>, #blocked>
    ttg.local_store %y_13, %y : tensor<1024xf32, #blocked> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    // CHECK: return
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.padded_shared<[1:+1] {order = [0], shape = [1024]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK: func.func public @add_kernel__writer()
  tt.func public @add_kernel__writer(%x_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %output_ptr: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %n_elements: i32 {tt.divisibility = 8 : i32}) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32

    // CHECK-DAG: %[[c4096_i32:.*]] = arith.constant 4096 : i32
    // CHECK-DAG: %[[c0_i8:.*]] = arith.constant 0 : i8
    // CHECK-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
    // CHECK-DAG: %[[c2_index:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[c3_i32:.*]] = arith.constant 3 : i32
    // CHECK-DAG: %[[c0_i32:.*]] = arith.constant 0 : i32

    // All three tensor inputs get a chained TensorAccessorArgs (anchors), but the
    // writer only builds the accessor for its output buffer (tagged 2).
    // CHECK-DAG: %[[X_ARGS:.*]] = ttkernel.TensorAccessorArgs(%[[c3_i32]], %[[c0_i32]]) {tt.accessor_base_arg_index = 0 : i32}
    // CHECK-DAG: %[[Y_ARGS:.*]] = ttkernel.TensorAccessorArgs(prev = %[[X_ARGS]]) {tt.accessor_base_arg_index = 1 : i32}
    // CHECK-DAG: %[[OUTPUT_ARGS:.*]] = ttkernel.TensorAccessorArgs(prev = %[[Y_ARGS]]) {tt.accessor_base_arg_index = 2 : i32}
    // CHECK-DAG: %[[OUTPUT_PTR:.*]] = ttkernel.get_common_arg_val(%[[c2_index]])
    // CHECK-DAG: %[[OUTPUT:.*]] = ttkernel.get_compile_time_arg_val(2)
    // CHECK-DAG: %[[OUTPUT_TILE_SIZE:.*]] = ttkernel.get_tile_size(%[[OUTPUT]])
    // CHECK-DAG: %[[OUTPUT_ACCESSOR:.*]] = ttkernel.TensorAccessor(%[[OUTPUT_ARGS]], %[[OUTPUT_PTR]], %[[OUTPUT_TILE_SIZE]])
    // CHECK-NOT: ttkernel.copy_tile_init
    // CHECK-NOT: ttkernel.copy_tile
    %0 = ttg.local_alloc {alloc_idx = 2 : i32} : () -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %pid = tt.get_program_id x : i32
    %block_start = arith.muli %pid, %c1024_i32 : i32
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %offsets_0 = tt.splat %block_start : i32 -> tensor<1024xi32, #blocked>
    %offsets_1 = arith.addi %offsets_0, %offsets : tensor<1024xi32, #blocked>
    %1 = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %2 = tt.addptr %1, %offsets_1 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: %[[PID:.*]] = ttkernel.get_arg_val(%[[c0_index:.*]])
    // CHECK: %[[OUTPUT_TILE_INDEX_BYTES:.*]] = arith.muli %[[PID]], %[[c4096_i32]] : i32

    // CHECK: ttkernel.cb_wait_front(%[[OUTPUT]], %[[c1_i32]])

    // CHECK: %[[OUTPUT_TILE_INDEX:.*]] = arith.divui %[[OUTPUT_TILE_INDEX_BYTES]], %[[OUTPUT_TILE_SIZE]]
    // CHECK: %[[OUTPUT_READ_PTR:.*]] = ttkernel.get_read_ptr(%[[OUTPUT]])
    // CHECK: ttkernel.noc_async_write_tile(%[[OUTPUT_TILE_INDEX]], %[[OUTPUT_ACCESSOR]], %[[OUTPUT_READ_PTR]], %[[c0_i8]])
    // CHECK: ttkernel.noc_async_write_barrier(%[[c0_i8]])
    %3 = ttg.local_load %0 : !ttg.memdesc<1024xf32, #shared, #smem, mutable> -> tensor<1024xf32, #blocked>
    tt.store %2, %3 : tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: ttkernel.cb_pop_front(%[[OUTPUT]], %[[c1_i32]])
    // CHECK: return
    tt.return
  }
}

// -----

// COM: Multi-block kernel lowering
#blocked = #ttg.blocked<{sizePerThread = [1024], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.padded_shared<[1:+1] {order = [0], shape = [1024]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  tt.func public @add_kernel__reader(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %c1_i32 = arith.constant 1 : i32
    %y = ttg.local_alloc {alloc_idx = 1 : i32} : () -> !ttg.memdesc<1024xf16, #shared, #smem, mutable>
    %x = ttg.local_alloc {alloc_idx = 0 : i32} : () -> !ttg.memdesc<1024xf16, #shared, #smem, mutable>
    %0 = ttc.block_end
    %1 = ttc.block_start
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %offsets_0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %mask = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %mask_1 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %x_2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %y_3 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    // CHECK-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
    // CHECK-DAG: %[[c0_index:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1_index:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[START:.*]] = ttkernel.get_arg_val(%[[c0_index]])
    // CHECK-DAG: %[[END:.*]] = ttkernel.get_arg_val(%[[c1_index]])
    // CHECK: scf.for %[[arg0:.*]] = %[[START]] to %[[END]] step %[[c1_i32]]
    scf.for %arg4 = %1 to %0 step %c1_i32  : i32 {
      // CHECK: %[[block_start:.*]] = arith.muli %[[arg0]]
      // CHECK-COUNT-2: ttkernel.noc_async_read_tile(
      // CHECK-NOT: ttkernel.noc_async_read_tile(
      %2 = ttc.current_block %arg4 : i32
      %block_start = arith.muli %2, %c1024_i32 : i32
      %offsets_4 = tt.splat %block_start : i32 -> tensor<1024xi32, #blocked>
      %offsets_5 = tt.splat %block_start : i32 -> tensor<1024xi32, #blocked>
      %offsets_6 = arith.addi %offsets_4, %offsets : tensor<1024xi32, #blocked>
      %offsets_7 = arith.addi %offsets_5, %offsets_0 : tensor<1024xi32, #blocked>
      %mask_8 = arith.cmpi slt, %offsets_6, %mask : tensor<1024xi32, #blocked>
      %mask_9 = arith.cmpi slt, %offsets_7, %mask_1 : tensor<1024xi32, #blocked>
      %x_10 = tt.addptr %x_2, %offsets_6 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
      %x_11 = tt.load %x_10, %mask_8 : tensor<1024x!tt.ptr<f16>, #blocked>
      ttg.local_store %x_11, %x : tensor<1024xf16, #blocked> -> !ttg.memdesc<1024xf16, #shared, #smem, mutable>
      %y_12 = tt.addptr %y_3, %offsets_7 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
      %y_13 = tt.load %y_12, %mask_9 : tensor<1024x!tt.ptr<f16>, #blocked>
      ttg.local_store %y_13, %y : tensor<1024xf16, #blocked> -> !ttg.memdesc<1024xf16, #shared, #smem, mutable>
    }
    // CHECK: }
    // CHECK: return
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  tt.func public @addptr_scalar__writer() attributes {noinline = false} {
    // CHECK: ttkernel.dprint("ptr address: ") : () -> ()
    tt.print "ptr address: " {hex = false, isSigned = array<i32: 0>}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  tt.func public @addptr_scalar__writer(%ptr: !tt.ptr<f32>, %offset: i32, %val: f32) attributes {noinline = false} {
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c4_i32:.*]] = arith.constant 4 : i32
    // CHECK-DAG: %[[PTR:.*]] = ttkernel.get_common_arg_val(%[[c0]])
    // CHECK-DAG: %[[OFFSET:.*]] = ttkernel.get_common_arg_val(%[[c1]])
    // CHECK: %[[OFFSET_BYTES:.*]] = arith.muli %[[OFFSET]], %[[c4_i32:.*]] : i32
    // CHECK: %[[PTR_OFFSET:.*]] = arith.addi %[[PTR]], %[[OFFSET_BYTES]] : i32
    %0 = tt.addptr %ptr, %offset : !tt.ptr<f32>, i32
    tt.print "ptr address: " {hex = false, isSigned = array<i32: 1>} : %0 : !tt.ptr<f32>
    tt.return
  }
}
