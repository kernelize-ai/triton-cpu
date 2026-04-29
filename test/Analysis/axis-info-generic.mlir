// RUN: triton-opt %s -split-input-file --convert-triton-cpu-to-llvm | FileCheck %s

// COM: Tests that load alignments are properly forwarded through generic op regions

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
    tt.func public @load(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
        %c512_i32 = arith.constant 512 : i32
        %pid = "ttc.block_id"() <{axis = 0 : i32}> : () -> i32
        %block_start = arith.muli %pid, %c512_i32 : i32

        %offsets = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
        %offsets_0 = tt.splat %block_start : i32 -> tensor<512xi32, #blocked>
        %offsets_1 = arith.addi %offsets_0, %offsets : tensor<512xi32, #blocked>
        %x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>

        %x_3 = tt.addptr %x, %offsets_1 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>

        // CHECK-COUNT-128: ttc.masked_load {{.*}} -> vector<4xf32>
        ttc.generic %x_3 attributes {blockShape = array<i32: 512>, tileShape = array<i32: 4>} body {
            ^bb0(%offset:i32, %arg0: tensor<4x!tt.ptr<f32>, #blocked>):
                %x_10 = tt.load %arg0 : tensor<4x!tt.ptr<f32>, #blocked>
                ttc.yield
        } combiners {}: (tensor<512x!tt.ptr<f32>, #blocked>) -> ()
        tt.return
    }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
    tt.func public @load_scalar(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
        ttc.generic %x_ptr attributes {blockShape = array<i32: 1024>, tileShape = array<i32: 4>} body {
            ^bb0(%tileOffset: i32, %ptr: !tt.ptr<f32>):
                %offsets = ttc.make_dynamic_range %tileOffset : tensor<4xi32, #blocked>
                %ptrs = tt.splat %ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>, #blocked>
                %offset_ptrs = tt.addptr %ptrs, %offsets : tensor<4x!tt.ptr<f32>, #blocked>, tensor<4xi32, #blocked>
                // COM: no un-rolled prologue, check for branch to loop pre-header
                // CHECK: llvm.br ^bb1
                // COM: loop branch
                // CHECK: llvm.cond_br {{.*}}, ^bb2, ^bb4
                // COM: branch to inlined block
                // CHECK: llvm.br ^bb3
                // CHECK: ttc.masked_load {{.*}} -> vector<4xf32>
                %ret = tt.load %offset_ptrs : tensor<4x!tt.ptr<f32>, #blocked>
                ttc.yield
        } combiners {}: (!tt.ptr<f32>) -> ()

        tt.return
    }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  tt.func public @store_2d_vectorized(
      %c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %data: tensor<4x8xf16, #blocked> {tt.divisibility = 16 : i32},
      %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32},
      %stride_cm: i32 {tt.divisibility = 16 : i32},
      %offs_am: i32 {tt.divisibility = 16 : i32}, %offs_bn: i32 {tt.divisibility = 16 : i32}) {

    // 4 M-tiles × 1 N-tile = 4 body invocations, each must emit one vector<8xf16> store.
    // CHECK-COUNT-4: ttc.masked_store {{.*}} : (!llvm.ptr<1>, vector<8xf16>, vector<8xi1>) -> ()
    // CHECK-NOT: ttc.masked_store {{.*}} : (!llvm.ptr<1>, vector<1xf16>
    ttc.generic %data, %c_ptr, %stride_cm, %M, %N, %offs_am, %offs_bn
        attributes {blockShape = array<i32: 4, 8>, tileShape = array<i32: 1, 8>} body {
    ^bb0(%tile_n: i32, %tile_m: i32,
         %tile_data: tensor<1x8xf16, #blocked>,
         %ptr: !tt.ptr<f16>, %stride: i32, %m_bound: i32, %n_bound: i32,
         %row_base: i32, %col_base: i32):

      // Column offsets: col_base + [0..7]
      %col_range = tt.make_range {end = 8 : i32, start = 0 : i32}
          : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %col_base_t = tt.splat %col_base : i32 -> tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %cols = arith.addi %col_base_t, %col_range
          : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %cols_2d = tt.expand_dims %cols {axis = 0 : i32}
          : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>

      // Row offset: row_base + M-tile index
      %row_range = ttc.make_dynamic_range %tile_m
          : tensor<1xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %row_base_t = tt.splat %row_base : i32 -> tensor<1xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %rows = arith.addi %row_base_t, %row_range
          : tensor<1xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %rows_2d = tt.expand_dims %rows {axis = 1 : i32}
          : tensor<1xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1xi32, #blocked>

      // Mask: (row < M) & (col < N)
      %n_t = tt.splat %n_bound : i32 -> tensor<1x8xi32, #blocked>
      %col_mask = arith.cmpi slt, %cols_2d, %n_t : tensor<1x8xi32, #blocked>
      %m_t = tt.splat %m_bound : i32 -> tensor<1x1xi32, #blocked>
      %row_mask = arith.cmpi slt, %rows_2d, %m_t : tensor<1x1xi32, #blocked>
      %row_mask_bcast = tt.broadcast %row_mask
          : tensor<1x1xi1, #blocked> -> tensor<1x8xi1, #blocked>
      %mask = arith.andi %row_mask_bcast, %col_mask : tensor<1x8xi1, #blocked>

      // Pointer: ptr + row * stride + col
      %stride_t = tt.splat %stride : i32 -> tensor<1x1xi32, #blocked>
      %row_off = arith.muli %rows_2d, %stride_t : tensor<1x1xi32, #blocked>
      %ptr_base = tt.splat %ptr : !tt.ptr<f16> -> tensor<1x1x!tt.ptr<f16>, #blocked>
      %ptrs_r = tt.addptr %ptr_base, %row_off
          : tensor<1x1x!tt.ptr<f16>, #blocked>, tensor<1x1xi32, #blocked>
      %ptrs_bcast = tt.broadcast %ptrs_r
          : tensor<1x1x!tt.ptr<f16>, #blocked> -> tensor<1x8x!tt.ptr<f16>, #blocked>
      %ptrs = tt.addptr %ptrs_bcast, %cols_2d
          : tensor<1x8x!tt.ptr<f16>, #blocked>, tensor<1x8xi32, #blocked>

      tt.store %ptrs, %tile_data, %mask : tensor<1x8x!tt.ptr<f16>, #blocked>
      ttc.yield
    } combiners {} : (tensor<4x8xf16, #blocked>, !tt.ptr<f16>, i32, i32, i32, i32, i32) -> ()

    tt.return
  }
}
