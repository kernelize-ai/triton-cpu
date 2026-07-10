// RUN: triton-opt %s -split-input-file --tritoncpu-tile-and-fuse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK-LABEL: tt.func public @test_fuse_trivial_cvt
  tt.func public @test_fuse_trivial_cvt(%ptr: !tt.ptr<f16>) attributes {noinline = false, ttc.persistent_kernel} {
    %c1 = arith.constant 1 : i32
    %start = ttc.block_start
    %end = ttc.block_end
    scf.for %bid = %start to %end step %c1 : i32 {
      %block = ttc.current_block %bid : i32
      %cst = arith.constant dense<0.0> : tensor<4x8xf16, #blocked1>
      %ptrs = tt.splat %ptr : !tt.ptr<f16> -> tensor<4x8x!tt.ptr<f16>, #blocked>
      // CHECK: ttc.generic
      // CHECK ttg.convert_layout
      // CHECK: tt.store
      // CHECK-NOT: ttc.generic
      %cvt = ttg.convert_layout %cst : tensor<4x8xf16, #blocked1> -> tensor<4x8xf16, #blocked>
      tt.store %ptrs, %cvt : tensor<4x8x!tt.ptr<f16>, #blocked>
    }
    tt.return
  }
}

// -----

// Register-reordering CVT where requiredTileShape == dst sizePerThread
// (sizePerThread=[1,4] → [1,8], lcm=[1,8]) is fusible
//


#src = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#dst = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK-LABEL: tt.func public @test_fuse_register_cvt
  tt.func public @test_fuse_register_cvt(%ptr: !tt.ptr<f16>) attributes {noinline = false, ttc.persistent_kernel} {
    %c1 = arith.constant 1 : i32
    %start = ttc.block_start
    %end = ttc.block_end
    scf.for %bid = %start to %end step %c1 : i32 {
    // CHECK:       ttc.generic
    // CHECK:         ttg.convert_layout
    // CHECK:         tt.store
    // CHECK-NOT:   ttc.generic
      %block = ttc.current_block %bid : i32
      %cst = arith.constant dense<0.0> : tensor<16x8xf16, #src>
      %ptrs = tt.splat %ptr : !tt.ptr<f16> -> tensor<16x8x!tt.ptr<f16>, #dst>
      %cvt = ttg.convert_layout %cst : tensor<16x8xf16, #src> -> tensor<16x8xf16, #dst>
      tt.store %ptrs, %cvt : tensor<16x8x!tt.ptr<f16>, #dst>
    }
    tt.return
  }
}

// -----

// Non-fusible register-reordering CVT (sizePerThread=[4,4] → [1,8],
// requiredTileShape=[4,8] ≠ defaultTileShape=[1,8]) gets wrapped in its own
// generic; the store gets a separate generic.

#blocked2 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  // CHECK-LABEL: tt.func public @test_wrap_nonfusible_cvt
  tt.func public @test_wrap_nonfusible_cvt(%ptr: !tt.ptr<f16>) attributes {noinline = false, ttc.persistent_kernel} {
    %c1 = arith.constant 1 : i32
    %start = ttc.block_start
    %end = ttc.block_end
    scf.for %bid = %start to %end step %c1 : i32 {

      %block = ttc.current_block %bid : i32
      %cst = arith.constant dense<0.0> : tensor<16x32xf16, #blocked2>
      %ptrs = tt.splat %ptr : !tt.ptr<f16> -> tensor<16x32x!tt.ptr<f16>, #blocked3>
      // CHECK:       ttc.generic
      // CHECK:         ttg.convert_layout
      // CHECK:       ttc.generic
      // CHECK:         tt.store
      %cvt = ttg.convert_layout %cst : tensor<16x32xf16, #blocked2> -> tensor<16x32xf16, #blocked3>
      tt.store %ptrs, %cvt : tensor<16x32x!tt.ptr<f16>, #blocked3>
    }
    tt.return
  }
}
