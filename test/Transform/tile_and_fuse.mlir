// RUN: triton-opt %s -split-input-file --tritoncpu-tile-and-fuse -canonicalize 2>/dev/null | FileCheck %s


// Basic fusion: make_range, splat, addptr, and load should all be fused into
// the store generic's body. No operands from these ops should remain as ins.
//
// CHECK-LABEL: tt.func public @test_basic_fusion
// CHECK:       ttc.generic
// CHECK-NOT:     tensor<4x!tt.ptr<f32>
// CHECK:         ttc.make_dynamic_range
// CHECK:         tt.splat
// CHECK:         tt.addptr
// CHECK:         tt.load
// CHECK:         tt.store

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  tt.func public @test_basic_fusion(%ptr: !tt.ptr<f32>, %val: !tt.ptr<f32>) attributes {noinline = false, ttc.persistent_kernel} {
    %c1 = arith.constant 1 : i32
    %start = ttc.block_start
    %end = ttc.block_end
    scf.for %bid = %start to %end step %c1 : i32 {
      %block = ttc.current_block %bid : i32
      %offsets = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32, #blocked>
      %base = tt.splat %val : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>, #blocked>
      %ptrs = tt.addptr %base, %offsets : tensor<4x!tt.ptr<f32>, #blocked>, tensor<4xi32, #blocked>
      %data = tt.load %ptrs : tensor<4x!tt.ptr<f32>, #blocked>
      %out_base = tt.splat %ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>, #blocked>
      %out_ptrs = tt.addptr %out_base, %offsets : tensor<4x!tt.ptr<f32>, #blocked>, tensor<4xi32, #blocked>
      tt.store %out_ptrs, %data : tensor<4x!tt.ptr<f32>, #blocked>
    }
    tt.return
  }
}

// -----

// Memory-effect ops (tt.load) defined in an outer block must not be fused
// into a ttc.generic that lives in an inner block, even though tt.load is
// otherwise considered fusible. The load must remain as a generic ins arg.
//
// CHECK-LABEL: tt.func public @test_no_cross_block_load_fusion
// CHECK:       tt.load
// CHECK:       scf.for
// CHECK:         ttc.generic
// CHECK-NOT:       tt.load
// CHECK:           tt.store

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  tt.func public @test_no_cross_block_load_fusion(%val_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %offsets = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32, #blocked>
    %val_base = tt.splat %val_ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>, #blocked>
    %val_ptrs = tt.addptr %val_base, %offsets : tensor<4x!tt.ptr<f32>, #blocked>, tensor<4xi32, #blocked>
    %data = tt.load %val_ptrs : tensor<4x!tt.ptr<f32>, #blocked>
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      %out_base = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>, #blocked>
      %out_ptrs = tt.addptr %out_base, %offsets : tensor<4x!tt.ptr<f32>, #blocked>, tensor<4xi32, #blocked>
      tt.store %out_ptrs, %data : tensor<4x!tt.ptr<f32>, #blocked>
    }
    tt.return
  }
}

// -----

// Shared op: %mask (arith.cmpi) feeds two separate loads. In the per-chain
// scheme, each generic operand gets its own clone of %mask and its producers — no
// shared block argument remains.
//
// CHECK-LABEL: tt.func public @test_shared_op
// CHECK:       ttc.generic
// CHECK:         arith.cmpi
// CHECK:         arith.cmpi

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  tt.func public @test_shared_op(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) attributes {noinline = false, ttc.persistent_kernel} {
    %c1 = arith.constant 1 : i32
    %start = ttc.block_start
    %end = ttc.block_end
    scf.for %bid = %start to %end step %c1 : i32 {
      %block = ttc.current_block %bid : i32
      %offsets = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32, #blocked>
      %splat_n = tt.splat %n : i32 -> tensor<4xi32, #blocked>
      %mask = arith.cmpi slt, %offsets, %splat_n : tensor<4xi32, #blocked>
      %cst = arith.constant dense<0.0> : tensor<4xf32, #blocked>
      %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>, #blocked>
      %src_ptrs = tt.addptr %src_base, %offsets : tensor<4x!tt.ptr<f32>, #blocked>, tensor<4xi32, #blocked>
      %data = tt.load %src_ptrs, %mask, %cst : tensor<4x!tt.ptr<f32>, #blocked>
      %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>, #blocked>
      %dst_ptrs = tt.addptr %dst_base, %offsets : tensor<4x!tt.ptr<f32>, #blocked>, tensor<4xi32, #blocked>
      tt.store %dst_ptrs, %data, %mask : tensor<4x!tt.ptr<f32>, #blocked>
    }
    tt.return
  }
}
