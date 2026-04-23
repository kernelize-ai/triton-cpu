// RUN: triton-opt %s -split-input-file --tritoncpu-tile-and-fuse -canonicalize 2>/dev/null | FileCheck %s


// Basic fusion: make_range, splat, addptr, and load should all be fused into
// the store generic's body. No operands from these ops should remain as ins.
//
// CHECK-LABEL: tt.func public @test_basic_fusion
// CHECK:       ttc.generic
// CHECK-NOT:     tensor<4x!tt.ptr<f32>
// CHECK:         tt.splat
// CHECK:         ttc.make_dynamic_range
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

// -----

// Iter-arg fusion: %init (tt.make_range) is the init value of an scf.for
// iter_arg. The generic inside the loop takes the iter_arg as ins. The fuser
// must see through the iter_arg to its init op and replace the block argument,
// rather than leaving the iter_arg as an unfused ins. Without the fix, the
// make_dynamic_range would be emitted but never replace the block arg, leaving
// a dead clone.
//
// CHECK-LABEL: tt.func public @test_iter_arg_fusion
// CHECK:       scf.for
// CHECK:         ttc.generic
// CHECK-NOT:       tensor<4xi32
// CHECK:           ttc.make_dynamic_range

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  tt.func public @test_iter_arg_fusion(%ptr: !tt.ptr<i32>, %n: i32) attributes {noinline = false, ttc.persistent_kernel} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c10 = arith.constant 10 : i32
    %init = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32, #blocked>
    scf.for %bid = %c0 to %c10 step %c1 iter_args(%carried = %init) -> (tensor<4xi32, #blocked>) : i32 {
      %base = tt.splat %ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #blocked>
      %ptrs = tt.addptr %base, %carried : tensor<4x!tt.ptr<i32>, #blocked>, tensor<4xi32, #blocked>
      tt.store %ptrs, %carried : tensor<4x!tt.ptr<i32>, #blocked>
      scf.yield %carried : tensor<4xi32, #blocked>
    }
    tt.return
  }
}
