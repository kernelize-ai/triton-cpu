// RUN: triton-opt %s -split-input-file --test-lane-map-analysis 2>&1 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cpu", "ttg.threads-per-warp" = 1 : i32} {
  tt.func public @matmul_kernel(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},  // A
      %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32},  // (unused, was B)
      %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32},  // C
      %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32
    ) attributes {noinline = false} {

    %cst = arith.constant dense<8> : tensor<4x8xi32, #blocked>          // A step per K-iter
    %c8_i32 = arith.constant 8 : i32
    %c4_i32 = arith.constant 4 : i32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c7_i32 = arith.constant 7 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<4x8xf16, #blocked>

    %0 = ttc.block_end
    %1 = ttc.block_start

    %2 = arith.addi %arg3, %c3_i32 : i32
    %3 = arith.divsi %2, %c4_i32 : i32
    %4 = arith.addi %arg4, %c7_i32 : i32
    %5 = arith.divsi %4, %c8_i32 : i32

    %6 = arith.cmpi sgt, %arg6, %c0_i32 : i32
    %7 = arith.cmpi sgt, %arg7, %c0_i32 : i32
    %8 = arith.cmpi sgt, %arg8, %c0_i32 : i32

    %9  = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %10 = tt.splat %arg3 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>

    %11 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %14 = tt.splat %arg6 : i32 -> tensor<4x1xi32, #blocked>

    %15 = tt.expand_dims %11 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %16 = tt.broadcast %15 : tensor<1x8xi32, #blocked> -> tensor<4x8xi32, #blocked>

    %17 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<4x8x!tt.ptr<f16>, #blocked>

    %24 = arith.addi %arg5, %c7_i32 : i32
    %25 = arith.divsi %24, %c8_i32 : i32

    // C base pieces (unchanged from your store path)
    %28 = tt.splat %arg8 : i32 -> tensor<4x1xi32, #blocked>
    %29 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<4x1x!tt.ptr<f16>, #blocked>
    %30 = tt.splat %arg3 : i32 -> tensor<4x1xi32, #blocked>
    %31 = tt.splat %arg4 : i32 -> tensor<1x8xi32, #blocked>

    scf.for %arg9 = %1 to %0 step %c1_i32 : i32 {
      %32 = ttc.current_block %arg9 : i32
      %33 = arith.divsi %32, %5 : i32
      %34 = arith.subi %3, %33 : i32
      %35 = arith.minsi %34, %c1_i32 : i32
      %36 = arith.remsi %32, %5 : i32
      %37 = arith.remsi %36, %35 : i32
      %38 = arith.addi %33, %37 : i32
      %39 = arith.divsi %36, %35 : i32

      %40 = arith.cmpi sge, %38, %c0_i32 : i32
      %41 = arith.cmpi sge, %39, %c0_i32 : i32

      // Compute A tile pointers
      %42 = arith.muli %38, %c4_i32 : i32
      %43 = tt.splat %42 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %44 = arith.addi %43, %9 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %45 = arith.remsi %44, %10 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>

      %46 = arith.muli %39, %c8_i32 : i32
      %47 = tt.splat %46 : i32 -> tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %49 = arith.addi %47, %11 : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>

      %52 = tt.expand_dims %45 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
      %53 = arith.muli %52, %14 : tensor<4x1xi32, #blocked>
      %54 = tt.broadcast %53 : tensor<4x1xi32, #blocked> -> tensor<4x8xi32, #blocked>
      %55 = arith.addi %54, %16 : tensor<4x8xi32, #blocked>
      %56 = tt.addptr %17, %55 : tensor<4x8x!tt.ptr<f16>, #blocked>, tensor<4x8xi32, #blocked>

      // Inner K loop preserved; just load A and advance ptr each iter.
      %61:2 = scf.for %arg10 = %c0_i32 to %25 step %c1_i32
          iter_args(%arg11 = %56, %arg12 = %cst_0)
          -> (tensor<4x8x!tt.ptr<f16>, #blocked>, tensor<4x8xf16, #blocked>) : i32 {
        %76 = arith.muli %arg10, %c8_i32 : i32
        %77 = arith.subi %arg5, %76 : i32
        %78 = tt.splat %77 : i32 -> tensor<1x8xi32, #blocked>
        %79 = arith.cmpi slt, %15, %78 : tensor<1x8xi32, #blocked>
        %80 = tt.broadcast %79 : tensor<1x8xi1, #blocked> -> tensor<4x8xi1, #blocked>

        %81 = tt.load %arg11, %80, %cst_0 : tensor<4x8x!tt.ptr<f16>, #blocked>
        %82 = tt.addptr %arg11, %cst : tensor<4x8x!tt.ptr<f16>, #blocked>, tensor<4x8xi32, #blocked>

        scf.yield %82, %81 : tensor<4x8x!tt.ptr<f16>, #blocked>, tensor<4x8xf16, #blocked>
      }

      // Store path
      %63 = tt.expand_dims %44 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
      %64 = arith.muli %28, %63 : tensor<4x1xi32, #blocked>
      %65 = tt.addptr %29, %64 : tensor<4x1x!tt.ptr<f16>, #blocked>, tensor<4x1xi32, #blocked>

      %66 = tt.expand_dims %49 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
      %67 = tt.broadcast %65 : tensor<4x1x!tt.ptr<f16>, #blocked> -> tensor<4x8x!tt.ptr<f16>, #blocked>
      %68 = tt.broadcast %66 : tensor<1x8xi32, #blocked> -> tensor<4x8xi32, #blocked>
      %69 = tt.addptr %67, %68 : tensor<4x8x!tt.ptr<f16>, #blocked>, tensor<4x8xi32, #blocked>

      %70 = arith.cmpi slt, %63, %30 : tensor<4x1xi32, #blocked>
      %71 = arith.cmpi slt, %66, %31 : tensor<1x8xi32, #blocked>
      %72 = tt.broadcast %70 : tensor<4x1xi1, #blocked> -> tensor<4x8xi1, #blocked>
      %73 = tt.broadcast %71 : tensor<1x8xi1, #blocked> -> tensor<4x8xi1, #blocked>
      %74 = arith.andi %72, %73 : tensor<4x8xi1, #blocked>

      tt.store %69, %61#1, %74 : tensor<4x8x!tt.ptr<f16>, #blocked>
    }

    tt.return
  }
}

// CHECK: STORE POINTWISE
