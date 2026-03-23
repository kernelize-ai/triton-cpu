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
        ttc.generic %x_3 attributes {blockShape = array<i32: 512>, vectorShape = array<i32: 4>} body {
            ^bb0(%arg0: tensor<4x!tt.ptr<f32>, #blocked>):
                %x_10 = tt.load %arg0 : tensor<4x!tt.ptr<f32>, #blocked>
                ttc.yield
        } combiners {}: (tensor<512x!tt.ptr<f32>, #blocked>) -> ()
        tt.return
    }
}
