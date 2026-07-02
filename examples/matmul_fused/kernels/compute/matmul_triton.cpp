#include <cstdint>
#include "api/compile_time_args.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"
inline uint32_t float_to_bits(const float f) { uint32_t r; __builtin_memcpy(&r, &f, sizeof(r)); return r; }
#ifndef INFINITY
#define INFINITY __builtin_inff()
#endif
void kernel_main() {
  int32_t v1 = 15;
  int32_t v2 = 7;
  int32_t v3 = 14;
  int32_t v4 = 6;
  int32_t v5 = 13;
  int32_t v6 = 5;
  int32_t v7 = 12;
  int32_t v8 = 4;
  int32_t v9 = 11;
  int32_t v10 = 3;
  int32_t v11 = 10;
  int32_t v12 = 2;
  int32_t v13 = 9;
  int32_t v14 = 8;
  int32_t v15 = 16;
  int32_t v16 = 1;
  int32_t v17 = 0;
  size_t v18 = 7;
  size_t v19 = 6;
  size_t v20 = 5;
  size_t v21 = 4;
  size_t v22 = 3;
  size_t v23 = 2;
  size_t v24 = 1;
  size_t v25 = 0;
  DeviceZoneScopedN("kernel_outer_matmul_kernel_fused__compute");
  int32_t v26 = get_common_arg_val<uint32_t>(42);
  CircularBuffer cb_ctarg_3(get_compile_time_arg_val(3));
  CircularBuffer cb_ctarg_2(get_compile_time_arg_val(2));
  CircularBuffer cb_ctarg_1(get_compile_time_arg_val(1));
  CircularBuffer cb_ctarg_0(get_compile_time_arg_val(0));
  compute_kernel_hw_startup<SrcOrder::Reverse>(get_compile_time_arg_val(0), get_compile_time_arg_val(1), get_compile_time_arg_val(3));
  matmul_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v17);
  int32_t v27 = get_arg_val<uint32_t>(v24);
  int32_t v28 = get_arg_val<uint32_t>(v25);
  for (int32_t i29 = v28; i29 < v27; i29 += v16) {
    tile_regs_acquire();
    matmul_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v17);
    for (int32_t j30 = v17; j30 < ((int32_t) ((uint32_t) v26 + (uint32_t) 255) / 256); j30 += v16) {
      cb_ctarg_0.wait_front(v15);
      cb_ctarg_1.wait_front(v15);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v17, v17, v25);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v17, v14, v24);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v14, v17, v23);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v14, v14, v22);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v16, v16, v25);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v16, v13, v24);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v13, v16, v23);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v13, v13, v22);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v12, v12, v25);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v12, v11, v24);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v11, v12, v23);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v11, v11, v22);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v10, v10, v25);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v10, v9, v24);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v9, v10, v23);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v9, v9, v22);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v8, v8, v25);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v8, v7, v24);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v7, v8, v23);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v7, v7, v22);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v6, v6, v25);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v6, v5, v24);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v5, v6, v23);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v5, v5, v22);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v4, v4, v25);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v4, v3, v24);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v3, v4, v23);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v3, v3, v22);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v2, v2, v25);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v2, v1, v24);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v1, v2, v23);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v1, v1, v22);
      cb_ctarg_0.pop_front(v15);
      cb_ctarg_1.pop_front(v15);
    }
    cb_ctarg_2.wait_front(v8);
    copy_tile_init(get_compile_time_arg_val(2));
    copy_tile(get_compile_time_arg_val(2), v25, v21);
    copy_tile(get_compile_time_arg_val(2), v24, v20);
    copy_tile(get_compile_time_arg_val(2), v23, v19);
    copy_tile(get_compile_time_arg_val(2), v22, v18);
    cb_ctarg_2.pop_front(v8);
    add_binary_tile_init();
    add_binary_tile(v25, v21, v25);
    add_binary_tile(v24, v20, v24);
    add_binary_tile(v23, v19, v23);
    add_binary_tile(v22, v18, v22);
    cb_ctarg_3.reserve_back(v8);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(v17, get_compile_time_arg_val(3), v17);
    pack_tile<true>(v16, get_compile_time_arg_val(3), v16);
    pack_tile<true>(v12, get_compile_time_arg_val(3), v12);
    pack_tile<true>(v10, get_compile_time_arg_val(3), v10);
    tile_regs_release();
    cb_ctarg_3.push_back(v8);
  }
  return;
}
