#include <cstdint>
#include "api/compile_time_args.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"
inline uint32_t float_to_bits(const float f) { uint32_t r; __builtin_memcpy(&r, &f, sizeof(r)); return r; }
#ifndef INFINITY
#define INFINITY __builtin_inff()
#endif
void kernel_main() {
  int32_t v1 = 3;
  int32_t v2 = 2;
  int32_t v3 = 4;
  int32_t v4 = 1;
  int32_t v5 = 0;
  size_t v6 = 3;
  size_t v7 = 2;
  size_t v8 = 1;
  size_t v9 = 0;
  DeviceZoneScopedN("kernel_outer_matmul_kernel_tma__compute");
  int32_t v10 = get_common_arg_val<uint32_t>(32);
  CircularBuffer cb_ctarg_2(get_compile_time_arg_val(2));
  CircularBuffer cb_ctarg_1(get_compile_time_arg_val(1));
  CircularBuffer cb_ctarg_0(get_compile_time_arg_val(0));
  compute_kernel_hw_startup<SrcOrder::Reverse>(get_compile_time_arg_val(0), get_compile_time_arg_val(1), get_compile_time_arg_val(2));
  matmul_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v5);
  int32_t v11 = get_arg_val<uint32_t>(v8);
  int32_t v12 = get_arg_val<uint32_t>(v9);
  for (int32_t i13 = v12; i13 < v11; i13 += v4) {
    tile_regs_acquire();
    matmul_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v5);
    for (int32_t j14 = v5; j14 < ((int32_t) ((uint32_t) v10 + (uint32_t) 63) / 64); j14 += v4) {
      cb_ctarg_0.wait_front(v3);
      cb_ctarg_1.wait_front(v3);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v5, v5, v9);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v5, v2, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v2, v5, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v2, v2, v6);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v4, v4, v9);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v4, v1, v8);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v1, v4, v7);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v1, v1, v6);
      cb_ctarg_0.pop_front(v3);
      cb_ctarg_1.pop_front(v3);
    }
    cb_ctarg_2.reserve_back(v3);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(v5, get_compile_time_arg_val(2), v5);
    pack_tile<true>(v4, get_compile_time_arg_val(2), v4);
    pack_tile<true>(v2, get_compile_time_arg_val(2), v2);
    pack_tile<true>(v1, get_compile_time_arg_val(2), v1);
    tile_regs_release();
    cb_ctarg_2.push_back(v3);
  }
  return;
}
