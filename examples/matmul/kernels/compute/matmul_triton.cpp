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
  int32_t v1 = 1;
  int32_t v2 = 0;
  size_t v3 = 0;
  DeviceZoneScopedN("kernel_outer_matmul_kernel__compute");
  int32_t v4 = get_common_arg_val<uint32_t>(5);
  CircularBuffer cb_ctarg_2(get_compile_time_arg_val(2));
  CircularBuffer cb_ctarg_1(get_compile_time_arg_val(1));
  CircularBuffer cb_ctarg_0(get_compile_time_arg_val(0));
  compute_kernel_hw_startup<SrcOrder::Reverse>(get_compile_time_arg_val(0), get_compile_time_arg_val(1), get_compile_time_arg_val(2));
  matmul_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v2);
  int32_t v5 = get_arg_val<uint32_t>(1);
  int32_t v6 = get_arg_val<uint32_t>(v3);
  for (int32_t i7 = v6; i7 < v5; i7 += v1) {
    tile_regs_acquire();
    matmul_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v2);
    for (int32_t j8 = v2; j8 < ((int32_t) ((uint32_t) v4 + (uint32_t) 31) / 32); j8 += v1) {
      cb_ctarg_0.wait_front(v1);
      cb_ctarg_1.wait_front(v1);
      matmul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1), v2, v2, v3);
      cb_ctarg_0.pop_front(v1);
      cb_ctarg_1.pop_front(v1);
    }
    cb_ctarg_2.reserve_back(v1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(v2, get_compile_time_arg_val(2), v2);
    tile_regs_release();
    cb_ctarg_2.push_back(v1);
  }
  return;
}
