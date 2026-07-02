#include <cstdint>
#include "api/compile_time_args.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
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
  int32_t v1 = 0;
  int32_t v2 = 1;
  size_t v3 = 1;
  size_t v4 = 0;
  DeviceZoneScopedN("kernel_outer_add_kernel__compute");
  CircularBuffer cb_ctarg_2(get_compile_time_arg_val(2));
  CircularBuffer cb_ctarg_1(get_compile_time_arg_val(1));
  CircularBuffer cb_ctarg_0(get_compile_time_arg_val(0));
  init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(2));
  int32_t v5 = get_arg_val<uint32_t>(v3);
  int32_t v6 = get_arg_val<uint32_t>(v4);
  for (int32_t i7 = v6; i7 < v5; i7 += v2) {
    tile_regs_acquire();
    cb_ctarg_0.wait_front(v2);
    copy_tile_init(get_compile_time_arg_val(0));
    copy_tile(get_compile_time_arg_val(0), v4, v4);
    cb_ctarg_0.pop_front(v2);
    cb_ctarg_1.wait_front(v2);
    copy_tile_init(get_compile_time_arg_val(1));
    copy_tile(get_compile_time_arg_val(1), v4, v3);
    cb_ctarg_1.pop_front(v2);
    add_binary_tile_init();
    add_binary_tile(v4, v3, v4);
    cb_ctarg_2.reserve_back(v2);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(v1, get_compile_time_arg_val(2), v1);
    tile_regs_release();
    cb_ctarg_2.push_back(v2);
  }
  return;
}
