#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"
void kernel_main() {
  int32_t v1 = 0;
  int32_t v2 = 3;
  int32_t v3 = 1;
  int8_t v4 = 0;
  Noc noc0(0);
  DeviceZoneScopedN("kernel_outer_add_kernel__writer");
  auto tensor_accessor_args_0 = TensorAccessorArgs<3, 0>();
  auto tensor_accessor_args_1 = TensorAccessorArgs<tensor_accessor_args_0.next_compile_time_args_offset(), tensor_accessor_args_0.next_common_runtime_args_offset()>();
  auto tensor_accessor_args_2 = TensorAccessorArgs<tensor_accessor_args_1.next_compile_time_args_offset(), tensor_accessor_args_1.next_common_runtime_args_offset()>();
  int32_t v5 = get_common_arg_val<uint32_t>(2);
  CircularBuffer cb_ctarg_2(get_compile_time_arg_val(2));
  int32_t v6 = get_tile_size(get_compile_time_arg_val(2));
  TensorAccessor v7 = TensorAccessor(tensor_accessor_args_2, v5, v6);
  int32_t v8 = get_arg_val<uint32_t>(1);
  int32_t v9 = get_arg_val<uint32_t>(0);
  for (int32_t i10 = v9; i10 < v8; i10 += v3) {
    cb_ctarg_2.wait_front(v3);
    noc0.async_write(CoreLocalMem<uint32_t>(cb_ctarg_2.get_read_ptr()), v7, v7.get_aligned_page_size(), {} , {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) ((int32_t) ((uint32_t) i10 * (uint32_t) 2048)) / (uint32_t) v6))});
    noc0.async_write_barrier();
    cb_ctarg_2.pop_front(v3);
  }
  return;
}
