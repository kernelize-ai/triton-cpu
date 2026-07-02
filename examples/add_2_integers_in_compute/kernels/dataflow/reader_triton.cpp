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
  size_t v3 = 0;
  size_t v4 = 1;
  int32_t v5 = 1;
  int8_t v6 = 1;
  Noc noc1(1);
  DeviceZoneScopedN("kernel_outer_add_kernel__reader");
  auto tensor_accessor_args_0 = TensorAccessorArgs<3, 0>();
  int32_t v7 = get_common_arg_val<uint32_t>(v3);
  auto tensor_accessor_args_1 = TensorAccessorArgs<tensor_accessor_args_0.next_compile_time_args_offset(), tensor_accessor_args_0.next_common_runtime_args_offset()>();
  int32_t v8 = get_common_arg_val<uint32_t>(v4);
  CircularBuffer cb_ctarg_1(get_compile_time_arg_val(1));
  int32_t v9 = get_tile_size(get_compile_time_arg_val(1));
  TensorAccessor v10 = TensorAccessor(tensor_accessor_args_1, v8, v9);
  CircularBuffer cb_ctarg_0(get_compile_time_arg_val(0));
  int32_t v11 = get_tile_size(get_compile_time_arg_val(0));
  TensorAccessor v12 = TensorAccessor(tensor_accessor_args_0, v7, v11);
  int32_t v13 = get_arg_val<uint32_t>(v4);
  int32_t v14 = get_arg_val<uint32_t>(v3);
  for (int32_t i15 = v14; i15 < v13; i15 += v5) {
    int32_t v16 = (int32_t) ((uint32_t) i15 * (uint32_t) 2048);
    cb_ctarg_0.reserve_back(v5);
    noc1.async_read(v12, CoreLocalMem<uint32_t>(cb_ctarg_0.get_write_ptr()), v12.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v16 / (uint32_t) v11))}, {});
    noc1.async_read_barrier();
    cb_ctarg_0.push_back(v5);
    cb_ctarg_1.reserve_back(v5);
    noc1.async_read(v10, CoreLocalMem<uint32_t>(cb_ctarg_1.get_write_ptr()), v10.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v16 / (uint32_t) v9))}, {});
    noc1.async_read_barrier();
    cb_ctarg_1.push_back(v5);
  }
  return;
}
