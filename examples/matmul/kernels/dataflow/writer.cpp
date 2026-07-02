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
  int32_t v1 = 3;
  size_t v2 = 0;
  int32_t v3 = 32;
  int32_t v4 = 1;
  int32_t v5 = 31;
  int32_t v6 = 1024;
  int32_t v7 = 2;
  int8_t v8 = 0;
  Noc noc0(0);
  DeviceZoneScopedN("kernel_outer_matmul_kernel__writer");
  auto tensor_accessor_args_0 = TensorAccessorArgs<3, 0>();
  int32_t v9 = get_common_arg_val<uint32_t>(v2);
  auto tensor_accessor_args_1 = TensorAccessorArgs<tensor_accessor_args_0.next_compile_time_args_offset(), tensor_accessor_args_0.next_common_runtime_args_offset()>();
  auto tensor_accessor_args_2 = TensorAccessorArgs<tensor_accessor_args_1.next_compile_time_args_offset(), tensor_accessor_args_1.next_common_runtime_args_offset()>();
  int32_t v10 = get_common_arg_val<uint32_t>(2);
  int32_t v11 = get_common_arg_val<uint32_t>(3);
  int32_t v12 = get_common_arg_val<uint32_t>(4);
  int32_t v13 = get_common_arg_val<uint32_t>(5);
  CircularBuffer cb_ctarg_2(get_compile_time_arg_val(2));
  int32_t v14 = get_tile_size(get_compile_time_arg_val(2));
  TensorAccessor v15 = TensorAccessor(tensor_accessor_args_2, v10, v14);
  CircularBuffer cb_ctarg_0(get_compile_time_arg_val(0));
  int32_t v16 = get_tile_size(get_compile_time_arg_val(0));
  TensorAccessor v17 = TensorAccessor(tensor_accessor_args_0, v9, v16);
  int32_t v18 = get_arg_val<uint32_t>(1);
  int32_t v19 = get_arg_val<uint32_t>(v2);
  int32_t v20 = (int32_t) ((uint32_t) v12 + (uint32_t) v5) / v3;
  int32_t v21 = (int32_t) ((uint32_t) v13 + (uint32_t) v5) / v3;
  for (int32_t i22 = v19; i22 < v18; i22 += v4) {
    int32_t v23 = i22 / v20;
    int32_t v24 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v11 + (uint32_t) v5) / v3) - (uint32_t) v23) < v4 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v11 + (uint32_t) v5) / v3) - (uint32_t) v23) : v4;
    int32_t v25 = i22 % v20;
    int32_t v26 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v23 + (uint32_t) (v25 % v24))) * (uint32_t) v3) % v11;
    int32_t v27 = v26 / v3;
    int32_t v28 = (int32_t) ((uint32_t) (v26 % v3) * (uint32_t) v3);
    for (int32_t j29 = 0; j29 < v21; j29 += v4) {
      cb_ctarg_0.reserve_back(v4);
      noc0.async_read(v17, CoreLocalMem<uint32_t>(cb_ctarg_0.get_write_ptr()), v17.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v27 * (uint32_t) v21)) + (uint32_t) ((int32_t) ((uint32_t) j29 * (uint32_t) v3) / v3))) * (uint32_t) v6)) + (uint32_t) ((int32_t) ((uint32_t) v28 + (uint32_t) ((int32_t) ((uint32_t) j29 * (uint32_t) v3) % v3))))) * (uint32_t) v7)) / (uint32_t) v16))}, {});
      noc0.async_read_barrier();
      cb_ctarg_0.push_back(v4);
    }
    cb_ctarg_2.wait_front(v4);
    noc0.async_write(CoreLocalMem<uint32_t>(cb_ctarg_2.get_read_ptr()), v15, v15.get_aligned_page_size(), {} , {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v27 * (uint32_t) v20)) + (uint32_t) (((int32_t) ((uint32_t) (v25 / v24) * (uint32_t) v3) % v12) / v3))) * (uint32_t) v6)) + (uint32_t) ((int32_t) ((uint32_t) v28 + (uint32_t) (((int32_t) ((uint32_t) (v25 / v24) * (uint32_t) v3) % v12) % v3))))) * (uint32_t) v7)) / (uint32_t) v14))});
    noc0.async_write_barrier();
    cb_ctarg_2.pop_front(v4);
  }
  return;
}
