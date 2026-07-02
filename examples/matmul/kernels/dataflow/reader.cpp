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
  size_t v2 = 1;
  int32_t v3 = 32;
  int32_t v4 = 1;
  int32_t v5 = 31;
  int8_t v6 = 1;
  Noc noc1(1);
  DeviceZoneScopedN("kernel_outer_matmul_kernel__reader");
  auto tensor_accessor_args_0 = TensorAccessorArgs<3, 0>();
  auto tensor_accessor_args_1 = TensorAccessorArgs<tensor_accessor_args_0.next_compile_time_args_offset(), tensor_accessor_args_0.next_common_runtime_args_offset()>();
  int32_t v7 = get_common_arg_val<uint32_t>(v2);
  int32_t v8 = get_common_arg_val<uint32_t>(3);
  int32_t v9 = get_common_arg_val<uint32_t>(4);
  int32_t v10 = get_common_arg_val<uint32_t>(5);
  CircularBuffer cb_ctarg_1(get_compile_time_arg_val(1));
  int32_t v11 = get_tile_size(get_compile_time_arg_val(1));
  TensorAccessor v12 = TensorAccessor(tensor_accessor_args_1, v7, v11);
  int32_t v13 = get_arg_val<uint32_t>(v2);
  int32_t v14 = get_arg_val<uint32_t>(0);
  for (int32_t i15 = v14; i15 < v13; i15 += v4) {
    for (int32_t j16 = 0; j16 < ((int32_t) ((uint32_t) v10 + (uint32_t) v5) / v3); j16 += v4) {
      cb_ctarg_1.reserve_back(v4);
      noc1.async_read(v12, CoreLocalMem<uint32_t>(cb_ctarg_1.get_write_ptr()), v12.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) j16 * (uint32_t) v3) / v3) * (uint32_t) ((int32_t) ((uint32_t) v9 + (uint32_t) v5) / v3))) + (uint32_t) (((int32_t) ((uint32_t) ((i15 % ((int32_t) ((uint32_t) v9 + (uint32_t) v5) / v3)) / ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v8 + (uint32_t) v5) / v3) - (uint32_t) (i15 / ((int32_t) ((uint32_t) v9 + (uint32_t) v5) / v3))) < v4 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v8 + (uint32_t) v5) / v3) - (uint32_t) (i15 / ((int32_t) ((uint32_t) v9 + (uint32_t) v5) / v3))) : v4)) * (uint32_t) v3) % v9) / v3))) * (uint32_t) 1024)) + (uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) j16 * (uint32_t) v3) % v3) * (uint32_t) v3)) + (uint32_t) (((int32_t) ((uint32_t) ((i15 % ((int32_t) ((uint32_t) v9 + (uint32_t) v5) / v3)) / ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v8 + (uint32_t) v5) / v3) - (uint32_t) (i15 / ((int32_t) ((uint32_t) v9 + (uint32_t) v5) / v3))) < v4 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v8 + (uint32_t) v5) / v3) - (uint32_t) (i15 / ((int32_t) ((uint32_t) v9 + (uint32_t) v5) / v3))) : v4)) * (uint32_t) v3) % v9) % v3))))) * (uint32_t) 2)) / (uint32_t) v11))}, {});
      noc1.async_read_barrier();
      cb_ctarg_1.push_back(v4);
    }
  }
  return;
}
