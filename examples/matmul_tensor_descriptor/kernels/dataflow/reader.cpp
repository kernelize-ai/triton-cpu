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
  int32_t v2 = 5;
  int32_t v3 = 63;
  int32_t v4 = 64;
  int32_t v5 = 1;
  int32_t v6 = 4;
  int32_t v7 = 32;
  int8_t v8 = 1;
  Noc noc1(1);
  DeviceZoneScopedN("kernel_outer_matmul_kernel_tma__reader");
  auto tensor_accessor_args_0 = TensorAccessorArgs<5, 0>();
  auto tensor_accessor_args_1 = TensorAccessorArgs<tensor_accessor_args_0.next_compile_time_args_offset(), tensor_accessor_args_0.next_common_runtime_args_offset()>();
  int32_t v9 = get_common_arg_val<uint32_t>(10);
  int32_t v10 = get_common_arg_val<uint32_t>(12);
  int32_t v11 = get_common_arg_val<uint32_t>(30);
  int32_t v12 = get_common_arg_val<uint32_t>(31);
  int32_t v13 = get_common_arg_val<uint32_t>(32);
  CircularBuffer cb_ctarg_1(get_compile_time_arg_val(1));
  int32_t v14 = get_tile_size(get_compile_time_arg_val(1));
  int32_t v15 = get_arg_val<uint32_t>(1);
  int32_t v16 = get_arg_val<uint32_t>(0);
  int32_t v17 = v10 != (int32_t) ((uint32_t) (v10 / v7) * (uint32_t) v7) && v10 < v1 == false ? (int32_t) ((uint32_t) (v10 / v7) + (uint32_t) v5) : v10 / v7;
  for (int32_t i18 = v16; i18 < v15; i18 += v5) {
    for (int32_t j19 = v1; j19 < ((int32_t) ((uint32_t) v13 + (uint32_t) v3) / v4); j19 += v5) {
      cb_ctarg_1.reserve_back(v6);
      TensorAccessor v20 = TensorAccessor(tensor_accessor_args_1, v9, v14);
      int32_t v21 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) j19 * (uint32_t) v4) / v7) * (uint32_t) v17)) + (uint32_t) ((int32_t) ((uint32_t) ((i18 % ((int32_t) ((uint32_t) v12 + (uint32_t) v3) / v4)) / ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v11 + (uint32_t) v3) / v4) - (uint32_t) (i18 / ((int32_t) ((uint32_t) v12 + (uint32_t) v3) / v4))) < v5 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v11 + (uint32_t) v3) / v4) - (uint32_t) (i18 / ((int32_t) ((uint32_t) v12 + (uint32_t) v3) / v4))) : v5)) * (uint32_t) v4) / v7));
      noc1.async_read(v20, CoreLocalMem<uint32_t>(cb_ctarg_1.get_write_ptr()), v20.get_aligned_page_size(), {.page_id = static_cast<uint32_t>(v21)}, {});
      int32_t v22 = (int32_t) ((uint32_t) v21 + (uint32_t) v5);
      noc1.async_read(v20, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_1.get_write_ptr() + (uint32_t) 4096)), v20.get_aligned_page_size(), {.page_id = static_cast<uint32_t>(v22)}, {});
      noc1.async_read(v20, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_1.get_write_ptr() + (uint32_t) 2048)), v20.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v21 + (uint32_t) v17))}, {});
      noc1.async_read(v20, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_1.get_write_ptr() + (uint32_t) 6144)), v20.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v22 + (uint32_t) v17))}, {});
      noc1.async_read_barrier();
      cb_ctarg_1.push_back(v6);
    }
  }
  return;
}
