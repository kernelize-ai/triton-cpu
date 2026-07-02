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
  bool v1 = false;
  int32_t v2 = 0;
  int32_t v3 = 63;
  int32_t v4 = 256;
  int32_t v5 = 64;
  int32_t v6 = 1;
  int32_t v7 = 16;
  int32_t v8 = 32;
  int8_t v9 = 1;
  int32_t v10 = 2048;
  int32_t v11 = 4096;
  int32_t v12 = 6144;
  int32_t v13 = 4;
  Noc noc1(1);
  DeviceZoneScopedN("kernel_outer_matmul_kernel_fused__reader");
  auto tensor_accessor_args_0 = TensorAccessorArgs<6, 0>();
  auto tensor_accessor_args_1 = TensorAccessorArgs<tensor_accessor_args_0.next_compile_time_args_offset(), tensor_accessor_args_0.next_common_runtime_args_offset()>();
  int32_t v14 = get_common_arg_val<uint32_t>(10);
  int32_t v15 = get_common_arg_val<uint32_t>(12);
  auto tensor_accessor_args_2 = TensorAccessorArgs<tensor_accessor_args_1.next_compile_time_args_offset(), tensor_accessor_args_1.next_common_runtime_args_offset()>();
  auto tensor_accessor_args_3 = TensorAccessorArgs<tensor_accessor_args_2.next_compile_time_args_offset(), tensor_accessor_args_2.next_common_runtime_args_offset()>();
  int32_t v16 = get_common_arg_val<uint32_t>(30);
  int32_t v17 = get_common_arg_val<uint32_t>(32);
  int32_t v18 = get_common_arg_val<uint32_t>(40);
  int32_t v19 = get_common_arg_val<uint32_t>(41);
  int32_t v20 = get_common_arg_val<uint32_t>(42);
  CircularBuffer cb_ctarg_2(get_compile_time_arg_val(2));
  int32_t v21 = get_tile_size(get_compile_time_arg_val(2));
  CircularBuffer cb_ctarg_1(get_compile_time_arg_val(1));
  int32_t v22 = get_tile_size(get_compile_time_arg_val(1));
  int32_t v23 = get_arg_val<uint32_t>(1);
  int32_t v24 = get_arg_val<uint32_t>(0);
  int32_t v25 = (int32_t) ((uint32_t) v19 + (uint32_t) v3) / v5;
  int32_t v26 = v15 != (int32_t) ((uint32_t) (v15 / v8) * (uint32_t) v8) && v15 < v2 == v1 ? (int32_t) ((uint32_t) (v15 / v8) + (uint32_t) v6) : v15 / v8;
  int32_t v27 = (int32_t) ((uint32_t) v26 * (uint32_t) 2);
  int32_t v28 = (int32_t) ((uint32_t) v26 * (uint32_t) 3);
  int32_t v29 = (int32_t) ((uint32_t) v26 * (uint32_t) v13);
  int32_t v30 = (int32_t) ((uint32_t) v26 * (uint32_t) 5);
  int32_t v31 = (int32_t) ((uint32_t) v26 * (uint32_t) 6);
  int32_t v32 = (int32_t) ((uint32_t) v26 * (uint32_t) 7);
  int32_t v33 = v17 != (int32_t) ((uint32_t) (v17 / v8) * (uint32_t) v8) && v17 < v2 == v1 ? (int32_t) ((uint32_t) (v17 / v8) + (uint32_t) v6) : v17 / v8;
  for (int32_t i34 = v24; i34 < v23; i34 += v6) {
    int32_t v35 = i34 / v25;
    int32_t v36 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v18 + (uint32_t) v3) / v5) - (uint32_t) v35) < v6 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v18 + (uint32_t) v3) / v5) - (uint32_t) v35) : v6;
    int32_t v37 = (int32_t) ((uint32_t) ((i34 % v25) / v36) * (uint32_t) v5) / v8;
    for (int32_t j38 = v2; j38 < ((int32_t) ((uint32_t) v20 + (uint32_t) 255) / v4); j38 += v6) {
      cb_ctarg_1.reserve_back(v7);
      TensorAccessor v39 = TensorAccessor(tensor_accessor_args_1, v14, v22);
      int32_t v40 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) j38 * (uint32_t) v4) / v8) * (uint32_t) v26)) + (uint32_t) v37);
      noc1.async_read(v39, CoreLocalMem<uint32_t>(cb_ctarg_1.get_write_ptr()), v39.get_aligned_page_size(), {.page_id = static_cast<uint32_t>(v40)}, {});
      int32_t v41 = (int32_t) ((uint32_t) v40 + (uint32_t) v6);
      noc1.async_read(v39, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_1.get_write_ptr() + (uint32_t) 16384)), v39.get_aligned_page_size(), {.page_id = static_cast<uint32_t>(v41)}, {});
      noc1.async_read(v39, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_1.get_write_ptr() + (uint32_t) v10)), v39.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v40 + (uint32_t) v26))}, {});
      noc1.async_read(v39, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_1.get_write_ptr() + (uint32_t) 18432)), v39.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v41 + (uint32_t) v26))}, {});
      noc1.async_read(v39, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_1.get_write_ptr() + (uint32_t) v11)), v39.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v40 + (uint32_t) v27))}, {});
      noc1.async_read(v39, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_1.get_write_ptr() + (uint32_t) 20480)), v39.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v41 + (uint32_t) v27))}, {});
      noc1.async_read(v39, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_1.get_write_ptr() + (uint32_t) v12)), v39.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v40 + (uint32_t) v28))}, {});
      noc1.async_read(v39, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_1.get_write_ptr() + (uint32_t) 22528)), v39.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v41 + (uint32_t) v28))}, {});
      noc1.async_read(v39, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_1.get_write_ptr() + (uint32_t) 8192)), v39.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v40 + (uint32_t) v29))}, {});
      noc1.async_read(v39, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_1.get_write_ptr() + (uint32_t) 24576)), v39.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v41 + (uint32_t) v29))}, {});
      noc1.async_read(v39, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_1.get_write_ptr() + (uint32_t) 10240)), v39.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v40 + (uint32_t) v30))}, {});
      noc1.async_read(v39, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_1.get_write_ptr() + (uint32_t) 26624)), v39.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v41 + (uint32_t) v30))}, {});
      noc1.async_read(v39, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_1.get_write_ptr() + (uint32_t) 12288)), v39.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v40 + (uint32_t) v31))}, {});
      noc1.async_read(v39, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_1.get_write_ptr() + (uint32_t) 28672)), v39.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v41 + (uint32_t) v31))}, {});
      noc1.async_read(v39, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_1.get_write_ptr() + (uint32_t) 14336)), v39.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v40 + (uint32_t) v32))}, {});
      noc1.async_read(v39, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_1.get_write_ptr() + (uint32_t) 30720)), v39.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v41 + (uint32_t) v32))}, {});
      noc1.async_read_barrier();
      cb_ctarg_1.push_back(v7);
    }
    cb_ctarg_2.reserve_back(v13);
    TensorAccessor v42 = TensorAccessor(tensor_accessor_args_3, v16, v21);
    int32_t v43 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v35 + (uint32_t) (i34 % v36))) * (uint32_t) v5) / v8) * (uint32_t) v33)) + (uint32_t) v37);
    noc1.async_read(v42, CoreLocalMem<uint32_t>(cb_ctarg_2.get_write_ptr()), v42.get_aligned_page_size(), {.page_id = static_cast<uint32_t>(v43)}, {});
    int32_t v44 = (int32_t) ((uint32_t) v43 + (uint32_t) v6);
    noc1.async_read(v42, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_2.get_write_ptr() + (uint32_t) v10)), v42.get_aligned_page_size(), {.page_id = static_cast<uint32_t>(v44)}, {});
    noc1.async_read(v42, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_2.get_write_ptr() + (uint32_t) v11)), v42.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v43 + (uint32_t) v33))}, {});
    noc1.async_read(v42, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_2.get_write_ptr() + (uint32_t) v12)), v42.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v44 + (uint32_t) v33))}, {});
    noc1.async_read_barrier();
    cb_ctarg_2.push_back(v13);
  }
  return;
}
