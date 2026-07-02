#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"

// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_SEMAPHORE_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_SEMAPHORE_H

namespace experimental {

FORCE_INLINE
void semaphore_wait(volatile tt_l1_ptr uint32_t *sem_addr, uint32_t val) {
  uint32_t sem_val;
  do {
    invalidate_l1_cache();
    sem_val = *sem_addr;
  } while (sem_val != val);
}

FORCE_INLINE
void semaphore_wait_min(volatile tt_l1_ptr uint32_t *sem_addr, uint32_t val) {
  uint32_t sem_val;
  do {
    invalidate_l1_cache();
    sem_val = *sem_addr;
  } while (sem_val < val);
}

} // namespace experimental

#endif


// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_COORD_TRANSLATION_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_COORD_TRANSLATION_H

namespace experimental {

FORCE_INLINE
std::uint32_t convert_logical_x_to_translated(std::uint32_t logical_x) {
  /*
      convert x coord from LOGICAL to TRANSLATED coordinate system
  */
  return worker_logical_col_to_virtual_col[logical_x];
}

FORCE_INLINE
std::uint32_t convert_logical_y_to_translated(std::uint32_t logical_y) {
  /*
      convert x coord from LOGICAL to TRANSLATED coordinate system
  */
  return worker_logical_row_to_virtual_row[logical_y];
}

} // namespace experimental

#endif

void kernel_main() {
  bool v1 = false;
  uint32_t v2 = 0;
  uint32_t v3 = 32;
  int32_t v4 = 0;
  int32_t v5 = 5;
  size_t v6 = 0;
  size_t v7 = 1;
  int32_t v8 = 63;
  int32_t v9 = 64;
  int32_t v10 = 1;
  int32_t v11 = 65535;
  int32_t v12 = 16;
  int32_t v13 = 4;
  int8_t v14 = 0;
  int32_t v15 = 32;
  int32_t v16 = 2048;
  int32_t v17 = 4096;
  int32_t v18 = 6144;
  UnicastEndpoint unicast_ep;
  MulticastEndpoint mcast_ep;
  Noc noc0(0);
  DeviceZoneScopedN("kernel_outer_matmul_kernel_tma__writer");
  auto tensor_accessor_args_0 = TensorAccessorArgs<5, 0>();
  int32_t v19 = get_common_arg_val<uint32_t>(v6);
  int32_t v20 = get_common_arg_val<uint32_t>(2);
  auto tensor_accessor_args_1 = TensorAccessorArgs<tensor_accessor_args_0.next_compile_time_args_offset(), tensor_accessor_args_0.next_common_runtime_args_offset()>();
  auto tensor_accessor_args_2 = TensorAccessorArgs<tensor_accessor_args_1.next_compile_time_args_offset(), tensor_accessor_args_1.next_common_runtime_args_offset()>();
  int32_t v21 = get_common_arg_val<uint32_t>(20);
  int32_t v22 = get_common_arg_val<uint32_t>(22);
  int32_t v23 = get_common_arg_val<uint32_t>(30);
  int32_t v24 = get_common_arg_val<uint32_t>(31);
  int32_t v25 = get_common_arg_val<uint32_t>(32);
  CircularBuffer cb_ctarg_2(get_compile_time_arg_val(2));
  int32_t v26 = get_tile_size(get_compile_time_arg_val(2));
  TensorAccessor v27 = TensorAccessor(tensor_accessor_args_2, v21, v26);
  CircularBuffer cb_ctarg_0(get_compile_time_arg_val(0));
  int32_t v28 = get_tile_size(get_compile_time_arg_val(0));
  int32_t v29 = get_arg_val<uint32_t>(v7);
  int32_t v30 = get_arg_val<uint32_t>(v6);
  int32_t v31 = (int32_t) ((uint32_t) v24 + (uint32_t) v8) / v9;
  int32_t v32 = get_arg_val<uint32_t>(3);
  int32_t v33 = get_arg_val<uint32_t>(4);
  int32_t v34 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v12 < v3 ? (uint32_t) v32 >> (uint32_t) v12 : v2)) & (uint32_t) v11);
  int32_t v35 = (int32_t) ((uint32_t) v32 & (uint32_t) v11);
  int32_t v36 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v12 < v3 ? (uint32_t) v33 >> (uint32_t) v12 : v2)) & (uint32_t) v11);
  int32_t v37 = (int32_t) ((uint32_t) v33 & (uint32_t) v11);
  int32_t v38 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) ((int32_t) ((uint32_t) v36 + (uint32_t) v10)) * (uint32_t) ((int32_t) ((uint32_t) v37 + (uint32_t) v10)))) - (uint32_t) v10);
  int32_t v39 = v22 != (int32_t) ((uint32_t) (v22 / v15) * (uint32_t) v15) && v22 < v4 == v1 ? (int32_t) ((uint32_t) (v22 / v15) + (uint32_t) v10) : v22 / v15;
  for (int32_t i40 = v30; i40 < v29; i40 += v10) {
    int32_t v41 = i40 / v31;
    int32_t v42 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v23 + (uint32_t) v8) / v9) - (uint32_t) v41) < v10 ? (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v23 + (uint32_t) v8) / v9) - (uint32_t) v41) : v10;
    int32_t v43 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v41 + (uint32_t) (i40 % v42))) * (uint32_t) v9);
    for (int32_t j44 = v4; j44 < ((int32_t) ((uint32_t) v25 + (uint32_t) v8) / v9); j44 += v10) {
      int32_t v45 = get_semaphore(get_compile_time_arg_val(3));
      int32_t v46 = get_semaphore(get_compile_time_arg_val(4));
      if (v34 == (int32_t) ((ptrdiff_t) get_absolute_logical_x()) && v35 == (int32_t) ((ptrdiff_t) get_absolute_logical_y())) {
        cb_ctarg_0.reserve_back(v13);
        TensorAccessor v47 = TensorAccessor(tensor_accessor_args_0, v19, v28);
        int32_t v48 = v20 != (int32_t) ((uint32_t) (v20 / v15) * (uint32_t) v15) && v20 < v4 == v1 ? (int32_t) ((uint32_t) (v20 / v15) + (uint32_t) v10) : v20 / v15;
        int32_t v49 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) (v43 / v15) * (uint32_t) v48)) + (uint32_t) ((int32_t) ((uint32_t) j44 * (uint32_t) v9) / v15));
        noc0.async_read(v47, CoreLocalMem<uint32_t>(cb_ctarg_0.get_write_ptr()), v47.get_aligned_page_size(), {.page_id = static_cast<uint32_t>(v49)}, {});
        int32_t v50 = (int32_t) ((uint32_t) v49 + (uint32_t) v10);
        noc0.async_read(v47, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_0.get_write_ptr() + (uint32_t) v16)), v47.get_aligned_page_size(), {.page_id = static_cast<uint32_t>(v50)}, {});
        noc0.async_read(v47, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_0.get_write_ptr() + (uint32_t) v17)), v47.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v49 + (uint32_t) v48))}, {});
        noc0.async_read(v47, CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_0.get_write_ptr() + (uint32_t) v18)), v47.get_aligned_page_size(), {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v50 + (uint32_t) v48))}, {});
        noc0.async_read_barrier();
        tt_l1_ptr uint32_t* v51 = reinterpret_cast<tt_l1_ptr uint32_t*>(v45);
        experimental::semaphore_wait(v51, v38);
        noc_semaphore_set(v51, v6);
        size_t v52 = experimental::convert_logical_y_to_translated(v37);
        size_t v53 = experimental::convert_logical_x_to_translated(v36);
        size_t v54 = experimental::convert_logical_y_to_translated(v35);
        size_t v55 = experimental::convert_logical_x_to_translated(v34);
        noc0.async_write_multicast(CoreLocalMem<uint32_t>(cb_ctarg_0.get_write_ptr()), mcast_ep, 8192, v38, {} , noc_traits_t<MulticastEndpoint>::dst_args_mcast_type{.noc_x_start = v55, .noc_y_start = v54, .noc_x_end = v53, .noc_y_end = v52, .addr = static_cast<uint32_t>(cb_ctarg_0.get_write_ptr())}, false);
        noc0.async_write_barrier();
        tt_l1_ptr uint32_t* v56 = reinterpret_cast<tt_l1_ptr uint32_t*>(v46);
        noc_semaphore_set(v56, v7);
        int64_t v57 = get_noc_multicast_addr(v55, v54, v53, v52, v46, v14);
        noc_semaphore_set_multicast(v46, v57, v38);
      } else {
        cb_ctarg_0.reserve_back(v13);
        size_t v58 = experimental::convert_logical_y_to_translated(v35);
        size_t v59 = experimental::convert_logical_x_to_translated(v34);
        uint64_t noc_addr_3 = unicast_ep.get_noc_unicast_addr(static_cast<uint32_t>(v59), static_cast<uint32_t>(v58), static_cast<uint32_t>(v45), noc0.get_noc_id());
        noc_semaphore_inc(noc_addr_3, v7, v14);
        tt_l1_ptr uint32_t* v60 = reinterpret_cast<tt_l1_ptr uint32_t*>(v46);
        experimental::semaphore_wait(v60, v7);
        noc_semaphore_set(v60, v6);
      }
      cb_ctarg_0.push_back(v13);
    }
    cb_ctarg_2.wait_front(v13);
    int32_t v61 = (int32_t) ((uint32_t) ((int32_t) ((uint32_t) (v43 / v15) * (uint32_t) v39)) + (uint32_t) ((int32_t) ((uint32_t) ((i40 % v31) / v42) * (uint32_t) v9) / v15));
    noc0.async_write(CoreLocalMem<uint32_t>(cb_ctarg_2.get_read_ptr()), v27, v27.get_aligned_page_size(), {} , {.page_id = static_cast<uint32_t>(v61)});
    int32_t v62 = (int32_t) ((uint32_t) v61 + (uint32_t) v10);
    noc0.async_write(CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_2.get_read_ptr() + (uint32_t) v16)), v27, v27.get_aligned_page_size(), {} , {.page_id = static_cast<uint32_t>(v62)});
    noc0.async_write(CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_2.get_read_ptr() + (uint32_t) v17)), v27, v27.get_aligned_page_size(), {} , {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v61 + (uint32_t) v39))});
    noc0.async_write(CoreLocalMem<uint32_t>((int32_t) ((uint32_t) cb_ctarg_2.get_read_ptr() + (uint32_t) v18)), v27, v27.get_aligned_page_size(), {} , {.page_id = static_cast<uint32_t>((int32_t) ((uint32_t) v62 + (uint32_t) v39))});
    noc0.async_write_barrier();
    cb_ctarg_2.pop_front(v13);
  }
  return;
}
