// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <bmm_op.hpp>
#include <tt-metalium/device.hpp>
#include <fmt/core.h>

#include <omp.h>

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
// Reference implementation of matrix multiplication.
// Array A is of size MxK, Array B is of size KxN, and the output C is of size MxN.
// The implementation is bare bones and does not include optimizations such as tiling or vectorization.
// This is intended to be used as a golden reference for testing the Metalium implementation.
void golden_matmul(
    std::vector<bfloat16>& a,
    std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K) {
 #pragma omp parallel for collapse(2) schedule(static)
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            uint32_t idx_a = i * K;
            uint32_t idx_b = j;
            for (uint32_t k = 0; k < K; ++k) {
                sum += static_cast<float>(a[idx_a + k]) *
                       static_cast<float>(b[idx_b + k * N]);
            }
            output[j + i * N] = bfloat16(sum);
        }
    }
}


/**
 * @brief Multi-core matrix multiplication using SPMD (Single Program, Multiple Data) parallelization.
 *
 * Performs C = A * B matrix multiplication by distributing output tiles across multiple cores.
 * Each core runs the same program but works on different portions of the output matrix,
 * making this a simple and efficient parallelization scheme.
 *
 * The function uses three types of kernels running in parallel:
 * - Reader: Loads input matrix tiles from DRAM into circular buffers
 * - Compute: Performs tile-wise matrix multiplication (A_tile * B_tile = C_tile)
 * - Writer: Stores computed output tiles back to DRAM
 *
 * Work distribution is handled automatically - if output tiles don't divide evenly
 * across cores, some cores get one extra tile to balance the workload.
 *
 * @param a Input matrix A in row-major format (bfloat16 elements)
 * @param b Input matrix B in row-major format (bfloat16 elements)
 * @param output Output matrix C to store A*B result (bfloat16 elements)
 * @param M Number of rows in matrix A and output matrix C
 * @param N Number of columns in matrix B and output matrix C
 * @param K Number of columns in matrix A and rows in matrix B
 * @param mesh_device Target mesh device (1x1 or larger) for computation
 *
 * @note Matrix dimensions must be divisible by tile size (32x32) for this implementation
 * @note Uses circular buffers with 2 tiles for double-buffering to overlap compute and data movement
 */
void matmul_multi_core(
    std::vector<bfloat16>& a,
    std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    // Check if the configuration is valid - matrices must be divisible by tile dimensions
    TT_ASSERT(
        (M * N) % TILE_HW == 0,
        "Matrix dimensions M={} and N={} must be divisible by TILE_HW={} to use this matmul implementation",
        M,
        N,
        TILE_HW);

    uint32_t BM = 64;  // block M in elements
    uint32_t BN = 64;  // block N in elements
    uint32_t BK = 64;  // block K in elements

    TT_ASSERT(BM % TILE_HEIGHT == 0 && BN % TILE_WIDTH == 0 && BK % TILE_WIDTH == 0,
          "Block sizes must be tile-multiple (32).");

    // Set up mesh command queue, workload, device range, and program for multi-core execution
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program{};

    // Get the compute grid size to determine how many cores are available
    auto core_grid = mesh_device->compute_with_storage_grid_size();
    fmt::print("Compute with storage grid size: {}\n", core_grid);
    auto ceil_div = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };

    // Extracting Matrix dimensions from input/output vectors and converting to tile coordinates.
    // The accelerator works with 32x32 tiles, so we need to convert from element dimensions
    // to tile dimensions for proper addressing and computation.
    const uint32_t Mt = M / TILE_HEIGHT;  // Number of tiles in M dimension
    const uint32_t Kt = K / TILE_WIDTH;   // Number of tiles in K dimension
    const uint32_t Nt = N / TILE_WIDTH;   // Number of tiles in N dimension

    const uint32_t BMt = BM / TILE_HEIGHT; // tiles per block in M
    const uint32_t BNt = BN / TILE_WIDTH;  // tiles per block in N
    const uint32_t BKt = BK / TILE_WIDTH;  // tiles per block in K

    const uint32_t grid_m = ceil_div(Mt, BMt);
    const uint32_t grid_n = ceil_div(Nt, BNt);
    const uint32_t num_output_blocks_total = grid_m * grid_n;

    // Use the split_work_to_cores utility function to distribute matrix multiplication work
    // across available cores for efficient SPMD (Single Program, Multiple Data) execution.
    // This function takes the total number of output tiles and available cores, then calculates
    // how to divide the work when it cannot be evenly distributed. It returns two groups of cores:
    // - Primary group: handles more tiles per core
    // - Secondary group: handles fewer tiles per core
    // The secondary group is empty if the work can be evenly distributed across all cores. This
    // approach minimizes workload imbalance between cores for optimal performance.
#define MULTICAST
#ifdef MULTICAST
    uint32_t num_cores = 40;
    CoreCoord start_core = {0, 0};
    // CoreCoord core_range = {8, 5};
    uint32_t start_core_x = start_core.x;
    uint32_t start_core_y = start_core.y;
    uint32_t num_cores_x = 8;
    uint32_t num_cores_y = 5;
    // this defines the first rectangle. build up the core range by adding rectangles    
    std::vector<CoreRange> ranges;
    for (uint32_t i = 0; i < num_cores_x; i++) {
        ranges.emplace_back(
            CoreCoord((std::size_t)start_core_x + i, (std::size_t)start_core_y),
            CoreCoord((std::size_t)start_core_x + i, (std::size_t)start_core_y + num_cores_y - 1));
    }
    CoreRangeSet all_cores;  
    all_cores = all_cores.merge(ranges);
    auto core_group_1 = all_cores;
    CoreRangeSet core_group_2; // empty
    uint32_t work_per_core1 = num_output_blocks_total / num_cores;
    uint32_t work_per_core2 = 0;
#else
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] =
        split_work_to_cores(core_grid, num_output_blocks_total);
#endif 

    fmt::print(
        "Distributing {} output tiles across {} cores: {} cores ({}) x {} tiles/core + {} cores ({}) x {} tiles/core\n",
        num_output_blocks_total,
        num_cores,
        core_group_1.num_cores(),
        core_group_1.str(),
        work_per_core1,
        core_group_2.num_cores(),
        core_group_2.str(),
        work_per_core2);
    fmt::print("All cores: {}\n", all_cores.str());

    fmt::print(
        "Matrix A: {}x{} ({} tiles), Matrix B: {}x{} ({} tiles), Output C: {}x{} ({} tiles)\n",
        M,
        K,
        Mt * Kt,
        K,
        N,
        Kt * Nt,
        M,
        N,
        Mt * Nt);

    // Create DRAM buffers for input and output matrices (replicated per device across the mesh).
    // We allocate DRAM buffers for the input matrices and output matrix.
    // Setting page_size to single_tile_size is the most common configuration for memory buffers in Metalium
    // as it is generic, works for most cases and achieves good performance.
    // Writing data from input vectors to source buffers.
    constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;  // 2 * 32 * 32 = 2048 bytes

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};

    distributed::ReplicatedBufferConfig buffer_config_A{.size = single_tile_size * Mt * Kt};

    distributed::ReplicatedBufferConfig buffer_config_B{.size = single_tile_size * Nt * Kt};

    distributed::ReplicatedBufferConfig buffer_config_C{.size = single_tile_size * Mt * Nt};

    auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config_A, dram_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config_B, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config_C, dram_config, mesh_device.get());
    // Each handle is a mesh-wide replicated allocation; on a unit mesh this is a single device buffer

    // Configure Circular Buffers
    // Circular buffers act as staging areas for data movement between DRAM and compute units.
    // Using 2 tiles per circular buffer to allow for double buffering (data movement can be reading from one tile while
    // the compute kernel is using the other tile). This number can be adjusted based on the use case, but generally
    // diminishing returns are observed after several tiles.
    // input tiles count is = 2 so one tile can be read while the other is being processed
    const uint32_t A_tiles_per_block = BMt * BKt;
    const uint32_t B_tiles_per_block = BKt * BNt;
    const uint32_t C_tiles_per_block = BMt * BNt;
    fmt::print(
        "Circular Buffer Config: A tiles per block={}, B tiles per block={}, C tiles per block={}\n",
        A_tiles_per_block,
        B_tiles_per_block,
        C_tiles_per_block);

    const auto cb_data_format = tt::DataFormat::Float16_b;
    uint32_t cb_buffer_depth = 2; 
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,  // create on all cores
        CircularBufferConfig(A_tiles_per_block * cb_buffer_depth * single_tile_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, single_tile_size));

    tt_metal::CreateCircularBuffer(
        program,
        all_cores,  // create on all cores
        CircularBufferConfig(B_tiles_per_block * cb_buffer_depth * single_tile_size, {{CBIndex::c_1, cb_data_format}})
            .set_page_size(CBIndex::c_1, single_tile_size));

    tt_metal::CreateCircularBuffer(
        program,
        all_cores,  // create on all cores
        CircularBufferConfig(C_tiles_per_block * cb_buffer_depth * single_tile_size, {{CBIndex::c_16, cb_data_format}})
            .set_page_size(CBIndex::c_16, single_tile_size));

    // Create Kernels (Reader, Writer, Compute)
    // - Reader kernel: Handles reading input data from DRAM into circular buffers
    // - Writer kernel: Handles writing output data from circular buffers back to DRAM
    // - Compute kernel: Performs the actual matrix multiplication computation
    // All kernels run across all cores to enable parallel execution
    MathFidelity math_fidelity = MathFidelity::HiFi4;  // High fidelity math for accurate results
    
    auto in0_mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, 0);
    auto in0_mcast_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, 0);
    std::vector<uint32_t> compile_args = {static_cast<uint32_t>(CBIndex::c_0),
                                        static_cast<uint32_t>(CBIndex::c_1),
                                        static_cast<uint32_t>(CBIndex::c_16), in0_mcast_sender_semaphore_id, in0_mcast_receiver_semaphore_id};

    // Each generated kernel builds a chain of TensorAccessors (one per tensor
    // input) reading their interleaved-DRAM layout from compile-time args.
    // Append one block per buffer *after* the CB indices and the two multicast
    // semaphores, in function-input order (A, B, C) -- the same order as the
    // buffer addresses in common_args. The codegen reserves the 2 semaphore
    // slots (see TritonFuncOpToFuncOp), so the accessor chain starts at
    // compile-time-arg offset 5 (3 CB ports + 2 semaphores).
    TensorAccessorArgs(src0_dram_buffer).append_to(compile_args);  // A (MxK)
    TensorAccessorArgs(src1_dram_buffer).append_to(compile_args);  // B (KxN)
    TensorAccessorArgs(dst_dram_buffer).append_to(compile_args);   // C (MxN)

    auto reader_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul_tensor_descriptor/kernels/dataflow/reader.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = compile_args});


    auto writer_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul_tensor_descriptor/kernels/dataflow/writer.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = compile_args});

    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul_tensor_descriptor/kernels/compute/matmul_triton.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compile_args});

    // Set Runtime Arguments for Kernels
    // Each core needs to know which portion of the work it's responsible for. We are parallelizing across output
    // tiles - each core computes different output tiles. Runtime arguments can be changed between program executions
    // without recompilation.
    uint32_t work_offset = 0;
    auto work_groups = {std::make_pair(core_group_1, work_per_core1), std::make_pair(core_group_2, work_per_core2)};


    uint32_t stride_AM = K;
    uint32_t stride_BK = N;
    uint32_t stride_CM = N;
    fmt::print("Matrix strides: AM={}, BK={}, CM={}\n", stride_AM, stride_BK, stride_CM);

    // Set common runtime args per kernel
    auto common_args = std::vector<uint32_t>{(uint32_t)src0_dram_buffer->address(),  // Address of matrix A in DRAM
                        M, // shape[0]
                        K, // shape[1]
                        stride_AM, // strides[0]
                        1, // strides[1]
                        0, // padding (bool)
                        M, // shape[0] // repeated? 
                        K, // shape[1]
                        stride_AM, // strides[0]
                        1, // strides[1]
                        (uint32_t)src1_dram_buffer->address(),  // Address of matrix B in DRAM
                        K, // shape[0]
                        N, // shape[1]
                        stride_BK, // strides[0]
                        1, // strides[1]
                        0, // padding (bool)
                        K, // shape[0]
                        N, // shape[1]
                        stride_BK, // strides[0]
                        1, // strides[1]
                        (uint32_t)dst_dram_buffer->address(),
                        M, // shape[0]
                        N, // shape[1]
                        stride_CM, // strides[0]
                        1, // strides[1]
                        0, // padding (bool)
                        M, // shape[0]
                        N, // shape[1]
                        stride_CM, // strides[0]
                        1, // strides[1]
                        M,                           
                        N,                           
                        K};

    tt_metal::SetCommonRuntimeArgs(program, reader_id, common_args);
    tt_metal::SetCommonRuntimeArgs(program, writer_id, common_args);
    tt_metal::SetCommonRuntimeArgs(program, compute_kernel_id, common_args);

    auto packMulticastCoord = [](uint32_t x, uint32_t y) {
        return ((x & 0xFFFFu) << 16) | (y & 0xFFFFu);
    };

    // Iterate through each work group and assign work to cores
    for (const auto& [ranges, work_per_core] : work_groups) {
        for (const auto& range : ranges.ranges()) {
            for (const auto& core : range) {  
                uint32_t block_start = work_offset;
                uint32_t block_end = work_offset + work_per_core < num_output_blocks_total ? work_offset + work_per_core : num_output_blocks_total;
                
                fmt::print("Core {} assigned grid range ({}, {})\n", core.str(), block_start, block_end);
                // Set arguments for the reader kernel (data input)
                tt_metal::SetRuntimeArgs(
                    program,
                    reader_id,
                    core,
                    {                        
                        block_start,
                        block_end,
                        num_output_blocks_total,
                        packMulticastCoord(0, 0),
                        packMulticastCoord(7, 4)
                    });          

                // Set arguments for the writer kernel (data output)
                tt_metal::SetRuntimeArgs(
                    program, writer_id, core,             
                    {                         
                        block_start,
                        block_end,
                        num_output_blocks_total,
                        packMulticastCoord(0, 0),
                        packMulticastCoord(7, 4)
                    });  

                // Set arguments for the compute kernel
                tt_metal::SetRuntimeArgs(
                    program,
                    compute_kernel_id,
                    core,
                    {                   
                        block_start,
                        block_end,
                        num_output_blocks_total,
                        packMulticastCoord(0, 0),
                        packMulticastCoord(7, 4)
                    });                   
                work_offset += work_per_core;  // Update offset for next core
            }
        }
    }
    // Launch program & read in output buffer result into the host vector
    // 1. Upload input data to DRAM buffers
    // 2. Execute the program (all kernels run in parallel across cores)
    // 3. Read back the result from DRAM to host memory
    // The 'true' parameter in EnqueueReadMeshBuffer ensures we wait for completion (so when the function
    // returns, the output vector is fully populated).
    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, a, false);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, b, false);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    // Blocking read waits for completion before returning and resizes 'output' as needed
    distributed::EnqueueReadMeshBuffer(cq, output, dst_dram_buffer, true);
}

///////////////////////////////////////

float bf16_to_float(bfloat16 v) {
    return static_cast<float>(v);
}

std::string slice_to_string(const std::vector<bfloat16>& mat,
                            int start, int stop, int step, int line_break_interval) {
    std::ostringstream oss;
    int line_ctr = 0;
    int crt_line = 0;
    oss << crt_line << " | ";
    for (int r = start; r < stop; r += step) {
        oss << bf16_to_float(mat[r]);
        if (r + step < stop) oss << " ";
        if (++line_ctr == line_break_interval) {
            if (r + step >= stop) break;
            oss << "\n";
            oss << ++crt_line << " | ";
            line_ctr = 0;
        }   
    }
    oss << "\n";

    return oss.str();
}

int main(int argc, char* argv[]) {
    bool pass = true;

    try {
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        // Create source data with specified matrix dimensions
        // Matrix dimensions - use command line args if provided, otherwise use defaults
        uint32_t M = 64;  // Number of rows in matrix A (default)
        uint32_t N = 2560;  // Number of columns in matrix B (default)
        uint32_t K = 9728;   // Inner dimension for multiplication (default)
        
        if (argc >= 4) {
            M = std::atoi(argv[1]);
            N = std::atoi(argv[2]);
            K = std::atoi(argv[3]);
            fmt::print("Using user-specified dimensions: M={}, N={}, K={}\n", M, N, K);
        } else {
            fmt::print("Using default dimensions: M={}, N={}, K={}\n", M, N, K);
        }

        // Calculate matrix dimensions in tiles for the accelerator
        uint32_t Mt = M / TILE_HEIGHT;
        uint32_t Nt = N / TILE_WIDTH;

        // Calculate buffer sizes needed for each matrix in bytes
        constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;  // 2 * 32 * 32 = 2048 bytes
        uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt;  // num_tiles of FP16_B

        // Create random input vectors for matrices A and B
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        std::vector<bfloat16> src0_vec(M * K, 0);  // Matrix A (MxK)
        std::vector<bfloat16> src1_vec(K * N, 0);  // Matrix B (KxN)
        // // Fill with random bfloat16 values
        for (bfloat16& v : src0_vec) {
            v = bfloat16(dist(rng));
        }
        for (bfloat16& v : src1_vec) {
            v = bfloat16(dist(rng));
        }

        // Golden Matmul running on CPU (Float) - reference implementation for verification
        std::vector<bfloat16> golden_vec(M * N, 0);
        golden_matmul(src0_vec, src1_vec, golden_vec, M, N, K);

        // Input vector tilizing to match device expected tiled layout
        // The Tenstorrent hardware operates on data in 32x32 tiles rather than standard row-major format.
        // tilize_nfaces() converts the input matrices from row-major layout to the tiled layout expected by the device.
        // This transformation groups elements into 32x32 blocks and reorders them in memory so that each tile
        // (32x32 elements) is stored contiguously. This matches the native data access patterns of the matrix engine
        // and enables efficient operations on the accelerator.
        src0_vec = tilize_nfaces(src0_vec, M, K);
        src1_vec = tilize_nfaces(src1_vec, K, N);

#ifdef DEBUG_PRINTS
        fmt::print("A post-tilize:\n{}\n", slice_to_string(src0_vec, /*start=*/0, /*stop=*/M * K, /*step=*/16, /*line_break_every=*/K / 16));
        fmt::print("B post-tilize:\n{}\n", slice_to_string(src1_vec, /*start=*/0, /*stop=*/K * N, /*step=*/16, /*line_break_every=*/N / 16));
#endif 

        /* Calling the MatMul host program. Read in result into a host vector */
        std::vector<bfloat16> result_vec(dram_buffer_C_size / sizeof(bfloat16));
        matmul_multi_core(src0_vec, src1_vec, result_vec, M, N, K, mesh_device);
        // Reverse the tilization to get the result in the row-major format that the CPU expects
        result_vec = untilize_nfaces(result_vec, M, N);

        fmt::print("Output vector of size {}\n", result_vec.size());

        // Calculate the Pearson correlation coefficient (PCC) between the golden vector and the result vector
        // This is a measure of how similar the two vectors are.
        // A PCC close to 1 indicates that the two vectors are very similar.
        float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
        fmt::print("Metalium vs Golden -- PCC = {}\n", pearson);
         
        TT_FATAL(pearson > 0.97, "PCC not high enough. Result PCC: {}, Expected PCC: 0.97", pearson);

        pass &= mesh_device->close();

    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());

        throw;
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
