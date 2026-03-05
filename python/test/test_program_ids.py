import torch
import triton
import triton.language as tl


# To verify that program IDs are correctly assigned, we create a 3D grid of
# blocks and have each block increment a single cell in an output tensor. If our
# lowering of program IDs is correct, each cell should be incremented exactly
# once and the final output should be a tensor of all ones.
def test_program_ids():
    # Force the CPU driver.
    driver = triton.backends.backends['cpu'].driver()
    triton.runtime.driver.set_active(driver)
    device = driver.get_active_torch_device()
    assert device.type == 'cpu'

    # Compile a Triton kernel that updates a cell in the `output` tensor. The
    # `output` tensor has the same dimensions as the block grid and tick off
    # each (x, y, z) cell by looking at the program ID.
    @triton.jit
    def kernel(output, GRID_SIZE: tl.constexpr):
        # Locate the cell in our output grid.
        x = tl.program_id(0)
        y = tl.program_id(1)
        z = tl.program_id(2)
        offset = x + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE
        addr = output + offset

        # Increment each cell of our output by 1 to verify each block is touched
        # exactly once.
        i = tl.load(addr) + 1
        tl.store(addr, i)

    # We divide the "problem" into a grid of 5x5x5 blocks; there is no input so
    # we do not worry about the number of items per block.
    GRID_SIZE = 5
    output = torch.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=torch.int32, device=device)
    assert torch.all(output == 0)

    grid = lambda _: (GRID_SIZE, GRID_SIZE, GRID_SIZE)
    kernel[grid](output, GRID_SIZE=GRID_SIZE)
    assert torch.all(output == 1)
