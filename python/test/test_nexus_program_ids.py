import os

os.environ["TRITON_CPU_USE_NEXUS"] = "1"

import torch
import triton
import triton.language as tl


def test_nexus_program_ids():
    # Force CPU backend driver, but through Nexus launch/load path.

    driver = triton.backends.backends["cpu"].driver()
    triton.runtime.driver.set_active(driver)

    device = driver.get_active_torch_device()

    @triton.jit
    def kernel(output, GRID_SIZE: tl.constexpr):
        x = tl.program_id(0)
        y = tl.program_id(1)
        z = tl.program_id(2)
        offset = x + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE
        addr = output + offset
        i = tl.load(addr) + 1
        tl.store(addr, i)

    grid_size = 5
    output = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.int32, device=device)
    assert torch.all(output == 0)

    grid = lambda _: (grid_size, grid_size, grid_size)
    kernel[grid](output, GRID_SIZE=grid_size)
    assert torch.all(output == 1)


if __name__ == "__main__":
    test_nexus_program_ids()
