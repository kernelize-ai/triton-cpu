import pytest
import torch
import triton
import triton.language as tl

pytest.importorskip("nexus")


@pytest.fixture
def nexus_cpu_driver(monkeypatch):
    old_active = triton.runtime.driver._active
    old_default = triton.runtime.driver._default
    monkeypatch.setenv("TRITON_CPU_USE_NEXUS", "1")
    driver = triton.backends.backends["cpu"].driver()
    triton.runtime.driver.set_active(driver)
    try:
        yield driver
    finally:
        triton.runtime.driver._active = old_active
        triton.runtime.driver._default = old_default


def test_nexus_program_ids(nexus_cpu_driver):
    # Force CPU backend driver, but through Nexus launch/load path.
    assert type(nexus_cpu_driver.utils).__name__ == "NexusCpuUtils"
    assert nexus_cpu_driver.launcher_cls.__name__ == "NexusCPULauncher"

    device = nexus_cpu_driver.get_active_torch_device()

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
