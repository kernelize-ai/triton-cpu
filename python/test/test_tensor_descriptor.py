import pytest
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
def test_host_tensor_descriptor(dtype):
    driver = triton.backends.backends['cpu'].driver()
    triton.runtime.driver.set_active(driver)
    device = driver.get_active_torch_device()

    @triton.jit
    def kernel(in_desc, out_desc, BLOCK_M: tl.constexpr):
        off = tl.program_id(0) * BLOCK_M
        out_desc.store([off, 0], in_desc.load([off, 0]) + 1)

    # M is not a multiple of BLOCK_M, so the last block is partly out of bounds.
    M, N, BLOCK_M = 70, 16, 32
    x = torch.arange(M * N, device=device).reshape(M, N).to(dtype)
    y = torch.zeros_like(x)
    in_desc = TensorDescriptor.from_tensor(x, [BLOCK_M, N])
    out_desc = TensorDescriptor.from_tensor(y, [BLOCK_M, N])
    kernel[(triton.cdiv(M, BLOCK_M), )](in_desc, out_desc, BLOCK_M)
    assert torch.equal(y, x + 1)
