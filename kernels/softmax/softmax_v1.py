import torch
import triton
import triton.language as tl
from core.metadata import TritonKernelInspector, DeviceProperties


@triton.jit
def softmax_kernel_v1(
    x_ptr,
    input_row_stride: int,
    output_row_stride: int,
    output_ptr,
    n_rows: int,
    n_cols: int,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    
    for i in tl.range(pid, n_rows, num_programs, num_stages=num_stages): #pyright: ignore
        input_ptr_start = x_ptr + i * input_row_stride
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        input_ptrs = input_ptr_start + offsets
        inputs = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        inputs_minus_max = inputs - tl.max(inputs, axis=0)
        
        numerator = tl.exp(inputs_minus_max)
        denominator = tl.sum(numerator, axis=0)
        output = numerator / denominator
        
        output_ptr_start = output_ptr + i * output_row_stride
        output_ptrs = output_ptr_start + offsets
        tl.store(output_ptrs, output, mask=mask)


_inspector = TritonKernelInspector(softmax_kernel_v1)


def softmax_v1(x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    x = x.contiguous()
    y = torch.empty_like(x)
    
    n_rows, n_cols = x.shape
    num_warps = 8
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_stages = 4 if DeviceProperties.get()["max_smem_per_sm"] > 200_000 else 2 
    
    compiled = _inspector.warmup(
        x, x.stride(0), y.stride(0), y, n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages, num_warps=num_warps
    )
    
    stats = _inspector.get_stats()
    if verbose:
        stats.display()
    
    num_programs = min(stats.num_programs, n_rows)
    compiled[(num_programs, 1, 1)](
        x, x.stride(0), y.stride(0), y, n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages
    )
    
    return y