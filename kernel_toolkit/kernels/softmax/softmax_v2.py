import torch
import triton
import triton.language as tl
from kernel_toolkit.core.metadata import TritonKernelInspector, DeviceProperties

""" 
    Online softmax with tiles instead of just rows.
"""
@triton.jit
def softmax_kernel_v2(
    x_ptr,
    input_row_stride: int,
    output_row_stride: int,
    output_ptr,
    n_rows: int,
    n_cols: int,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    num_stages: tl.constexpr
):
    pid_m = tl.program_id(axis=0)

    row_start = pid_m * BLOCK_SIZE_M
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)[:, None]
    row_mask = rows < n_rows
    
    m = tl.full(value=-float('inf'), shape=(BLOCK_SIZE_M,), dtype=tl.float32)
    l = tl.zeros_like(m)
    
    for column_block in tl.range(0, n_cols, BLOCK_SIZE_N, num_stages=num_stages):
        cols = column_block + tl.arange(0, BLOCK_SIZE_N)[None, :]
        col_mask = cols < n_cols
        mask = row_mask & col_mask
        input_ptrs = x_ptr + rows * input_row_stride + cols
        inputs = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        
        new_max = tl.maximum(tl.max(inputs, axis=1), m)
        block_denom = tl.sum(tl.exp(inputs - new_max[:, None]), axis=1)
        l = l * tl.exp(m - new_max) + block_denom
        m = new_max
        
    for column_block in tl.range(0, n_cols, BLOCK_SIZE_N, num_stages=num_stages):
        cols = column_block + tl.arange(0, BLOCK_SIZE_N)[None, :]
        col_mask = cols < n_cols
        mask = row_mask & col_mask
        input_ptrs = x_ptr + rows * input_row_stride + cols
        inputs = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        
        output_ptrs = output_ptr + rows * output_row_stride + cols
        outputs = tl.exp(inputs - m[:, None]) / l[:, None]
        tl.store(output_ptrs, outputs, mask=mask)
    
_inspector = TritonKernelInspector(softmax_kernel_v2)


def softmax_v2(x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    x = x.contiguous()
    y = torch.empty_like(x)
    
    n_rows, n_cols = x.shape
    num_warps = 8
    BLOCK_SIZE_M = triton.next_power_of_2(n_cols // 8)
    BLOCK_SIZE_N = triton.next_power_of_2(n_rows // 8)
    num_stages = 4 if DeviceProperties.get()["max_smem_per_sm"] > 200_000 else 2 
    
    compiled = _inspector.warmup(
        x, x.stride(0), y.stride(0), y, n_rows, n_cols,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, num_stages=num_stages, num_warps=num_warps
    )
    
    stats = _inspector.get_stats()
    if verbose:
        stats.display()
    
    num_programs = min(stats.num_programs, tl.ceil(n_rows / BLOCK_SIZE_M))
    compiled[(num_programs, 1, 1)](
        x, x.stride(0), y.stride(0), y, n_rows, n_cols, BLOCK_SIZE_M, BLOCK_SIZE_N, num_stages
    )
    
    return y