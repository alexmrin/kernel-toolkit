import torch
import triton
import triton.language as tl
from kernel_toolkit.core.metadata import TritonKernelInspector, DeviceProperties
import math

"""
    Implements flash attention 2 from https://arxiv.org/abs/2307.08691
"""

@triton.heuristics(
    values=dict(
        RCP_LN2=lambda _: math.log2(math.e),
    )
)
@triton.jit
def flash_attention_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    qb_stride: tl.constexpr, qn_stride: tl.constexpr, qd_stride: tl.constexpr,
    kb_stride: tl.constexpr, kn_stride: tl.constexpr, kd_stride: tl.constexpr,
    vb_stride: tl.constexpr, vn_stride: tl.constexpr, vd_stride: tl.constexpr,
    ob_stride: tl.constexpr, on_stride: tl.constexpr, od_stride: tl.constexpr,
    N: int,                         # sequence dimension of QKVO matrix
    softmax_scale: float,
    HEAD_DIM: tl.constexpr,         # hidden dimension of QKVO matrix
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    num_stages: tl.constexpr,
    causal: tl.constexpr,
    RCP_LN2: tl.constexpr,
):
    pid_h = tl.program_id(axis=0)
    pid_q = tl.program_id(axis=1)
    q_row_idx = pid_q * BLOCK_SIZE_Q
    PROGRAM_OFFSET_Q = pid_h * qb_stride
    PROGRAM_OFFSET_K = pid_h * kb_stride 
    PROGRAM_OFFSET_V = pid_h * vb_stride
    PROGRAM_OFFSET_O = pid_h * ob_stride
    
    q_tile_ptrs = tl.make_block_ptr(
        base=q_ptr + PROGRAM_OFFSET_Q,
        offsets=(q_row_idx, 0),
        shape=(N, HEAD_DIM),
        strides=(qn_stride, qd_stride),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0)
    )
    k_tile_ptrs = tl.make_block_ptr(
        base=k_ptr + PROGRAM_OFFSET_K,
        offsets=(0, 0),
        shape=(HEAD_DIM, N),
        strides=(kd_stride, kn_stride),
        block_shape=(HEAD_DIM, BLOCK_SIZE_K),
        order=(0, 1)
    )
    v_tile_ptrs = tl.make_block_ptr(
        base=v_ptr + PROGRAM_OFFSET_V,
        offsets=(0, 0),
        shape=(N, HEAD_DIM),
        strides=(vn_stride, vd_stride),
        block_shape=(BLOCK_SIZE_K, HEAD_DIM),
        order=(1, 0)
    )
    o_tile_ptrs = tl.make_block_ptr(
        base=o_ptr + PROGRAM_OFFSET_O,
        offsets=(q_row_idx, 0),
        shape=(N, HEAD_DIM),
        strides=(on_stride, od_stride),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0)
    )
    
    o_tile = tl.zeros(shape=(BLOCK_SIZE_Q, HEAD_DIM), dtype=tl.float32)
    l = tl.zeros(shape=(BLOCK_SIZE_Q,), dtype=tl.float32)
    m = tl.full(value=-float('inf'), shape=(BLOCK_SIZE_Q,), dtype=tl.float32)
    
    q_tile = tl.load(q_tile_ptrs, boundary_check=(0,))
    input_dtype = q_tile.dtype
    q_tile = (q_tile * (softmax_scale * RCP_LN2).to(input_dtype))
    
    for kv_tile in tl.range(0, tl.cdiv(N, BLOCK_SIZE_K), num_stages=num_stages):
        kv_start = kv_tile * BLOCK_SIZE_K
        last = kv_tile == (tl.cdiv(N, BLOCK_SIZE_K) - 1)
        
        if last:
            k_tile = tl.load(tl.advance(k_tile_ptrs, (0, kv_start)), boundary_check=(1,))
            v_tile = tl.load(tl.advance(v_tile_ptrs, (kv_start, 0)), boundary_check=(0,))
        else:
            k_tile = tl.load(tl.advance(k_tile_ptrs, (0, kv_start)))
            v_tile = tl.load(tl.advance(v_tile_ptrs, (kv_start, 0)))
            
        s_tile = tl.dot(q_tile, k_tile, out_dtype=tl.float32)

        if causal:
            offset_m = q_row_idx + tl.arange(0, BLOCK_SIZE_Q)
            offset_n = kv_start + tl.arange(0, BLOCK_SIZE_K)
            mask = offset_m[:, None] >= offset_n[None, :]
            s_tile = tl.where(mask, s_tile, -float('inf'))
            
        new_max = tl.maximum(m, tl.max(s_tile, axis=1))
        p_tile = tl.math.exp2(s_tile - new_max[:, None])
        
        alpha = tl.math.exp2(m - new_max)
        l = alpha * l + tl.sum(p_tile, axis=1)
        o_tile = o_tile * alpha[:, None]
        o_tile = tl.dot(p_tile.to(v_tile.dtype), v_tile, o_tile, out_dtype=tl.float32)

        m = new_max
        
    o_tile = (o_tile / l[:, None]).to(input_dtype)
    
    tl.store(o_tile_ptrs, o_tile, boundary_check=(0, ))
                
def flash_attention_fwd(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor,
    softmax_scale: float,
    *,
    causal: bool = False,
) -> torch.Tensor:
    assert q.shape == k.shape == v.shape, "QKV tensors must all have the same shape."
    assert q.ndim == 3, "QKV tensors must all have 3 dimensions."
    
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    o = torch.empty_like(q)
            
    B, N, D = q.shape
    
    num_warps = 4
    BLOCK_SIZE_Q = 128
    BLOCK_SIZE_K = 64
    num_stages = 1
    
    flash_attention_fwd_kernel[(B, triton.cdiv(N, BLOCK_SIZE_Q), 1)](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        N, softmax_scale, HEAD_DIM=D,
        BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_Q=BLOCK_SIZE_Q, 
        num_stages=num_stages, causal=causal, num_warps=num_warps
    )

    return o