import os
import math
import argparse

import torch
import torch.nn.functional as F
import triton
import triton.testing

from kernel_toolkit.bench.correctness import test_correctness
from kernel_toolkit.kernels.attention.flash_attention2 import flash_attention_fwd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", type=str, default=os.path.join(os.path.dirname(__file__), "results"))
    p.add_argument("--causal", action="store_true", default=False)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    return p


def torch_attention_naive(q, k, v, causal: bool, softmax_scale: float):
    scores = (q @ k.transpose(-2, -1)) * softmax_scale
    if causal:
        n = scores.shape[-1]
        mask = torch.triu(torch.ones((n, n), device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    p = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return p @ v


def torch_sdpa(q, k, v, causal: bool, softmax_scale: float):
    """Wrapper for PyTorch's scaled_dot_product_attention.
    
    SDPA expects shape (B, H, N, D) or (B*H, N, D) with is_causal flag.
    Our inputs are (B*H, N, D), so we can use them directly.
    """
    return F.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
        scale=softmax_scale,
    )


def bench_attention(save_path: str, causal: bool, dtype: torch.dtype) -> None:
    HEAD_DIM = 64
    NUM_HEADS = 12
    B = 2

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=[128 * i for i in range(1, 48)],
            x_log=False,
            line_arg="provider",
            line_vals=["triton_fa", "torch_sdpa", "torch_naive"],
            line_names=["Triton_FA", "Torch_SDPA", "Torch_Naive"],
            styles=[("blue", "-"), ("green", "-"), ("purple", "-")],
            ylabel="TFLOPS",
            plot_name=f"flash-attention-fwd-D{HEAD_DIM}-H{NUM_HEADS}-B{B}-causal={causal}",
            args={"B": B, "HEAD_DIM": HEAD_DIM, "NUM_HEADS": NUM_HEADS, "causal": causal},
        )
    )
    def benchmark(B: int, HEAD_DIM: int, NUM_HEADS: int, N_CTX: int, causal: bool, provider: str) -> float:
        device = "cuda"
        q = torch.randn((B * NUM_HEADS, N_CTX, HEAD_DIM), device=device, dtype=dtype)
        k = torch.randn((B * NUM_HEADS, N_CTX, HEAD_DIM), device=device, dtype=dtype)
        v = torch.randn((B * NUM_HEADS, N_CTX, HEAD_DIM), device=device, dtype=dtype)
        softmax_scale = 1.0 / math.sqrt(HEAD_DIM)

        if provider == "torch_naive":
            fn = lambda: torch_attention_naive(q, k, v, causal=causal, softmax_scale=softmax_scale)
        elif provider == "torch_sdpa":
            fn = lambda: torch_sdpa(q, k, v, causal=causal, softmax_scale=softmax_scale)
        elif provider == "triton_fa":
            fn = lambda: flash_attention_fwd(q, k, v, softmax_scale=softmax_scale, causal=causal)
        else:
            raise ValueError(provider)

        ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
        median_ms = ms[0]  # pyright: ignore
        flops = 4.0 * B * NUM_HEADS * N_CTX * N_CTX * HEAD_DIM
        if causal:
            flops *= 0.5
        tflops = flops * 1e-12 / (median_ms * 1e-3)
        return tflops

    out_dir = os.path.join(save_path, "attention_fwd")
    os.makedirs(out_dir, exist_ok=True)
    benchmark.run(show_plots=False, print_data=True, save_path=out_dir)


def main():
    args = build_parser().parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    match args.dtype:
        case "fp32":
            dtype = torch.float32
        case "fp16":
            dtype = torch.float16
        case "bf16":
            dtype = torch.bfloat16
        case _:
            raise NotImplementedError("Dtype not found.")

    softmax_scale = 1.0 / math.sqrt(64)  # HEAD_DIM = 64

    # Test correctness first (uses power-of-2 head dims)
    test_correctness(
        kernel=flash_attention_fwd,
        baseline=torch_attention_naive,
        kernel_kwargs={"softmax_scale": softmax_scale, "causal": args.causal},
        baseline_kwargs={"softmax_scale": softmax_scale, "causal": args.causal},
        shapes=[
            (2, 128, 64),  # (B, N, D)
            (4, 256, 64),
            (2, 512, 128),
            (1, 1024, 64),
        ],
        num_inputs=3,  # q, k, v
        dtype=dtype,
        atol=1e-2,
        rtol=1e-2,
    )

    bench_attention(save_path=args.save_dir, causal=args.causal, dtype=dtype)
    
if __name__ == "__main__":
    main()