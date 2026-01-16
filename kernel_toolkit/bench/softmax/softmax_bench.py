import os
import argparse
import torch
import triton
from kernel_toolkit.bench.correctness import test_correctness
from kernel_toolkit.kernels.softmax.softmax_v1 import softmax_v1
from kernel_toolkit.kernels.softmax.softmax_v2 import softmax_v2

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.path.dirname(__file__), "results"))
    return parser


def bench_throughput(save_path: str) -> None:
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],
            x_vals=[128 * i for i in range(2, 100)],
            line_arg='provider',
            line_vals=['triton_v1', 'triton_v2', 'torch'],
            line_names=["Triton_v1", 'Triton_v2', "Torch"],
            styles=[('blue', '-'), ('red', '-'), ('green', '-')],
            ylabel="GB/s",
            plot_name="softmax-performance",
            args={'M': 4096},
        )
    )
    def benchmark(M: int, N: int, provider: str) -> float:
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        
        if provider == 'torch':
            ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=-1), quantiles=[0.5, 0.2, 0.8])
        elif provider == 'triton_v1':
            ms = triton.testing.do_bench(lambda: softmax_v1(x), quantiles=[0.5, 0.2, 0.8])
        elif provider == 'triton_v2':
            ms =  triton.testing.do_bench(lambda: softmax_v2(x), quantiles=[0.5, 0.2, 0.8])
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        median_ms = ms[0] #pyright: ignore
        gbps = 2 * x.numel() * x.element_size() * 1e-9 / (median_ms * 1e-3)
        return gbps
    
    out_dir = os.path.join(save_path, "throughput")
    os.makedirs(out_dir, exist_ok=True)
    benchmark.run(show_plots=False, print_data=True, save_path=out_dir)

def main():
    parser = build_parser()
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    test_correctness(
        kernel=softmax_v1,
        baseline=torch.softmax,
        baseline_kwargs={"dim": -1}
    )
    
    test_correctness(
        kernel=softmax_v2,
        baseline=torch.softmax,
        baseline_kwargs={"dim": -1}
    )
    
    bench_throughput(save_path=args.save_dir)
    
if __name__ == "__main__":
    main()