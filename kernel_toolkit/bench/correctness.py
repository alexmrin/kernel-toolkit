from typing import Callable, Dict, Optional
import torch

"""
    Actually write this later
"""
def test_correctness(
    kernel: Callable[..., torch.Tensor],
    baseline: Callable[..., torch.Tensor],
    kernel_kwargs: Optional[Dict] = None,
    baseline_kwargs: Optional[Dict] = None,
    shapes: Optional[list[tuple[int, ...]]] = None,
    seed: int = 0,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    device = torch.cuda.current_device()
    torch.manual_seed(seed)
    
    shapes = shapes or [(1823, 781), (128, 256), (4096, 4096)]
    
    for shape in shapes:
        x = torch.randn(*shape, device=device)
        y_kernel = kernel(x, **(kernel_kwargs or {}))
        y_baseline = baseline(x, **(baseline_kwargs or {}))
        
        if torch.allclose(y_kernel, y_baseline, atol=atol, rtol=rtol):
            print(f"Passed shape={shape}")
        else:
            max_diff = (y_kernel - y_baseline).abs().max().item()
            raise AssertionError(f"Failed shape={shape}, max_diff={max_diff} for {kernel.__name__} kernel.")
    
    print(f"All {len(shapes)} correctness tests passed for {kernel.__name__} kernel!")