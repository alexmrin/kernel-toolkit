from typing import Callable, Dict, Optional, Union, List, Tuple
import torch

def test_correctness(
    kernel: Callable[..., torch.Tensor],
    baseline: Callable[..., torch.Tensor],
    kernel_kwargs: Optional[Dict] = None,
    baseline_kwargs: Optional[Dict] = None,
    shapes: Optional[List[Tuple[int, ...]]] = None,
    num_inputs: int = 1,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    """
    Test kernel correctness against a baseline implementation.
    
    Args:
        kernel: Kernel function to test
        baseline: Reference implementation
        kernel_kwargs: Additional kwargs for kernel
        baseline_kwargs: Additional kwargs for baseline
        shapes: List of input shapes to test
        num_inputs: Number of input tensors (1 for softmax, 3 for attention)
        dtype: Data type for input tensors
        seed: Random seed
        atol: Absolute tolerance
        rtol: Relative tolerance
    """
    device = torch.cuda.current_device()
    torch.manual_seed(seed)
    
    shapes = shapes or [(1823, 781), (128, 256), (4096, 4096)]
    kernel_kwargs = kernel_kwargs or {}
    baseline_kwargs = baseline_kwargs or {}
    
    for shape in shapes:
        # Generate inputs
        inputs = [torch.randn(*shape, device=device, dtype=dtype) for _ in range(num_inputs)]
        
        # Run both implementations
        if num_inputs == 1:
            y_kernel = kernel(inputs[0], **kernel_kwargs)
            y_baseline = baseline(inputs[0], **baseline_kwargs)
        else:
            y_kernel = kernel(*inputs, **kernel_kwargs)
            y_baseline = baseline(*inputs, **baseline_kwargs)
        
        if torch.allclose(y_kernel, y_baseline, atol=atol, rtol=rtol):
            print(f"✓ Passed shape={shape}")
        else:
            max_diff = (y_kernel - y_baseline).abs().max().item()
            raise AssertionError(
                f"✗ Failed shape={shape}, max_diff={max_diff} for {kernel.__name__} kernel."
            )
    
    print(f"All {len(shapes)} correctness tests passed for {kernel.__name__} kernel!\n")