from typing import Callable, Tuple, Any, Dict, Optional, Union
from dataclasses import dataclass, asdict
import torch
import triton

class DeviceProperties:
    _props: Optional[Dict[str, Any]] = None
    
    @classmethod
    def get(cls) -> Dict[str, Any]:
        if cls._props is None:
            device_id = torch.cuda.current_device()
            import triton.runtime.driver as driver
            raw_metadata = driver.active.utils.get_device_properties(device_id) #pyright: ignore
            props = torch.cuda.get_device_properties(device_id)
            major, minor = torch.cuda.get_device_capability(device_id)
            
            cc_map = {
                (7, 0): 32, (7, 5): 16,
                (8, 0): 32, (8, 6): 16, (8, 9): 24,
                (9, 0): 32,
            }
            
            cls._props = {
                "num_sm": props.multi_processor_count,
                "max_regs_per_sm": raw_metadata.get("max_num_regs"),
                "max_threads_per_sm": props.max_threads_per_multi_processor,
                "max_smem_per_sm": props.shared_memory_per_multiprocessor,
                "warp_size": 32,
                "capability": (major, minor),
                "max_blocks_per_sm": cc_map.get((major, minor), 16),
            }
        return cls._props
    
    @classmethod
    def reset(cls) -> None:
        """Reset cached properties (useful for multi-GPU)."""
        cls._props = None


@dataclass
class KernelStats:
    regs_per_thread: int
    shared_mem_bytes: int
    occupancy_pct: float
    max_blocks_reg: int
    max_blocks_smem: int
    max_blocks_threads: int
    active_blocks_per_sm: int
    num_programs: int
    
    def display(self) -> None:
        import pprint
        print(f"\n--- {self.__class__.__name__} ---")
        pprint.pprint(asdict(self), sort_dicts=False, indent=4)


class TritonKernelInspector:
    def __init__(self, kernel):
        self.kernel = kernel
        self._compiled: Optional[Any] = None
        self._num_warps: Optional[int] = None
        self._kernel_stats: Optional[KernelStats] = None
        self._device_meta = DeviceProperties.get()
        
    def warmup(self, *args, **kwargs) -> Any:
        if "num_warps" not in kwargs:
            raise RuntimeError("'num_warps' must be specified in warmup.")
        self._num_warps = kwargs["num_warps"]
        self._kernel_stats = None  # Reset stats on new warmup
        
        self._compiled = self.kernel.warmup(*args, **kwargs, grid=(1,))
        return self._compiled
    
    def get_stats(self) -> KernelStats:
        if self._compiled is None:
            raise RuntimeError("Call warmup() before get_stats().")
        
        if self._kernel_stats is not None:
            return self._kernel_stats
        
        meta = self._compiled.metadata
        regs_per_thread = getattr(self._compiled, "n_regs", getattr(meta, "num_regs", 0))
        shared_mem = getattr(meta, "shared", 0)
        
        threads_per_block = self._num_warps * self._device_meta["warp_size"]
        
        max_blocks_reg = (
            self._device_meta["max_regs_per_sm"] // (regs_per_thread * threads_per_block)
            if regs_per_thread > 0 else self._device_meta["max_blocks_per_sm"]
        )
        max_blocks_smem = (
            self._device_meta["max_smem_per_sm"] // shared_mem
            if shared_mem > 0 else self._device_meta["max_blocks_per_sm"]
        )
        max_blocks_threads = self._device_meta["max_threads_per_sm"] // threads_per_block

        active_blocks_per_sm = min(
            max_blocks_reg, 
            max_blocks_smem, 
            max_blocks_threads, 
            self._device_meta["max_blocks_per_sm"]
        )
        
        occupancy = (active_blocks_per_sm * threads_per_block) / self._device_meta["max_threads_per_sm"]
        num_programs = active_blocks_per_sm * self._device_meta["num_sm"]
        
        self._kernel_stats = KernelStats(
            regs_per_thread=regs_per_thread,
            shared_mem_bytes=shared_mem,
            occupancy_pct=occupancy * 100,
            max_blocks_reg=int(max_blocks_reg),
            max_blocks_smem=int(max_blocks_smem),
            max_blocks_threads=int(max_blocks_threads),
            active_blocks_per_sm=int(active_blocks_per_sm),
            num_programs=int(num_programs),
        )
        return self._kernel_stats
        
    def __getitem__(self, grid: Union[Tuple[int, ...], Callable]) -> Callable:
        if self._compiled is None:
            raise RuntimeError("Call warmup() before launching.")
        return self._compiled[grid]