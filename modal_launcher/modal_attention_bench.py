import sys
import modal

from modal_helper.helpers import get_run_id, ModalEnv

run_id = get_run_id("attention-bench")
app = modal.App(name=run_id)
env = ModalEnv(results_vol="results_vol")

@app.local_entrypoint()
def modal_run(causal: bool = False, dtype: str = "fp16", label: str = "attention_bench"):
    # Generate run ID
    remote_path = f"/{run_id}"

    sys.argv = [
        "run_bench.py", 
        "--save_dir", f"/results{remote_path}",
        "--dtype", dtype
    ]
    if causal:
        sys.argv.append("--causal")

    print(f"Dispatching to Modal: {run_id}")
    
    remote_benchmark_executor.remote(sys.argv)
    
    env.download_results(remote_subdirs=[remote_path])

@app.function(
    image=env.get_image(),
    volumes={"/results": env.get_results_vol()},
    gpu="A100"
)
def remote_benchmark_executor(mock_argv):
    from kernel_toolkit.bench.attention.attention_bench import main as bench_main
    import time
    
    sys.argv = mock_argv
    
    try:
        bench_main()
        print("Success!")
    finally:
        start_time = time.time()
        print(f"committing results into volume: {env.get_results_vol()}")
        env.get_results_vol().commit()
        print(f"Committed!. Time taken: {(time.time() - start_time):.2f}s")