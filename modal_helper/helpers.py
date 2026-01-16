import os
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm
import modal
                    
def get_run_id(label: str):
    import datetime
    import subprocess
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except:
        git_hash = "no-git"
    return f"{ts}_{git_hash}_{label}"


class ModalEnv:
    def __init__(
        self,
        results_vol = "results_vol",
    ):
        self._root = Path(__file__).parent.parent.resolve()
        self._results_vol_name = results_vol
        self._results_vol = modal.Volume.from_name(results_vol, create_if_missing=True)
        
    def get_image(self, extra_pip: Optional[List[str]] = None) -> modal.Image:
        img = modal.Image.debian_slim().pip_install_from_pyproject(
            pyproject_toml=str(self._root / "pyproject.toml")
        )
        if extra_pip:
            img.pip_install(*extra_pip)
            
        return img.add_local_python_source("kernel_toolkit").add_local_python_source("modal_helper")
    
    def get_results_vol(self) -> modal.Volume:
        return self._results_vol
    
    def get_root(self) -> Path:
        return self._root

    def download_results(
        self,
        local_dir: str = "local_results",
        remote_subdirs: Optional[List[str]] = None,
    ) -> None:
        """Downloads files from the cloud volume to your local disk."""
        print(f"Syncing results to {local_dir}...")
        
        local_path = self._root / local_dir
        local_path.mkdir(exist_ok=True)
                
        # Define which remote paths to sync
        paths_to_sync = remote_subdirs if remote_subdirs else ["/"]
        files_to_download = []
        
        for root_path in paths_to_sync:
            print(f"Listing '{root_path}' recursive=True...")
            entries = list(self._results_vol.listdir(root_path, recursive=True))
            print(f"Found {len(entries)} total entries.")
            for entry in entries:
                is_file = (
                        entry.type == "file" or 
                        str(entry.type).lower() == "file" or
                        getattr(entry.type, 'name', '').lower() == 'file' or
                        (hasattr(entry, 'is_dir') and not entry.is_dir()) #pyright: ignore
                    )
                if is_file:
                    files_to_download.append(entry)
        
        if not files_to_download:
            print("Done. No new files found.")
            return
            
        for entry in tqdm(files_to_download, desc="Downloading...", unit="file"):
            dest = local_path / entry.path.lstrip("/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            with open(dest, "wb") as f:
                for chunk in self._results_vol.read_file(entry.path):
                    f.write(chunk)
                            
        print("Results downloaded.")