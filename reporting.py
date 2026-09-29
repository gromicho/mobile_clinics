"""Record the environment and export full-precision teaching results."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
import subprocess

import psutil


def environment_snapshot(cfg, input_hashes):
    packages = {p: metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "gurobipy",
                "gurobi-machinelearning", "osmnx", "pandana", "geopandas", "rasterio", "contextily", "folium"]}
    try:
        import cpuinfo
        cpu = cpuinfo.get_cpu_info().get("brand_raw", platform.processor())
    except (ImportError, OSError):
        cpu = platform.processor() or "unavailable"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", None
    sources = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
               ["clinic_routing.py", "clinic_data.py", "build_notebook.py", "requirements.txt"]}
    return dict(utc=datetime.now(timezone.utc).isoformat(), git_commit=commit, working_tree_dirty=dirty,
                source_sha256=sources, input_sha256=input_hashes, cpu=cpu,
                physical_cores=psutil.cpu_count(logical=False), logical_cores=psutil.cpu_count(),
                ram_gib=psutil.virtual_memory().total / 2**30, os=platform.platform(),
                python=platform.python_version(), gurobi=packages["gurobipy"], packages=packages,
                threads=cfg.threads, numerical_library_threads_during_fit=1,
                background_load="uncontrolled and unmeasured", gpu="not used",
                timing="Wall time: fit, model build and Gurobi solve reported separately; input loading and plotting excluded from those timings.",
                cold_start="Cached geographic inputs; independent models, no imported incumbents or warm starts.",
                settings=asdict(cfg))


def save_results(directory, environment, cfg, frames):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    environment["settings"] = asdict(cfg)
    (directory / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf8")
    for name, frame in frames.items():
        frame.to_json(directory / f"{name}.json", orient="records", indent=2, double_precision=15)
