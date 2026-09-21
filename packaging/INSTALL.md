# Installing python3-flagscale

After `apt install python3-flagscale` (Debian/Ubuntu) or `dnf install python3-flagscale` (Fedora),
install the ML runtime separately — it is intentionally **not** declared as a
hard Depends/Requires because the distro versions are CPU-only (torch) or
too old (triton) for GPU workloads.

FlagScale needs `hydra-core` at runtime (no distro package available):

```bash
pip install hydra-core
```

A `venv` is recommended to isolate pip-installed packages from the system
Python:

```bash
python3 -m venv ~/.venv/flagos
source ~/.venv/flagos/bin/activate
# then the pip install lines above
```

Note: the distro `python3-torch` package (CPU-only build) is intentionally
not pulled in — FlagOS workloads need a GPU build, which PyTorch upstream
distributes via PyPI (per-CUDA-version wheels), not as a `.deb` / `.rpm`.
