from __future__ import annotations

import platform
from importlib.metadata import version

print(f"Python architecture: {platform.machine()}")
print(f"NeMo Curator package installed: {version('nemo-curator')}")
print(f"Data Designer: {version('data-designer')}")
try:
    import nemo_curator  # noqa: F401
except ValueError as exc:
    if "only supports Linux" not in str(exc):
        raise
    print("NeMo Curator runtime: unavailable on macOS (upstream Linux-only guard)")
else:
    print("NeMo Curator runtime: available")
print("Data Designer remote-inference environment: installed (no CUDA required).")
