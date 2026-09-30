import json
import time
from pathlib import Path

import psutil
import torch
from transformers import TrainerCallback


class ResourceCallback(TrainerCallback):
    def __init__(self, destination: Path):
        self.destination, self.started_at = destination, time.perf_counter()
        self.process, self.peaks = psutil.Process(), {}

    def sample(self, **extra):
        memory = {"rss_gb": self.process.memory_info().rss / 1024**3}
        if torch.backends.mps.is_available():
            memory.update(mps_allocated_gb=torch.mps.current_allocated_memory() / 1024**3,
                          mps_driver_gb=torch.mps.driver_allocated_memory() / 1024**3)
        for key, value in memory.items():
            self.peaks[key] = max(self.peaks.get(key, 0.0), value)
        record = {"elapsed_seconds": round(time.perf_counter() - self.started_at, 3),
                  **{key: round(value, 3) for key, value in memory.items()}, **extra}
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        with self.destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.sample(phase="trainer", step=state.global_step, **logs)

