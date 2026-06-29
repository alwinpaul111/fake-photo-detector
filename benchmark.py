"""Measure predict.py latency on one image."""

from __future__ import annotations

import statistics
import sys
import time

from predict import predict


def main(image_path: str, runs: int = 25) -> None:
    predict(image_path)
    times = []
    for _ in range(runs):
        started = time.perf_counter()
        predict(image_path)
        times.append((time.perf_counter() - started) * 1000.0)
    print(f"runs: {runs}")
    print(f"median_ms: {statistics.median(times):.2f}")
    print(f"mean_ms: {statistics.mean(times):.2f}")
    print(f"p95_ms: {sorted(times)[int(0.95 * (len(times) - 1))]:.2f}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("Usage: python benchmark.py image.jpg [runs]")
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) == 3 else 25)
