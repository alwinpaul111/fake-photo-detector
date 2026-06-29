"""Evaluate predict.py against real/ and screen/ folders.

Expected layout:
    dataset/
      real/*.jpg
      screen/*.jpg

Usage:
    python evaluate.py dataset
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

from predict import predict


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic"}


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def main(dataset_dir: str, threshold: float = 0.5) -> None:
    root = Path(dataset_dir)
    samples = [(p, 0) for p in _images(root / "real")]
    samples += [(p, 1) for p in _images(root / "screen")]
    if not samples:
        raise SystemExit("No images found. Expected dataset/real and dataset/screen.")

    correct = 0
    latencies_ms = []
    tp = fp = tn = fn = 0

    for path, label in samples:
        started = time.perf_counter()
        score = predict(str(path))
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        pred = int(score >= threshold)
        correct += int(pred == label)
        tp += int(pred == 1 and label == 1)
        fp += int(pred == 1 and label == 0)
        tn += int(pred == 0 and label == 0)
        fn += int(pred == 0 and label == 1)
        print(f"{score:.4f}\tlabel={label}\tpred={pred}\t{path}")

    n = len(samples)
    print()
    print(f"images: {n}")
    print(f"threshold: {threshold:.2f}")
    print(f"accuracy: {correct / n:.3%}")
    print(f"tp={tp} fp={fp} tn={tn} fn={fn}")
    print(f"median_latency_ms: {statistics.median(latencies_ms):.2f}")
    print(f"mean_latency_ms: {statistics.mean(latencies_ms):.2f}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("Usage: python evaluate.py dataset_dir [threshold]")
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) == 3 else 0.5)
