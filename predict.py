"""Screen recapture detector.

Usage:
    python predict.py some_image.jpg

Prints one number in [0, 1]:
    0 = real photo, 1 = photo of a screen / recapture.

The detector is deliberately small and model-free. It scores artifacts that are
common when a camera photographs an illuminated display: periodic screen-grid
energy, chromatic sub-pixel aliasing, regular horizontal/vertical line energy,
and display-like clipping/glare.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


MAX_EDGE = 768
FFT_SIZE = 512
EPS = 1e-8


def _resize_for_features(img: Image.Image, max_edge: int = MAX_EDGE) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_edge:
        return img
    scale = max_edge / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def _box_blur(a: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return a.copy()
    kernel = radius * 2 + 1
    mode = "reflect" if min(a.shape) > radius else "edge"
    padded = np.pad(a, ((radius, radius), (radius, radius)), mode=mode)
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    total = (
        integral[kernel:, kernel:]
        - integral[:-kernel, kernel:]
        - integral[kernel:, :-kernel]
        + integral[:-kernel, :-kernel]
    )
    return total / float(kernel * kernel)


def _moving_average_1d(a: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return a.copy()
    kernel = radius * 2 + 1
    padded = np.pad(a, (radius, radius), mode="reflect" if a.size > radius else "edge")
    integral = np.pad(padded, (1, 0), mode="constant").cumsum()
    return (integral[kernel:] - integral[:-kernel]) / float(kernel)


def _center_square(gray: np.ndarray, size: int = FFT_SIZE) -> np.ndarray:
    h, w = gray.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = gray[y0 : y0 + side, x0 : x0 + side]
    pil = Image.fromarray(np.uint8(np.clip(crop, 0, 1) * 255), mode="L")
    return np.asarray(pil.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0


def _robust_peak_ratio(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    baseline = np.median(values) + EPS
    peak = np.percentile(values, 99.7)
    return float(peak / baseline)


def _fft_screen_features(gray: np.ndarray) -> tuple[float, float]:
    """Return periodic peak and axis-aligned frequency evidence in [0, 1]."""
    patch = _center_square(gray)
    patch = patch - np.mean(patch)
    window = np.outer(np.hanning(patch.shape[0]), np.hanning(patch.shape[1])).astype(np.float32)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(patch * window)))
    log_spectrum = np.log1p(spectrum)

    h, w = log_spectrum.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    annulus = (rr > 18) & (rr < min(h, w) * 0.46)
    peak_ratio = _robust_peak_ratio(log_spectrum[annulus])
    periodic_peak = np.clip((peak_ratio - 2.8) / 3.8, 0.0, 1.0)

    axis_band = (((np.abs(yy - cy) <= 2) | (np.abs(xx - cx) <= 2)) & annulus)
    off_axis = annulus & ~axis_band
    axis_energy = np.mean(log_spectrum[axis_band]) / (np.mean(log_spectrum[off_axis]) + EPS)
    axis_score = np.clip((axis_energy - 1.05) / 0.55, 0.0, 1.0)
    return float(periodic_peak), float(axis_score)


def _line_regularity(gray: np.ndarray) -> float:
    """Score repeated row/column structure caused by screen pixels or scan lines."""
    patch = _center_square(gray)
    hp = patch - _box_blur(patch, 5)
    gx = np.abs(np.diff(hp, axis=1))
    gy = np.abs(np.diff(hp, axis=0))
    col_profile = gx.mean(axis=0)
    row_profile = gy.mean(axis=1)

    ratios = []
    for profile in (col_profile, row_profile):
        profile = profile - _moving_average_1d(profile, 9)
        power = np.abs(np.fft.rfft(profile))
        if power.size > 24:
            band = power[6 : power.size // 2]
            ratios.append(_robust_peak_ratio(band))
    if not ratios:
        return 0.0
    return float(np.clip((max(ratios) - 3.0) / 10.0, 0.0, 1.0))


def _chroma_alias_score(rgb: np.ndarray, gray: np.ndarray) -> float:
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    luma_hp = gray - _box_blur(gray, 2)
    rg_hp = (r - g) - _box_blur(r - g, 2)
    bg_hp = (b - g) - _box_blur(b - g, 2)
    chroma = (np.std(rg_hp) + np.std(bg_hp)) * 0.5
    luma = np.std(luma_hp) + EPS
    ratio = chroma / luma
    return float(np.clip((ratio - 0.34) / 0.58, 0.0, 1.0))


def _display_tone_score(rgb: np.ndarray) -> float:
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    saturation = (mx - mn) / (mx + EPS)

    clipped = np.mean((mx > 0.985) | (mn < 0.015))
    bright_low_sat = np.mean((mx > 0.90) & (saturation < 0.16))
    very_saturated = np.mean((saturation > 0.72) & (mx > 0.35))

    clipping_score = np.clip((clipped - 0.010) / 0.080, 0.0, 1.0)
    glare_score = np.clip((bright_low_sat - 0.006) / 0.050, 0.0, 1.0)
    saturation_score = np.clip((very_saturated - 0.055) / 0.180, 0.0, 1.0)
    return float(np.clip(0.45 * clipping_score + 0.35 * glare_score + 0.20 * saturation_score, 0.0, 1.0))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _features(image_path: str | Path) -> dict[str, float]:
    img = Image.open(image_path).convert("RGB")
    img = _resize_for_features(img)
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    gray = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

    fft_peak, axis_peak = _fft_screen_features(gray)
    return {
        "fft_peak": fft_peak,
        "axis_peak": axis_peak,
        "line_regularity": _line_regularity(gray),
        "chroma_alias": _chroma_alias_score(rgb, gray),
        "display_tone": _display_tone_score(rgb),
    }


def predict(image_path: str) -> float:
    f = _features(image_path)

    # Hand-tuned, conservative ensemble. The main signal is agreement between
    # periodic FFT peaks and regular line/axis structure; color and tone only
    # nudge the decision when periodic evidence is already present.
    grid_agreement = math.sqrt(f["fft_peak"]) * f["line_regularity"] * f["axis_peak"]
    periodic_support = f["fft_peak"] * max(f["axis_peak"], f["line_regularity"])
    chroma_support = f["chroma_alias"] * f["fft_peak"]
    raw = (
        -2.40
        + 5.20 * grid_agreement
        + 2.20 * periodic_support
        + 0.80 * chroma_support
        + 0.60 * f["display_tone"]
    )
    return float(np.clip(_sigmoid(raw), 0.0, 1.0))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python predict.py image.jpg")
    print(f"{predict(sys.argv[1]):.6f}")
