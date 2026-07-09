# Spot the Fake Photo

This repository contains a small, fast detector for the take-home assignment:

```bash
python predict.py image.jpg
```

It prints a single score from `0` to `1`, where `1` means the image is likely a photo of a screen / recapture

## Approach

I used a deterministic computer-vision detector instead of a trained neural model. A screen recapture usually contains artifacts that are rare in a direct camera photo

- periodic frequency peaks from the display pixel grid or moire patterns
- horizontal/vertical frequency energy from screen scan/grid structure
- chromatic aliasing from RGB sub-pixels interacting with the camera sensor
- display-like highlights , clipping , and saturated backlit color

`predict.py` extracts these features with Pillow + NumPy, combines them with a small hand-tuned logistic score, and returns the resulting probability-like value. The implementation resizes the image to a maximum long edge of 768 px, so runtime is mostly constant across large input files.

## Files

- `predict.py` - required one-line predictor
- `evaluate.py` - optional evaluator for `real/` and `screen/` folders
- `benchmark.py` - optional latency measurement script
- `requirements.txt` - minimal dependencies
- `NOTES.md` - short assignment note

## Evaluate

Create a dataset like this:-

```text
dataset/
  real/
    image1.jpg
  screen/
    image2.jpg
```

Then run:

```bash
python evaluate.py dataset
```

The evaluator reports accuracy, confusion counts, and latency. The default cutoff is `0.5`; pass another cutoff as the second argument if a validation set suggests a better operating point.

## Cost

The detector runs locally and does not call any API  On-device cost is effectively `$0` per image, apart from normal device CPU/battery use.
