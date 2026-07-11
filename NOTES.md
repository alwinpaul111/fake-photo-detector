

## Summary

I implemented a small, model-free detector in `predict.py`. It treats screen recapture as an image-forensics problem rather than an object-recognition problem: the model looks for screen-grid / moire frequency peaks, axis-aligned display structure, chromatic sub-pixel aliasing, and display-like clipping or glare. The final output is a single score from `0` to `1`, where higher means "photo of a screen".

## Accuracy

No `real/` and `screen/` image folders were included with the provided starter package, so I am not claiming a measured validation accuracy from the submitted files alone. I included `evaluate.py` so the exact accuracy can be measured on any collected validation set:

```bash
python evaluate.py dataset
```

With more time, I would collect the requested ~50 real photos and ~50 recaptures, tune the cutoff on that set, and then keep a separate holdout set for the honest accuracy number. I would also split by capture scene and display device so the validation set does not accidentally contain near-duplicates.

## Latency

The detector is designed to be instant: it downsizes the long edge to 768 px and uses only NumPy/Pillow operations. On this local Darwin arm64 laptop runtime, a 20-run benchmark on a 1200x900 JPEG reported about `58 ms` median latency per image. Run `python benchmark.py image.jpg` on the target device for the exact deployment number.

## Cost Per Image

Cost per image is effectively `$0` because inference runs locally and does not use a cloud API. At scale, the main cost is device CPU/battery. If run on a server, the code is lightweight enough to batch many images per CPU core, so cost should be negligible compared with a neural-network API call

## Improvements

The biggest improvement would be calibration on real collected data. I would add a small validation harness that logs failure cases, tune the cutoff to the desired false-positive rate, and maintain a rolling evaluation set as cheaters adapt. If a pure heuristic plateaus, the next step would be a tiny mobile CNN trained on crops plus the current handcrafted features, distilled/quantized for on-device inference
