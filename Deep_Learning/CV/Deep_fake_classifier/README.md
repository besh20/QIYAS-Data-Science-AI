# Real vs. Manipulated Face Detector

A CNN-based classifier that detects digitally manipulated (spliced) face images, swapped eyes, nose, or mouth, using Error Level Analysis, transfer learning, and Grad-CAM explainability. Includes a Streamlit app for interactive and batch inference.

## Why this project

Face manipulation detection is directly relevant to platform trust & safety and identity verification. Unlike fully GAN-generated faces, the manipulations here are subtle, localized edits genuinely hard to catch by eye, which makes both the modeling problem and the debugging story more interesting than a typical classification tutorial.

## The approach (and what didn't work first)

This project went through three real iterations, not a straight line:

1. **Frozen MobileNetV2 on raw pixels** → ~50% accuracy (chance level). Generic ImageNet features don't capture the fine-grained blending artifacts this task depends on.
2. **Fine-tuned MobileNetV2 (last 30 layers) on raw pixels** → only marginally better (~51-58%), slow to converge.
3. **Switched input to Error Level Analysis (ELA) maps** : recompressing each image at a known JPEG quality and taking the pixel-wise difference from the original, which surfaces compression mismatches left by a splice — then fine-tuned on that. This got the model to **65% test accuracy**, a real, non-chance result for a genuinely hard forensic task.

## Results

- Test accuracy: **65%**
- FAKE: precision 0.60, recall 0.75
- REAL: precision 0.72, recall 0.56
- Accuracy by manipulation difficulty: easy 71.4%, mid 78.9%, hard 69.7%, real 55.8%

**Key finding:** the model is biased toward flagging images as fake it catches manipulations more reliably than it correctly recognizes real, unaltered faces. This makes sense given what it's actually reading: **compression-artifact evidence of editing, not visual realism.** Its errors don't line up neatly with what a human eye would find obvious or subtle.

**A note on reliability:** with only ~2,000 total images, results shifted meaningfully between training runs of the same setup a reminder to treat small-dataset results as a range, not a fixed number, and to say so honestly rather than cherry-picking the best run.

## Project structure

```
├── deepfake_classifier.ipynb   # full training pipeline, 10 phases: data → EDA → ELA →
│                                # augmentation → model → training → evaluation → Grad-CAM →
│                                # save model → findings & conclusion
├── app.py                      # Streamlit app: single-image detect, batch upload,
│                                # example gallery, model insights, about
├── requirements.txt
└── real_vs_fake_face_model.keras   # produced by running the notebook (not included in repo)
```

## Dataset

[Real and Fake Face Detection](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection) (CIP Lab, Yonsei University) 1,081 real faces, 960 manipulated faces (~215MB), manipulations graded by difficulty (`easy_`, `mid_`, `hard_` in filenames).

## Running it

**1. Train the model**
- Download the dataset above and extract it so you have `real_and_fake_face/training_real/` and `real_and_fake_face/training_fake/`
- Open `deepfake_classifier.ipynb` and run all cells top to bottom
- This saves `real_vs_fake_face_model.keras` at the end

**2. Run the app**
```bash
pip install -r requirements.txt
streamlit run app.py
```
Place `app.py`, the saved `.keras` model, and (optionally, for the example gallery) the `real_and_fake_face/` folder in the same directory.

## App features

- **Detect** : upload one image, see the original, its ELA map, a Grad-CAM overlay, and a verdict with confidence. Includes a face-detection guard (OpenCV Haar Cascade) so it won't produce a meaningless prediction on a faceless image.
- **Batch** : upload multiple images at once, get a results table (verdict + confidence per file), summary counts, and a CSV export.
- **Try Examples** : one-click preloaded easy/mid/hard fakes and real faces, for testing without your own image.
- **Model Insights** : the real evaluation numbers and the accuracy-by-difficulty breakdown, with an explanation of what the model is actually detecting.
- **About** : the full project story, including the debugging path and known limitations.

## Limitations & what I'd change for production

- Small dataset: unclear how well this generalizes to manipulation techniques not represented here (e.g. full GAN-generated faces, which this model was never trained to catch).
- ELA's usefulness depends on the JPEG quality parameter used for recompression (90 here) — untested how sensitive results are to that choice.
- The model is biased toward predicting FAKE  a tuned decision threshold (rather than the default 0.5) could reduce false positives on real faces.
- Face detection filters out obviously faceless images but isn't perfect at unusual angles/lighting.
- For real-time or video use: single-frame ELA + inference is fast enough per-frame, but the model should ship with a clear disclaimer about its actual detection basis so predictions aren't over-trusted.

## Stack

TensorFlow/Keras (MobileNetV2 transfer learning + fine-tuning), OpenCV, scikit-learn, Streamlit.
