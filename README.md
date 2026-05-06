# Pneumonia Detection with SE-Augmented CNNs

An ablation study comparing ResNet50+SE, EfficientNetB0, and EfficientNetB0+SE on chest X-ray classification — trained with 3-way split, two-phase fine-tuning, and GradCAM explainability.

## Key Results

- **Best AUC-ROC**: 0.9630 (ResNet50+SE)
- **Highest Sensitivity**: 99.2% (ResNet50+SE) — critical for pneumonia detection
- **Best F1 Score**: 0.8866 (EfficientNetB0+SE)
- **Highest Accuracy**: 84.3% (EfficientNetB0+SE)

## Models Compared

1. **ResNet50 + SE** — Best by AUC, highest clinical sensitivity
2. **EfficientNetB0** — Baseline with built-in SE blocks
3. **EfficientNetB0+SE** — Ablation study (redundant SE stacking)

## Key Finding

Adding an extra SE block on top of EfficientNetB0's built-in SE yielded an AUC delta of −0.0063 — confirming that stacking SE on a model with internal channel attention provides **negligible additional gain**. ResNet50, which lacks native SE, benefits most from the added attention mechanism.

## Dataset

- **Source**: HuggingFace Chest X-Ray Pneumonia
- **Train/Val/Test Split**: 3-way stratified split

## Training Pipeline

- Phase 1: Frozen backbone, train head only (LR=1e-3)
- Phase 2: Unfreeze top 20 layers, fine-tune (LR=1e-5)
- Class weighting for imbalance handling
- GradCAM explainability for clinical validation

## Framework

- TensorFlow / Keras
- GradCAM for visual explainability
- Ablation study methodology per IEEE standards

---

For detailed results and visualizations, see `pneumonia_demo.html`.
