# 🧠 Brain Tumor Detection using Deep Learning (MRI Classification)

A Convolutional Neural Network–based system that classifies brain MRI scans into four categories — **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor** — using Transfer Learning with MobileNetV2.

---

## 👩‍🎓 Author

**Shraddha Patel**
M.Sc, IIIT Lucknow

---

## 📌 Overview

This project builds an end-to-end pipeline for automated brain tumor detection from MRI images:

1. Downloads and verifies a public brain MRI dataset
2. Trains a MobileNetV2-based transfer learning model in two phases (frozen → fine-tuned)
3. Evaluates the model with per-class metrics to catch issues like class imbalance or collapse
4. Saves a deployable model + label mapping
5. Serves predictions through a simple **Streamlit web app**

---

## 🗂️ Project Structure

```
.
├── Brain_Tumor_Detection_v2.ipynb   # Main training notebook (use this one)
├── app.py                          # Streamlit app for inference
├── brain_tumor_model_final.keras   # Saved trained model (generated after running the notebook)
├── class_indices.json              # Label ↔ index mapping (generated after running the notebook)
└── README.md                       # This file
```

> `brain_tumor_model_final.keras` and `class_indices.json` are **not included in this repo** — they are generated when you run the notebook. Keep them in the same folder as `app.py` before launching the app.

---

## 📊 Dataset

**Brain Tumor MRI Dataset** (Kaggle, by Masoud Nickparvar)
🔗 https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

- ~7,023 MRI images across 4 classes: `glioma`, `meningioma`, `notumor`, `pituitary`
- Pre-split into `Training/` and `Testing/` folders
- Automatically downloaded inside the notebook using `kagglehub` — no manual download needed

---

## ⚙️ Installation

```bash
pip install tensorflow streamlit kagglehub scikit-learn seaborn matplotlib pillow numpy
```

> 💡 Training on CPU is slow. Use a free GPU runtime (Google Colab or Kaggle Notebooks) for much faster training.

---

## 🚀 How to Run

### 1. Train the Model

Open and run `Brain_Tumor_Detection_v2.ipynb` top to bottom (Colab/Kaggle/Jupyter). It will:

- Download and verify the dataset
- Train the model in two phases (frozen base → full fine-tuning)
- Evaluate it on the test set with a confusion matrix and per-class report
- Save `brain_tumor_model_final.keras` and `class_indices.json`

### 2. Launch the Web App

Place `brain_tumor_model_final.keras` and `class_indices.json` next to `app.py`, then:

```bash
streamlit run app.py
```

Upload an MRI image in the browser to get the predicted class and raw model probabilities.

---

## 🧠 Model Architecture

| Stage | Details |
|---|---|
| Base | MobileNetV2 (ImageNet pre-trained) |
| Head | GlobalAveragePooling2D → BatchNormalization → Dense(256, ReLU) → Dropout(0.4) → Dense(4, Softmax) |
| Phase 1 | Base frozen, train head only (15 epochs, LR `1e-3`) |
| Phase 2 | Entire base unfrozen, full fine-tune (25 epochs, LR `1e-5`) |
| Regularization | Data augmentation (rotation/zoom/shift/flip), Dropout, class weighting |
| Callbacks | `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau` |
| Input | 150×150 RGB images, normalized to `[0, 1]` |

---

## ✅ Evaluation

The notebook reports, on the held-out test set:

- Overall test accuracy & loss
- **Per-class** precision / recall / F1-score (via `classification_report`)
- Confusion matrix heatmap
- A predicted-vs-actual count table per class — used to verify the model isn't biased toward only 1–2 classes

---

## 🐞 Key Issues Found & Fixed During Development

This project went through real debugging — documented here for transparency and as a learning reference:

1. **Model never saved correctly** — an earlier version accidentally saved a stale/unrelated model object instead of the actually-trained one, so the deployed app was loading the wrong weights. Fixed by giving every model a unique variable name and explicitly saving right after training.
2. **Model collapse (only predicting 2 of 4 classes)** — caused by a frozen backbone with no data augmentation and very limited fine-tuning, which wasn't enough to learn subtle differences between tumor types. Fixed by adding image augmentation, class weighting, and fine-tuning the **entire** MobileNetV2 backbone with a low learning rate.
3. **Inference/training preprocessing mismatch** — the app must resize, RGB-convert, and normalize (`/255.0`) images in **exactly** the same way the training pipeline does, or predictions become unreliable.

---

## 🔮 Future Improvements

- Try alternative backbones (EfficientNetB0, ResNet50) for comparison
- Add Grad-CAM visualizations to highlight tumor regions the model focuses on
- Deploy the Streamlit app on a public host (Streamlit Community Cloud / Hugging Face Spaces)
- Expand the dataset with additional verified MRI sources for better generalization

---

## 🙏 Acknowledgments

- Dataset: [Masoud Nickparvar – Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle)
- Base model: MobileNetV2 (Keras Applications, ImageNet weights)

---

*Project by Shraddha Patel — M.Sc, IIIT Lucknow.*
