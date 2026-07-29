# Spine Disease Detection System

A deep learning system for detecting lumbar spine degenerative conditions from MRI scans, built on the [RSNA 2024 Lumbar Spine Degenerative Classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification) Kaggle dataset.

Deploy link:
https://bte7cckf8olaeo4gk7h9fe.streamlit.app/

## Overview

Given a cropped lumbar spine MRI slice centered on a specific region, the model predicts whether one of five degenerative conditions is present:

- Spinal Canal Stenosis
- Left / Right Neural Foraminal Narrowing
- Left / Right Subarticular Stenosis

## Model

- **Backbone:** EfficientNetV2S (ImageNet pretrained, fine-tuned)
- **Architecture:** Shared backbone with 5 independent binary classification heads (multi-task learning)
- **Input:** 224×224 cropped MRI slice, percentile-normalized
- **Training:** ~9,800 labeled crops from RSNA dataset, study-level train/val split (no patient leakage), class-balanced sample weighting, data augmentation (vertical flip, brightness/contrast)

## Results (Validation Set)

| Condition | Accuracy | Positive Recall |
|---|---|---|
| Spinal Canal Stenosis | 94% | 82% |
| Left Foraminal Narrowing | 95% | 94% |
| Right Foraminal Narrowing | 96% | 96% |
| Left Subarticular Stenosis | 95% | 96% |
| Right Subarticular Stenosis | 95% | 95% |

## Project Structure

```
Spine-disease-Detection-/
├── app.py                              # Streamlit web app
├── requirements.txt                    # Python dependencies
├── runtime.txt                         # Python version config
├── classification_v2_best.weights.h5   # Trained model weights (via Git LFS)
├── .gitattributes                      # Git LFS tracking config
└── README.md
```

## Running Locally

```bash
git clone <your-repo-url>
cd spine-detector-app
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Tech Stack

Python · TensorFlow/Keras · EfficientNetV2S · Streamlit · pydicom · OpenCV

## Disclaimer

This project is for academic and portfolio purposes only. It is **not** a certified clinical diagnostic tool and should not be used for real medical decision-making.

## Author

**Muhammad Adeel** — [LinkedIn](https://www.linkedin.com/in/muhammad-adeel-ml) · [GitHub](https://github.com/muhammadadeel0a1-commits)
