# Chest X-ray Pneumonia Classification Using Transfer Learning and Grad-CAM

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Deep learning project for binary classification of chest X-ray images into **Pneumonia** and **Normal** classes using **DenseNet121 transfer learning** and **Grad-CAM explainability**.

---

## Overview

This project includes:

* transfer learning with DenseNet121
* image preprocessing and augmentation
* class imbalance handling
* staged fine-tuning
* Grad-CAM visualization
* evaluation using accuracy, precision, recall, F1-score, and AUC

---

## Dataset

Dataset: **Chest X-Ray Images (Pneumonia)** from Kaggle.

Classes:

* NORMAL
* PNEUMONIA

---

## Model

Architecture:

```text
DenseNet121 → GlobalAveragePooling → Dropout → Dense(128) → Sigmoid Output
```

---

## Results

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 89.58% |
| Precision | 90.73% |
| Recall    | 92.82% |
| F1-score  | 91.76% |
| AUC       | 0.9615 |

---

## Project Structure

```text
├── data/
├── notebooks/
├── models/
├── results/
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/chest-xray-pneumonia-classification.git
cd chest-xray-pneumonia-classification
pip install -r requirements.txt
```

---

## Features

* DenseNet121 transfer learning
* data augmentation
* class weighting
* Grad-CAM explainability
* ROC curve and confusion matrix evaluation

---

## Future Improvements

* EfficientNet comparison
* external validation
* web deployment
* multicenter datasets

---

## License

MIT License.
