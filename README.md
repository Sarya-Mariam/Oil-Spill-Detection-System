# 🌊 AI SpillGuard – Oil Spill Detection

## 📌 Project Overview

Oil spills pose a serious threat to marine ecosystems, coastal areas, and economies. This project leverages **deep learning (U-Net)** to automatically detect and segment oil spills in satellite imagery. The solution enables rapid intervention and supports environmental monitoring agencies.

## 🚀 Features

* Preprocessing pipeline for oil spill dataset (image + binary mask conversion).
* Data augmentation for imbalance handling (flips, rotations, brightness, scaling).
* U-Net model for binary segmentation (`0 = non-spill`, `1 = spill`).
* Training with combined **Binary Cross-Entropy + Dice Loss**.
* Evaluation metrics: Accuracy, Precision, Recall, IoU, Dice Coefficient.
* Visualizations: Ground truth vs. predicted masks + overlays.
* **Deployment via Streamlit app** for real-time inference.

## 📂 Project Structure

```
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
├── test/
│   ├── images/
│   └── masks/
├── Oil_Spill_M1_M2_M3.ipynb   # Notebook (Milestones 1–3)
├── app.py                      # Streamlit deployment (Milestone 4)
├── unet_oilspill_final.h5      # Trained model
└── README.md                   # Documentation
```

## 🧑‍💻 How to Run the Notebook

1. Open `Oil_Spill_M1_M2_M3.ipynb` in Jupyter/Colab.
2. Update `BASE_DIR` with your dataset path.
3. Run cells for preprocessing, training, and visualization.

## 🖥️ How to Run the Streamlit App

1. Install requirements:

```bash
pip install -r requirements.txt
```

Typical requirements:

```
tensorflow
opencv-python
streamlit
numpy
pillow
```

2. Run the app:

```bash
streamlit run app.py
```

3. Upload a satellite image and view predicted oil spill regions.

## 📊 Results

* Dice Coefficient: \~X.XX (replace with your value)
* IoU: \~X.XX
* Precision/Recall: reported from validation set.
* Example visualizations included in notebook.

## 📑 Milestones

* **Milestone 1**: Data collection, preprocessing.
* **Milestone 2**: Model development + training.
* **Milestone 3**: Visualization of results.
* **Milestone 4**: Deployment + documentation.

## 📌 Future Work

* Extend model to multi-class segmentation (different spill severity levels).
* Integrate real-time satellite data feeds.
* Add alerting system for emergency response teams.

---

👨‍💻 Developed as part of **AI SpillGuard Project** for oil spill monitoring and detection.
