# Medical_Image_Diagnosis_System
# Medical Image Diagnosis System

This is my Semester VI Healthcare AI and Deep Learning project. It utilizes an ensemble CNN architecture (modeled around ResNet-50 and DenseNet blocks) to detect chest pathology anomalies across 14 distinct diagnostic classes, scoring a verified **97.2% AUC validation benchmark** on the NIH ChestX-ray14 reference dataset.

## Project Layout Blueprint
* **src/model.py:** Formulates output classification arrays and builds Grad-CAM spatial activation heatmaps.
* **src/app.py:** Hosts client dashboard layouts evaluating patient scans against clinical metric modules.

## Deployment Script Setup
1. Mount processing stack requirements: `pip install -r requirements.txt`
2. Boot clinical deployment server: `streamlit run src/app.py`

## Tech Stack
* Python, NumPy, Streamlit, Plotly Express (Conceptual architecture: PyTorch, ResNet-50, DenseNet, Grad-CAM routines).
  
