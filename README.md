# Facial Age Group Classification using Deep Learning

## Overview

This project presents a deep learning-based approach for **facial age group classification** using transfer learning with **ResNet-18**. The model classifies facial images into five predefined age groups and demonstrates the complete machine learning pipeline, including data preprocessing, model training, evaluation, and performance visualization.

The project was developed using **PyTorch** and showcases the application of computer vision techniques for image classification tasks.

---

## Features

* Transfer Learning with **ResNet-18**
* Image preprocessing and data augmentation
* Multi-class age group classification
* Training and validation pipeline using PyTorch
* Performance evaluation using multiple metrics
* Training Accuracy and Loss visualization
* Confusion Matrix and Classification Report
* Modular and reproducible implementation

---

## Age Groups

* 18 – 20
* 21 – 30
* 31 – 40
* 41 – 50
* 51 – 60

---

## Tech Stack

* Python
* PyTorch
* Torchvision
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Google Colab

---

## Dataset

The dataset consists of facial images grouped into five age categories.

> **Note:** The dataset (~1.2 GB) is not included in this repository due to GitHub's file size limitations.

### Dataset Structure

```
Split_DataSet/
│
├── train/
│   ├── 18-20
│   ├── 21-30
│   ├── 31-40
│   ├── 41-50
│   └── 51-60
│
└── test/
    ├── 18-20
    ├── 21-30
    ├── 31-40
    ├── 41-50
    └── 51-60
```

---

## Project Workflow

1. Load Dataset
2. Image Preprocessing
3. Data Augmentation
4. Transfer Learning using ResNet-18
5. Model Training
6. Model Evaluation
7. Performance Visualization
8. Model Saving

---

## Model Architecture

* Backbone: **ResNet-18 (Pretrained on ImageNet)**
* Dropout Layer (0.5)
* Fully Connected Classification Layer
* Output Classes: **5**

---

## Training Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Model         | ResNet-18        |
| Optimizer     | Adam             |
| Loss Function | CrossEntropyLoss |
| Epochs        | 10               |
| Batch Size    | 64               |
| Image Size    | 224 × 224        |

---

## Results

### Training Performance

| Metric                  | Value      |
| ----------------------- | ---------- |
| Final Training Accuracy | **85.31%** |
| Final Training Loss     | **0.3847** |

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

---

## Repository Structure

```
facial-age-group-classification/

├── notebook/
│   └── Age_Group_Detection.ipynb
│
├── images/
│   ├── training_accuracy_curve.png
│   ├── training_loss_curve.png
│   ├── confusion_matrix.png
│   └── sample_prediction.png
│
├── model/
│   └── best_model.pth
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/facial-age-group-classification.git
```

Move into the project directory:

```bash
cd facial-age-group-classification
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Run the Project

Open the notebook:

```bash
jupyter notebook
```

or run directly in **Google Colab**.

---

## Future Improvements

* Fine-tune deeper ResNet architectures (ResNet-50, EfficientNet)
* Hyperparameter optimization
* Real-time webcam age prediction
* Model deployment using Streamlit or Flask
* Improve class balance and dataset diversity

---

## Author

**Sumit Jha**

B.Tech, Engineering Physics
Delhi Technological University (DTU)

---

## Connect With Me

* **LinkedIn:** https://linkedin.com/in/your-linkedin
* **GitHub:** https://github.com/sumitjha-2004

---

## License

This project is licensed under the MIT License.
