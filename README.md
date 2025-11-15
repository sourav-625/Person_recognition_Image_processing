# Real-Time Face Recognition Pipeline

A complete end-to-end **Face Recognition System** that trains on your own images, predicts new faces, and performs **real-time recognition via webcam**—both on your **local machine** and **Jupyter notebook environment**.

---

## 🚀 Features

- **Face Embeddings using FaceNet**
  - Pre-trained FaceNet (`keras-facenet`)
  - Generates 128-D embedding vectors
- **Trainable Classifier**
  - SVM / Logistic Regression / KNN
- **Flexible Input Pipeline**
  - Train on images in `person_images/`
  - Predict on external images
- **Real-Time Recognition**
  - **Local:** Webcam via OpenCV  
  - **hosted jupyter notebook:** Browser-based webcam capture using JavaScript
- **Modular Codebase**
  - Works on both local code editor and notebook with minimal edits

---

## 🛠️ Project Structure

```
project/
├── person_images/
│   ├── firstName_lastName_01.jpg
│   ├── firstName_lastName_02.png
│   ├── anotherPerson_01.jpg
│   └── ...
├── train_model.ipynb
├── predict_local.py
└── predict_colab.ipynb
```

---

## 📂 Dataset Naming Convention

Each image MUST follow this rule:

```
firstName_lastName_<number>.jpg
```

**Examples:**

```
sourav_pati_01.jpg
niranjan_pati_02.png
anik_banerjee_03.jpg
```

The prefix (e.g., `sourav_pati`) becomes the **class label**.

---

## 📦 Installation / Requirements

Install dependencies:

```bash
pip install tensorflow keras-facenet numpy opencv-python scikit-learn pillow
```

---

## 📘 Usage Guide

### ✅ 1. Train the Model  
Use: **image_processing_notebook.ipynb**

This notebook will:

- Load the FaceNet model  
- Read all images from `person_images/`  
- Detect faces (if applicable)  
- Convert each face into 128-D embeddings  
- Train a classifier  
- Save the following:

```
embeddings.npy
labels.npy
classifier.pkl
label_map.json
```

---

### 🎥 2. Real-Time Prediction

### 🖥️ A. Local Machine (VS Code / Python)

Run:

```bash
python predict_local.py
```

---

### 🌐 B. Google Colab (Browser Webcam)

Use: **image_processing_notebook.ipynb**

---

## 🧠 Tech Stack

- Python  
- TensorFlow / Keras  
- Keras-FaceNet  
- OpenCV  
- NumPy  
- Scikit-learn  

---

## 📌 Notes for Best Accuracy

- Add **15–40 images per person**
- Include different lighting, angles, expressions
