# Real-Time Face Recognition with SCRFD + ArcFace (ONNX)

This project implements a **modern, high‑accuracy**, **production‑grade**, and **fully free** face‑recognition pipeline using:

- **SCRFD (face detector)**
- **ArcFace-ResNet100 (face embedding model)**
- **SVM classifier** for personalized recognition
- **Real‑time webcam inference**

The system is optimized for **small datasets**, making it ideal for events, access‑control, or personalized recognition tasks.

---

## 🚀 Features

### 🔍 Face Detection — SCRFD
- Extremely fast
- Highly accurate
- Robust across lighting, pose, angle, small faces, etc.
- Runs entirely on CPU via **ONNX Runtime**

### 🧠 Face Embeddings — ArcFace ResNet100
- State‑of‑the‑art model used in industry
- Produces **512‑dimensional embeddings**
- Highly discriminative and stable
- Excellent performance even with **1–3 images per person**

### 🎯 Classifier — SVM
- Works great with small datasets
- No retraining of deep models required
- Very fast training (<1 second typically)

### 🎥 Real‑Time Recognition
- Uses OpenCV webcam feed
- No caching: each frame is processed fresh
- Works fluently on CPU

---

## 📦 Project Structure

```
project/
│
├── models/
│   ├── scrfd_500m_bnkps.onnx
│   ├── scrfd_500m_kps.onnx
│   └── arcfaceresnet100-8.onnx
│
├── person_images/
│   └── person_name/ (one folder per person)
│
├── outputs/
│   └── (embeddings + classifier are saved here)
│
└── Main.py
```

---

## 📥 Installing Dependencies

Activate your virtual environment and run:

```bash
pip install onnxruntime opencv-python numpy scikit-learn joblib requests
```

---

## 📘 How to Use

### 1️⃣ Place Your Training Images
Inside:

```
person_images/
```

Use the structure:

```
person_images/
   Sourav/
       1.jpg
       2.jpg
       3.jpg
   Alice/
       1.jpg
       2.jpg
```

You need **1 or more** images per person.

---

### 2️⃣ Run the Script

```bash
python Main.py --mode train
python Main.py --mode webcam
```

What happens:

- SCRFD detects faces in each training image
- ArcFace converts each face → 512‑D embedding
- SVM classifier trains
- Webcam opens
- Each frame is analyzed and recognized live

---

## 📊 Output Files

Saved in `outputs/`:

| File | Purpose |
|------|---------|
| `svm_classifier.pkl` | Trained classifier |
| `label_encoder.pkl` | Mapping between numeric labels ↔ names |
| `embeddings.npy` | Stored embeddings |
| `labels.npy` | Stored corresponding labels |

---

## 🎯 Notes for Best Accuracy

- 3–10 photos per person recommended  
- Ensure face is clear and well-lit  
- Include multiple angles if possible  
- Avoid sunglasses or masks  

---

## 🔐 License & Cost

- **SCRFD** — MIT License  
- **ArcFace-ResNet100** — Apache 2.0  
- **ONNX Runtime** — MIT  
- Everything used is **100% free** and requires **no subscription**.

---

## 📎 Download This README

A downloadable version has been generated below.
