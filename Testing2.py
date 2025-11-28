#!/usr/bin/env python3
"""
Testing2.py - ArcFace ONNX + OpenCV DNN detector pipeline (single-file)

Usage:
  python Testing2.py --mode train        # trains classifier from person_images/<label>/*.jpg
  python Testing2.py --mode webcam       # runs live webcam recognition (predicts on each live frame)

Requirements:
  pip install onnxruntime opencv-python numpy scikit-learn joblib requests

Place these model files in ./models/ (next to this script):
  - arcface_r50.onnx
      (ArcFace ONNX model. Many community ArcFace ONNXs exist; typical input 1x3x112x112, normalization (img-127.5)/128.)
  - deploy.prototxt
  - res10_300x300_ssd_iter_140000.caffemodel
      (OpenCV DNN res10 SSD face detector files)

Notes:
  - Adjust FACE_SIZE or normalization if you use a different ArcFace ONNX variant.
  - By default the code uses an SVM classifier (linear) on top of L2-normalized embeddings.
  - If you prefer a simpler nearest-mean classifier, set USE_SVM = False.
  - The webcam loop runs detection+embedding on every frame and classifies the latest frame (no image caching).
"""

import os
import sys
import argparse
from pathlib import Path
from glob import glob
import time

import cv2
import numpy as np
import onnxruntime as ort
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, Normalizer
from joblib import dump, load

# ----------------- Configuration -----------------
BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "models"
ARCFACE_ONNX = MODEL_DIR / "arcface_r50.onnx"
DETECT_PROTOTXT = MODEL_DIR / "deploy.prototxt"
DETECT_CAFFEMODEL = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

PERSON_IMAGES_DIR = BASE_DIR / "person_images"   # expected structure: person_images/<label>/*.jpg
CLASSIFIER_PATH = BASE_DIR / "face_model.pkl"

FACE_SIZE = 112          # ArcFace typical input size (112x112). Change if your ONNX model expects different.
EMBED_DIM = 512          # typical ArcFace embedding dim (some models differ)
CONF_THRESHOLD = 0.5     # detector confidence threshold
USE_SVM = True           # if False -> use nearest-mean classifier
UNKNOWN_PROB_THRESHOLD = 0.2   # for SVM probability thresholding to mark UNKNOWN
UNKNOWN_SIM_THRESHOLD = 0.5    # for nearest-mean similarity threshold (cosine)
# -------------------------------------------------

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def check_models_exit_if_missing():
    missing = []
    if not ARCFACE_ONNX.exists():
        missing.append(str(ARCFACE_ONNX))
    if not DETECT_PROTOTXT.exists():
        missing.append(str(DETECT_PROTOTXT))
    if not DETECT_CAFFEMODEL.exists():
        missing.append(str(DETECT_CAFFEMODEL))
    if missing:
        eprint("[error] Missing model files:")
        for m in missing:
            eprint("  -", m)
        eprint("Put required files in:", MODEL_DIR)
        sys.exit(1)

def load_detector():
    # OpenCV DNN res10 SSD face detector
    net = cv2.dnn.readNetFromCaffe(str(DETECT_PROTOTXT), str(DETECT_CAFFEMODEL))
    return net

def load_arcface_onnx_session():
    # Create an onnxruntime session for ArcFace model; CPU provider by default.
    sess = ort.InferenceSession(str(ARCFACE_ONNX), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    return sess, input_name, output_name

def preprocess_face_for_arcface(face_bgr):
    """
    face_bgr: HxW BGR numpy array crop
    returns: 1x3xFACE_SIZExFACE_SIZE float32 input for ONNX
    Note: normalization used here: (img - 127.5) / 128.0
    If your ONNX expects different, adjust accordingly.
    """
    img = cv2.resize(face_bgr, (FACE_SIZE, FACE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - 127.5) / 128.0
    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0).astype(np.float32)
    return img

def get_embedding(session, input_name, face_bgr):
    inp = preprocess_face_for_arcface(face_bgr)
    outs = session.run(None, {input_name: inp})
    emb = np.asarray(outs[0]).flatten().astype(np.float32)
    # L2 normalize
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb

def detect_faces(net, frame, conf_threshold=CONF_THRESHOLD):
    """
    Returns list of ((x1,y1,x2,y2), confidence) for each detection in frame.
    Uses OpenCV DNN; coordinate clipping included.
    """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()
    results = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < conf_threshold:
            continue
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        results.append(((x1, y1, x2, y2), conf))
    return results

def train(person_images_dir=str(PERSON_IMAGES_DIR)):
    check_models_exit_if_missing()
    detector = load_detector()
    session, input_name, _ = load_arcface_onnx_session()

    X = []
    y = []

    person_images_dir = Path(person_images_dir)
    if not person_images_dir.exists():
        eprint("[error] person_images directory does not exist:", person_images_dir)
        return

    people = sorted([d.name for d in person_images_dir.iterdir() if d.is_dir()])
    if not people:
        eprint("[error] No person subdirectories found inside", person_images_dir)
        eprint("Expected structure: person_images/<label>/*.jpg")
        return

    print("[info] Found persons:", people)
    for person in people:
        files = list((person_images_dir / person).glob("*"))
        if not files:
            print(f"[warn] no files for {person}, skipping")
            continue
        for img_path in files:
            img = cv2.imread(str(img_path))
            if img is None:
                print("[warn] could not read", img_path)
                continue
            boxes = detect_faces(detector, img, conf_threshold=0.45)
            if not boxes:
                print("[warn] no face detected in", img_path)
                continue
            # select largest detected face
            boxes_sorted = sorted(boxes, key=lambda b: (b[0][2]-b[0][0])*(b[0][3]-b[0][1]), reverse=True)
            (x1, y1, x2, y2), conf = boxes_sorted[0]
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                print("[warn] empty crop for", img_path)
                continue
            emb = get_embedding(session, input_name, face)
            X.append(emb)
            y.append(person)
            print(f"[debug] added embedding for {person} from {img_path}")

    if len(X) == 0:
        eprint("[error] No embeddings were extracted; training aborted.")
        return

    X = np.vstack(X)
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)
    normalizer = Normalizer(norm='l2')
    X_norm = normalizer.transform(X)

    if USE_SVM:
        clf = SVC(kernel='linear', probability=True)
        print("[info] Training SVM classifier on", X_norm.shape[0], "examples...")
        clf.fit(X_norm, y_enc)
        model_obj = {"clf": clf, "le": le, "normalizer": normalizer}
    else:
        # nearest-mean classifier
        class_means = {}
        for lbl in np.unique(y_enc):
            class_means[int(lbl)] = X_norm[y_enc == lbl].mean(axis=0)
        model_obj = {"class_means": class_means, "le": le, "normalizer": normalizer}

    dump(model_obj, CLASSIFIER_PATH)
    print("[info] Training complete. Model saved to:", CLASSIFIER_PATH)

def webcam_loop():
    check_models_exit_if_missing()
    detector = load_detector()
    session, input_name, _ = load_arcface_onnx_session()
    if not CLASSIFIER_PATH.exists():
        eprint("[error] classifier not found. Run --mode train first.")
        return

    model_obj = load(CLASSIFIER_PATH)
    normalizer = model_obj.get("normalizer")
    le = model_obj.get("le")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        eprint("[error] cannot open webcam (index 0). Check camera or try another index.")
        return

    print("[info] Webcam opened. Predictions run on each live frame. Press ESC to exit.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            eprint("[error] failed to read frame from webcam")
            break

        # detect faces on current frame
        boxes = detect_faces(detector, frame, conf_threshold=0.2)
        label_text = "No face"
        if boxes:
            # pick largest
            boxes_sorted = sorted(boxes, key=lambda b: (b[0][2]-b[0][0])*(b[0][3]-b[0][1]), reverse=True)
            (x1, y1, x2, y2), conf = boxes_sorted[0]
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                label_text = "Bad crop"
            else:
                emb = get_embedding(session, input_name, face)
                emb_norm = normalizer.transform([emb])[0] if normalizer is not None else emb
                if USE_SVM and ("clf" in model_obj):
                    probs = model_obj["clf"].predict_proba([emb_norm])[0]
                    idx = int(np.argmax(probs))
                    prob = float(probs[idx])
                    label = model_obj["le"].inverse_transform([idx])[0]
                    if prob < UNKNOWN_PROB_THRESHOLD:
                        label_text = f"UNKNOWN ({prob:.2f})"
                    else:
                        label_text = f"{label} ({prob:.2f})"
                else:
                    best_lbl = None
                    best_sim = -1.0
                    for lbl, mean_vec in model_obj["class_means"].items():
                        sim = float(np.dot(emb_norm, mean_vec))
                        if sim > best_sim:
                            best_sim = sim
                            best_lbl = lbl
                    label_name = model_obj["le"].inverse_transform([int(best_lbl)])[0]
                    if best_sim < UNKNOWN_SIM_THRESHOLD:
                        label_text = f"UNKNOWN ({best_sim:.2f})"
                    else:
                        label_text = f"{label_name} ({best_sim:.2f})"

            # draw rectangle and label on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, label_text, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # fps display
        now = time.time()
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.imshow("Live Face Recognition", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="ArcFace ONNX + OpenCV DNN face recognition (train/webcam)")
    parser.add_argument("--mode", choices=["train", "webcam"], required=True)
    parser.add_argument("--data", default=str(PERSON_IMAGES_DIR), help="person_images directory (for training)")
    args = parser.parse_args()

    if args.mode == "train":
        train(person_images_dir=args.data)
    elif args.mode == "webcam":
        webcam_loop()
    else:
        eprint("Unknown mode:", args.mode)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[info] Interrupted by user, exiting.")
    except Exception as exc:
        eprint("[fatal] Unhandled exception:", exc)
        raise
