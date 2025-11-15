"""
Person Identification (filename-labeled) - Executable script using TensorFlow/Keras.

This revised script strictly follows the user's constraints:
- Loads images from a local directory path (DATA_ROOT).
- If the path does NOT exist, it creates a directory structure and saves
  **actual image files** (solid-color PNGs) to that path so the loader can run.
  (The simulation runs ONLY when the provided DATA_ROOT does not exist.)
- IMPORTANT: class labels are NOT inferred from folder names. Instead, the label
  for each image is extracted from the **image filename** (e.g. "sourav_pati_01.png"
  -> class "sourav_pati"). The loader creates one-hot encoded labels from those names.
- Uses 100% of the loaded data for training (no train/test split).
- Builds a multi-class CNN (softmax) to classify images into the person names.
- Contains a function `predict_people_in_image(image_path)` that simulates
  detecting multiple faces in a single image by creating a small batch (2-3)
  of resized/modified versions of the same image (since real face detection is not included),
  passes them to the model, and prints the identified names.
- Includes a comment confirming that a standard multi-class CNN CANNOT
  natively identify multiple people in a single image frame without
  a pre-processing step like face detection (this is a constraint, not a bug).

Requirements:
- Python 3.8+
- tensorflow (2.x), pillow (PIL), numpy

Run:
    python person_identification_filename_labels.py
"""

from google.colab import output
from google.colab.output import eval_js
from base64 import b64decode
import os
import re
import sys
import math
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
from keras_facenet import FaceNet
from tensorflow.keras import layers, models, optimizers
import numpy as np
from PIL import Image, ImageDraw
import cv2
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------
# Configuration
# -------------------------
SEED = 1234
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_ROOT = "person_images"
IMG_SIZE = (96, 96)
CHANNELS = 3
BATCH_SIZE = 8
EPOCHS = 50
VERBOSE = 2

FILENAME_LABEL_RE = re.compile(r"^([A-Za-z0-9\-\.]+?)(?:_[0-9]+)?\.(?:jpg|jpeg|png)$", re.IGNORECASE)

# -------------------------
# Helper: simulate dataset (only if DATA_ROOT not present)
# -------------------------
def simulate_dataset_if_missing(root_path: str, num_people: int = 3, imgs_per_person: int = 10):
    """
    Create 'root_path' with subfolders and actual image files (solid-color PNGs)
    if the path does not already exist.

    Note: Although the directory structure is created, labels will NOT be inferred
    from folder names. Filenames will be used to determine labels (see loader).
    """
    root = Path(root_path)
    if root.exists():
        print(f"[simulate_dataset_if_missing] Found existing dataset at '{root_path}'. Simulation skipped.")
        return

    print(f"[simulate_dataset_if_missing] Dataset not found at '{root_path}'. Creating simulated dataset...")

    root.mkdir(parents=True, exist_ok=True)

    base_names = [f"person{idx+1}" for idx in range(num_people)]

    colors = [
        (220, 120, 120),
        (120, 220, 120),
        (120, 120, 220),
        (200, 180, 100),
        (180, 100, 200),
    ]

    for pi, name in enumerate(base_names):
        folder = root / f"group_{name}"
        folder.mkdir(exist_ok=True)
        color = colors[pi % len(colors)]

        for img_idx in range(imgs_per_person):
            filename = f"{name}_{img_idx:02d}.png"  # e.g., person1_00.png
            path = folder / filename

            img = Image.new("RGB", IMG_SIZE, color=color)
            draw = ImageDraw.Draw(img)

            w, h = IMG_SIZE
            rect_w = max(6, w // 8)
            rect_h = max(6, h // 8)
            offset_x = (img_idx * 7) % (w - rect_w)
            offset_y = (img_idx * 11) % (h - rect_h)
            rect_color = ((color[0] + 80) % 256, (color[1] + 80) % 256, (color[2] + 80) % 256)
            draw.rectangle([offset_x, offset_y, offset_x + rect_w, offset_y + rect_h], outline=rect_color)

            img.save(path, format="PNG")

    print(f"[simulate_dataset_if_missing] Simulation complete. Created {num_people} people with {imgs_per_person} images each.")
    print(f"Files are under: {root.resolve()}\n")


# -------------------------
# Data loader: filenames -> labels (one-hot)
# -------------------------
def gather_image_paths_and_labels(root_path: str) -> Tuple[List[str], List[str]]:
    """
    Gathers all image file paths recursively from the dataset directory.
    Derives class labels directly from filenames (not folder names).
    Works with filenames like:
        - sourav_pati.jpg
        - sourav_pati_01.jpg
        - anik_banerjee.jpg
        - some_person_12.jpg
    Returns:
        - file_paths: list of image file paths
        - labels: list of labels (person names)
    """
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Data root path '{root_path}' does not exist.")

    file_paths = []
    labels = []

    for p in root.rglob("*.jpg"):
        if p.is_file():
            name_part = p.stem
            parts = name_part.split("_")
            if parts[-1].isdigit():
                parts = parts[:-1]
            label = "_".join(parts).lower().strip()
            file_paths.append(str(p.resolve()))
            labels.append(label)

    if not file_paths:
        raise RuntimeError(f"No valid .jpg images found under '{root_path}'.")

    print(f"[gather_image_paths_and_labels] Collected {len(file_paths)} images for {len(set(labels))} unique labels.")
    return file_paths, labels


def build_label_mapping(labels: List[str]) -> Tuple[Dict[str, int], List[str]]:
    """
    From a list of label strings, build a mapping label->index and a sorted list of class names.
    Returns:
      - label_to_index: dict
      - class_names: list where index corresponds to class index (sorted for determinism)
    """
    unique = sorted(set(labels))
    label_to_index = {lab: idx for idx, lab in enumerate(unique)}
    print(f"[build_label_mapping] label->index mapping: {label_to_index}")
    return label_to_index, unique


def labels_to_one_hot(labels: List[str], label_to_index: Dict[str, int]) -> np.ndarray:
    """
    Convert list of label strings to one-hot encoded numpy array of shape (N, C)
    """
    indices = np.array([label_to_index[l] for l in labels], dtype=np.int32)
    num_classes = len(label_to_index)
    one_hot = np.eye(num_classes, dtype=np.float32)[indices]
    return one_hot

# -------------------------
# Image preprocessing for FaceNet input
# -------------------------
def preprocess_image(img):
    """
    Resize and normalize image for FaceNet input (160x160, RGB, [-1,1] range).
    """
    img = tf.image.resize(img, [160, 160])
    img = (img - 127.5) / 128.0
    return img

# -------------------------
# TensorFlow dataset construction (from file paths + one-hot labels)
# -------------------------
def build_tf_dataset(file_paths: List[str], one_hot_labels: np.ndarray, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
                     shuffle=True) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from lists of file paths and one-hot labels.
    Uses 100% of data for training (no split).
    """

    path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(one_hot_labels)

    ds = tf.data.Dataset.zip((path_ds, label_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths), seed=SEED)

    def _load_and_preprocess(path, one_hot):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=CHANNELS, expand_animations=False)
        image.set_shape([None, None, CHANNELS])
        image = preprocess_image(image)
        return image, one_hot

    autotune = tf.data.AUTOTUNE
    ds = ds.map(_load_and_preprocess, num_parallel_calls=autotune)
    ds = ds.batch(batch_size)
    ds = ds.cache()
    ds = ds.prefetch(autotune)
    return ds


# -------------------------
# Model architecture
# -------------------------
def build_model(num_classes):
    """
    Builds a FaceNet-based classifier.
    Uses pretrained FaceNet for embeddings + trainable dense head for person classification.
    """

    print("[model] Loading pretrained FaceNet model...")
    embedder = FaceNet()
    facenet = embedder.model

    for layer in facenet.layers:
        layer.trainable = False

    inputs = layers.Input(shape=(160, 160, 3))
    x = facenet(inputs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="facenet_person_identifier")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model

def capture_from_camera():
    js = """
    async function takePhoto() {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.width = 320;
      video.height = 240;
      div.appendChild(video);

      document.body.appendChild(div);

      const stream = await navigator.mediaDevices.getUserMedia({video: true});
      video.srcObject = stream;
      await video.play();

      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = 320;
      canvas.height = 240;
      canvas.getContext('2d').drawImage(video, 0, 0, 320, 240);

      stream.getTracks().forEach(t => t.stop());
      div.remove();

      return canvas.toDataURL('image/jpeg', 0.9);
    }
    takePhoto();
    """

    data = eval_js(js)
    img_bytes = b64decode(data.split(',')[1])
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

# -------------------------
# Prediction: simulate multi-person detection (as required)
# -------------------------
# Constraint comment required by the user:
# NOTE: A standard multi-class CNN cannot natively locate or identify multiple people in a single image frame.
# It expects a single centered input corresponding to one individual. To identify multiple people in one frame,
# an external pre-processing step (e.g., face/person detection like MTCNN, Haar cascades, or YOLO) is required
# to generate crops for each detected face/person. This script simulates that cropping step.
#
def predict_people_in_image(model: keras.Model, class_names: List[str], image_path: str, simulate_count: int = 1) -> List[str]:
    """
    Same as before but now supports camera-captured image as input.
    """
    if image_path is None:
        raise ValueError("No image path provided for prediction.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image for prediction not found: {image_path}")

    pil = Image.open(image_path).convert("RGB")
    w, h = pil.size

    crops = []
    for i in range(simulate_count):
        crop_arr = tf.convert_to_tensor(np.array(pil), dtype=tf.float32)
        crop_arr = preprocess_image(crop_arr)
        crop_arr = crop_arr.numpy()
        crops.append(crop_arr)

    batch = np.stack(crops, axis=0)

    preds = model.predict(batch, verbose=0)
    pred_indices = np.argmax(preds, axis=1)
    identified = [class_names[int(idx)] for idx in pred_indices]

    print("\n[predict_people_in_image] Identification result:")
    for i, name in enumerate(identified, start=1):
        print(f"  - Face #{i}: {name}")
    print("")
    return identified

# -------------------------
# Main execution: assemble everything
# -------------------------
def main():
    simulate_dataset_if_missing(DATA_ROOT, num_people=3, imgs_per_person=12)

    file_paths, raw_labels = gather_image_paths_and_labels(DATA_ROOT)
    label_to_index, class_names = build_label_mapping(raw_labels)
    one_hot = labels_to_one_hot(raw_labels, label_to_index)
    dataset = build_tf_dataset(file_paths, one_hot, img_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True)

    num_classes = len(class_names)
    model = build_model(num_classes)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print(f"\n[main] Training on full dataset (N={len(file_paths)} images, classes={num_classes}) for {EPOCHS} epochs.")
    model.fit(dataset, epochs=EPOCHS, verbose=VERBOSE)

    # ---------- NEW CAMERA TESTING (COLAB) ----------
    print("\n[main] Opening camera to capture test image...")

    img = capture_from_camera()  # JS-based webcam
    cv2.imwrite("camera_test.jpg", img)
    cam_path = "camera_test.jpg"

    print("[main] Running model on captured image...")
    identified = predict_people_in_image(model, class_names, cam_path, simulate_count=1)
    print("[main] Identified:", identified)

    print("\n[main] Demo complete.\n")
    
    # model.save("person_recog_model.h5")


if __name__ == "__main__":
    main()
