import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import pickle
import queue
import threading

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "person_images"
DB_PATH = BASE_DIR / "face_db.pkl"

MATCH_THRESHOLD = 0.65
UNCERTAIN_THRESHOLD = 0.55

def init_face_model():
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def train():
    app = init_face_model()
    database = {}

    for person_dir in DATA_DIR.iterdir():
        if not person_dir.is_dir():
            continue

        embeddings = []
        for img_path in person_dir.glob("*"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            faces = app.get(img)
            if not faces:
                continue

            emb = faces[0].embedding
            embeddings.append(emb)

        if embeddings:
            database[person_dir.name] = np.vstack(embeddings)

    with open(DB_PATH, "wb") as f:
        pickle.dump(database, f)

    print("Training complete.")
    print("Stored identities:", list(database.keys()))

def classify(embedding, database):
    best_name = None
    best_score = -1

    for name, embs in database.items():
        sims = cosine_similarity([embedding], embs)[0]
        score = sims.mean()
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= MATCH_THRESHOLD:
        return best_name, best_score, "MATCH"
    elif best_score >= UNCERTAIN_THRESHOLD:
        return best_name, best_score, "UNCERTAIN"
    else:
        return "UNKNOWN", best_score, "REJECT"

speech_queue = queue.Queue()

def speech_worker():
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    speaker = win32com.client.Dispatch("SAPI.SpVoice")

    while True:
        text = speech_queue.get()
        if text is None:
            break
        speaker.Speak(text)

def webcam():
    if not DB_PATH.exists():
        print("Run training first.")
        return

    with open(DB_PATH, "rb") as f:
        database = pickle.load(f)

    app = init_face_model()
    cap = cv2.VideoCapture(0)
    displayed_names = set()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)

        for face in faces:
            box = face.bbox.astype(int)
            emb = face.embedding

            name, score, status = classify(emb, database)

            color = (0,255,0) if status=="MATCH" else (0,255,255) if status=="UNCERTAIN" else (0,0,255)
            label = f"Welcome {name} | {status} | {score:.2f}" if name != "UNKNOWN" else f"{name} | {status} | {score:.2f}"

            display_name = name.replace("_", " ")
            if (display_name not in displayed_names and display_name != "UNKNOWN"):
                print(f"\nWelcome {display_name}\n")
                speech_queue.put(f"Welcome {display_name}")
                displayed_names.add(display_name)

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, label, (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "webcam"], required=True)
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        threading.Thread(target=speech_worker, daemon=True).start()
        webcam()
