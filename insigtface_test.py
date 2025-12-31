from insightface.app import FaceAnalysis
import cv2

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0)

img = cv2.imread("test.jpg")  # put a face image here
faces = app.get(img)

print("Faces detected:", len(faces))
if faces:
    print("Embedding shape:", faces[0].embedding.shape)
