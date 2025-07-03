from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, cv2
from typing import List
from pathlib import Path as FilePath
import numpy as np
import face_recognition
import pickle
import asyncio
from fastapi.responses import StreamingResponse
import subprocess

from scripts.test_model import predict_person


app = FastAPI()

BASE_DIR1 = FilePath(__file__).resolve().parent


app.mount("/static", StaticFiles(directory=BASE_DIR1 / "static"), name="static")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
BASE_DIR = FilePath(__file__).resolve().parent.parent
print(BASE_DIR,"BASE_DIR")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "saved_model/face_recognition_model.keras")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)


from fastapi.responses import FileResponse

@app.get("/", response_class=FileResponse)
async def get_home():
    return FileResponse(BASE_DIR1 / "templates/index.html")

from fastapi import FastAPI, UploadFile, File, Path
@app.post("/upload-dataset/{label}/")
async def upload_dataset(label: str = Path(...), file: UploadFile = File(...)):
    # Create label directory inside the dataset folder
    label_dir = os.path.join(DATASET_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    # Extract just the file name (strip any folders)
    filename = os.path.basename(file.filename)

    # Final save path
    file_path = os.path.join(label_dir, filename)

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"status": "success", "label": label, "filename": filename}


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = await asyncio.to_thread(predict_person, str(file_path))
        os.remove(file_path)
        return {"status": "success", "result": result}
    except Exception as e:
        os.remove(file_path)
        return {"status": "error", "message": str(e)}



@app.post("/capture-image/")
async def capture_image():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    if not ret:
        return {"error": "Failed to capture image"}
    filename = "captured.jpg"
    path = os.path.join(UPLOAD_DIR, filename)
    cv2.imwrite(path, frame)
    cam.release()
    return {"filename": filename}


@app.post("/train/")
async def train_model():
    parent_dir = FilePath(__file__).parent.parent

    def stream():
        process = subprocess.Popen(
            ["python3", str(parent_dir / "scripts" / "train_model.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in iter(process.stdout.readline, ""):
            yield line
        process.stdout.close()
        process.wait()

    return StreamingResponse(stream(), media_type="text/plain")


@app.post("/predict/")
async def predict_face(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not trained"}

    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    image = face_recognition.load_image_file(file_path)
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        return {"result": "No face found"}

    face_enc = face_recognition.face_encodings(image, face_locations)[0]
    matches = face_recognition.compare_faces(data["encodings"], face_enc)
    name = "Unknown"

    if True in matches:
        match_index = matches.index(True)
        name = data["names"][match_index]

    return {"result": name}
