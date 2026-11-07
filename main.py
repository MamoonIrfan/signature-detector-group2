from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from model import predict_image
from r2_upload import upload_to_r2

app = FastAPI(title="Defect Detection API")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 1. Model se prediction
    result = predict_image(file_path)
    
    # 2. R2 pe upload
    r2_url = upload_to_r2(file_path, file.filename)
    
    return JSONResponse(content={
        "status": "success",
        "filename": file.filename,
        "r2_url": r2_url,
        "defects_found": result
    })

@app.get("/result/{filename}")
async def get_result(filename: str):
    # Yahan DB se result nikalna hai. Filhal dummy
    return {"filename": filename, "status": "processed"}

@app.get("/")
def home():
    return {"message": "FastAPI Defect Detection is Running"}
