from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from uuid import uuid4
from app.core.redis_client import redis_client
import io

router = APIRouter()

# Upload image and store in Redis
@router.post("/")
async def upload_photo(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image format")

    image_bytes = await file.read()
    image_id = str(uuid4())

    # Save image with 1-hour expiry
    redis_client.setex(f"image:{image_id}", 3600, image_bytes)

    return {"id": image_id, "message": "Image saved in Redis"}

# Retrieve image by ID from Redis
@router.get("/{image_id}")
async def get_image(image_id: str):
    data = redis_client.get(f"image:{image_id}")
    if not data:
        raise HTTPException(status_code=404, detail="Image not found")

    return StreamingResponse(io.BytesIO(data), media_type="image/jpeg")
# Delete image by ID from Redis
@router.delete("/{image_id}")
async def delete_image(image_id: str):
    if not redis_client.delete(f"image:{image_id}"):
        raise HTTPException(status_code=404, detail="Image not found")

    return {"message": "Image deleted successfully"}    


