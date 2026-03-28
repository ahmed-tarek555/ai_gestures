from fastapi import APIRouter, HTTPException, status
from fastapi.templating import Jinja2Templates
from camera_utils import detect

templates = Jinja2Templates(directory="templates")
router = APIRouter(prefix="/detect")

@router.get("/")
def detect_api():
    text = detect()
    return {"text": text}
