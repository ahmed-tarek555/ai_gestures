from fastapi import Request, APIRouter, HTTPException, status, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from camera_utils import register

templates = Jinja2Templates(directory="templates")
router = APIRouter(prefix="/register")

@router.post("/")
def register_api(gesture: str =Form(...)):
    register(gesture)
    return {"details": "Success"}