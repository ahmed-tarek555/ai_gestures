from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from routers import detect, register

templates = Jinja2Templates(directory="templates")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect.router)
app.include_router(register.router)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})