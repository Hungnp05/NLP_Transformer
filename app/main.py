from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.model import predict
from app.self_built_model import predict_self_built 

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None, "mode": "pretrained"})

@app.post("/predict", response_class=HTMLResponse)
async def make_prediction(request: Request, text: str = Form(...)):
    result = predict(text)
    return templates.TemplateResponse("index.html", {"request": request, "result": result, "mode": "pretrained"})

@app.get("/self-built", response_class=HTMLResponse)
async def self_built_page(request: Request):
    return templates.TemplateResponse("self_built.html", {"request": request, "result": None})

@app.post("/self-built-predict", response_class=HTMLResponse)
async def self_built_prediction(request: Request, text: str = Form(...)):
    result = predict_self_built(text)
    return templates.TemplateResponse("self_built.html", {"request": request, "result": result})