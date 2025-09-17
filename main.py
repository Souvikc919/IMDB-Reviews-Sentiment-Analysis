from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib

# Load models and vectorizer
vectorizer = joblib.load("model/vectorizer.pkl")
log_reg = joblib.load("model/log_reg.pkl")
svm = joblib.load("model/svm.pkl")
nb = joblib.load("model/nb.pkl")

# FastAPI app
app = FastAPI(title="IMDB Sentiment Analysis")

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, text: str = Form(...), model: str = Form("svm")):
    text_vector = vectorizer.transform([text])

    if model == "log_reg":
        pred = log_reg.predict(text_vector)[0]
        model_name = "Logistic Regression"
    elif model == "nb":
        pred = nb.predict(text_vector)[0]
        model_name = "MultinomialNB"
    else:
        pred = svm.predict(text_vector)[0]
        model_name = "Linear SVM"

    sentiment = "Positive" if pred == 1 else "Negative"

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": sentiment, "model": model_name, "text": text}
    )
