import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Templates (Folder where HTML files are stored)
templates = Jinja2Templates(directory="templates")

# 1. Load the Model and Scaler (No encoders needed)
with open("financial_regression_model.pkl", "rb") as f:
    saved_data = pickle.load(f)
    model = saved_data["model"]
    scaler = saved_data["scaler"]

# The exact list of 40+ columns the scaler saw during training and expects now
expected_columns = scaler.feature_names_in_


# 2. Pydantic Model (We only request a single feature from the user!)
class GoldFeatures(BaseModel):
    gold_close: float


# 3. GET Request - Display the Home Page (HTML)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 4. POST Request - Make AI Prediction
@app.post("/predict")
async def predict(features: GoldFeatures):
    # TRICK: Create a dummy dataframe filled with 0s and only update 'gold close'
    input_data = pd.DataFrame(0, index=[0], columns=expected_columns)
    input_data['gold close'] = features.gold_close

    # Scaling and Prediction
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    predicted_price = float(prediction[0])

    # CHAMPION'S MARGIN OF ERROR (MAE = 1.32 Dollars)
    margin_of_error = 1.32
    lower_bound = predicted_price - margin_of_error
    upper_bound = predicted_price + margin_of_error

    # Return not just the prediction, but also the bounds!
    return {
        "predicted_price": predicted_price,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound
    }