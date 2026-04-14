from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Initialize FastAPI app
app = FastAPI(title="Cultivar Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and preprocessors
model_sustainability = load_model('model_sustainability.keras')
model_market_price = load_model('model_market_price.keras')

with open('preprocessor_sus.pkl', 'rb') as f:
    scaler_sus = pickle.load(f)
with open('preprocessor_market.pkl', 'rb') as f:
    scaler_market = pickle.load(f)

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY or GOOGLE_API_KEY == 'YOUR_GOOGLE_API_KEY':
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=GOOGLE_API_KEY)
gen_model = genai.GenerativeModel('gemini-1.5-pro-001')


# Define request models
class SustainabilityInput(BaseModel):
    Soil_pH: float
    Soil_Moisture: float
    Temperature_C: float
    Rainfall_mm: float
    Crop_Type: str
    Fertilizer_Usage_kg: float
    Pesticide_Usage_kg: float
    Crop_Yield_ton: float

class MarketInput(BaseModel):
    Product: str
    Demand_Index: float
    Supply_Index: float
    Competitor_Price_per_ton: float
    Economic_Indicator: float
    Weather_Impact_Score: float
    Seasonal_Factor: str
    Consumer_Trend_Index: float

class PredictionResponse(BaseModel):
    prediction: float
    analysis: str

@app.post("/predict/sustainability", response_model=PredictionResponse)
async def predict_sustainability(input_data: SustainabilityInput):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.model_dump()])
        
        # Transform using preprocessor
        scaled_input = scaler_sus.transform(input_df)
        
        # Make prediction
        prediction = float(model_sustainability.predict(scaled_input)[0][0])
        
        # Get analysis
        chat = gen_model.start_chat(history=[])
        prompt = f"""
        Analyze this agricultural sustainability score: {prediction}
        These are the factors that influenced the prediction: {input_data.model_dump()}
        Consider:
        The sustainability score is a measure of the environmental impact of farming practices.
        Higher score means more sustainable practices.
        It is based on various factors including soil health, water usage, and crop yield.
        Consider:
        1. What this sustainability score means for the farmer
        2. Recommendations for improving sustainability
        3. Potential environmental impact
        Keep it practical and detailed, give numbers.
        Keep it under 300 words.
        """
        analysis = chat.send_message(prompt).text
        
        return PredictionResponse(prediction=prediction, analysis=analysis)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/market", response_model=PredictionResponse)
async def predict_market(input_data: MarketInput):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.model_dump()])
        
        # Transform using preprocessor
        scaled_input = scaler_market.transform(input_df)
        
        # Make prediction
        prediction = float(model_market_price.predict(scaled_input)[0][0])
        
        # Get analysis
        chat = gen_model.start_chat(history=[])
        prompt = f"""
        Analyze this predicted market price per ton: {prediction}
        These are the factors that influenced the prediction: {input_data.model_dump()}
        Consider:
        The market price is influenced by various factors including demand, supply, and competitor pricing.
        Consider:
        1. Market implications of this price
        2. Recommendations for the farmer
        3. Potential pricing strategy
        Keep it practical and detailed, give numbers.
        Keep it under 300 words.
        """
        analysis = chat.send_message(prompt).text
        
        return PredictionResponse(prediction=prediction, analysis=analysis)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}