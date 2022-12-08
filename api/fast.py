from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd

from ofnd.ml_logic.preprocessor import clean_data
from ofnd.ml_logic.scraping_module import scraping
from ofnd.interface.main import predict
from ofnd.ml_logic.classifier import label
import json

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# $WIPE_BEGIN
# 💡 Preload the model to accelerate the predictions
# We want to avoid loading the heavy deep-learning model from MLflow at each `get("/predict")`
# The trick is to load the model in memory when the uvicorn server starts
# Then to store the model in an `app.state.model` global variable accessible across all routes!
# This will prove very useful for demo days
# $WIPE_END


# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get("/pred")
def pred(text_or_url):

    #    check if textorurl is text; or url
    # .    if url, get the text
    print("Now checking text_or_url")
    if text_or_url[:7] == 'http://' or text_or_url[:8] == 'https://':
        print("Now going to scrape the text")
        text = scraping(text_or_url)
    else:
        text =  pd.DataFrame({'news': [text_or_url]})
    if text.empty:
        return { "error": 'No text scraped' }
    print("Now going to clean the text")
    preprocessed_text = clean_data(text)
    if preprocessed_text.size == 0:
        return { "error": 'No preprocessed text' }
    print("Now going to predict")
    prediction_result = predict(preprocessed_text)
    if not prediction_result:
        return { "error": 'No prediction result' }
    print("Now going to label the result")
    outcome = label(prediction_result)

    return {
         "text_or_url": text_or_url,
         "prediction": outcome,
         }
