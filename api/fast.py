
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from GEofnd.interface.main import predict
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
@app.get("/predict")
def predict(text = None):
#if it is a website
    if text.startswith('http') or text.startswith('www'):
        prediction = predict(text)
        return prediction
#if it is a text
    else :
        pass
        # The get request expects the URL input to be a string
    """
    we use type hinting to indicate the data types expected
    for the parameters of the function
    FastAPI uses this information in order to hand errors
    to the developpers providing incompatible parameters
    FastAPI also provides variables of the expected data type to use
    without type hinting we need to manually convert
    the parameters of the functions which are all received as strings
    """
    # $CHA_BEGIN

    # ⚠️ if the timezone conversion was not handled here the user would be assumed to provide an UTC datetime
    # create datetime object from user provided date
    # pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")

    # localize the user provided datetime with the NYC timezone
     #eastern = pytz.timezone("US/Eastern")
     #localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)

    # convert the user datetime to UTC and format the datetime as expected by the pipeline
     #utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)
     #formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")

    # fixing a value for the key, unused by the model
    # in the future the key might be removed from the pipeline input
     #key = "2013-07-06 17:18:00.000000119"

     # X_pred = pd.DataFrame(dict(
        # key=[key],  # useless but the pipeline requires it
        # pickup_datetime=[formatted_pickup_datetime],
        # pickup_longitude=[pickup_longitude],
        # pickup_latitude=[pickup_latitude],
        # dropoff_longitude=[dropoff_longitude],
        # dropoff_latitude=[dropoff_latitude],
        # passenger_count=[passenger_count]))

    # model = app.state.model
    # X_processed = preprocess_features(X_pred)
    # y_pred = model.predict(X_processed)

    # ⚠️ fastapi only accepts simple python data types as a return value
    # among which dict, list, str, int, float, bool
    # in order to be able to convert the api response to json
    # return dict(fare=float(y_pred))
    # $CHA_END


# @app.get("/")
# def root():
    # $CHA_BEGIN
    # return dict(greeting="Hello")
    # $CHA_END
