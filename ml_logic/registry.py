import time
from ml_logic.params import LOCAL_REGISTRY_PATH
import os
import glob
from joblib import dump, load

LOCAL_REGISTRY_PATH = '..'

def save_model (model = None):

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", timestamp)
    #model.save(model_path)
    dump(model, model_path)

    return None

def load_model():

    model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")


    results = glob.glob(f"{model_directory}/*")
    if not results:
        return None

    model_path = sorted(results)[-1]

    #model = models.load_model(model_path)
    model = load(model_path)
    return model
