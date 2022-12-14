import time
from GEofnd.ml_logic.params import LOCAL_REGISTRY_PATH, MODEL_TARGET, BUCKET_NAME, DESTINATION_BLOB_NAME
import os
import glob
from joblib import dump, load
from google.cloud import storage
from io import BytesIO



def save_model (model=None):

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if MODEL_TARGET == 'local':

        model_path = os.path.join(LOCAL_REGISTRY_PATH,"models", timestamp)
        dump(model, model_path)

        return None

    elif MODEL_TARGET == 'gcloud':

        bytes_container = BytesIO()
        dump(model, bytes_container)
        bytes_container.seek(0)
        bytes_model = bytes_container.read()


        timestamp = time.strftime("%Y%m%d-%H%M%S")
        storage_client = storage.Client()
        bucket = storage_client.bucket(f'{BUCKET_NAME}')
        blob = bucket.blob(f'{DESTINATION_BLOB_NAME}{timestamp}.joblib')
        blob.upload_from_string(bytes_model)

        return None

def load_model():

    if MODEL_TARGET == 'local':

        model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")

        results = glob.glob(f"{model_directory}/*")
        if not results:

            return None
        else:
            model_path = sorted(results)[-1]
            model = load(model_path)
            return model

    elif MODEL_TARGET == 'gcloud':
        models_list=[]

        model_directory = storage.Client().list_blobs(f'{BUCKET_NAME}',\
            prefix=f'{DESTINATION_BLOB_NAME}2')

        for i in model_directory:
            models_list.append(i.name)
        model_path = models_list[-1]

        storage_client = storage.Client()
        bucket = storage_client.bucket(f'{BUCKET_NAME}')
        blob = bucket.blob(f'{model_path}')

        model_file = BytesIO()

        blob.download_to_file(model_file)

        model = load(model_file)
        return model
