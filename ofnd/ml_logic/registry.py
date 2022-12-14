import time
from ofnd.ml_logic.params import LOCAL_REGISTRY_PATH, MODEL_TARGET, BUCKET_NAME, DESTINATION_BLOB_NAME, MODEL_TYPE, TF_DEST_BLOB_NAME
import os
import glob
from joblib import dump, load
from google.cloud import storage
from io import BytesIO
import tensorflow as tf



def save_model (model=None):

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if MODEL_TYPE == 'ml':
        if MODEL_TARGET == 'local':

            model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", timestamp)
            #model.save(model_path)
            dump(model, model_path)

            return None

        bytes_container = BytesIO()
        dump(model, bytes_container)
        bytes_container.seek(0)
        bytes_model = bytes_container.read()


        timestamp = time.strftime("%Y%m%d-%H%M%S")
        storage_client = storage.Client()
        bucket = storage_client.bucket(f'{BUCKET_NAME}')
        blob = bucket.blob(f'{DESTINATION_BLOB_NAME}{timestamp}.joblib')
        blob.upload_from_string(bytes_model)

        #     blob

        return None

    if MODEL_TYPE == 'tensorflow':

        model.save(f'gs://{BUCKET_NAME}/{TF_DEST_BLOB_NAME}/{timestamp}', save_format='tf')

        return None



def load_model():

    if MODEL_TYPE == 'ml':

        if MODEL_TARGET == 'local':

            model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")


            results = glob.glob(f"{model_directory}/*")
            if not results:
                return None

            model_path = sorted(results)[-1]

            model = load(model_path)
            return model

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

    if MODEL_TYPE == 'tensorflow':

        model = tf.keras.models.load_model('gs://{BUCKET_NAME}/{TF_DEST_BLOB_NAME}tf_model')

        return model
