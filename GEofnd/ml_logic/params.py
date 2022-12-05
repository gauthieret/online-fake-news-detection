
import os
import numpy as np

TARGET_COLUMN = os.environ.get("TARGET_COLUMN")
FEATURE_COLUMN = os.environ.get("FEATURE_COLUMN")

DATASET_LOCAL_PATH = os.environ.get("DATASET_LOCAL_PATH")

LOCAL_REGISTRY_PATH = os.path.expanduser(os.environ.get("LOCAL_REGISTRY_PATH"))

MODEL_TARGET = os.environ.get('MODEL_TARGET')

BUCKET_NAME = os.environ.get('BUCKET_NAME')

DESTINATION_BLOB_NAME = os.environ.get('DESTINATION_BLOB_NAME')
