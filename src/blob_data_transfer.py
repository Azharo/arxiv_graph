import time
import re
import sys
import glob
import os
import gzip
import json
import math
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from pathlib import Path
sys.path.append(os.path.abspath("/home/arxiv/doc_intel_etl"))
import config

CONTAINER = config.azure_blob['container']

# credentials for blob with our raw data
storage_name = config.azure_blob['storage_name']
key = config.azure_blob['key']
connect_str = config.azure_blob['connect_str']

# Instantiate a BlobServiceClient using a connection string
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
# Instantiate a ContainerClient
container_client = blob_service_client.get_container_client(CONTAINER)


def get_blob_list(prefix):
    # List the blobs in a container
    full_blob_list = list(container_client.list_blobs(name_starts_with=prefix))
    return full_blob_list

def get_blob_file_list(file_type, full_blob_list, year_del):
    blob_list = []
    year_list = []
    for blob in full_blob_list:
        if blob.name.split('.')[-1]==file_type:
            blob_list.append(blob.name)
            year_list.append(blob.name.split('/')[year_del])
    year_list = list(set(year_list))
    return blob_list, year_list

def copy_blob(blob):
    # copy a given blob to local directory
    # ensure there is a directory to copy the blobs to
    blob_path = '/'.join(blob.split('/')[:-1])
    Path(blob_path).mkdir(parents=True, exist_ok=True)
    
    blob_client = blob_service_client.get_blob_client(container=CONTAINER, blob=blob)
    with open(blob, 'wb') as download_file:
        download_file.write(blob_client.download_blob().readall())
    return blob

def stream_blob(blob):
    blob_client = blob_service_client.get_blob_client(container=CONTAINER, blob=blob)
    blob_stream = blob_client.download_blob().readall()
    return blob_stream

def send_to_blob(file, split):
    file_blob_path = '/'.join(file.split('/')[:split])+'/'+file.split('/')[-1]
    blob_client = blob_service_client.get_blob_client(container=CONTAINER, blob=file_blob_path)
    with open(file, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)