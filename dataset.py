#!/usr/bin/env python3
"""
This module loads the matlab datasets to be used for training the LTE/NR
classifier.
"""
import os
import threading
import requests
from os.path import basename, isdir, isfile, join
import zipfile

DATA_URIS = ["https://www.mathworks.com/supportfiles/spc/SpectrumSensing/SpectrumSensingTrainingData128x128.zip",
             "https://www.mathworks.com/supportfiles/spc/SpectrumSensing/SpectrumSensingTrainedCustom_2024.zip",
             "https://www.mathworks.com/supportfiles/spc/SpectrumSensing/SpectrumSensingCapturedData128x128.zip",
]
DATA_DIR = "data"

def _get_resource(uri: str, data_dir=DATA_DIR):
    """Download a zip file and unzip it into a basename directory.

    Args:
        uri: web resource
    """
    filename = basename(uri)
    if not isfile(join(DATA_DIR, filename)):
        print(f"Starting download of data files from:\n\t{uri}")
        rsp = requests.get(uri, stream=True, timeout=60 * 5)
        rsp.raise_for_status()
        with open(join(data_dir, filename), 'wb') as file:
            for chunk in rsp.iter_content(chunk_size=8192):
                file.write(chunk)

    with zipfile.ZipFile(join(DATA_DIR, filename), 'r') as zip_r:
        zip_r.extractall(join(DATA_DIR, filename.replace(".zip", "")))
    assert isdir(join(DATA_DIR, filename.replace(".zip", ""))), f"Failed to extract {filename}"

def download_dataset(clean=False):
    """Download the datasets from matlab if they don't exist.

    Args:
        clean (): delete already downloaded dataset if it exists before download
    """
    if clean and isdir(DATA_DIR):
        os.remove(DATA_DIR)
    if not isdir(DATA_DIR):
        os.mkdir(DATA_DIR)

    threads = []
    for uri in DATA_URIS:
        x = threading.Thread(target=_get_resource, args=(uri,))
        x.start()
        threads.append(x)
    _ = [t.join() for t in threads]

def generate_training_data(num_images=800, sample_rate=61_440_000):
    pass
