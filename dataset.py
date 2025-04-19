#!/usr/bin/env python3
"""
This module loads the matlab datasets to be used for training the LTE/NR
classifier.
"""
import os
import shutil
import threading
import zipfile
from glob import glob
from os.path import basename, isdir, isfile, join

import numpy as np
import requests
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATA_URIS = [
    "https://www.mathworks.com/supportfiles/spc/SpectrumSensing/SpectrumSensingTrainingData128x128.zip",
    "https://www.mathworks.com/supportfiles/spc/SpectrumSensing/SpectrumSensingTrainedCustom_2024.zip",
    "https://www.mathworks.com/supportfiles/spc/SpectrumSensing/SpectrumSensingCapturedData128x128.zip",
]
DATA_DIR = "data"
IMG_SIZE = (128, 128)


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
        with open(join(data_dir, filename), "wb") as file:
            for chunk in rsp.iter_content(chunk_size=8192):
                file.write(chunk)

    with zipfile.ZipFile(join(DATA_DIR, filename), "r") as zip_r:
        zip_r.extractall(join(DATA_DIR, filename.replace(".zip", "")))
    assert isdir(
        join(DATA_DIR, filename.replace(".zip", ""))
    ), f"Failed to extract {filename}"


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


CLASSES = ["Noise", "NR", "LTE", "Unknown"]


def format_for_logistic_regression():
    """
    Move image files into training directory under CLASSES directory
    for label inference.
    """
    lr_dir = join(DATA_DIR, "lr_training_data")
    if not isdir(lr_dir):
        os.mkdir(lr_dir)
    for my_class in CLASSES:
        class_dir = join(lr_dir, my_class)
        if not isdir(class_dir):
            os.mkdir(class_dir)

    img_paths = glob(f"{DATA_DIR}/**/*.png", recursive=True)
    assert img_paths, f"No png files found in {DATA_DIR}, run download_dataset?"
    for img_path in img_paths:
        if lr_dir in img_path:
            continue
        # TODO: multiple signals in one spectrogram?
        if "lte_nr" in img_path.lower():
            continue
        for my_class in CLASSES:
            if my_class.lower() in img_path.lower():
                class_dir = join(lr_dir, my_class)
                shutil.copy2(img_path, class_dir)


def load_data(data_dir, img_size, test_split, normalize=True):
    """Load data from training dir

    Args:
        data_dir (): training data dir
        img_size (): tuple of image size e.g., (128, 128)
        test_split (): ratio of training data to testing data
        normalize: whether or not to normalize all images

    Returns: split of images and labels X_train, X_test, y_train, y_test, and
    list of class names

    """
    # Class names are from subfolders in the lr_training_data dir
    class_names = sorted(os.listdir(data_dir))

    images = []
    labels = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = tf.keras.utils.load_img(img_path, target_size=img_size)
            img_array = tf.keras.utils.img_to_array(img)
            images.append(img_array)
            labels.append(class_idx)

    images = np.array(images)
    labels = np.array(labels)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_split, random_state=42, stratify=labels
    )
    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    return X_train, X_test, y_train, y_test, class_names


# kkelso: this did not work loading the hdf files from matlab
# from osgeo import gdal
# from pyhdf import SD
# import pandas as pd
# from netCDF4 import Dataset
# import tables
# from PIL import Image
# def load_data():
#     """Load the downloaded data into numpy arrays for training the network.
#
#     Returns: tuple of spectogram images, and pixel labels
#
#     """
#     img_paths = glob(f"{DATA_DIR}/**/*.png", recursive=True)
#     assert img_paths, f"No png files found in {DATA_DIR}, run download_dataset?"
#     images = []
#     labels = []
#     for img_path in img_paths:
#         img = Image.open(img_path)
#         img = img.resize(IMG_SIZE)
#         label_path = img_path.replace('.png', '.hdf')
#         assert isfile(label_path), f"{img_path} has no corresponding .hdf"
#
#         hdf = tables.open_file(label_path)
#         print(hdf)
#         break
#
#
#         # print(pd.read_hdf(label_path))
#         # hdf_file = SD.SD(label_path, SD.SDC.READ)
#
#         # datasets = hdf_file.datasets()
#         # print("Datasets in the HDF4 file:", datasets)
#
#         # h5f = h5py.File(label_path, "r")
#         # print(h5f.keys())
#         # gdal_dataset = gdal.Open(label_path)
#         # print(f"gdal_dataset = {gdal_dataset.GetSubDatasets()}")
#         # subdataset_path = gdal_dataset.GetSubDatasets()[0][0]
#         # with h5py.File(subdataset_path, 'r') as f:
#         #     labels.append(np.array(f['data']))
#         #     images.append(np.array(img))
#         #     print("Success")
#
#     return np.array(images), np.array(labels)


def generate_training_data(num_images=800, sample_rate=61_440_000):
    pass
