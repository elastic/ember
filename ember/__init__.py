# -*- coding: utf-8 -*-

import os
import json
import tqdm
import numpy as np
import pandas as pd
import lightgbm as lgb
import multiprocessing
from .features import PEFeatureExtractor


def raw_feature_iterator(file_paths):
    """
    Yield raw feature strings from the inputed file paths
    """
    for path in file_paths:
        with open(path, "r") as fin:
            for line in fin:
                yield line


def vectorize(irow, raw_features_string, X_path, y_path, nrows):
    """
    Vectorize a single sample of raw features and write to a large numpy file
    """
    extractor = PEFeatureExtractor()
    raw_features = json.loads(raw_features_string)
    feature_vector = extractor.process_raw_features(raw_features)

    y = np.memmap(y_path, dtype=np.float32, mode="r+", shape=nrows)
    y[irow] = raw_features["label"]

    X = np.memmap(X_path, dtype=np.float32, mode="r+", shape=(nrows, extractor.dim))
    X[irow] = feature_vector


def vectorize_unpack(args):
    """
    Pass through function for unpacking vectorize arguments
    """
    return vectorize(*args)


def vectorize_subset(X_path, y_path, raw_feature_paths, nrows):
    """
    Vectorize a subset of data and write it to disk
    """
    # Create space on disk to write features to
    extractor = PEFeatureExtractor()
    X = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(nrows, extractor.dim))
    y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=nrows)
    del X, y

    # Distribute the vectorization work
    pool = multiprocessing.Pool()
    argument_iterator = ((irow, raw_features_string, X_path, y_path, nrows)
                         for irow, raw_features_string in enumerate(raw_feature_iterator(raw_feature_paths)))
    for _ in tqdm.tqdm(pool.imap_unordered(vectorize_unpack, argument_iterator), total=nrows):
        pass


def create_vectorized_features(data_dir):
    """
    Create feature vectors from raw features and write them to disk
    """
    print("Vectorizing training set")
    X_path = os.path.join(data_dir, "X_train.dat")
    y_path = os.path.join(data_dir, "y_train.dat")
    raw_feature_paths = [os.path.join(data_dir, "train_features_{}.jsonl".format(i)) for i in range(6)]
    vectorize_subset(X_path, y_path, raw_feature_paths, 900000)

    print("Vectorizing test set")
    X_path = os.path.join(data_dir, "X_test.dat")
    y_path = os.path.join(data_dir, "y_test.dat")
    raw_feature_paths = [os.path.join(data_dir, "test_features.jsonl")]
    vectorize_subset(X_path, y_path, raw_feature_paths, 200000)


def read_vectorized_features(data_dir, subset=None):
    """
    Read vectorized features into memory mapped numpy arrays
    """
    if subset is not None and subset not in ["train", "test"]:
        return None

    ndim = PEFeatureExtractor.dim
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    if subset is None or subset == "train":
        X_train_path = os.path.join(data_dir, "X_train.dat")
        y_train_path = os.path.join(data_dir, "y_train.dat")
        X_train = np.memmap(X_train_path, dtype=np.float32, mode="r", shape=(900000, ndim))
        y_train = np.memmap(y_train_path, dtype=np.float32, mode="r", shape=900000)
        if subset == "train":
            return X_train, y_train

    if subset is None or subset == "test":
        X_test_path = os.path.join(data_dir, "X_test.dat")
        y_test_path = os.path.join(data_dir, "y_test.dat")
        X_test = np.memmap(X_test_path, dtype=np.float32, mode="r", shape=(200000, ndim))
        y_test = np.memmap(y_test_path, dtype=np.float32, mode="r", shape=200000)
        if subset == "test":
            return X_test, y_test

    return X_train, y_train, X_test, y_test


def read_metadata_record(raw_features_string):
    """
    Decode a raw features stringa and return the metadata fields
    """
    full_metadata = json.loads(raw_features_string)
    return {"sha256": full_metadata["sha256"], "appeared": full_metadata["appeared"], "label": full_metadata["label"]}


def create_metadata(data_dir):
    """
    Write metadata to a csv file and return its dataframe
    """
    pool = multiprocessing.Pool()

    train_feature_paths = [os.path.join(data_dir, "train_features_{}.jsonl".format(i)) for i in range(6)]
    train_records = list(pool.imap(read_metadata_record, raw_feature_iterator(train_feature_paths)))
    train_records = [dict(record, **{"subset": "train"}) for record in train_records]

    test_feature_paths = [os.path.join(data_dir, "test_features.jsonl")]
    test_records = list(pool.imap(read_metadata_record, raw_feature_iterator(test_feature_paths)))
    test_records = [dict(record, **{"subset": "test"}) for record in test_records]

    metadf = pd.DataFrame(train_records + test_records)[["sha256", "appeared", "subset", "label"]]
    metadf.to_csv(os.path.join(data_dir, "metadata.csv"))
    return metadf


def read_metadata(data_dir):
    """
    Read an already created metadata file and return its dataframe
    """
    return pd.read_csv(os.path.join(data_dir, "metadata.csv"), index_col=0)


def train_model(data_dir):
    """
    Train the LightGBM model from the EMBER dataset from the vectorized features
    """
    # Read data
    X_train, y_train = read_vectorized_features(data_dir, subset="train")

    # Filter unlabeled data
    train_rows = (y_train != -1)

    # Train
    lgbm_dataset = lgb.Dataset(X_train[train_rows], y_train[train_rows])
    lgbm_model = lgb.train({"application": "binary"}, lgbm_dataset)

    return lgbm_model


def predict_sample(lgbm_model, file_data):
    """
    Predict a PE file with an LightGBM model
    """
    extractor = PEFeatureExtractor()
    features = np.array(extractor.feature_vector(file_data), dtype=np.float32)
    return lgbm_model.predict([features])[0]
