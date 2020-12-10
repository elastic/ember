#!/usr/bin/env python

import os
import json
import ember
import argparse


def main():
    prog = "train_ember"
    descr = "Train an ember model from a directory with raw feature files"
    parser = argparse.ArgumentParser(prog=prog, description=descr)
    parser.add_argument("-v", "--featureversion", type=int, default=2, help="EMBER feature version")
    parser.add_argument("-m", "--metadata", action="store_true", help="Create metadata CSVs")
    parser.add_argument("-t", "--train", action="store_true", help="Train an EMBER model")
    parser.add_argument("datadir", metavar="DATADIR", type=str, help="Directory with raw features")
    parser.add_argument("--optimize", help="gridsearch to find best parameters", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.datadir) or not os.path.isdir(args.datadir):
        parser.error("{} is not a directory with raw feature files".format(args.datadir))

    X_train_path = os.path.join(args.datadir, "X_train.dat")
    y_train_path = os.path.join(args.datadir, "y_train.dat")
    if not (os.path.exists(X_train_path) and os.path.exists(y_train_path)):
        print("Creating vectorized features")
        ember.create_vectorized_features(args.datadir, args.featureversion)
        
    if args.metadata:
        ember.create_metadata(args.datadir)

    if args.train:
        params = {
            "boosting": "gbdt",
            "objective": "binary",
            "num_iterations": 1000,
            "learning_rate": 0.05,
            "num_leaves": 2048,
            "max_depth": 15,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.5
        }
        if args.optimize:
            params = ember.optimize_model(args.datadir)
            print("Best parameters: ")
            print(json.dumps(params, indent=2))

        print("Training LightGBM model")
        lgbm_model = ember.train_model(args.datadir, params, args.featureversion)
        lgbm_model.save_model(os.path.join(args.datadir, "model.txt"))


if __name__ == "__main__":
    main()
