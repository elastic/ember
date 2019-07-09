#!/usr/bin/env python

import os
import ember
import argparse
import json


def main():
    prog = "train_ember"
    descr = "Train an ember model from a directory with raw feature files"
    parser = argparse.ArgumentParser(prog=prog, description=descr)
    parser.add_argument("-y", "--year", type=int, default=2018, help="EMBER data year")
    parser.add_argument("datadir", metavar="DATADIR", type=str, help="Directory with raw features")
    parser.add_argument("--optimize", help="gridsearch to find best parameters", action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.datadir) or not os.path.isdir(args.datadir):
        parser.error("{} is not a directory with raw feature files".format(args.datadir))

    X_train_path = os.path.join(args.datadir, "X_train.dat")
    y_train_path = os.path.join(args.datadir, "y_train.dat")
    if not (os.path.exists(X_train_path) and os.path.exists(y_train_path)):
        print("Creating vectorized features")
        ember.create_vectorized_features(args.datadir, args.year)

    params = {}
    if args.optimize:
        params = ember.optimize_model( args.datadir )
        print("Best parameters: ")
        print( json.dumps( params, indent=2 ) )

    print("Training LightGBM model")
    lgbm_model = ember.train_model(args.datadir, params, args.year)
    lgbm_model.save_model(os.path.join(args.datadir, "model.txt"))

if __name__ == "__main__":
    main()
