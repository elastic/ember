#!/usr/bin/env python

import os
import ember
import argparse
import lightgbm as lgb


def main():
    prog = "classify_binaries"
    descr = "Use a trained ember model to make predictions on PE files"
    parser = argparse.ArgumentParser(prog=prog, description=descr)
    parser.add_argument("-v", "--featureversion", type=int, default=2, help="EMBER feature version")
    parser.add_argument("-m", "--modelpath", type=str, default=None, required=True, help="Ember model")
    parser.add_argument("binaries", metavar="BINARIES", type=str, nargs="+", help="PE files to classify")
    args = parser.parse_args()

    if not os.path.exists(args.modelpath):
        parser.error("ember model {} does not exist".format(args.modelpath))
    lgbm_model = lgb.Booster(model_file=args.modelpath)

    for binary_path in args.binaries:
        if not os.path.exists(binary_path):
            print("{} does not exist".format(binary_path))

        file_data = open(binary_path, "rb").read()
        score = ember.predict_sample(lgbm_model, file_data, args.featureversion)

        if len(args.binaries) == 1:
            print(score)

        else:
            print("\t".join((binary_path, str(score))))


if __name__ == "__main__":
    main()
