
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

import config
import ml_features
import nn_features

from result import Result

import models.logistic_regression
import models.cnn

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nn', action='store_true',
                    help='Run all neural network models')
parser.add_argument('-m', '--ml', action='store_true',
                    help='Run machine learning models')
args = parser.parse_args()
run_all_models = not args.nn and not args.ml

# Set up environment for NN
if run_all_models or args.nn:
    config.configure_tensorflow()

# Gather features
if run_all_models or args.ml:
    y_ml, x_ml = ml_features.extract_features()
if run_all_models or args.nn:
    y_nn, x_nn = nn_features.extract_features()

# Split training/testing data
if run_all_models or args.ml:
    x_ml_train, x_ml_test, y_ml_train, y_ml_test = train_test_split(x_ml,
                                                                    y_ml,
                                                                    test_size=config.test_split,
                                                                    stratify=y_ml)
    x_ml_train = np.array(x_ml_train)
    x_ml_test = np.array(x_ml_test)
    y_ml_train = np.array(y_ml_train)
    y_ml_test = np.array(y_ml_test)
if run_all_models or args.nn:
    x_nn_train, x_nn_test, y_nn_train, y_nn_test = train_test_split(x_nn,
                                                                    y_nn,
                                                                    test_size=config.test_split,
                                                                    stratify=y_nn)
    x_nn_train = np.array(x_nn_train)
    x_nn_test = np.array(x_nn_test)
    y_nn_train = np.array(y_nn_train)
    y_nn_test = np.array(y_nn_test)

# Determine which models to run
ml_models = []
nn_models = []
if args.nn or run_all_models:
    nn_models.append(
        ("CNN", models.cnn.run_model))
if args.ml or run_all_models:
    ml_models.append(
        ("Logistic Regression", models.logistic_regression.run_model))

# Run all models
ml_results = []
for name, model in ml_models:
    try:
        ml_results.append((name, model(
            x_ml_train, x_ml_test, y_ml_train, y_ml_test)))
    except Exception as e:
        ml_results.append((name, Result(error=str(e))))
nn_results = []
for name, model in nn_models:
    try:
        nn_results.append((name, model(
            x_nn_train, x_nn_test, y_nn_train, y_nn_test)))
    except Exception as e:
        nn_results.append((name, Result(error=str(e))))

# Print results
if args.nn or run_all_models:
    print("-------------------------------------")
    print("Neural network results")
    print("-------------------------------------")
    for name, result in nn_results:
        print(f"{name:<15}")
        if result.error:
            print(f"\tError: {result.error}")
            continue
        if result.test_acc:
            print(f"\tTest Accuracy: {result.test_acc: < 7}")
if args.ml or run_all_models:
    print("-------------------------------------")
    print("Machine Learning results")
    print("-------------------------------------")
    for name, result in ml_results:
        print(f"{name:<15}")
        if result.error:
            print(f"\tError: {result.error}")
            continue
        if result.test_acc:
            print(f"\tTest Accuracy: {result.test_acc: < 7}")
