
import numpy as np
from sklearn.model_selection import StratifiedKFold
import argparse

import config
import ml_features
import nn_features

from result import Result

import models.logistic_regression
import models.cnn
import models.svm

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
    y_ml, x_ml = np.array(y_ml), np.array(x_ml)
if run_all_models or args.nn:
    y_nn, x_nn = nn_features.extract_features(
        desired_size=config.cnn_input_size)
    y_nn, x_nn = np.array(y_nn), np.array(x_nn)

# Use KFold on the data
kf = StratifiedKFold(n_splits=10, shuffle=True)
index = 1
acc_per_fold = []
loss_per_fold = []

nn_results = []
ml_results = []
if args.ml or run_all_models:
    for train_index, test_index in kf.split(x_ml, y_ml):
        x_ml_train, x_ml_test = x_ml[train_index], x_ml[test_index]
        y_ml_train, y_ml_test = y_ml[train_index], y_ml[test_index]

        # Add models to run
        ml_models = []
        ml_models.append(
            ("Logistic Regression", models.logistic_regression.run_model))
        ml_models.append(
            ("SVM", models.svm.run_model))

        # Run all models
        for name, model in ml_models:
            try:
                ml_results.append((name, model(
                    x_ml_train, x_ml_test, y_ml_train, y_ml_test)))
            except Exception as e:
                ml_results.append((name, Result(error=str(e))))
if args.nn or run_all_models:
    for train_index, test_index in kf.split(x_nn, y_nn):
        x_nn_train, x_nn_test = x_nn[train_index], x_nn[test_index]
        y_nn_train, y_nn_test = y_nn[train_index], y_nn[test_index]

        # Add models to run
        nn_models = []
        nn_models.append(
            ("CNN", models.cnn.run_model))

        # Run all models
        for name, model in nn_models:
            try:
                nn_results.append((name, model(
                    x_nn_train, x_nn_test, y_nn_train, y_nn_test)))
            except Exception as e:
                nn_results.append((name, Result(error=str(e))))
                print(str(e))

# Agreggate the results for the k fold runs
if args.ml or run_all_models:
    ml_results_dict = dict()
    for name, ml_result in ml_results:
        if name in ml_results_dict:
            ml_results_dict[name].append(ml_result)
        else:
            ml_results_dict[name] = [ml_result]

    # find mean and average accuracy for each model
    ml_results = []
    for name, results in ml_results_dict.items():
        accuracies = np.array([result.test_acc for result in results])
        test_acc_mean = np.mean(accuracies)
        test_acc_std = np.std(accuracies)
        ml_results.append(
            (name, Result(test_acc_mean=test_acc_mean, test_acc_std=test_acc_std)))

if args.nn or run_all_models:
    nn_results_dict = dict()
    for name, nn_result in nn_results:
        if name in nn_results_dict:
            nn_results_dict[name].append(nn_result)
        else:
            nn_results_dict[name] = [nn_result]

    # find mean and average accuracy for each model
    nn_results = []
    for name, results in nn_results_dict.items():
        accuracies = np.array([result.test_acc for result in results])
        test_acc_mean = np.mean(accuracies)
        test_acc_std = np.std(accuracies)
        nn_results.append(
            (name, Result(test_acc_mean=test_acc_mean, test_acc_std=test_acc_std)))


# Print results
if args.nn or run_all_models:
    print("-------------------------------------")
    print("Neural network results")
    print("-------------------------------------")
    for name, result in nn_results:
        print(f"{name}")
        if result.error:
            print(f"\tError: {result.error}")
            continue
        if result.test_acc_mean and result.test_acc_std:
            print(
                f"\tTest Accuracy: {result.test_acc_mean} (+- {result.test_acc_std})')")
if args.ml or run_all_models:
    print("-------------------------------------")
    print("Machine Learning results")
    print("-------------------------------------")
    for name, result in ml_results:
        print(f"{name}")
        if result.error:
            print(f"\tError: {result.error}")
            continue
        if result.test_acc_mean and result.test_acc_std:
            print(
                f"\tTest Accuracy: {result.test_acc_mean} (+- {result.test_acc_std})')")
