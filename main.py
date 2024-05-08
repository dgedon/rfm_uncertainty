import argparse
import json
import math
import os
import pickle
import time
from datetime import datetime
import ipdb
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_tabular_benchmark import DataTabularBenchmarkRegression
from models import GPRBF, GPLaplace, GPDeepKL, GPLaplaceARDFull, GPRFMLaplace, NGBoost, CatBoostEnsemble
from utils import get_config, compute_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general arguments
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_false')
    parser.add_argument('--config_folder', type=str, default='config/tabularbenchmark', help='set to default if want to use args values')
    parser.add_argument('--test_ratio', type=float, default=0.3)
    parser.add_argument('--dataset_idx', type=int, default=0) # 3

    # GP specific arguments
    parser.add_argument('--n_iter', type=int, default=250, help='Number of iterations for CatBoost or GPs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for all GP-based methods')
    parser.add_argument('--ard_num_dims', type=int, default=1, help='Number of dimensions for ARD')
    # RFM specific arguments
    parser.add_argument('--rfm_reg', type=float, default=0.1, help='Regularization for RFM')
    parser.add_argument('--rfm_ridge', type=float, default=0.1, help='Ridge for RFM')
    parser.add_argument('--rfm_n_iter', type=int, default=5, help='Number of iterations for RFM')
    # NGBoost specific arguments
    parser.add_argument('--n_estimators', type=int, default=500, help='Number of estimators for NGBoost')
    # CatBoost specific arguments
    parser.add_argument('--depth', type=int, default=6, help='Depth for CatBoost')
    parser.add_argument('--n_ensembles', type=int, default=10, help='Number of ensembles for CatBoost')
    # preprocessing arguments
    parser.add_argument('--standardize', action='store_false', help='Standardize data')  # boosting False, else True
    parser.add_argument('--to_numpy', action='store_true', help='Convert data to numpy')  # boosting True, else False
    args = parser.parse_args()

    # seed everything
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # list of methods
    methods_list = [
        "GP-RBF", "GP-Laplace",
        "GP-deepKL-RBF",
        "GP-ARD-RBF", "GP-ARD-Laplace",
        "GP-ARD-Laplace-full",
        "GP-RFM-Laplace", "GP-RFM-Laplace-diag",
        "NG-Boost", "Cat-Boost-Ensemble",
    ]

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ############################################
    # dataset
    ############################################
    dataset = DataTabularBenchmarkRegression(test_ratio=args.test_ratio,
                                             random_state=seed,
                                             verbose=args.verbose)
    dataset_len = len(dataset)

    ############################################
    # metrics
    ############################################
    # initialize metrics data
    metrics_data = dict()

    # dummy call to compute_metrics to get the list of metrics
    metrics_list = list(compute_metrics(torch.ones(2), torch.ones(2), torch.ones(2), torch.ones(2)).keys()) + ["Time"]

    ############################################
    # get data
    ############################################
    data_train, data_test, dataset_name = dataset.get_data(args.dataset_idx)

    n_train, n_test = data_train[0].shape[0], data_test[0].shape[0]
    n_features = data_train[0].shape[1]

    metrics_data["dataset"] = dataset_name.replace("_", "-")
    metrics_data["samples"] = n_train + n_test
    metrics_data["features"] = n_features

    ############################################
    # Loop over all methods
    ############################################
    for method in methods_list:
        # start timer
        time_start = time.time()

        # get hyperparameters
        config = get_config(args, method, dataset_name)
        config["ard_num_dims"] = n_features if "ARD" in method else 1

        if method == "GP-RBF":
            model = GPRBF(data_train, data_test, config, device=device, verbose=args.verbose)
        elif method == "GP-Laplace":
            model = GPLaplace(data_train, data_test, config, device=device, verbose=args.verbose)
        elif method == "GP-ARD-RBF":
            model = GPRBF(data_train, data_test, config, device=device, verbose=args.verbose)
        elif method == "GP-ARD-Laplace":
            model = GPLaplace(data_train, data_test, config, device=device, verbose=args.verbose)
        elif method == "GP-ARD-Laplace-full":
            model = GPLaplaceARDFull(data_train, data_test, config, device=device, verbose=args.verbose)
        elif method == "GP-RFM-Laplace":
            model = GPRFMLaplace(data_train, data_test, config, device=device, verbose=args.verbose)
        elif method == "GP-RFM-Laplace-diag":
            model = GPRFMLaplace(data_train, data_test, config, device=device, verbose=args.verbose, diag=True)
        elif method == "NG-Boost":
            model = NGBoost(data_train, data_test, config, device=device, verbose=args.verbose)
        elif method == "Cat-Boost-Ensemble":
            model = CatBoostEnsemble(data_train, data_test, config, device=device, verbose=args.verbose)
        elif method == "GP-deepKL-RBF":
            model = GPDeepKL(data_train, data_test, config, device=device, verbose=args.verbose)
        else:
            raise NotImplementedError(f"Method {method} not implemented")

        try:
            # get preprocessed data
            X_train, y_train, X_test, y_test = model.preprocessed_data()

            # fit model
            model.fit(X_train, y_train, desc=f"Training {method}")

            # predict
            y_mean, y_std = model.predict(X_test)

            # postprocessing
            y_mean, y_std, y_test, y_train = model.postprocess_data(y_mean, y_std, y_test, y_train)

            # compute metrics
            time_end = time.time()
            metrics = compute_metrics(y_test, y_mean, y_std, y_train)
            metrics["Time"] = time_end - time_start
        except Exception as e:
            tqdm.write(f"Error in {method} for {dataset_name}. Error: {e}")

            # set metrics to NaN
            metrics = {metric: np.nan for metric in metrics_list}
            # set time to NaN
            metrics["Time"] = np.nan

        # save metrics
        for metric in metrics:
            metrics_data[method + ' ' + metric] = metrics[metric]

        # clear memory
        try:
            del model, X_train, y_train, X_test, y_test, y_mean, y_std
        except:
            pass
        torch.cuda.empty_cache()

    # print results
    if args.verbose:
        tqdm.write(f'{"Model":20} | {"RMSE":>10} | {"NLL":>10} | {"Covergage":>10} | {"Intv. Len":>10} | {"Time":>10}')
        tqdm.write(f'{"-" * 20} | {"-" * 10} | {"-" * 10} |{"-" * 10} | {"-" * 10} | {"-" * 10} ')
        for method in methods_list:
            tqdm.write(f'{method:20} '
                       f'| {metrics_data[method + " RMSE"]:10.3f} '
                       f'| {metrics_data[method + " NLPD"]:10.3f} '
                       f'| {metrics_data[method + " Coverage"]:10.3f} '
                       f'| {metrics_data[method + " Interval Len"]:10.3f} '
                       f'| {metrics_data[method + " Time"]:10.3f}')
