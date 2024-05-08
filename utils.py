import os
from typing import Optional
import ipdb
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal, Normal


###########################################
# Config
###########################################
def get_config(args, method, dataset_name):
    config = vars(args).copy()

    if args.config_folder == "default":
        # define hyperparameters manually from args
        tqdm.write("Use default hyperparameters from argparser.")
        arg_dict = vars(args)
        config.update(arg_dict)

        if "boost" not in method.lower():
            config['standardize'], config['to_numpy'] = True, False
        else:
            config['standardize'], config['to_numpy'] = False, True
    else:
        file_path = os.path.join(args.config_folder, f"config_{method.replace('-', '_')}.csv")
        # check if config file exists
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            # filter dataset
            dataset_filter = df["dataset_name"] == dataset_name
            filtered_df = df[dataset_filter]
            # check if dataset and seed exists
            if len(filtered_df) > 0:
                for key in config:
                    if key in df.columns:
                        config[key] = filtered_df[key].values[0]
            else:
                raise Exception(f"No config file found for {method}, {dataset_name}, {seed} in {file_path}")
        else:
            raise Exception(f"No config file found for {method} in {file_path}")

    return config


###########################################
# Metrics
###########################################

def compute_metrics(
        y_true: Tensor,
        y_pred: Tensor,
        y_std: Tensor,
        y_train: Optional[Tensor] = None,
        quantile: Optional[float] = 95.0,
):
    metrics = {}

    # root mean squared error
    metrics['RMSE'] = mean_squared_error(y_true, y_pred, squared=False)
    # negative log predictive density
    metrics['NLPD'] = negative_log_predictive_density(y_true, y_pred, y_std)
    # coverage error
    metrics['Coverage'] = quantile_coverage_error(y_true, y_pred, y_std, quantile=quantile)
    # average interval length
    metrics['Interval Len'] = average_interval_length(y_std, quantile=quantile)

    return metrics


def mean_squared_error(
        y_true: Tensor,
        y_pred: Tensor,
        squared: Optional[bool] = True,
) -> float:
    if squared:
        return torch.mean((y_pred - y_true) ** 2, dim=-1).item()
    else:
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=-1)).item()


def negative_log_predictive_density(
        y_true: Tensor,
        y_pred: Tensor,
        y_std: Tensor,
) -> float:
    covariance_matrix = torch.diag(y_std ** 2)
    mvn = MultivariateNormal(y_pred, covariance_matrix)
    nlpd = -mvn.log_prob(y_true) / y_true.shape[0]
    return nlpd.item()


def quantile_coverage_error(
        y_true: Tensor,
        y_pred: Tensor,
        y_std: Tensor,
        quantile: float = 95.0,
) -> float:
    if quantile <= 0 or quantile >= 100:
        raise NotImplementedError("Quantile must be between 0 and 100")
    standard_normal = Normal(loc=0.0, scale=1.0)
    deviation = standard_normal.icdf(torch.as_tensor(0.5 + 0.5 * (quantile / 100)))
    lower = y_pred - deviation * y_std
    upper = y_pred + deviation * y_std
    n_samples_within_bounds = ((y_true > lower) * (y_true < upper)).sum(-1)
    fraction = n_samples_within_bounds / y_true.shape[-1]
    cov_error = torch.abs(fraction - quantile / 100)
    return cov_error.item()


def average_interval_length(
        y_std: Tensor,
        quantile: float = 95.0,
) -> float:
    if quantile <= 0 or quantile >= 100:
        raise NotImplementedError("Quantile must be between 0 and 100")
    standard_normal = Normal(loc=0.0, scale=1.0)
    deviation = standard_normal.icdf(torch.as_tensor(0.5 + 0.5 * (quantile / 100)))
    interval_length = 2 * deviation * y_std
    return interval_length.mean().item()


###########################################
# Preprocessing
###########################################
class Preprocessing:
    def __init__(
            self,
            config: dict,
            device: torch.device = torch.device('cpu'),
    ):
        """
        by default the data is in the forward pass converted to a torch tensor
        and in the backward pass also converted to a torch tensor
        """
        self.config = config
        self.standardize = config['standardize']
        self.to_numpy = config['to_numpy']
        self.device = device
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_scaler.fit(X)
        self.y_scaler.fit(y.reshape(-1, 1))

    def transform(self, X: np.ndarray, y: np.ndarray):
        X = self.X_scaler.transform(X)
        y = self.y_scaler.transform(y.reshape(-1, 1)).reshape(-1)
        return X, y

    def inverse_transform(self, y: np.ndarray):
        y = y * self.y_scaler.scale_.item() + self.y_scaler.mean_.item()
        return y

    def preprocess(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray
    ):
        if self.standardize:
            # if data is torch tensor
            flag = 0
            if type(X_train) == torch.Tensor:
                flag = 1
                device = X_train.device
                # convert to numpy first
                X_train, y_train = X_train.cpu().numpy(), y_train.cpu().numpy()
                X_test, y_test = X_test.cpu().numpy(), y_test.cpu().numpy()
            self.fit(X_train, y_train)
            X_train, y_train = self.transform(X_train, y_train)
            X_test, y_test = self.transform(X_test, y_test)
            if flag == 1:
                # convert back to torch tensor
                X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
                y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
                X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
                y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

        if not self.to_numpy:
            # check if already torch tensor
            if type(X_train) != torch.Tensor:
                X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
                y_train = torch.tensor(y_train, dtype=torch.float32, device=self.device)
                X_test = torch.tensor(X_test, dtype=torch.float32, device=self.device)
                y_test = torch.tensor(y_test, dtype=torch.float32, device=self.device)
            else:
                X_train = X_train.to(self.device)
                y_train = y_train.to(self.device)
                X_test = X_test.to(self.device)
                y_test = y_test.to(self.device)
        return X_train, y_train, X_test, y_test

    def postprocess(
            self,
            y_mean,
            y_std,
            y_test,
            y_train,
    ):
        if self.to_numpy:
            y_mean = torch.tensor(y_mean, dtype=torch.float32, device=self.device)
            y_std = torch.tensor(y_std, dtype=torch.float32, device=self.device)
            # if y_test is not tensor
            if type(y_test) != torch.Tensor:
                y_test = torch.tensor(y_test, dtype=torch.float32, device=self.device)
            else:
                y_test = y_test.to(self.device)
            if type(y_train) != torch.Tensor:
                y_train = torch.tensor(y_train, dtype=torch.float32, device=self.device)
            else:
                y_train = y_train.to(self.device)
        if self.standardize:
            y_mean = self.inverse_transform(y_mean)
            y_std = self.y_scaler.scale_.item() * y_std
            y_test = self.inverse_transform(y_test)
            y_train = self.inverse_transform(y_train)
        return y_mean, y_std, y_test, y_train
