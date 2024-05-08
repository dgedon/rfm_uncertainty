from typing import Tuple, List, Optional, Dict
import ipdb

import numpy as np
import torch
from torch import Tensor
from torch import nn
from tqdm import tqdm, trange
from catboost import CatBoostRegressor
from gpytorch.likelihoods import GaussianLikelihood
from models_gp import GPARDModel, GPLaplaceMahalanobisModel, GPMahalanobisARDFullModel, GPDeepKLModel, train_gp, predict_gp
from ngboost import NGBRegressor
from utils import Preprocessing
from recursive_feature_machine import LaplaceRFM


class ModelGeneric(nn.Module):
    def __init__(self,
                 data_train: Tuple[Tensor, Tensor],
                 data_test: Tuple[Tensor, Tensor],
                 config: Dict,
                 device: torch.device = torch.device("cpu"),
                 verbose: Optional[bool] = False,
                 ):
        super(ModelGeneric, self).__init__()
        self.config = config
        self.device = device
        self.verbose = verbose

        # define preprocessing
        self.preprocessing = Preprocessing(config, device)
        X_train, y_train, X_test, y_test = self.preprocessing.preprocess(*data_train, *data_test)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.likelihood = None
        self.model = None
        self.weight_matrix = None
        self.def_model()

    def def_model(self):
        raise NotImplementedError

    def preprocessed_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test

    def postprocess_data(self, y_mean: Tensor, y_std: Tensor,
                         y_test: Tensor, y_train: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.preprocessing.postprocess(y_mean, y_std, y_test, y_train)

    def fit(self, X: Tensor, y: Tensor, desc: str = "Training") -> None:
        raise NotImplementedError

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class GPGeneric(ModelGeneric):
    def __init__(self,
                 data_train: Tuple[Tensor, Tensor],
                 data_test: Tuple[Tensor, Tensor],
                 config: Dict,
                 device: torch.device = torch.device("cpu"),
                 verbose: Optional[bool] = False,
                 ):
        super(GPGeneric, self).__init__(data_train, data_test, config, device, verbose)

    def def_model(self):
        # define GP
        self.likelihood = GaussianLikelihood().to(self.device)
        self.model = GPARDModel(
            self.X_train, self.y_train,
            self.likelihood,
            ard_num_dims=self.config["ard_num_dims"]
        ).to(self.device)

    def fit(self, X: Tensor, y: Tensor, desc: str = "Training") -> None:
        self.model, self.likelihood = train_gp(
            X, y,
            self.model, self.likelihood,
            num_iter=self.config["n_iter"],
            lr=self.config["lr"],
            desc=desc,
            verbose=self.verbose,
        )

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        y_mean, y_std = predict_gp(X, self.model, self.likelihood)
        return y_mean, y_std


class GPRBF(GPGeneric):
    def __init__(self,
                 data_train: Tuple[Tensor, Tensor],
                 data_test: Tuple[Tensor, Tensor],
                 config: Dict,
                 device: torch.device = torch.device("cpu"),
                 verbose: Optional[bool] = False,
                 ):
        super(GPRBF, self).__init__(data_train, data_test, config, device, verbose)

    def def_model(self):
        # define GP
        self.likelihood = GaussianLikelihood().to(self.device)
        self.model = GPARDModel(
            self.X_train, self.y_train,
            self.likelihood,
            ard_num_dims=self.config["ard_num_dims"]
        ).to(self.device)


class GPLaplace(GPGeneric):
    def __init__(self,
                 data_train: Tuple[Tensor, Tensor],
                 data_test: Tuple[Tensor, Tensor],
                 config: Dict,
                 device: torch.device = torch.device("cpu"),
                 verbose: Optional[bool] = False,
                 ):
        super(GPLaplace, self).__init__(data_train, data_test, config, device, verbose)

    def def_model(self):
        # define GP
        self.likelihood = GaussianLikelihood().to(self.device)
        self.model = GPLaplaceMahalanobisModel(
            self.X_train, self.y_train,
            self.likelihood,
            ard_num_dims=self.config["ard_num_dims"]
        ).to(self.device)


class GPLaplaceARDFull(GPGeneric):
    def __init__(self,
                 data_train: Tuple[Tensor, Tensor],
                 data_test: Tuple[Tensor, Tensor],
                 config: Dict,
                 device: torch.device = torch.device("cpu"),
                 verbose: Optional[bool] = False,
                 ):
        super(GPLaplaceARDFull, self).__init__(data_train, data_test, config, device, verbose)

    def def_model(self):
        # define GP
        self.likelihood = GaussianLikelihood().to(self.device)
        self.model = GPMahalanobisARDFullModel(
            self.X_train, self.y_train,
            self.likelihood,
            ard_num_dims=1,
            squared=False,
        ).to(self.device)


class GPDeepKL(GPGeneric):
    def __init__(self,
                 data_train: Tuple[Tensor, Tensor],
                 data_test: Tuple[Tensor, Tensor],
                 config: Dict,
                 device: torch.device = torch.device("cpu"),
                 verbose: Optional[bool] = False,
                 ):
        super(GPDeepKL, self).__init__(data_train, data_test, config, device, verbose)

    def def_model(self):
        # define GP
        self.likelihood = GaussianLikelihood().to(self.device)
        self.model = GPDeepKLModel(
            self.X_train, self.y_train,
            self.likelihood,
        ).to(self.device)


class GPRFMLaplace(GPGeneric):
    def __init__(self,
                 data_train: Tuple[Tensor, Tensor],
                 data_test: Tuple[Tensor, Tensor],
                 config: Dict,
                 device: torch.device = torch.device("cpu"),
                 verbose: Optional[bool] = False,
                 diag: Optional[bool] = False,
                 ):
        self.diag = diag
        super(GPRFMLaplace, self).__init__(data_train, data_test, config, device, verbose)

    def def_model(self, weight_matrix=None):
        # define RFM
        rfm = LaplaceRFM(
            mem_gb=38,  # 38 GB, 2 GB for other stuff
            bandwidth=1,
            device=self.device,
            reg=self.config["rfm_reg"],
            ridge=self.config["rfm_ridge"],
            verbose=False,
            diag=self.diag,
            centering=True,
        )
        # fit RFM
        rfm.fit(
            (self.X_train, self.y_train), (self.X_test, self.y_test),
            loader=False,
            iters=self.config["rfm_n_iter"],
            classif=False,
        )

        # weight matrix
        if weight_matrix is None:
            self.weight_matrix = rfm.M if not self.diag else torch.diag(rfm.M)
        else:
            self.weight_matrix = weight_matrix

        # define GP
        self.likelihood = GaussianLikelihood().to(self.device)
        self.model = GPLaplaceMahalanobisModel(
            self.X_train, self.y_train,
            self.likelihood,
            # diag=self.diag,
            weight_matrix=self.weight_matrix,
            ard_num_dims=1
        ).to(self.device)
        del rfm
        torch.cuda.empty_cache()


class NGBoost(ModelGeneric):
    def __init__(self,
                 data_train: Tuple[Tensor, Tensor],
                 data_test: Tuple[Tensor, Tensor],
                 config: Dict,
                 device: torch.device = torch.device("cpu"),
                 verbose: Optional[bool] = False,
                 ):
        super(NGBoost, self).__init__(data_train, data_test, config, device, verbose)

    def def_model(self):
        self.model = NGBRegressor(
            natural_gradient=True,
            n_estimators=self.config["n_estimators"],
            minibatch_frac=1.0,
            verbose=False,
            random_state=self.config["seed"],
        )

    def fit(self, X: Tensor, y: Tensor, desc: str = "Training") -> None:
        tqdm.write(desc) if self.verbose else None
        self.model.fit(X, y)

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        y_dists = self.model.pred_dist(X)
        y_mean, y_std = y_dists.loc, y_dists.scale
        return y_mean, y_std


class CatBoostEnsemble(ModelGeneric):
    def __init__(self,
                 data_train: Tuple[Tensor, Tensor],
                 data_test: Tuple[Tensor, Tensor],
                 config: Dict,
                 device: torch.device = torch.device("cpu"),
                 verbose: Optional[bool] = False,
                 ):
        super(CatBoostEnsemble, self).__init__(data_train, data_test, config, device, verbose)

    def def_model(self):
        self.model = []
        for i in range(self.config["n_ensembles"]):
            self.model.append(CatBoostRegressor(
                random_seed=self.config["seed"] + i,
                iterations=self.config["n_iter"],
                learning_rate=self.config["lr"],
                depth=self.config["depth"],
                loss_function='RMSEWithUncertainty',
                posterior_sampling=True,
                bootstrap_type='No',
                verbose=False,
                # task_type=task_type,  # posterior_sampling is unimplemented for task type GPU
            ))

    def fit(self, X: Tensor, y: Tensor, desc: str = "Training") -> None:
        for i in trange(self.config["n_ensembles"], desc=desc, disable=not self.verbose):
            self.model[i].fit(X, y)

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        ens_preds = np.zeros((self.config["n_ensembles"], X.shape[0], 2))
        for i in range(self.config["n_ensembles"]):
            ens_preds[i] = self.model[i].predict(X)
        y_mean = ens_preds.mean(axis=0)[:, 0]
        var_data = ens_preds.mean(axis=0)[:, 1]
        var_knowledge = ens_preds.var(axis=0)[:, 0]
        y_std = np.sqrt(var_data + var_knowledge)  # total std
        return y_mean, y_std
