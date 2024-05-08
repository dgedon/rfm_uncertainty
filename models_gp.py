from typing import Tuple, Optional

import ipdb
import gpytorch
import torch
from gpytorch.constraints import Positive, Interval
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import Prior
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm, trange


def train_gp(
        X_train: Tensor,
        y_train: Tensor,
        model: torch.nn.Module,
        likelihood: Likelihood,
        num_iter: int = 200,
        lr: float = 0.1,
        use_scheduler: Optional[bool] = True,
        desc: Optional[str] = '',
        verbose: Optional[bool] = False,
        eta_min: Optional[float] = 1e-7,
):
    """
    Trains a GP model with the given training data and returns the trained model and likelihood.

    Args:
        X_train: The training data.
        y_train: The training labels.
        model: The GP model.
        likelihood: The likelihood for the GP model.
        num_iter: The number of iterations to train for.
        lr: The learning rate.
        use_scheduler: Whether to use a learning rate scheduler.
        verbose: Whether to print training progress.

    Returns:
        The trained model and likelihood.
    """
    model.train()
    likelihood.train()

    # Create an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # , weight_decay=1e-8)

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=eta_min) if use_scheduler else None

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Training loop
    with trange(num_iter, desc=desc, disable=not verbose) as pbar:
        for _ in pbar:
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(X_train)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()
            if use_scheduler:
                scheduler.step()
            pbar.set_postfix(train_loss=loss.item())
    del loss, output, X_train, y_train, optimizer, mll, scheduler
    torch.cuda.empty_cache()
    return model, likelihood


def predict_gp(
        X_test: Tensor,
        model: torch.nn.Module,
        likelihood: Likelihood,
):
    model.eval()
    likelihood.eval()
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(X_test))

    del X_test, model, likelihood
    torch.cuda.empty_cache()
    return observed_pred.mean, observed_pred.stddev


def sq_dist_M(x1: Tensor, x2: Tensor, M: Tensor, x1_eq_x2: bool = False) -> Tensor:
    """Compute the squared Mahalanobis distance ||x1 - x2||_M^2."""
    adjustment = x1.mean(dim=-2, keepdim=True)
    x1 = x1 - adjustment

    # Compute squared distance matrix using quadratic expansion
    x1_norm = (x1.matmul(M) * x1).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)

    if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
        x2, x2_norm, x2_pad = x1, x1_norm, x1_pad
    else:
        x2 = x2 - adjustment
        x2_norm = (x2.matmul(M) * x2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)

    x1_ = torch.cat([-2.0 * x1.matmul(M), x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))
    del x1_, x2_, x1_norm, x2_norm, x1_pad, x2_pad, adjustment
    torch.cuda.empty_cache()

    if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
        res.diagonal(dim1=-2, dim2=-1).fill_(0)

    # Zero out negative values
    return res.clamp_min_(0)


def dist_M(x1: Tensor, x2: Tensor, M: Tensor, x1_eq_x2: bool = False) -> Tensor:
    if not x1_eq_x2:
        # res = torch.cdist(x1,x2)  # batched_mahalanobis_distance(x1, x2, M)
        res = batched_mahalanobis_distance(x1, x2, M)
        """
        This gives negative values which are clamped to zeros
        But also this is only if x1neq_x2 so figure out when this happens and try to recreate in test.py file to 
        see where the problem lies
        """
        return res.clamp_min(1e-15)
    res = sq_dist_M(x1, x2, M, x1_eq_x2=x1_eq_x2)
    return res.clamp_min_(1e-30).sqrt_()


def batched_mahalanobis_distance(x1, x2, weight_matrix):
    x1_norm2 = ((x1 @ weight_matrix) * x1).sum(-1)
    x2_norm2 = ((x2 @ weight_matrix) * x2).sum(-1)

    dist = -2 * (x1 @ weight_matrix) @ x2.T
    dist.add_(x1_norm2.view(-1, 1))
    dist.add_(x2_norm2)
    dist.clamp_(min=0).sqrt_()
    del x1_norm2, x2_norm2, x1, x2, weight_matrix
    torch.cuda.empty_cache()
    return dist


class ExpMahalanobisDistanceKernel(Kernel):
    """
    Mahalanobis distance kernel with exponential transformation
    can be used with squared distance (for RBF like kernel or without for Laplace like kernel)
    """
    has_lengthscale = True

    def __init__(
            self,
            weight_matrix: Tensor,
            squared: Optional[bool] = False,
            ard_num_dims: Optional[int] = None,
            lengthscale_prior: Optional[Prior] = None,
            lengthscale_constraint: Optional[Interval] = None,
            eps: Optional[float] = 1e-6,
            **kwargs,
    ):
        super(ExpMahalanobisDistanceKernel, self).__init__()
        self.covariance_matrix = weight_matrix
        self.squared = squared
        self.ard_num_dims = ard_num_dims
        self.eps = eps

        if lengthscale_constraint is None:
            lengthscale_constraint = Positive()

        if self.has_lengthscale:
            lengthscale_num_dims = 1 if ard_num_dims is None else ard_num_dims
            self.register_parameter(
                name="raw_lengthscale",
                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, lengthscale_num_dims)),
            )
            if lengthscale_prior is not None:
                if not isinstance(lengthscale_prior, Prior):
                    raise TypeError("Expected gpytorch.priors.Prior but got " + type(lengthscale_prior).__name__)
                self.register_prior(
                    "lengthscale_prior", lengthscale_prior, self._lengthscale_param, self._lengthscale_closure
                )

            self.register_constraint("raw_lengthscale", lengthscale_constraint)

    def _lengthscale_param(self, m: Kernel) -> Tensor:
        # Used by the lengthscale_prior
        return m.lengthscale

    def _lengthscale_closure(self, m: Kernel, v: Tensor) -> Tensor:
        # Used by the lengthscale_prior
        return m._set_lengthscale(v)

    def _set_lengthscale(self, value: Tensor):
        # Used by the lengthscale_prior
        if not self.has_lengthscale:
            raise RuntimeError("Kernel has no lengthscale.")

        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)

        self.initialize(raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value))

    def covar_dist_M(
            self,
            x1: Tensor,
            x2: Tensor,
            M: Tensor,
            diag: Optional[bool] = False,
            last_dim_is_batch: Optional[bool] = False,
            square_dist: Optional[bool] = False,
            **params,
    ) -> Tensor:

        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        x1_eq_x2 = torch.equal(x1, x2)
        res = None

        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                return torch.zeros(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
            else:
                # M = torch.diag(M)
                res = torch.sum(torch.matmul((x1 - x2), M) * (x1 - x2), dim=-1)  # mahalanobis distance norm
                return res if square_dist else torch.sqrt(res)
        else:
            dist_func = sq_dist_M if square_dist else dist_M
            return dist_func(x1, x2, M, x1_eq_x2)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # for RBF, lengthscale is 1/(2*L^2) ---->> TODO somehow?!
        # for Laplace, lengthscale is 1/L --->> TODO
        # include lengthscale directly in the weight matrix.

        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)

        covar_dist = - self.covar_dist_M(
            x1_,
            x2_,
            self.covariance_matrix,
            diag=diag,
            last_dim_is_batch=last_dim_is_batch,
            square_dist=self.squared,
            **params,
        )
        del x1_, x2_
        torch.cuda.empty_cache()
        return covar_dist.exp_()


class ExpMahalanobisDistanceKernelARDFull(Kernel):
    """
    Mahalanobis distance kernel with exponential transformation
    can be used with squared distance (for RBF like kernel or without for Laplace like kernel)
    """
    has_lengthscale = True
    has_weight_matrix = True

    def __init__(
            self,
            squared: Optional[bool] = False,
            ard_num_dims: Optional[int] = None,
            feature_dim: Optional[int] = None,
            lengthscale_prior: Optional[Prior] = None,
            lengthscale_constraint: Optional[Interval] = None,
            weight_matrix_diag_prior: Optional[Prior] = None,
            weight_matrix_diag_constraint: Optional[Interval] = None,
            weight_matrix_triu_prior: Optional[Prior] = None,
            weight_matrix_triu_constraint: Optional[Interval] = None,
            eps: Optional[float] = 1e-6,
            **kwargs,
    ):
        super(ExpMahalanobisDistanceKernelARDFull, self).__init__()
        self.squared = squared
        self.ard_num_dims = ard_num_dims
        self.feature_dim = feature_dim
        self.eps = eps

        if lengthscale_constraint is None:
            lengthscale_constraint = Positive()

        if self.has_lengthscale:
            lengthscale_num_dims = 1 if ard_num_dims is None else ard_num_dims
            self.register_parameter(
                name="raw_lengthscale",
                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, lengthscale_num_dims)),
            )
            if lengthscale_prior is not None:
                if not isinstance(lengthscale_prior, Prior):
                    raise TypeError("Expected gpytorch.priors.Prior but got " + type(lengthscale_prior).__name__)
                self.register_prior(
                    "lengthscale_prior", lengthscale_prior, self._lengthscale_param, self._lengthscale_closure
                )
            self.register_constraint("raw_lengthscale", lengthscale_constraint)

        if weight_matrix_diag_constraint is None:
            weight_matrix_diag_constraint = Positive()

        if weight_matrix_triu_constraint is None:
            weight_matrix_triu_constraint = Positive()

        if self.has_weight_matrix:
            weight_matrix_num_dims = feature_dim
            self.register_parameter(
                name="raw_weight_matrix_diag",
                parameter=torch.nn.Parameter(torch.zeros(weight_matrix_num_dims)),
            )
            # number of elements in the upper triangular part of the weight matrix
            num_triu = int(weight_matrix_num_dims * (weight_matrix_num_dims - 1) / 2)
            self.register_parameter(
                name="raw_weight_matrix_triu",
                parameter=torch.nn.Parameter(-2 * torch.ones(1, num_triu)),
            )
            if weight_matrix_diag_prior is not None:
                if not isinstance(weight_matrix_diag_prior, Prior):
                    raise TypeError("Expected gpytorch.priors.Prior but got " + type(weight_matrix_diag_prior).__name__)
                self.register_prior(
                    "weight_matrix_diag_prior", weight_matrix_diag_prior, self._weight_matrix_diag_param,
                    self._weight_matrix_diag_closure
                )
            if weight_matrix_triu_prior is not None:
                if not isinstance(weight_matrix_triu_prior, Prior):
                    raise TypeError("Expected gpytorch.priors.Prior but got " + type(weight_matrix_triu_prior).__name__)
                self.register_prior(
                    "weight_matrix_triu_prior", weight_matrix_triu_prior, self._weight_matrix_triu_param,
                    self._weight_matrix_triu_closure
                )
            self.register_constraint("raw_weight_matrix_diag", weight_matrix_diag_constraint)
            self.register_constraint("raw_weight_matrix_triu", weight_matrix_triu_constraint)

    def _lengthscale_param(self, m: Kernel) -> Tensor:
        # Used by the lengthscale_prior
        return m.lengthscale

    def _lengthscale_closure(self, m: Kernel, v: Tensor) -> Tensor:
        # Used by the lengthscale_prior
        return m._set_lengthscale(v)

    def _set_lengthscale(self, value: Tensor):
        # Used by the lengthscale_prior
        if not self.has_lengthscale:
            raise RuntimeError("Kernel has no lengthscale.")
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.initialize(raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value))

    def _weight_matrix_diag_param(self, m: Kernel) -> Tensor:
        # Used by the weight_matrix_diag_prior
        return m.weight_matrix_diag

    def _weight_matrix_diag_closure(self, m: Kernel, v: Tensor) -> Tensor:
        # Used by the weight_matrix_diag_prior
        return m._set_weight_matrix_diag(v)

    def _set_weight_matrix_diag(self, value: Tensor):
        # Used by the weight_matrix_diag_prior
        if not self.has_weight_matrix:
            raise RuntimeError("Kernel has no weight_matrix_diag.")
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_weight_matrix_diag)
        self.initialize(raw_weight_matrix_diag=self.raw_weight_matrix_diag_constraint.inverse_transform(value))

    def _weight_matrix_triu_param(self, m: Kernel) -> Tensor:
        # Used by the weight_matrix_triu_prior
        return m.weight_matrix_triu

    def _weight_matrix_triu_closure(self, m: Kernel, v: Tensor) -> Tensor:
        # Used by the weight_matrix_triu_prior
        return m._set_weight_matrix_triu(v)

    def _set_weight_matrix_triu(self, value: Tensor):
        # Used by the weight_matrix_triu_prior
        if not self.has_weight_matrix:
            raise RuntimeError("Kernel has no weight_matrix_triu.")
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_weight_matrix_triu)
        self.initialize(raw_weight_matrix_triu=self.raw_weight_matrix_triu_constraint.inverse_transform(value))

    def covar_dist_M(
            self,
            x1: Tensor,
            x2: Tensor,
            M: Tensor,
            diag: Optional[bool] = False,
            last_dim_is_batch: Optional[bool] = False,
            square_dist: Optional[bool] = False,
            **params,
    ) -> Tensor:
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        x1_eq_x2 = torch.equal(x1, x2)
        res = None

        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                return torch.zeros(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
            else:
                # M = torch.diag(M)
                res = torch.sum(torch.matmul((x1 - x2), M) * (x1 - x2), dim=-1)  # mahalanobis distance norm
                return res if square_dist else torch.sqrt(res)
        else:
            dist_func = sq_dist_M if square_dist else dist_M
            return dist_func(x1, x2, M, x1_eq_x2)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # M consists of self.weight_matrix_diag and self.weight_matrix_triu
        weight_matrix_diag = self.raw_weight_matrix_diag_constraint.transform(self.raw_weight_matrix_diag)
        weight_matrix_triu = self.raw_weight_matrix_triu_constraint.transform(self.raw_weight_matrix_triu)
        weight_matrix = torch.diag_embed(weight_matrix_diag)
        idx = torch.triu_indices(x1.shape[-1], x1.shape[-1], offset=1)
        weight_matrix[idx[0], idx[1]] = weight_matrix_triu
        M = (weight_matrix + weight_matrix.transpose(-1, -2)) / 2

        # for RBF, lengthscale is 1/(2*L^2) ---->> TODO somehow?!
        # for Laplace, lengthscale is 1/L --->> TODO
        # include lengthscale directly in the weight matrix.
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)

        covar_dist = - self.covar_dist_M(
            x1_,
            x2_,
            M,
            diag=diag,
            last_dim_is_batch=last_dim_is_batch,
            square_dist=self.squared,
            **params,
        )
        del x1_, x2_
        torch.cuda.empty_cache()
        return covar_dist.exp_()


class GPMahalanobisModel(gpytorch.models.ExactGP):
    """
    Custom Mahalanobis distance GP based on custom ExpMahalanobisDistanceKernel
    """

    def __init__(
            self,
            train_x: Tensor,
            train_y: Tensor,
            likelihood: Likelihood,
            weight_matrix: Optional[Tensor] = None,
            ard_num_dims: Optional[int] = None,
            squared: Optional[bool] = True,
    ):
        super(GPMahalanobisModel, self).__init__(train_x, train_y, likelihood)
        if weight_matrix is None:
            self.weight_matrix = torch.eye(train_x.size(-1), device=train_x.device)
        else:
            self.weight_matrix = weight_matrix
        if self.weight_matrix.dim() == 2:
            assert torch.allclose(self.weight_matrix, self.weight_matrix.T)
        assert self.weight_matrix.size(0) == train_x.size(-1)
        if not self.weight_matrix.device == train_x.device:
            self.weight_matrix = self.weight_matrix.to(train_x.device)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            ExpMahalanobisDistanceKernel(
                weight_matrix=self.weight_matrix,
                squared=squared,
                ard_num_dims=ard_num_dims,
            )
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class GPMahalanobisARDFullModel(gpytorch.models.ExactGP):
    """
    Custom Mahalanobis distance GP based on custom ExpMahalanobisDistanceKernel
    """

    def __init__(
            self,
            train_x: Tensor,
            train_y: Tensor,
            likelihood: Likelihood,
            ard_num_dims: Optional[int] = None,
            squared: Optional[bool] = True,
    ):
        super(GPMahalanobisARDFullModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            ExpMahalanobisDistanceKernelARDFull(
                squared=squared,
                ard_num_dims=ard_num_dims,
                feature_dim=train_x.shape[-1],
            )
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class GPRBFMahalanobisModel(GPMahalanobisModel):
    """
    GP with Gaussian Mahalanobis distance GP based on custom Mahalanobis distance GP
    """

    # inhert from GPMahalanobisModel with squared = True
    def __init__(
            self,
            train_x: Tensor,
            train_y: Tensor,
            likelihood: Likelihood,
            weight_matrix: Optional[Tensor] = None,
            ard_num_dims: Optional[int] = None,
    ):
        super().__init__(
            train_x,
            train_y,
            likelihood,
            weight_matrix,
            squared=True,
            ard_num_dims=ard_num_dims,
        )


class GPLaplaceMahalanobisModel(GPMahalanobisModel):
    """
    GP with Laplace Mahalanobis distance GP based on custom Mahalanobis distance GP
    """

    # inherit from GPMahalanobisModel with squared = False
    def __init__(
            self,
            train_x: Tensor,
            train_y: Tensor,
            likelihood: Likelihood,
            weight_matrix: Optional[Tensor] = None,
            ard_num_dims: Optional[int] = None,
    ):
        super().__init__(
            train_x,
            train_y,
            likelihood,
            weight_matrix=weight_matrix,
            squared=False,
            ard_num_dims=ard_num_dims,
        )


class GPARDModel(gpytorch.models.ExactGP):
    """
    GP with Automatic Relevance Determination (ARD) kernel based on original GPyTorch code
    """

    def __init__(self, train_x, train_y, likelihood, ard_num_dims=1):
        super(GPARDModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class LargeFeatureExtractorDKL(torch.nn.Sequential):
    """
    from:
    https://docs.gpytorch.ai/en/stable/examples/06_PyTorch_NN_Integration_DKL/KISSGP_Deep_Kernel_Regression_CUDA.html
    """

    def __init__(self, data_dim: int):
        super(LargeFeatureExtractorDKL, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1000, 500))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(500, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, 2))


class GPDeepKLModel(gpytorch.models.ExactGP):
    """
    GP with Deep Kernel Learning (DKL) kernel based on original GPyTorch code
    from:
    https://docs.gpytorch.ai/en/stable/examples/06_PyTorch_NN_Integration_DKL/KISSGP_Deep_Kernel_Regression_CUDA.html
    """

    def __init__(self, train_x, train_y, likelihood, ard_num_dims=2):
        super(GPDeepKLModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.GridInterpolationKernel(
        #     gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)),
        #     num_dims=2, grid_size=100
        # ) for some reason this slows everything massively down...
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims))
        self.feature_extractor = LargeFeatureExtractorDKL(data_dim=train_x.size(-1))

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
