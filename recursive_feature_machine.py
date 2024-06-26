try:
    from eigenpro2 import KernelModel
    EIGENPRO_AVAILABLE = True
except ModuleNotFoundError:
    # print('`eigenpro2` is not installed...')
    # print('Using `torch.linalg.solve` for training the kernel model\n')
    # print('WARNING: `torch.linalg.solve` scales poorly with the size of training dataset,\n '
    # '         and may cause an `Out-of-Memory` error')
    # print('`eigenpro2` is a more scalable solver. To use, pass `method="eigenpro"` to `model.fit()`')
    # print('To install `eigenpro2` visit https://github.com/EigenPro/EigenPro-pytorch/tree/pytorch/')
    EIGENPRO_AVAILABLE = False

import hickle
import numpy as np
import torch
from kernels import laplacian_M, gaussian_M, euclidean_distances_M
from torchmetrics.functional.classification import accuracy
from tqdm import tqdm
from tqdm.contrib import tenumerate


class RecursiveFeatureMachine(torch.nn.Module):
    def __init__(self,
                 device=torch.device('cpu'),
                 mem_gb=32,  # leave some Gb for other stuff
                 diag=False,
                 centering=False,
                 reg=1e-3,
                 ridge=0,
                 verbose=True):
        super().__init__()
        self.M = None
        self.model = None
        self.diag = diag  # if True, Mahalanobis matrix M will be diagonal
        self.centering = centering  # if True, update_M will center the gradients before taking an outer product
        self.device = device
        self.mem_gb = mem_gb
        self.reg = reg  # only used when fit using direct solve
        self.ridge = ridge  # only used for non-diagonal M during f_M update
        self.verbose = verbose

    def get_data(self, data_loader):
        X, y = [], []
        for idx, batch in enumerate(data_loader):
            inputs, labels = batch
            X.append(inputs)
            y.append(labels)
        return torch.cat(X, dim=0), torch.cat(y, dim=0)

    def update_M(self):
        raise NotImplementedError("Must implement this method in a subclass")

    def fit_predictor(self, centers, targets, **kwargs):
        self.centers = centers
        if self.M is None:
            if self.diag:
                self.M = torch.ones(centers.shape[-1], device=self.device)
            else:
                self.M = torch.eye(centers.shape[-1], device=self.device)
        if self.fit_using_eigenpro and EIGENPRO_AVAILABLE:
            self.weights = self.fit_predictor_eigenpro(centers, targets, **kwargs)
        else:
            self.weights = self.fit_predictor_lstsq(centers, targets)

    def fit_predictor_lstsq(self, centers, targets):
        return torch.linalg.solve(
            self.kernel(centers, centers)
            + self.reg * torch.eye(len(centers), device=centers.device),
            targets
        )

    def fit_predictor_eigenpro(self, centers, targets, **kwargs):
        n_classes = 1 if targets.dim() == 1 else targets.shape[-1]
        self.model = KernelModel(self.kernel, centers, n_classes, device=self.device)
        _ = self.model.fit(centers, targets, mem_gb=self.mem_gb, **kwargs)
        return self.model.weight

    def predict(self, samples):
        return self.kernel(samples, self.centers) @ self.weights

    def fit(self, train_loader, test_loader=None,
            iters=3, name=None, method='lstsq',
            loader=True, classif=True, **kwargs):
        # if method=='eigenpro':
        #     raise NotImplementedError(
        #         "EigenPro method is not yet supported. "+
        #         "Please try again with `method='lstlq'`")
        self.fit_using_eigenpro = (method.lower() == 'eigenpro')
        # self.fit_using_eigenpro = True

        if loader:
            print("Loaders provided") if self.verbose else None
            X_train, y_train = self.get_data(train_loader)
            if test_loader is not None:
                X_test, y_test = self.get_data(test_loader)
        else:
            X_train, y_train = train_loader
            if test_loader is not None:
                X_test, y_test = test_loader
        # reshape labels to be of size (n_samples, n_classes)
        if y_train.dim() == 1:
            y_train = y_train.unsqueeze(-1)
        if test_loader is not None:
            if y_test.dim() == 1:
                y_test = y_test.unsqueeze(-1)

        with tqdm(range(iters), disable=not self.verbose) as pbar:
            pbar_postfix = {}
            for i in pbar:
                self.fit_predictor(X_train, y_train, **kwargs)

                if classif:
                    train_acc = self.score(X_train, y_train, metric='accuracy')
                    pbar_postfix['train_acc'] = f"{train_acc:.2f}%"
                    # print(f"Round {i}, Train Acc: {train_acc:.2f}%")
                    if test_loader is not None:
                        test_acc = self.score(X_test, y_test, metric='accuracy')
                        pbar_postfix['test_acc'] = f"{test_acc:.2f}%"
                        # print(f"Round {i}, Test Acc: {test_acc:.2f}%")

                if test_loader is not None:
                    test_mse = self.score(X_test, y_test, metric='mse')
                    pbar_postfix['test_mse'] = f"{test_mse:.4f}"
                    # print(f"Round {i}, Test MSE: {test_mse:.4f}")

                self.fit_M(X_train, y_train, ridge=self.ridge, **kwargs)

                if name is not None:
                    hickle.dump(self.M, f"saved_Ms/M_{name}_{i}.h")

                pbar.set_postfix(pbar_postfix)

        self.fit_predictor(X_train, y_train, **kwargs)
        if test_loader is not None:
            final_mse = self.score(X_test, y_test, metric='mse')
            print(f"Final MSE: {final_mse:.4f}") if self.verbose else None
            if classif:
                final_test_acc = self.score(X_test, y_test, metric='accuracy')
                print(f"Final Test Acc: {final_test_acc:.2f}%") if self.verbose else None

            # return final_mse

    def _compute_optimal_M_batch(self, p, c, d, scalar_size=4):
        """Computes the optimal batch size for EGOP."""
        THREADS_PER_BLOCK = 512  # pytorch default

        def tensor_mem_usage(numels):
            """Calculates memory footprint of tensor based on number of elements."""
            return np.ceil(scalar_size * numels / THREADS_PER_BLOCK) * THREADS_PER_BLOCK

        def max_tensor_size(mem):
            """Calculates maximum possible tensor given memory budget (bytes)."""
            return int(np.floor(mem / THREADS_PER_BLOCK) * (THREADS_PER_BLOCK / scalar_size))

        curr_mem_use = torch.cuda.memory_allocated()  # currently allocated memory in bytes
        M_mem = tensor_mem_usage(d if self.diag else d ** 2)
        centers_mem = tensor_mem_usage(p * d)
        mem_available = (self.mem_gb * 1024 ** 3) - curr_mem_use - (M_mem + centers_mem) * scalar_size
        M_batch_size = max_tensor_size(
            (mem_available - 3 * tensor_mem_usage(p) - tensor_mem_usage(p * c * d)) / (2 * scalar_size * (1 + p)))
        return M_batch_size

    def fit_M(self, samples, labels, M_batch_size=None, ridge=0, **kwargs):
        """Applies EGOP to update the Mahalanobis matrix M."""

        n, d = samples.shape
        M = torch.zeros_like(self.M) if self.M is not None else (
            torch.zeros(d, dtype=samples.dtype) if self.diag else torch.zeros(d, d, dtype=samples.dtype))

        if M_batch_size is None:
            BYTES_PER_SCALAR = self.M.element_size()
            p, d = samples.shape
            c = labels.shape[-1]
            M_batch_size = self._compute_optimal_M_batch(p, c, d, scalar_size=BYTES_PER_SCALAR)
            print(f"Using batch size of {M_batch_size}") if self.verbose else None
            if M_batch_size < 1:
                raise ValueError(f"Memory budget of {self.mem_gb} GB is too small to fit even one batch of M. "
                                 f"Please increase `mem_gb` or decrease `p` or `d`\n")

        with torch.no_grad():
            batches = torch.randperm(n).split(M_batch_size)
            for i, bids in enumerate(batches):
                torch.cuda.empty_cache()
                M.add_(self.update_M(samples[bids]))

        # some kind of normalization
        self.M = M / n # divide by number of samples
        # normalization by max
        # self.M /= self.M.max()
        del M
        # add ridge regularization
        if not self.diag:
            self.M.add_(ridge * torch.eye(self.M.shape[0], device=self.M.device))

    def score(self, samples, targets, metric='mse'):
        preds = self.predict(samples)
        if metric == 'accuracy':
            if preds.shape[-1] == 1:
                num_classes = len(torch.unique(targets))
                if num_classes == 2:
                    return accuracy(preds, targets, task="binary").item()
                else:
                    return accuracy(preds, targets, task="multiclass", num_classes=num_classes).item()
            else:
                return accuracy(preds, targets, task="multilabel", num_labels=preds.shape[-1]).item()

        elif metric == 'mse':
            return (targets - preds).pow(2).mean()


class LaplaceRFM(RecursiveFeatureMachine):

    def __init__(self, bandwidth=1., **kwargs):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self.kernel = lambda x, z: laplacian_M(x, z, self.M, self.bandwidth)  # must take 3 arguments (x, z, M)

    def update_M(self, samples):
        """Performs a batched update of M."""
        K = self.kernel(samples, self.centers)

        dist = euclidean_distances_M(samples, self.centers, self.M, squared=False)
        dist = torch.where(dist < 1e-10, torch.zeros(1, device=dist.device).float(), dist)

        K.div_(dist)
        del dist
        torch.cuda.empty_cache()
        K[K == float("Inf")] = 0.

        # p, d = self.centers.shape
        p, c = self.weights.shape
        n, d = samples.shape

        samples_term = (K @ self.weights).reshape(n, c, 1)  # (n, p)  # (p, c)

        if self.diag:
            centers_term = (K  # (n, p)
                    @ (self.weights.view(p, c, 1) * (self.centers * self.M).view(p, 1, d)).reshape(p, c * d)  # (p, cd)
            ).view(n, c, d)  # (n, c, d)
            samples_term = samples_term * (samples * self.M).reshape(n, 1, d)

        else:
            centers_term = (K  # (n, p)
                    @ (self.weights.view(p, c, 1) * (self.centers @ self.M).view(p, 1, d)).reshape(p, c * d)  # (p, cd)
            ).view(n, c, d)  # (n, c, d)
            samples_term = samples_term * (samples @ self.M).reshape(n, 1, d)

        G = (centers_term - samples_term) / self.bandwidth  # (n, c, d)

        del centers_term, samples_term, K
        torch.cuda.empty_cache()

        if self.centering:
            G = G - G.mean(0)  # (n, c, d)

        # return quantity to be added to M. Division by len(samples) will be done in parent function.
        if self.diag:
            return torch.einsum('ncd, ncd -> d', G, G)
        else:
            return torch.einsum("ncd, ncD -> dD", G, G)


class GaussRFM(RecursiveFeatureMachine):

    def __init__(self, bandwidth=1., **kwargs):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self.kernel = lambda x, z: gaussian_M(x, z, self.M, self.bandwidth)  # must take 3 arguments (x, z, M)

    def update_M(self, samples):

        K = self.kernel(samples, self.centers)

        p, d = self.centers.shape
        p, c = self.weights.shape
        n, d = samples.shape

        samples_term = (
                K  # (n, p)
                @ self.weights  # (p, c)
        ).reshape(n, c, 1)

        if self.diag:
            centers_term = (
                    K  # (n, p)
                    @ (
                            self.weights.view(p, c, 1) * (self.centers * self.M).view(p, 1, d)
                    ).reshape(p, c * d)  # (p, cd)
            ).view(n, c, d)  # (n, c, d)

            samples_term = samples_term * (samples * self.M).reshape(n, 1, d)

        else:
            centers_term = (
                    K  # (n, p)
                    @ (
                            self.weights.view(p, c, 1) * (self.centers @ self.M).view(p, 1, d)
                    ).reshape(p, c * d)  # (p, cd)
            ).view(n, c, d)  # (n, c, d)

            samples_term = samples_term * (samples @ self.M).reshape(n, 1, d)

        G = (centers_term - samples_term) / self.bandwidth ** 2  # (n, c, d)

        if self.centering:
            G = G - G.mean(0)  # (n, c, d)

        if self.diag:
            return torch.einsum('ncd, ncd -> d', G, G)
        else:
            return torch.einsum("ncd, ncD -> dD", G, G)
