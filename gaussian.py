import abc
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

class GaussianProcess(abc.ABC):
    """
    Interface for modeling a Gaussian process
    """

    @abc.abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        """
        Fit the Gaussian process to the given training data
        """
        pass
    
    @abc.abstractmethod
    def predict(
        self,
        X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the predicted mean and standard deviation
        """
        pass
    
    @abc.abstractmethod
    def is_fitted(self) -> bool:
        """
        Whether or not the data has been fitted
        """
        pass

class SklearnGP(GaussianProcess):
    def __init__(self) -> None:
        kernel = (
            ConstantKernel(1.0, (1e-30, 1e3))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-30, 1e3))
            + WhiteKernel(noise_level = 1e-5, noise_level_bounds=(1e-30, 1e2))
        )

        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True
        )
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        self._gp.fit(X, y)
        self._fitted = True

    def predict(
        self,
        X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mean, std = self._gp.predict(X, return_std=True)

        return mean, std

    def is_fitted(self) -> bool:
        return self._fitted
