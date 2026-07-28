import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import concurrent.futures

from functools import partial
from pauli import *
from surrogate import SurrogateModel
def get_full_ground_state(
    training_point,
    model
):
    H_full = model.build_H_full(training_point)
    v0 = np.ones(H_full.shape[0]) / np.sqrt(H_full.shape[0])

    evals, evecs = sps.linalg.eigsh(
        H_full,
        #k=5,
        k=min(int(model.size * model.sparse_proportion) + 1, 4),
        v0=v0,
        which='SA'
    )

    return evals[0]


class Tester:
    def __init__(
        self,
        model,
        param_bounds,
        processes = 1,
        num_tests = 200,
        seed = None
    ):
        model.log("Initializing Testing framework...")
        self.model = model
        self.param_bounds = param_bounds
        self.processes = processes
        self.num_tests = num_tests
        self.rng = np.random.default_rng(seed)

        self.thetas = []

        for i in range(self.num_tests):
            points = self.rng.random(len(model.selected_params))

            theta = [
                (b[1] - b[0]) * s + b[0]
                for b, s in zip(self.param_bounds, points)
            ]

            self.thetas.append(theta)

        if self.processes == 1:
            self.training_grid = []
            for theta in self.thetas:
                training_point = model.theta_to_training_point(theta)
                self.training_grid.append(training_point)

            self.ground_states = []

            for training_point in self.training_grid:
                self.ground_states.append(get_full_ground_state(training_point, self.model))
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.processes
            ) as pool:
                batch_size = int(np.ceil(len(self.thetas) / self.processes))

                self.training_grid = list(pool.map(
                    model.theta_to_training_point,
                    self.thetas,
                    chunksize=batch_size
                ))

                self.model.log("Testing grid created")

                self.ground_states = list(pool.map(
                    partial(
                        get_full_ground_state,
                        model=self.model
                    ),
                    self.training_grid,
                    chunksize=batch_size
                ))
        self.model.log("Testing framework initialized")

    def test_model(self):
        self.model.log(f"Testing model on {self.num_tests} tests")
        batch_size = int(np.ceil(len(self.training_grid) / self.processes))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.processes
        ) as pool:
            errors = list(pool.map(
                self.get_error,
                np.arange(len(self.training_grid)),
                chunksize=batch_size
            ))

        return errors

    def get_error(
        self,
        index
    ):
        evals, evecs = self.model.solve(self.training_grid[index])
        true_gse = self.ground_states[index]

        if abs(evals[0]) < 1e-12:
            error = np.abs(true_gse - evals[0])
        else:
            error = np.abs(true_gse - evals[0]) / np.abs(true_gse)

        return error
