import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

from pauli import *
from surrogate import SurrogateModel
from pathos.multiprocessing import ProcessPool

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
        self.pp = ProcessPool(nodes=self.processes)
        self.rng = np.random.default_rng(seed)

        self.thetas = []

        for i in range(self.num_tests):
            points = np.random.random(len(model.selected_params))

            theta = [
                (b[1] - b[0]) * s + b[0]
                for b, s in zip(self.param_bounds, points)
            ]

            self.thetas.append(theta)

        batch_size = int(np.ceil(len(points) / self.processes))
        self.training_grid = list(self.pp.map(
            model.theta_to_training_point,
            self.thetas,
            chunksize=batch_size
        ))
        self.training_grid = np.array(self.training_grid, dtype=dict)

        batch_size = int(np.ceil(len(self.training_grid) / self.processes))
        self.ground_states = list(self.pp.map(
            self.get_full_ground_state,
            self.training_grid,
            chunksize=batch_size
        ))
        self.model.log("Testing framework initialized")

    def test_model(self):
        self.model.log(f"Testing model on {self.num_tests} tests")
        batch_size = int(np.ceil(len(self.training_grid) / self.processes))
        errors = list(self.pp.map(
            self.get_error,
            np.arange(len(self.training_grid)),
            chunksize=batch_size
        ))

        return errors

    def get_full_ground_state(
        self,
        training_point
    ):
        H_full = self.model.build_H_full(training_point)

        evals, evecs = np.linalg.eigh(H_full)

        return evals[0]

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
