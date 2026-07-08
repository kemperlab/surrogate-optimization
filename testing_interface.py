import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

from pauli import *
from surrogate import SurrogateModel

def test_model(model, param_bounds, num_tests):
    model.log(f"Testing model on {num_tests} tests")
    errors = []
    for i in range(num_tests):
        points = np.random.random(len(model.selected_params))

        theta = [
            (b[1] - b[0]) * s + b[0]
            for b, s in zip(param_bounds, points)
        ]
        model.log(f"Test {i + 1}: {theta}")

        training_point = model.theta_to_training_point(theta)
        H_full = model.build_H_full(training_point)
        model.log("Built H_full...")

        if model.sparse:
            evals, evecs = sps.linalg.eigsh(
                H_full,
                k=int(model.sparse_proportion*model.size),
                which='SA'
            )
        else:
            evals, evecs = np.linalg.eigh(H_full)
        model.log(f"Actual ground state energy: {evals[0]}")

        test_evals, test_evecs = model.solve(training_point)
        model.log(f"Reduced ground state energy: {test_evals[0]}")
        if abs(evals[0]) < 1e-12:
            errors.append(np.abs(evals[0] - test_evals[0]))
        else:
            errors.append(np.abs(evals[0] - test_evals[0]) / np.abs(evals[0]))
        model.log(f"Error: {errors[-1]}")


    return errors
