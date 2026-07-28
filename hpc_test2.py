################################################################################
# HPC TEST 2
#
# Graph with lowering thresholds
################################################################################

import datetime
import concurrent.futures
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sys
import time

from examples import (
    training_grid_generator,
    ResidualCostFunction,
    VarianceCostFunction2,
    VarianceCostFunction
)
from surrogate import SurrogateModel
from testing_interface import Tester

def main():
    TEST_START = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    TEST_NAME = "HPC_TEST2"
    SAVE_FOLDER = TEST_NAME
    PROCESSES = 20
    NUM_TESTS = 200

    SEED = 1

    MODEL_NAME = "AIM"
    MODEL_N = 10
    SELECTED_PARAMETERS = (
        "U",
        "vb1", "vb2", "vb3", "vb4", "vb5",
        "eb2", "eb3", "eb4", "eb5"
    )
    PARAMETER_SPACE = (
        (0.01, 5.0),
        (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0),
        (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)
    )
    INIT_THETA = tuple(
        [PARAMETER_SPACE[i][0] for i in range(len(PARAMETER_SPACE))]
    )
    PARTICLE_SELECTION = (MODEL_N // 2, MODEL_N // 2)
    SPARSE = True

    if not os.path.isdir(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)

    #### COST FUNCTION SETUP ####
    GRID_SIZE = 8_000
    POINTS_PER_ITERATION = 100

    ### VARIANCE COST FUNCTION SETUP ###
    VARIANCE_THRESHOLDS = [
        1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10
    ]

    ### RESIDUAL COST FUNCTION SETUP ###
    RESIDUAL_THRESHOLDS = [
        1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10
    ]
    
    #### RUN ####
    model = SurrogateModel(
        name = MODEL_NAME,
        selected_params = SELECTED_PARAMETERS,
        N = MODEL_N,
        particle_selection = PARTICLE_SELECTION,
        sparse = SPARSE,
        processes = PROCESSES,
        save_folder = SAVE_FOLDER
    )

    model.build_terms()

    model.log("Generating test points...")

    tester = Tester(
        model,
        PARAMETER_SPACE,
        processes = PROCESSES,
        num_tests = NUM_TESTS,
        seed = 0 # Testing seed should ALWAYS be zero -- for consistency
    )

    model.log("Test points generated")

    var_basis_sizes = []
    var_iterations = []
    var_max_errors = []
    var_efficiencies = []
    var_times = []

    for VARIANCE_THRESHOLD in VARIANCE_THRESHOLDS:
        order = int(-np.log10(VARIANCE_THRESHOLD))
        model.log(f"OPTIMIZING BATCHED VARIANCE WITH ORDER {order}")

        var_cf = VarianceCostFunction2(
            model,
            VARIANCE_THRESHOLD,
            INIT_THETA,
            PARAMETER_SPACE,
            GRID_SIZE,
            POINTS_PER_ITERATION,
            seed = SEED
        )

        model.optimize(
            var_cf,
            INIT_THETA,
            f"Batched_VarianceResults_T{order}_S{SEED}"
        )

        var_cf_time = model.optimization_time
        var_basis_sizes.append(model.opt_basis.shape[1])
        var_iterations.append(model.n_iterations)
        var_max_errors.append(max(tester.test_model()))
        var_n_full_diag = model.n_full_diag
        var_efficiencies.append(
            var_basis_sizes[-1] / var_n_full_diag
            if var_n_full_diag else float("nan")
        )
        var_times.append(var_cf_time)

        model.log(f"Variance Optimization Time: {var_cf_time} seconds")
        model.log(f"Variance Basis Size {var_basis_sizes[-1]}")
        model.log(f"Variance Iterations {var_iterations[-1]}")
        model.log(f"Variance Max Error {var_max_errors[-1]}")
        model.log(f"Variance Full Diagonalizations {var_n_full_diag}")
        model.log(f"Variance Efficiency {var_efficiencies[-1]}")

        model.reset()

    res_basis_sizes = []
    res_iterations = []
    res_max_errors = []
    res_efficiencies = []
    res_times = []

    for RESIDUAL_THRESHOLD in RESIDUAL_THRESHOLDS:
        order = int(-np.log10(RESIDUAL_THRESHOLD))
        model.log(f"OPTIMIZING BATCHED RESIDUAL WITH ORDER {order}")

        res_cf = ResidualCostFunction(
            model,
            RESIDUAL_THRESHOLD,
            INIT_THETA,
            PARAMETER_SPACE,
            GRID_SIZE,
            POINTS_PER_ITERATION,
            seed = SEED
        )

        model.optimize(
            res_cf,
            INIT_THETA,
            f"Batched_ResidualResults_T{order}_S{SEED}"
        )

        res_cf_time = model.optimization_time
        res_basis_sizes.append(model.opt_basis.shape[1])
        res_iterations.append(model.n_iterations)
        res_max_errors.append(max(tester.test_model()))
        res_n_full_diag = model.n_full_diag
        res_efficiencies.append(
            res_basis_sizes[-1] / res_n_full_diag
            if res_n_full_diag else float("nan")
        )
        res_times.append(res_cf_time)

        model.log(f"Residual Optimization Time: {res_cf_time} seconds")
        model.log(f"Residual Basis Size {res_basis_sizes[-1]}")
        model.log(f"Residual Iterations {res_iterations[-1]}")
        model.log(f"Residual Max Error {res_max_errors[-1]}")
        model.log(f"Residual Full Diagonalizations {res_n_full_diag}")
        model.log(f"Residual Efficiency {res_efficiencies[-1]}")

        model.reset()

    plt.loglog(VARIANCE_THRESHOLDS, var_max_errors, label="Variance")
    plt.loglog(RESIDUAL_THRESHOLDS, res_max_errors, label="Residual")
    plt.xlabel("Threshold")
    plt.ylabel("Max Error")
    plt.title("Batched, Max Error vs Threshold")
    plt.legend()
    plt.savefig(f"{SAVE_FOLDER}/{TEST_NAME}_{TEST_START}_ERROR.svg")
    plt.clf()

    plt.semilogx(VARIANCE_THRESHOLDS, var_basis_sizes, label="Variance")
    plt.semilogx(RESIDUAL_THRESHOLDS, res_basis_sizes, label="Residual")
    plt.xlabel("Threshold")
    plt.ylabel("Basis Size")
    plt.title("Batched, Basis Size vs Threshold")
    plt.legend()
    plt.savefig(f"{SAVE_FOLDER}/{TEST_NAME}_{TEST_START}_BASIS_SIZE.svg")
    plt.clf()

    plt.semilogx(VARIANCE_THRESHOLDS, var_iterations, label="Variance")
    plt.semilogx(RESIDUAL_THRESHOLDS, res_iterations, label="Residual")
    plt.xlabel("Threshold")
    plt.ylabel("Iterations")
    plt.title("Batched, Iterations vs Threshold")
    plt.legend()
    plt.savefig(f"{SAVE_FOLDER}/{TEST_NAME}_{TEST_START}_ITERATIONS.svg")
    plt.clf()

    plt.semilogx(VARIANCE_THRESHOLDS, var_efficiencies, label="Variance")
    plt.semilogx(RESIDUAL_THRESHOLDS, res_efficiencies, label="Residual")
    plt.xlabel("Threshold")
    plt.ylabel("Efficiency")
    plt.title("Batched, Efficiency vs Threshold")
    plt.legend()
    plt.savefig(f"{SAVE_FOLDER}/{TEST_NAME}_{TEST_START}_EFFICIENCY.svg")
    plt.clf()

    plt.semilogx(VARIANCE_THRESHOLDS, var_times, label="Variance")
    plt.semilogx(RESIDUAL_THRESHOLDS, res_times, label="Residual")
    plt.xlabel("Threshold")
    plt.ylabel("Time")
    plt.title("Batched, Time vs Threshold")
    plt.legend()
    plt.savefig(f"{SAVE_FOLDER}/{TEST_NAME}_{TEST_START}_TIME.svg")
    plt.clf()

if __name__ == "__main__":
    main()
