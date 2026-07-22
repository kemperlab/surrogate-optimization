################################################################################
# MOCK TEST 1
# 
# SIX SITE ANDERSON IMPURITY MODEL
################################################################################

import datetime
import concurrent.futures
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sys

from examples import (
    ResidualCostFunction,
    VarianceCostFunction2,
    VarianceCostFunction,
    NaiveMethod
)
from surrogate import SurrogateModel
from testing_interface import Tester

#### MODEL SETUP ####
def main():
    TEST_START = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    TEST_NAME = "MOCK_TEST1"
    SAVE_FOLDER = TEST_NAME
    PROCESSES = 4
    NUM_TESTS = 200

    SEED = 4

    MODEL_NAME = "AIM"
    MODEL_N = 6
    SELECTED_PARAMETERS = ("U", "vb1", "vb2", "vb3", "eb2", "eb3")
    PARAMETER_SPACE = (
        (0.01, 5.0),
        (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0),
        (-5.0, 5.0), (-5.0, 5.0)
    )
    INIT_THETA = (
        PARAMETER_SPACE[0][0],
        PARAMETER_SPACE[1][0],PARAMETER_SPACE[2][0],PARAMETER_SPACE[3][0],
        PARAMETER_SPACE[4][0],PARAMETER_SPACE[5][0]
    )
    PARTICLE_SELECTION = (MODEL_N // 2, MODEL_N // 2)
    SPARSE = True

    if not os.path.isdir(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)

    LOG_FILENAME = f"{SAVE_FOLDER}/{TEST_NAME}_{TEST_START}.log"

    #### COST FUNCTION SETUP ####
    TOTAL_SOBOL_POINTS = 1000
    POINTS_PER_ITERATION = 10

    ### VARIANCE COST FUNCTION SETUP ###
    VARIANCE_THRESHOLD = 1e-9

    ### RESIDUAL COST FUNCTION SETUP ###
    RESIDUAL_THRESHOLD = 1e-9

    #### RUN ####
    model = SurrogateModel(
        MODEL_NAME,
        SELECTED_PARAMETERS,
        MODEL_N,
        particle_selection = PARTICLE_SELECTION,
        sparse=SPARSE,
        save_folder = SAVE_FOLDER,
        keep_on_disk = True,
        processes = PROCESSES
    )

    model.build_terms()

    model.log("Generating test points...")

    tester = Tester(
        model,
        PARAMETER_SPACE,
        processes = PROCESSES,
        num_tests = NUM_TESTS,
        seed = SEED + 1
    )

    model.log("Test points generated")
    model.log("Generating training grid...")

    sobol_gen = sp.stats.qmc.Sobol(len(PARAMETER_SPACE),
        rng=np.random.default_rng(SEED))
    # round up to the nearest power of two
    power = int(np.log2(TOTAL_SOBOL_POINTS) + 0.5)
    points = np.array(sobol_gen.random_base2(power))
    for i, point in enumerate(points):
        for coord, param_range in enumerate(PARAMETER_SPACE):
            points[i][coord] = (
                (param_range[1] - param_range[0]) * points[i][coord]
                + param_range[0]
            )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=PROCESSES
    ) as pool:
        batch_size = int(np.ceil(len(points) / PROCESSES))
        training_grid = list(pool.map(
            model.theta_to_training_point,
            points,
            chunksize=batch_size
        ))
    training_grid = np.array(training_grid, dtype=dict)

    model.log("Training grid generated")

    nve_cf = NaiveMethod(
        model,
        PARAMETER_SPACE,
        TOTAL_SOBOL_POINTS,
        POINTS_PER_ITERATION
    )

    var_training_cf = VarianceCostFunction(
        model,
        training_grid,
        VARIANCE_THRESHOLD
    )

    var_cf = VarianceCostFunction2(
        model,
        VARIANCE_THRESHOLD,
        INIT_THETA,
        PARAMETER_SPACE,
        TOTAL_SOBOL_POINTS,
        POINTS_PER_ITERATION,
        seed = SEED
    )

    res_cf = ResidualCostFunction(
        model,
        RESIDUAL_THRESHOLD,
        INIT_THETA,
        PARAMETER_SPACE,
        TOTAL_SOBOL_POINTS,
        POINTS_PER_ITERATION,
        seed = SEED
    )

    sob_cf = VarianceCostFunction2(
        model,
        VARIANCE_THRESHOLD,
        INIT_THETA,
        PARAMETER_SPACE,
        TOTAL_SOBOL_POINTS,
        1,
        seed = SEED
    )

    model.optimize(nve_cf, INIT_THETA, "NaiveResults")

    nve_basis_size = model.opt_basis.shape[1]
    nve_iterations = len(model.iteration_costs)
    nve_basis_growth = model.basis_growth
    nve_errors = tester.test_model()

    model.log(f"Naive Basis Size {nve_basis_size}")
    model.log(f"Naive Iterations {nve_iterations}")
    model.log(f"Naive Max Error {max(nve_errors)}")

    model.reset()

    model.optimize(var_cf, INIT_THETA, "VarianceResults")

    var_basis_size = model.opt_basis.shape[1]
    var_iterations = len(model.iteration_costs)
    var_basis_growth = model.basis_growth
    var_errors = tester.test_model()

    model.log(f"Variance Basis Size {var_basis_size}")
    model.log(f"Variance Iterations {var_iterations}")
    model.log(f"Variance Max Error {max(var_errors)}")

    model.reset()

    model.optimize(res_cf, INIT_THETA, "ResidualResults")

    res_basis_size = model.opt_basis.shape[1]
    res_iterations = len(model.iteration_costs)
    res_basis_growth = model.basis_growth
    res_errors = tester.test_model()

    model.log(f"Residual Basis Size {res_basis_size}")
    model.log(f"Residual Iterations {res_iterations}")
    model.log(f"Residual Max Error {max(res_errors)}")

    model.reset()

    model.optimize(sob_cf, INIT_THETA, "SobolResults")

    sob_basis_size = model.opt_basis.shape[1]
    sob_iterations = len(model.iteration_costs)
    sob_basis_growth = model.basis_growth
    sob_errors = tester.test_model()

    model.log(f"Sobol Basis Size {sob_basis_size}")
    model.log(f"Sobol Iterations {sob_iterations}")
    model.log(f"Sobol Max Error {max(sob_errors)}")

    model.reset()

    model.optimize(var_training_cf, INIT_THETA, "VarianceTResults")

    vart_basis_size = model.opt_basis.shape[1]
    vart_iterations = len(model.iteration_costs)
    vart_basis_growth = model.basis_growth
    vart_errors = tester.test_model()

    model.log(f"Variance (Training Grid) Basis Size {vart_basis_size}")
    model.log(f"Variance (Training Grid) Iterations {vart_iterations}")
    model.log(f"Variance (Training Grid) Max Error {max(vart_errors)}")

    model.log("########## RESULTS ##########")
    model.log(f"Hilbert Space Size {model.size}")
    model.log(f"Variance (Training Grid) Basis Size {vart_basis_size}")
    model.log(f"Variance (Training Grid) Iterations {vart_iterations}")
    model.log(f"Variance (Training Grid) Max Error {max(vart_errors)}")
    model.log(f"Variance Basis Size {var_basis_size}")
    model.log(f"Variance Iterations {var_iterations}")
    model.log(f"Variance Max Error {max(var_errors)}")
    model.log(f"Residual Basis Size {res_basis_size}")
    model.log(f"Residual Iterations {res_iterations}")
    model.log(f"Residual Max Error {max(res_errors)}")
    model.log(f"Sobol Basis Size {sob_basis_size}")
    model.log(f"Sobol Iterations {sob_iterations}")
    model.log(f"Sobol Max Error {max(sob_errors)}")
    
    plt.semilogy(var_errors, label="Variance")
    plt.semilogy(vart_errors, label="Variance (Training Grid)")
    plt.semilogy(res_errors, label="Residual")
    plt.semilogy(sob_errors, label="Sobol")
    plt.xlabel("Test #")
    plt.ylabel("Relative Error")
    plt.title("Relative Errors for Each Selection Method")
    plt.legend()
    plt.savefig(f"{SAVE_FOLDER}/{TEST_NAME}_ERRORS_{TEST_START}.svg")
    plt.clf()

    plt.plot(np.arange(var_iterations), var_basis_growth, label="Variance")
    plt.plot(np.arange(vart_iterations), vart_basis_growth,
        label="Variance (Training Grid)")
    plt.plot(np.arange(res_iterations), res_basis_growth, label="Residual")
    plt.xlabel("Iteration #")
    plt.ylabel("States Added")
    plt.title("States Added Per Iteration for Each Cost Function")
    plt.legend()
    plt.savefig(f"{SAVE_FOLDER}/{TEST_NAME}_STATES_ADDED_{TEST_START}.svg")

if __name__ == "__main__":
    main()
