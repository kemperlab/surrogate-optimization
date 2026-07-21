################################################################################
# HPC TEST 1
# 
# TEN SITE ANDERSON IMPURITY MODEL
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
    ResidualCostFunction,
    VarianceCostFunction2,
    VarianceCostFunction
)
from surrogate import SurrogateModel
from testing_interface import Tester

#### MODEL SETUP ####
def main():
    TEST_START = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    TEST_NAME = "HPC_TEST1"
    SAVE_FOLDER = TEST_NAME
    PROCESSES = 20
    NUM_TESTS = 200

    SEED = 4

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
    TOTAL_SOBOL_POINTS = 16_000
    POINTS_PER_ITERATION = 100

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

    # Same logic as Herbst et al. 2022
    var_training_cf = VarianceCostFunction(
        model,
        training_grid,
        VARIANCE_THRESHOLD
    )

    # Variance based, capping the number of points for variance
    # calculation and basis addition to POINTS_PER_ITERATION
    var_cf = VarianceCostFunction2(
        model,
        VARIANCE_THRESHOLD,
        INIT_THETA,
        PARAMETER_SPACE,
        TOTAL_SOBOL_POINTS,
        POINTS_PER_ITERATION,
        seed = SEED
    )

    # Residual based, capping the number of points for residual 
    # calculation to POINTS_PER_ITERATION
    res_cf = ResidualCostFunction(
        model,
        RESIDUAL_THRESHOLD,
        INIT_THETA,
        PARAMETER_SPACE,
        TOTAL_SOBOL_POINTS,
        POINTS_PER_ITERATION,
        seed = SEED
    )

    # Variance based, search for points in the Sobol sequence,
    # accept/reject points one-at-a-time based on variance. 
    # Terminates upon first rejection
    sob_cf = VarianceCostFunction2(
        model,
        VARIANCE_THRESHOLD,
        INIT_THETA,
        PARAMETER_SPACE,
        TOTAL_SOBOL_POINTS,
        1,
        seed = SEED
    )

    time_start = time.time()
    model.optimize(var_cf, INIT_THETA, "VarianceResults")
    var_cf_time = time.time() - time_start
    model.log(f"Variance Optimization Time: {var_cf_time} seconds")

    var_basis_size = model.opt_basis.shape[1]
    var_iterations = model.n_iterations
    var_basis_growth = model.basis_growth
    var_errors = tester.test_model()
    var_n_full_diag = model.n_full_diag
    var_efficiency = (
        var_basis_size / var_n_full_diag if var_n_full_diag else float("nan")
    )

    model.log(f"Variance Basis Size {var_basis_size}")
    model.log(f"Variance Iterations {var_iterations}")
    model.log(f"Variance Max Error {max(var_errors)}")
    model.log(f"Variance Full Diagonalizations {var_n_full_diag}")
    model.log(f"Variance Efficiency {var_efficiency}")

    model.reset()

    time_start = time.time()
    model.optimize(res_cf, INIT_THETA, "ResidualResults")
    res_cf_time = time.time() - time_start
    model.log(f"Residual Optimization Time: {res_cf_time} seconds")

    res_basis_size = model.opt_basis.shape[1]
    res_iterations = model.n_iterations
    res_basis_growth = model.basis_growth
    res_errors = tester.test_model()
    res_n_full_diag = model.n_full_diag
    res_efficiency = (
        res_basis_size / res_n_full_diag if res_n_full_diag else float("nan")
    )

    model.log(f"Residual Basis Size {res_basis_size}")
    model.log(f"Residual Iterations {res_iterations}")
    model.log(f"Residual Max Error {max(res_errors)}")
    model.log(f"Residual Full Diagonalizations {res_n_full_diag}")
    model.log(f"Residual Efficiency {res_efficiency}")

    model.reset()

    time_start = time.time()
    model.optimize(sob_cf, INIT_THETA, "SobolResults")
    sob_cf_time = time.time() - time_start
    model.log(f"Sobol Optimization Time: {sob_cf_time} seconds")

    sob_basis_size = model.opt_basis.shape[1]
    sob_iterations = model.n_iterations
    sob_basis_growth = model.basis_growth
    sob_errors = tester.test_model()
    sob_n_full_diag = model.n_full_diag
    sob_efficiency = (
        sob_basis_size / sob_n_full_diag if sob_n_full_diag else float("nan")
    )

    model.log(f"Sobol Basis Size {sob_basis_size}")
    model.log(f"Sobol Iterations {sob_iterations}")
    model.log(f"Sobol Max Error {max(sob_errors)}")
    model.log(f"Sobol Full Diagonalizations {sob_n_full_diag}")
    model.log(f"Sobol Efficiency {sob_efficiency}")

    model.reset()

    time_start = time.time()
    model.optimize(var_training_cf, INIT_THETA, "VarianceTResults")
    var_training_cf_time = time.time() - time_start
    model.log(f"Variance (Training Grid) Optimization Time: {var_training_cf_time} seconds")

    vart_basis_size = model.opt_basis.shape[1]
    vart_iterations = model.n_iterations
    vart_basis_growth = model.basis_growth
    vart_errors = tester.test_model()
    vart_n_full_diag = model.n_full_diag
    vart_efficiency = (
        vart_basis_size / vart_n_full_diag if vart_n_full_diag else float("nan")
    )

    model.log(f"Variance (Training Grid) Basis Size {vart_basis_size}")
    model.log(f"Variance (Training Grid) Iterations {vart_iterations}")
    model.log(f"Variance (Training Grid) Max Error {max(vart_errors)}")
    model.log(f"Variance (Training Grid) Full Diagonalizations {vart_n_full_diag}")
    model.log(f"Variance (Training Grid) Efficiency {vart_efficiency}")

    model.log("########## RESULTS ##########")
    model.log(f"Hilbert Space Size {model.size}")
    model.log(f"Variance (Training Grid) Basis Size {vart_basis_size}")
    model.log(f"Variance (Training Grid) Iterations {vart_iterations}")
    model.log(f"Variance (Training Grid) Max Error {max(vart_errors)}")
    model.log(f"Variance (Training Grid) Efficiency {vart_efficiency}")
    model.log(f"Variance (Training Grid) total time: {var_training_cf_time} seconds")
    model.log("")
    model.log(f"Variance Basis Size {var_basis_size}")
    model.log(f"Variance Iterations {var_iterations}")
    model.log(f"Variance Max Error {max(var_errors)}")
    model.log(f"Variance Efficiency {var_efficiency}")
    model.log(f"Variance total time: {var_cf_time} seconds")
    model.log("")
    model.log(f"Residual Basis Size {res_basis_size}")
    model.log(f"Residual Iterations {res_iterations}")
    model.log(f"Residual Max Error {max(res_errors)}")
    model.log(f"Residual Efficiency {res_efficiency}")
    model.log(f"Residual total time: {res_cf_time} seconds")
    model.log("")
    model.log(f"Sobol Basis Size {sob_basis_size}")
    model.log(f"Sobol Iterations {sob_iterations}")
    model.log(f"Sobol Max Error {max(sob_errors)}")
    model.log(f"Sobol Efficiency {sob_efficiency}")
    model.log(f"Sobol total time: {sob_cf_time} seconds")
    
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

    plt.plot(np.arange(len(var_basis_growth)), var_basis_growth, label="Variance")
    plt.plot(np.arange(len(vart_basis_growth)), vart_basis_growth,
        label="Variance (Training Grid)")
    plt.plot(np.arange(len(res_basis_growth)), res_basis_growth, label="Residual")
    plt.xlabel("Iteration #")
    plt.ylabel("States Added")
    plt.title("States Added Per Iteration for Each Cost Function")
    plt.legend()
    plt.savefig(f"{SAVE_FOLDER}/{TEST_NAME}_STATES_ADDED_{TEST_START}.svg")

if __name__ == "__main__":
    main()
