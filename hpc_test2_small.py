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
    TEST_NAME = "HPC_TEST2_SMALL"
    SAVE_FOLDER = TEST_NAME
    PROCESSES = 4
    NUM_TESTS = 200

    SEED = 1

    #MODEL_NAMES = ["AIM", "disordered_fermi_hubbard"]
    MODEL_NAMES = ["AIM"]
    MODEL_N = 8
    SELECTED_PARAMETERS = {
        "AIM": (
            "U",
            "vb1", "vb2", "vb3", "vb4",
            "eb2", "eb3", "eb4"
        ),
        "disordered_fermi_hubbard": (
            "xi0", "xi1", "xi2", "xi3",
            "xi4", "xi5", "xi6", "xi7"
        )
    }
    PARAMETER_SPACE = {
        "AIM": (
            (0.01, 5.0),
            (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0),
            (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)
        ),

        "disordered_fermi_hubbard": (
            (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0),
            (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)
        )
    }
    INIT_THETA = {
        "AIM": tuple(
            [PARAMETER_SPACE["AIM"][i][0]
            for i in range(len(PARAMETER_SPACE["AIM"]))]
        ),
        "disorderd_fermi_hubbard": tuple(
            [PARAMETER_SPACE["disordered_fermi_hubbard"][i][0]
            for i in range(len(PARAMETER_SPACE["disordered_fermi_hubbard"]))]
        )
    }
    PARTICLE_SELECTION = (MODEL_N // 2, MODEL_N // 2)
    SPARSE = True

    if not os.path.isdir(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)

    #### COST FUNCTION SETUP ####
    GRID_SIZE = 8_000
    POINTS_PER_ITERATION = 50

    ### VARIANCE COST FUNCTION SETUP ###
    VARIANCE_THRESHOLDS = [
        1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10
    ]

    ### RESIDUAL COST FUNCTION SETUP ###
    RESIDUAL_THRESHOLDS = [
        1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10
    ]
    
    #### RUN ####
    b_var_basis_sizes = {}
    b_var_iterations = {}
    b_var_max_errors = {}
    b_var_efficiencies = {}
    b_var_times = {}

    g_var_basis_sizes = {}
    g_var_iterations = {}
    g_var_max_errors = {}
    g_var_efficiencies = {}
    g_var_times = {}

    b_res_basis_sizes = {}
    b_res_iterations = {}
    b_res_max_errors = {}
    b_res_efficiencies = {}
    b_res_times = {}

    for MODEL_NAME in MODEL_NAMES:
        b_var_basis_sizes[MODEL_NAME] = []
        b_var_iterations[MODEL_NAME] = []
        b_var_max_errors[MODEL_NAME] = []
        b_var_efficiencies[MODEL_NAME] = []
        b_var_times[MODEL_NAME] = []

        g_var_basis_sizes[MODEL_NAME] = []
        g_var_iterations[MODEL_NAME] = []
        g_var_max_errors[MODEL_NAME] = []
        g_var_efficiencies[MODEL_NAME] = []
        g_var_times[MODEL_NAME] = []

        model = SurrogateModel(
            name = MODEL_NAME,
            selected_params = SELECTED_PARAMETERS[MODEL_NAME],
            N = MODEL_N,
            particle_selection = PARTICLE_SELECTION,
            sparse = SPARSE,
            processes = PROCESSES,
            save_folder = SAVE_FOLDER,
        )

        model.build_terms()

        model.log("Generating test points...")

        tester = Tester(
            model,
            PARAMETER_SPACE[MODEL_NAME],
            processes = PROCESSES,
            num_tests = NUM_TESTS,
            seed = 0 # Testing seed should ALWAYS be zero -- for consistency
        )

        model.log("Test points generated")

        for VARIANCE_THRESHOLD in VARIANCE_THRESHOLDS:
            order = int(-np.log10(VARIANCE_THRESHOLD))
            model.log(
                f"OPTIMIZING {MODEL_NAME} WITH BATCHED VARIANCE WITH ORDER "
                + f"{order}"
            )

            b_var_cf = VarianceCostFunction2(
                model,
                VARIANCE_THRESHOLD,
                PARAMETER_SPACE[MODEL_NAME],
                GRID_SIZE,
                POINTS_PER_ITERATION,
                seed = SEED
            )

            model.optimize(
                b_var_cf,
                INIT_THETA[MODEL_NAME],
                f"Batched_VarianceResults_T{order}_S{SEED}"
            )

            b_var_basis_sizes[MODEL_NAME].append(model.opt_basis.shape[1])
            b_var_iterations[MODEL_NAME].append(model.n_iterations)
            b_var_max_errors[MODEL_NAME].append(max(tester.test_model()))
            b_var_n_full_diag = model.n_full_diag
            b_var_efficiencies[MODEL_NAME].append(
                b_var_basis_sizes[MODEL_NAME][-1] / b_var_n_full_diag
                if b_var_n_full_diag else float("nan")
            )
            b_var_times[MODEL_NAME].append(model.optimization_time)

            model.log(
                f"Batched Variance Optimization Time: "
                + f"{b_var_times[MODEL_NAME][-1]} seconds"
            )
            model.log(
                f"Batched Variance Basis Size "
                + f"{b_var_basis_sizes[MODEL_NAME][-1]}"
            )
            model.log(
                f"Batched Variance Iterations "
                + f"{b_var_iterations[MODEL_NAME][-1]}"
            )
            model.log(
                f"Batched Variance Max Error {b_var_max_errors[MODEL_NAME][-1]}"
            )
            model.log(
                f"Batched Variance Full Diagonalizations {b_var_n_full_diag}"
            )
            model.log(
                f"Batched Variance Efficiency "
                + f"{b_var_efficiencies[MODEL_NAME][-1]}"
            )

            model.reset()

            model.log(
                f"OPTIMIZING {MODEL_NAME} WITH GLOBAL VARIANCE WITH ORDER "
                + f"{order}"
            )
            model.log(f"Building training grid...")

            training_grid = training_grid_generator(
                PARAMETER_SPACE[MODEL_NAME],
                GRID_SIZE,
                model,
                PROCESSES,
                seed = SEED
            )

            model.log(f"Training grid build")

            g_var_cf = VarianceCostFunction(
                model,
                training_grid,
                VARIANCE_THRESHOLD
            )

            model.optimize(
                g_var_cf,
                INIT_THETA[MODEL_NAME],
                f"Global_VarianceResults_T{order}_S{SEED}"
            )

            g_var_basis_sizes[MODEL_NAME].append(model.opt_basis.shape[1])
            g_var_iterations[MODEL_NAME].append(model.n_iterations)
            g_var_max_errors[MODEL_NAME].append(max(tester.test_model()))
            g_var_n_full_diag = model.n_full_diag
            g_var_efficiencies[MODEL_NAME].append(
                g_var_basis_sizes[MODEL_NAME][-1] / g_var_n_full_diag
                if g_var_n_full_diag else float("nan")
            )
            g_var_times[MODEL_NAME].append(model.optimization_time)

            model.log(
                f"Batched Variance Optimization Time: "
                + f"{g_var_times[MODEL_NAME][-1]} seconds"
            )
            model.log(
                f"Batched Variance Basis Size "
                + f"{g_var_basis_sizes[MODEL_NAME][-1]}"
            )
            model.log(
                f"Batched Variance Iterations "
                + f"{g_var_iterations[MODEL_NAME][-1]}"
            )
            model.log(
                f"Batched Variance Max Error {g_var_max_errors[MODEL_NAME][-1]}"
            )
            model.log(
                f"Batched Variance Full Diagonalizations {g_var_n_full_diag}"
            )
            model.log(
                f"Batched Variance Efficiency "
                + f"{g_var_efficiencies[MODEL_NAME][-1]}"
            )

            model.reset()

        b_res_basis_sizes[MODEL_NAME] = []
        b_res_iterations[MODEL_NAME] = []
        b_res_max_errors[MODEL_NAME] = []
        b_res_efficiencies[MODEL_NAME] = []
        b_res_times[MODEL_NAME] = []

        for RESIDUAL_THRESHOLD in RESIDUAL_THRESHOLDS:
            order = int(-np.log10(RESIDUAL_THRESHOLD))
            model.log(f"OPTIMIZING BATCHED RESIDUAL WITH ORDER {order}")

            b_res_cf = ResidualCostFunction(
                model,
                RESIDUAL_THRESHOLD,
                PARAMETER_SPACE[MODEL_NAME],
                GRID_SIZE,
                POINTS_PER_ITERATION,
                seed = SEED
            )

            model.optimize(
                b_res_cf,
                INIT_THETA[MODEL_NAME],
                f"Batched_ResidualResults_T{order}_S{SEED}"
            )

            b_res_basis_sizes[MODEL_NAME].append(model.opt_basis.shape[1])
            b_res_iterations[MODEL_NAME].append(model.n_iterations)
            b_res_max_errors[MODEL_NAME].append(max(tester.test_model()))
            b_res_n_full_diag = model.n_full_diag
            b_res_efficiencies[MODEL_NAME].append(
                res_basis_sizes[MODEL_NAME][-1] / res_n_full_diag
                if res_n_full_diag else float("nan")
            )
            b_res_times[MODEL_NAME].append(model.optimization_time)

            model.log(
                f"Batched Residual Optimization Time: "
                + f"{g_res_times[MODEL_NAME][-1]} seconds"
            )
            model.log(
                f"Batched Residual Basis Size "
                + f"{g_res_basis_sizes[MODEL_NAME][-1]}"
            )
            model.log(
                f"Batched Residual Iterations "
                + f"{g_res_iterations[MODEL_NAME][-1]}"
            )
            model.log(
                f"Batched Residual Max Error {g_res_max_errors[MODEL_NAME][-1]}"
            )
            model.log(
                f"Batched Residual Full Diagonalizations {g_res_n_full_diag}"
            )
            model.log(
                f"Batched Residual Efficiency "
                + f"{g_res_efficiencies[MODEL_NAME][-1]}"
            )

            model.reset()

    for MODEL_NAME in MODEL_NAMES:
        plt.loglog(
            VARIANCE_THRESHOLDS, b_var_max_errors[MODEL_NAME],
            label="Batched Variance"
        )
        plt.loglog(
            VARIANCE_THRESHOLDS, g_var_max_errors[MODEL_NAME],
            label="Global Variance"
        )
        plt.loglog(
            RESIDUAL_THRESHOLDS, b_res_max_errors[MODEL_NAME],
            label="Batched Residual"
        )
    plt.xlabel("Threshold")
    plt.ylabel("Max Error")
    plt.title("Max Error vs Threshold")
    plt.legend()
    plt.savefig(f"{SAVE_FOLDER}/{TEST_NAME}_{TEST_START}_ERROR.svg")
    plt.clf()

    for MODEL_NAME in MODEL_NAMES:
        plt.semilogx(
            VARIANCE_THRESHOLDS, b_var_basis_sizes[MODEL_NAME],
            label="Batched Variance"
        )
        plt.semilogx(
            VARIANCE_THRESHOLDS, g_var_basis_sizes[MODEL_NAME],
            label="Global Variance"
        )
        plt.semilogx(
            RESIDUAL_THRESHOLDS, b_res_basis_sizes[MODEL_NAME],
            label="Batched Residual"
        )
    plt.xlabel("Threshold")
    plt.ylabel("Basis Size")
    plt.title("Basis Size vs Threshold")
    plt.legend()
    plt.savefig(f"{SAVE_FOLDER}/{TEST_NAME}_{TEST_START}_BASIS_SIZE.svg")
    plt.clf()

    for MODEL_NAME in MODEL_NAMES:
        plt.semilogx(
            VARIANCE_THRESHOLDS, b_var_iterations[MODEL_NAME],
            label="Batched Variance"
        )
        plt.semilogx(
            VARIANCE_THRESHOLDS, g_var_iterations[MODEL_NAME],
            label="Global Variance"
        )
        plt.semilogx(
            RESIDUAL_THRESHOLDS, b_res_iterations[MODEL_NAME],
            label="Batched Residual"
        )
    plt.xlabel("Threshold")
    plt.ylabel("Iterations")
    plt.title("Iterations vs Threshold")
    plt.legend()
    plt.savefig(f"{SAVE_FOLDER}/{TEST_NAME}_{TEST_START}_ITERATIONS.svg")
    plt.clf()

    for MODEL_NAME in MODEL_NAMES:
        plt.semilogx(
            VARIANCE_THRESHOLDS, b_var_efficiencies[MODEL_NAME],
            label="Batched Variance"
        )
        plt.semilogx(
            VARIANCE_THRESHOLDS, g_var_efficiencies[MODEL_NAME],
            label="Global Variance"
        )
        plt.semilogx(
            RESIDUAL_THRESHOLDS, b_res_efficiencies[MODEL_NAME],
            label="Batched Residual"
        )
    plt.xlabel("Threshold")
    plt.ylabel("Efficiency")
    plt.title("Efficiency vs Threshold")
    plt.legend()
    plt.savefig(f"{SAVE_FOLDER}/{TEST_NAME}_{TEST_START}_EFFICIENCY.svg")
    plt.clf()

    for MODEL_NAME in MODEL_NAMES:
        plt.semilogx(
            VARIANCE_THRESHOLDS, b_var_times[MODEL_NAME],
            label="Batched Variance"
        )
        plt.semilogx(
            VARIANCE_THRESHOLDS, g_var_times[MODEL_NAME],
            label="Global Variance"
        )
        plt.semilogx(
            RESIDUAL_THRESHOLDS, b_res_times[MODEL_NAME],
            label="Batched Residual"
        )
    plt.xlabel("Threshold")
    plt.ylabel("Time")
    plt.title("Time vs Threshold")
    plt.legend()
    plt.savefig(f"{SAVE_FOLDER}/{TEST_NAME}_{TEST_START}_TIME.svg")
    plt.clf()

if __name__ == "__main__":
    main()
