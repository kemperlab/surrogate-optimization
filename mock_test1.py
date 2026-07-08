################################################################################
# MOCK TEST 1
# 
# SIX SITE ANDERSON IMPURITY MODEL
################################################################################

import datetime
import matplotlib.pyplot as plt

from examples import (
    ResidualCostFunction,
    VarianceCostFunction2,
    VarianceCostFunction
)
from surrogate import *
from testing_interface import *

#### MODEL SETUP ####
if __name__ == "__main__":
    TEST_START = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    TEST_NAME = "MOCK_TEST1"
    SAVE_FOLDER = TEST_NAME
    PROCESSES = 4

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
    with open(LOG_FILENAME, "w") as log_stream:
        model = SurrogateModel(
            MODEL_NAME,
            SELECTED_PARAMETERS,
            MODEL_N,
            particle_selection = PARTICLE_SELECTION,
            sparse=SPARSE,
            #output_stream=log_stream,
            save_folder = SAVE_FOLDER,
            keep_on_disk = True,
            processes = PROCESSES
        )

        model.build_terms()

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

        training_grid = []
        
        for point in points:
            training_grid.append(model.theta_to_training_point(point))
        training_grid = np.array(training_grid, dtype=dict)

        model.log("Training grid generated")

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

        model.optimize(var_training_cf, INIT_THETA)

        vart_basis_size = model.opt_basis.shape[1]
        vart_iterations = len(model.iteration_costs)
        vart_basis_growth = model.basis_growth
        vart_errors = test_model(model, PARAMETER_SPACE, 200)

        model.reset()

        model.optimize(var_cf, INIT_THETA)

        var_basis_size = model.opt_basis.shape[1]
        var_iterations = len(model.iteration_costs)
        var_basis_growth = model.basis_growth
        var_errors = test_model(model, PARAMETER_SPACE, 200)

        model.reset()

        model.optimize(res_cf, INIT_THETA)

        res_basis_size = model.opt_basis.shape[1]
        res_iterations = len(model.iteration_costs)
        res_basis_growth = model.basis_growth
        res_errors = test_model(model, PARAMETER_SPACE, 200)

        model.reset()

        model.optimize(sob_cf, INIT_THETA)

        sob_basis_size = model.opt_basis.shape[1]
        sob_iterations = len(model.iteration_costs)
        sob_basis_growth = model.basis_growth
        sob_errors = test_model(model, PARAMETER_SPACE, 200)

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
