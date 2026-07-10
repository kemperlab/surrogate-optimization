import numpy as np
import scipy as sp
import scipy.sparse as sps
import matplotlib.pyplot as plt
import datetime
from pathos.multiprocessing import ProcessPool

from advisor import SurrogateAdvisor
from examples import ResidualCostFunction, VarianceCostFunction
from pauli import *
from surrogate import SurrogateModel
from testing_interface import Tester

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

    pp = ProcessPool(nodes=PROCESSES)

    model = SurrogateModel(
        MODEL_NAME,
        SELECTED_PARAMETERS,
        MODEL_N,
        particle_selection = (MODEL_N // 2, MODEL_N // 2),
        processes = PROCESSES
    )

    model.build_terms()

    tester = Tester(
        model,
        PARAMETER_SPACE,
        processes = PROCESSES,
        num_tests = 50,
        seed = SEED
    )

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

    model.log("Generating training grid...")

    batch_size = int(np.ceil(len(points) / PROCESSES))
    training_grid = list(pp.map(
        model.theta_to_training_point,
        points,
        chunksize=batch_size
    ))
    training_grid = np.array(training_grid, dtype=dict)

    model.log("Training grid generated")

    var_cfi = VarianceCostFunction(
        model,
        training_grid,
        VARIANCE_THRESHOLD
    )

    res_cfi = ResidualCostFunction(
        model,
        RESIDUAL_THRESHOLD,
        INIT_THETA,
        PARAMETER_SPACE,
        TOTAL_SOBOL_POINTS,
        POINTS_PER_ITERATION,
        seed=SEED
    )

    model.basis_growth = []
    model.iteration_costs = [[model.init_optimize(var_cfi, INIT_THETA)]]
    model.set_optimal()

    var_basis_sizes = [model.basis.shape[1]]
    var_max_errors = [max(tester.test_model())]

    for i in range(model.max_it):
        model.log(f"Iteration {i + 1}")
        if model.optimize_step(var_cfi):
            break
        model.set_optimal()
        errors = tester.test_model()
        max_error = max(errors)

        var_basis_sizes.append(model.basis.shape[1])
        var_max_errors.append(max_error)

    model.reset()

    model.basis_growth = []
    model.iteration_costs = [[model.init_optimize(var_cfi, INIT_THETA)]]
    model.set_optimal()

    res_basis_sizes = [model.basis.shape[1]]
    res_max_errors = [max(tester.test_model())]

    for i in range(model.max_it):
        model.log(f"Iteration {i + 1}")
        if model.optimize_step(res_cfi):
            break
        model.set_optimal()
        errors = tester.test_model()
        max_error = max(errors)

        res_basis_sizes.append(model.basis.shape[1])
        res_max_errors.append(max_error)

    plt.semilogy(var_basis_sizes, var_max_errors, label="Variance")
    plt.semilogy(res_basis_sizes, res_max_errors, label="Residual")
    plt.xlabel("Basis Size")
    plt.ylabel("Max Error")
    plt.title("Max Error VS Basis Size, AIM N=6")
    plt.legend()
    plt.show()
