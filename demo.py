import numpy as np
import matplotlib.pyplot as plt

from advisor import SurrogateAdvisor
from examples import ResidualCostFunction, VarianceCostFunction
from pauli import model_to_paulis, param_to_paulis
from surrogate2 import SurrogateModel

MODEL = "TFIM"
N = 4
BASE_PARAMS = {"J": 1.0, "h": 1.0, "periodic": False}
PARAMS = ("J", "h")
PARAM_BOUNDS = ((-2.5, 2.5), (-2.5, 2.5))
INIT_THETA = (PARAM_BOUNDS[0][0], PARAM_BOUNDS[1][0])

model_paulis = model_to_paulis(N, MODEL, BASE_PARAMS)
H_paulis = [t[0] for t in model_paulis]

model = SurrogateModel(
    MODEL,
    PARAMS,
    N
)

INIT_TRAINING_POINT = param_to_paulis(
    INIT_THETA,
    model.params,
    model.name,
    model.N
)

model.build_terms()

cfi = ResidualCostFunction(
    model,
    INIT_THETA,
    PARAM_BOUNDS,
    1000,
    10
)
#cfi = VarianceCostFunction(model, np.zeros((1, 1)), 1e-8)

model.init_optimize(cfi, INIT_THETA)

for i in range(10):
    advisor = SurrogateAdvisor(
        model,
        cfi,
        PARAM_BOUNDS
    )

    advisor.sobol_sample()

    points = np.zeros((100, 100))
    fake_points = np.zeros((100, 100))
    for x, J in enumerate(np.linspace(-2.5, 2.5, 100)):
        for y, h in enumerate(np.linspace(-2.5, 2.5, 100)):
            theta = (J, h)
            training_point = param_to_paulis(
                theta,
                model.params,
                model.name,
                model.N
            )
            predicted_cost = advisor.predict(theta)
            real_cost = cfi.cost_function(training_point)
            points[x, y] = real_cost
            fake_points[x, y] = predicted_cost

    print("Max")
    print(advisor.find_max())
    new_theta = advisor.find_max()
    new_training_point = param_to_paulis(
        new_theta,
        model.params,
        model.name,
        model.N
    )
    plt.imshow(points)
    plt.colorbar()
    plt.show()
    plt.imshow(fake_points)
    plt.colorbar()
    plt.show()

    plt.imshow(np.abs(points - fake_points) / points, vmin = 0, vmax = 1)
    plt.colorbar()
    plt.show()
    new_cost = advisor.predict(new_theta)

    basis_addition = model.find_basis_addition([new_cost], [new_training_point])
    model.compress_basis(basis_addition)
