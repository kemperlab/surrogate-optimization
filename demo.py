import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

from advisor import SurrogateAdvisor
from examples import ResidualCostFunction, VarianceCostFunction
from pauli import *
from surrogate2 import SurrogateModel

MODEL = "AIM"
N = 5
NI = 1
NB = N - NI
PARAM_BOUNDS = ((-5.0, 5.0), (-5.0, 5.0))
SELECTED_PARAMS = ("vb", "eb")
INIT_THETA = (PARAM_BOUNDS[0][0], PARAM_BOUNDS[1][0])

sparse_proportion = .2

INIT_PARAM = theta_to_param(MODEL, N, SELECTED_PARAMS, INIT_THETA)
BASE_PARAMS = get_model_base_parameters(MODEL, N)
BASE_PARAMS2 = get_model_base_parameters(MODEL, N)
PARAMS = get_model_parameters(MODEL)

model_paulis = model_to_paulis(N, MODEL, BASE_PARAMS)
H_paulis = [t[0] for t in model_paulis]
H_paulis_order = {}

for i, pauli in enumerate(H_paulis):
    H_paulis_order[pauli] = i

model = SurrogateModel(
    MODEL,
    SELECTED_PARAMS,
    N
)

model2 = SurrogateModel(
    MODEL,
    SELECTED_PARAMS,
    N
)

model.build_terms()
model2.build_terms()
print("Built terms")

"""
mu = np.linspace(-5.0, 5.0, 20)
mu_2 = np.linspace(-5.0, 5.0, 20)

training_grid = np.array([])
for m1 in mu:
    for m2 in mu_2:
        if MODEL == "TFIM":
            BASE_PARAMS2["J"] = m1
            BASE_PARAMS2["h"] = m2
        elif MODEL == "TFXY":
            BASE_PARAMS2["Jx"] = m1
            BASE_PARAMS2["Jy"] = m1
            BASE_PARAMS2["h"] = m2
        elif MODEL == "heisenberg":
            BASE_PARAMS2["Jx"] = m1
            BASE_PARAMS2["Jy"] = m1
            BASE_PARAMS2["Jz"] = m2
            BASE_PARAMS2["h"] = 0.1
        elif MODEL == "fermi_hubbard":
            BASE_PARAMS2["t"] = m1
            BASE_PARAMS2["mu"] = mu_chem
            BASE_PARAMS2["U"] = m2
        elif MODEL == "AIM":
            BASE_PARAMS2["vb"] = np.array(
                [0.01] * ((NB) % 2) + [m1] * (NB - (NB) % 2)
            )

            BASE_PARAMS2["eb"] = np.array(
                [0.0] * ((NB) % 2)
                + [m2] * ((NB - (NB) % 2) // 2)
                + [-m2] * ((NB - (NB) % 2) // 2)
            )
        model_paulis2 = model_to_paulis(N, MODEL, BASE_PARAMS2)
        params = np.zeros(len(H_paulis), dtype=tuple)

        for t in model_paulis2:
            try:
                params[H_paulis_order[t[0]]] = t[1]
            except:
                raise Exception("Failed to generate all terms in model")

        model_paulis2 = list(zip(H_paulis, params))
        paulis_dict = {}
        for t in model_paulis2:
            paulis_dict[t[0]] = t[1]
        training_grid = np.append(training_grid, paulis_dict)
"""

cfi = ResidualCostFunction(
    model,
    INIT_THETA,
    PARAM_BOUNDS,
    1000,
    10
)
#cfi = VarianceCostFunction(model, np.zeros((1, 1)), 1e-8)
cfi2 = ResidualCostFunction(
    model2,
    INIT_THETA,
    PARAM_BOUNDS,
    1000,
    10
)
"""
cfi2 = VarianceCostFunction(model, training_grid, 1e-9)
"""

first_cost = model.init_optimize(cfi, INIT_THETA)

iteration_costs = [[first_cost]]

chosen = [INIT_THETA]

for i in range(100):
    print(f"It {i}")
    advisor = SurrogateAdvisor(
        model,
        cfi,
        PARAM_BOUNDS,
        log_sample_size=4
    )

    advisor.sobol_sample()

    grid_points = 1
    points = np.zeros((grid_points, grid_points))
    fake_points = np.zeros((grid_points, grid_points))
    for y, h in enumerate(np.linspace(-5.0, 5.0, grid_points)):
        for x, J in enumerate(np.linspace(-5.0, 5.0, grid_points)):
            theta = (J, h)
            training_point = model.theta_to_training_point(theta)
            predicted_cost = advisor.predict(theta)
            real_cost = cfi.cost_function(training_point)
            points[y, x] = real_cost
            fake_points[y, x] = predicted_cost

    print("Max")
    print(advisor.find_max())
    new_theta = advisor.find_max()

    new_training_point = model.theta_to_training_point(new_theta)

    """
    plt.imshow(points, extent=[-5.0, 5.0, 5.0, -5.0])
    plt.scatter(
        *np.array(chosen).T,
        marker="o",
        color="orange",
        s=20,
        label="Chosen Points",
    )
    plt.scatter(
        new_theta[0],
        new_theta[1],
        marker="x",
        color="red",
        s=20,
        label="Chosen Points",
    )
    plt.colorbar()
    plt.title("Actual Cost")
    plt.xlabel(SELECTED_PARAMS[0])
    plt.ylabel(SELECTED_PARAMS[1])
    plt.show()

    plt.imshow(fake_points, extent=[-5.0, 5.0, 5.0, -5.0])
    plt.scatter(
        *np.array(chosen).T,
        marker="o",
        color="orange",
        s=20,
        label="Chosen Points",
    )
    plt.scatter(
        new_theta[0],
        new_theta[1],
        marker="x",
        color="red",
        s=20,
        label="Chosen Points",
    )
    plt.scatter(
        *np.array(advisor.samples).T,
        marker="*",
        color="red",
        s=20,
        label="Chosen Points",
    )
    plt.colorbar()
    plt.title("Cost from Guassian Process")
    plt.xlabel(SELECTED_PARAMS[0])
    plt.ylabel(SELECTED_PARAMS[1])
    plt.show()

    plt.imshow(
        np.abs(points - fake_points) / points,
        vmin = 0,
        vmax = 1,
        extent=[-5.0, 5.0, 5.0, -5.0]
    )
    plt.colorbar()
    plt.title("Relative Error")
    plt.xlabel(SELECTED_PARAMS[0])
    plt.ylabel(SELECTED_PARAMS[1])
    plt.show()
    """

    chosen.append(new_theta)

    new_cost = advisor.predict(new_theta)

    basis_addition = model.find_basis_addition([new_cost], [new_training_point])
    model.compress_basis(basis_addition)

    iteration_costs.append([new_cost])

    if new_cost < 1e-6:
        print(model.basis.shape)
        break

    #if cfi.check_termination(iteration_costs):
    #    break

model.set_optimal()
model2.optimize(
    cfi2,
    INIT_THETA
)
### Testing the surrogate model against random parameters
errors = []
errors2 = []
all_ps = []
for i in range(200):
    H_full = np.zeros((model.size, model.size), dtype=complex)

    if MODEL == "TFIM":
        J = 2 * np.random.randn()
        h = 2 * np.random.randn()
        model_paulis = model_to_paulis(
            N,
            MODEL,
            {
                "J": J,
                "h": h,
                "periodic": False,
            },
        )
    elif MODEL == "TFXY":
        Jx = 2 * np.random.randn()
        Jy = Jx
        h = 2 * np.random.randn()
        model_paulis = model_to_paulis(
            N,
            MODEL,
            {
                "Jx": Jx,
                "Jy": Jy,
                "h": h,
                "periodic": False,
            },
        )
    elif MODEL == "heisenberg":
        Jx = 2 * np.random.randn()
        Jy = Jx
        Jz = 2 * np.random.randn()
        h = 2 * np.random.randn()
        model_paulis = model_to_paulis(
            N,
            MODEL,
            {
                "Jx": Jx,
                "Jy": Jy,
                "Jz": Jz,
                "h": h,
                "periodic": False,
            },
        )
    elif MODEL == "fermi_hubbard":
        model_paulis = model_to_paulis(
            N,
            MODEL,
            {
                "mu": mu_chem,
                "t": 2 * np.random.randn(),
                "U": (5.0 - 1.0) * np.random.rand() + 1.0,
            },
        )
    elif MODEL == "AIM":
        NI = 1
        NB = N - NI
        U = 4.0
        vb_test = np.array(
            [0.01] * ((NB) % 2) + [2 * np.random.randn()] * (NB - (NB) % 2)
        )
        eb_r = 2.0 * np.random.randn()
        eb_test = np.array(
            [0.0] * ((NB) % 2)
            + [eb_r] * ((NB - (NB) % 2) // 2)
            + [-eb_r] * ((NB - (NB) % 2) // 2)
        )
        print("vb_test", vb_test)
        print("eb_test", eb_test)
        model_paulis = model_to_paulis(
            N,
            MODEL,
            {
                "NI": NI,
                "NB": NB,
                "U": U,
                "ei": [0.0] * NI,
                "vb": vb_test,
                "eb": eb_test,
                "mu": U / 2,
                "periodic": False,
            },
        )

    paulis_dict = {}
    for t in model_paulis:
        paulis_dict[t[0]] = t[1]
    parameters = paulis_dict

    for pauli in model.H_terms.keys():
        H_full += parameters[pauli] * model.H_terms[pauli]

    if model.sparse:
        evals, evecs = sps.linalg.eigsh(
            H_full.real,
            k=int(sparse_proportion*model.size),
            which='SA'
        )
    else:
        evals, evecs = np.linalg.eigh(H_full)

    print(parameters)
    print("REAL")
    test_evals, test_evecs = model2.solve(parameters)
    print("Real", evals[0])
    print("Approx", test_evals[0])
    print("GAUSSIAN")
    if abs(evals[0]) < 1e-12:
        errors2.append(np.abs(evals[0] - test_evals[0]))
    else:
        errors2.append(np.abs(evals[0] - test_evals[0]) / np.abs(evals[0]))
    print(
        "Relative Error",
        errors2[-1],
    )
    print()
    test_evals, test_evecs = model.solve(parameters)
    print("Real", evals[0])
    print("Approx", test_evals[0])
    if abs(evals[0]) < 1e-12:
        errors.append(np.abs(evals[0] - test_evals[0]))
    else:
        errors.append(np.abs(evals[0] - test_evals[0]) / np.abs(evals[0]))
    print(
        "Relative Error",
        errors[-1],
    )
    print()
    all_ps.append(parameters)
print("Gaussian")
print("Basis Size:", model.opt_basis.shape[1])
print("Full Hilbert Size:", model.size)
print("Real")
print("Basis Size:", model2.opt_basis.shape[1])
print("Full Hilbert Size:", model2.size)
plt.plot(errors, "o-")
plt.plot(errors2, "x-")
plt.xlabel("Test Case")
plt.ylabel("Relative Error")
plt.title("Surrogate Model Relative Errors")
plt.yscale("log")
plt.ylim(1e-20, 1)
# plt.xticks(
#     range(len(errors)),
#     [f"({p[0]:.2f}, {p[n2b_terms + 1]:.2f})" for p in all_ps],
#     rotation=90,
# )
plt.show()
