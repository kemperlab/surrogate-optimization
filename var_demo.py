import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import datetime

from advisor import SurrogateAdvisor
from examples import ResidualCostFunction, VarianceCostFunction2
from pauli import *
from surrogate import SurrogateModel

from gep_advisor import GEPAdvisor

MODEL = "AIM"
N = 4
NI = 1
NB = N - NI
PARAM_BOUNDS = ((-5.0, 5.0), (-5.0, 5.0))
SELECTED_PARAMS = ("vb", "eb")
INIT_THETA = (PARAM_BOUNDS[0][0], PARAM_BOUNDS[1][0])
"""
N = 6
NI = 1
NB = N - NI
PARAM_BOUNDS = ((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0))
SELECTED_PARAMS = ("U", "vb1", "vb2", "vb3", "eb2", "eb3")
INIT_THETA = (PARAM_BOUNDS[0][0], PARAM_BOUNDS[1][0], PARAM_BOUNDS[2][0], PARAM_BOUNDS[3][0], PARAM_BOUNDS[4][0], PARAM_BOUNDS[5][0])
"""

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
    N,
    processes=1
)

model2 = SurrogateModel(
    MODEL,
    SELECTED_PARAMS,
    N,
    processes=1
)

model.build_terms()
model2.build_terms()
print("Built terms")

mu = np.linspace(-5.0, 5.0, 20)
mu_2 = np.linspace(-5.0, 5.0, 20)

training_grid = []
"""
for i, m1 in enumerate(mu):
    for j, m2 in enumerate(mu_2):
        training_grid.append(model.theta_to_training_point((m1, m2)))
"""
training_grid = np.array(training_grid)

cfi = VarianceCostFunction2(
    model,
    1e-9,
    INIT_THETA,
    PARAM_BOUNDS,
    1000,
    10
)
cfi2 = VarianceCostFunction2(
    model2,
    1e-9,
    INIT_THETA,
    PARAM_BOUNDS,
    1000,
    10
)

t1 = datetime.datetime.now()
first_cost = model.init_optimize(cfi, INIT_THETA)

model.build_Hr_terms()

iteration_costs = [[first_cost]]

chosen = [INIT_THETA]

for i in range(100):
    print(f"It {i}")
    cfi.preiteration()
    advisor = SurrogateAdvisor(
        model,
        cfi,
        PARAM_BOUNDS,
        log_sample_size=5
    )

    advisor.sobol_sample()

    grid_points = 40
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

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (13, 4))

    fig.suptitle(f"Iteration {i}")

    im1 = ax1.imshow(points, extent=[-5.0, 5.0, 5.0, -5.0])
    ax1.scatter(
        *np.array(chosen).T,
        marker="o",
        color="orange",
        s=20,
        label="Chosen Points",
    )
    ax1.scatter(
        new_theta[0],
        new_theta[1],
        marker="x",
        color="red",
        s=20,
        label="Chosen Points",
    )
    fig.colorbar(im1, ax=ax1)
    ax1.set_title("Actual Cost")
    ax1.set_xlabel(SELECTED_PARAMS[0])
    ax1.set_ylabel(SELECTED_PARAMS[1])

    im2 = ax2.imshow(fake_points, extent=[-5.0, 5.0, 5.0, -5.0])
    ax2.scatter(
        *np.array(chosen).T,
        marker="o",
        color="orange",
        s=20,
        label="Chosen Points",
    )
    ax2.scatter(
        new_theta[0],
        new_theta[1],
        marker="x",
        color="red",
        s=20,
        label="Chosen Points",
    )
    ax2.scatter(
        *np.array(advisor.samples).T,
        marker="*",
        color="red",
        s=20,
        label="Chosen Points",
    )
    fig.colorbar(im1, ax=ax2)
    ax2.set_title("Cost from Guassian Process")
    ax2.set_xlabel(SELECTED_PARAMS[0])
    ax2.set_ylabel(SELECTED_PARAMS[1])

    im3 = ax3.imshow(
        np.abs(points - fake_points) / points,
        vmin = 0,
        vmax = 1,
        extent=[-5.0, 5.0, 5.0, -5.0]
    )
    fig.colorbar(im3, ax=ax3)
    ax3.set_title("Relative Error")
    ax3.set_xlabel(SELECTED_PARAMS[0])
    ax3.set_ylabel(SELECTED_PARAMS[1])

    fig.show()
    input()

    chosen.append(new_theta)

    new_cost = advisor.predict(new_theta)

    basis_addition = model.find_basis_addition([new_cost], [new_training_point])
    model.compress_basis(basis_addition)
    model.build_Hr_terms()

    iteration_costs.append([new_cost])

    if new_cost < 1e-3:
        print(model.basis.shape)
        break

    #if cfi.check_termination(iteration_costs):
    #    break

model.set_optimal()
print(model.basis.shape)
t2 = datetime.datetime.now()
model2.optimize(
    cfi2,
    INIT_THETA
)
t3 = datetime.datetime.now()

print("Gaussian")
print("Basis Size:", model.opt_basis.shape[1])
print("Full Hilbert Size:", model.size)
print("Time:", t2 - t1)
print("Real")
print("Basis Size:", model2.opt_basis.shape[1])
print("Full Hilbert Size:", model2.size)
print("Time:", t3 - t2)

### Testing the surrogate model against random parameters
errors = []
errors2 = []
all_ps = []
for i in range(10):
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
    #test_evals, test_evecs = model2.solve(parameters)
    #print("Real", evals[0])
    #print("Approx", test_evals[0])
    #print("GAUSSIAN")
    #if abs(evals[0]) < 1e-12:
    #    errors2.append(np.abs(evals[0] - test_evals[0]))
    #else:
    #    errors2.append(np.abs(evals[0] - test_evals[0]) / np.abs(evals[0]))
    #print(
    #    "Relative Error",
    #    errors2[-1],
    #)
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
