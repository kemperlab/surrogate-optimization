import numpy as np
import scipy as sp
import scipy.sparse as sps
import matplotlib.pyplot as plt
import datetime

from advisor import SurrogateAdvisor
from examples import ResidualCostFunction, VarianceCostFunction, VarianceCostFunction2
from pauli import *
from surrogate import SurrogateModel

from gep_advisor import GEPAdvisor

MODEL = "AIM"
N = 6
NI = 1
NB = N - NI
PARAM_BOUNDS = ((0.01, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0))
SELECTED_PARAMS = ("U", "vb1", "vb2", "vb3", "eb2", "eb3")
INIT_THETA = (PARAM_BOUNDS[0][0], PARAM_BOUNDS[1][0], PARAM_BOUNDS[2][0], PARAM_BOUNDS[3][0], PARAM_BOUNDS[4][0], PARAM_BOUNDS[5][0])

sparse_proportion = .5

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
    particle_selection=(3, 3),
    basis_ordering="udud",
    sparse_proportion=sparse_proportion,
    processes=1
)

print(model.pauli_strings)

model2 = SurrogateModel(
    MODEL,
    SELECTED_PARAMS,
    N,
    particle_selection=(3, 3),
    basis_ordering="udud",
    sparse_proportion=sparse_proportion,
    processes=1
)

model3 = SurrogateModel(
    MODEL,
    SELECTED_PARAMS,
    N,
    particle_selection=(3, 3),
    basis_ordering="udud",
    sparse_proportion=sparse_proportion,
    processes=1
)

model.build_terms()
model2.build_terms()
model3.build_terms()
print(model.size)
mu = np.linspace(-5.0, 5.0, 20)
mu_2 = np.linspace(-5.0, 5.0, 20)


cfi = ResidualCostFunction(
    model,
    INIT_THETA,
    PARAM_BOUNDS,
    1000,
    10
)
cfi2 = ResidualCostFunction(
    model2,
    INIT_THETA,
    PARAM_BOUNDS,
    1000,
    10
)

cfi3 = VarianceCostFunction2(
    model3,
    1e-9,
    INIT_THETA,
    PARAM_BOUNDS,
    1000,
    10
)

print("Built terms")
model3.optimize(
    cfi3,
    INIT_THETA
)

print(model3.opt_basis.shape)
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

model2.optimize(
    cfi2,
    INIT_THETA
)
print(model2.opt_basis.shape)

model2_max_costs = []
model2_states_added = []
for costs in model2.iteration_costs:
    if len(costs) != 0:
        model2_max_costs.append(float(max(costs)))
    else:
        model2_max_costs.append(0)
    model2_states_added.append(len(costs))

model3_max_costs = []
model3_states_added = []
for costs in model3.iteration_costs:
    if len(costs) != 0:
        model3_max_costs.append(float(max(costs)))
    else:
        model3_max_costs.append(0)
    model3_states_added.append(len(costs))

model2_norm = max(model2_max_costs)
model2_max_costs = np.array(model2_max_costs)
model3_norm = max(model3_max_costs)
model3_max_costs = np.array(model3_max_costs)

plt.plot(np.arange(len(model2_max_costs)), model2_max_costs / model2_norm, label = "Residual")
plt.plot(np.arange(len(model3_max_costs)), model3_max_costs / model3_norm, label = "Variance")
plt.legend()
plt.title("Max Cost Over Each Iteration")
plt.xlabel("Iteration")
plt.ylabel("Max Cost (Normalized)")
plt.show()

plt.plot(np.arange(len(model2_max_costs)), model2_states_added, label = "Residual")
plt.plot(np.arange(len(model3_max_costs)), model3_states_added, label = "Variance")
plt.legend()
plt.title("States Added Over Each Iteration")
plt.xlabel("Iteration")
plt.ylabel("States added")
plt.show()

rng = np.random.default_rng(None)
sobol = sp.stats.qmc.Sobol(len(PARAM_BOUNDS), rng = rng)
samples = sobol.random_base2(7)

first_cost = model.init_optimize(cfi, INIT_THETA)

iteration_costs = [[first_cost]]
end_it = 0

for i, sample in enumerate(samples):
    theta = [
        (b[1] - b[0]) * s + b[0]
        for b, s in zip(PARAM_BOUNDS, sample)
    ]

    training_point = model.theta_to_training_point(theta)
    H_full = model.build_H_full(training_point)

    if model.sparse:
        evals, evecs = sps.linalg.eigsh(
            H_full.real,
            k=int(sparse_proportion*model.size),
            which='SA'
        )
    else:
        evals, evecs = np.linalg.eigh(H_full)
    basis_addition = None

    # find degeneracy of the ground state
    eps = 1e-10 # for comparing floating points of GSE
    degeneracy = 0
    for e in evals:
        if e - evals[0] < eps:
            degeneracy += 1
        else:
            break
        if degeneracy >= 5:
            break

    if type(basis_addition) == type(None):
        basis_addition = evecs[:, 0:degeneracy]
    else:
        basis_addition = np.append(
            basis_addition,
            evecs[:, 0:degeneracy],
            axis = 1
        )

    if cfi.cost_function(training_point) < 1e-10:
        end_it = i
        break

    model.compress_basis(basis_addition)

chosen = [INIT_THETA]

model.set_optimal()
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
    test_evals, test_evecs = model3.solve(parameters)
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

print(end_it)
