import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

from advisor import SurrogateAdvisor
from examples import ResidualCostFunction, VarianceCostFunction
from pauli import model_to_paulis, param_to_paulis
from surrogate2 import SurrogateModel

MODEL = "TFIM"
sparse_proportion = .2
model_type = MODEL
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

first_cost = model.init_optimize(cfi, INIT_THETA)

iteration_costs = [[first_cost]]

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

    iteration_costs.append([new_cost])

    print("COST =========================================================")
    print(new_cost)

    if new_cost < 1e-6:
        print(model.basis.shape)
        break

    #if cfi.check_termination(iteration_costs):
    #    break

model.set_optimal()
### Testing the surrogate model against random parameters
errors = []
all_ps = []
for i in range(200):
    H_full = np.zeros((model.size, model.size), dtype=complex)

    if model_type == "TFIM":
        J = 2 * np.random.randn()
        h = 2 * np.random.randn()
        model_paulis = model_to_paulis(
            N,
            model_type,
            {
                "J": J,
                "h": h,
                "periodic": False,
            },
        )
    elif model_type == "TFXY":
        Jx = 2 * np.random.randn()
        Jy = Jx
        h = 2 * np.random.randn()
        model_paulis = model_to_paulis(
            N,
            model_type,
            {
                "Jx": Jx,
                "Jy": Jy,
                "h": h,
                "periodic": False,
            },
        )
    elif model_type == "heisenberg":
        Jx = 2 * np.random.randn()
        Jy = Jx
        Jz = 2 * np.random.randn()
        h = 2 * np.random.randn()
        model_paulis = model_to_paulis(
            N,
            model_type,
            {
                "Jx": Jx,
                "Jy": Jy,
                "Jz": Jz,
                "h": h,
                "periodic": False,
            },
        )
    elif model_type == "fermi_hubbard":
        model_paulis = model_to_paulis(
            N,
            model_type,
            {
                "mu": mu_chem,
                "t": 2 * np.random.randn(),
                "U": (5.0 - 1.0) * np.random.rand() + 1.0,
            },
        )
    elif model_type == "AIM":
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
            model_type,
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

    # print(parameters)
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
print("Basis Size:", model.opt_basis.shape[1])
print("Full Hilbert Size:", model.size)
plt.plot(errors, "o-")
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
