import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

from advisor import SurrogateAdvisor
from examples import ResidualCostFunction, VarianceCostFunction2
from pauli import *
from surrogate import SurrogateModel

def test_model(model, sparse_proportion=0.2):
    errors = []
    for i in range(200):
        H_full = np.zeros((model.size, model.size), dtype=complex)

        if model.name == "TFIM":
            J = 2 * np.random.randn()
            h = 2 * np.random.randn()
            model_paulis = model_to_paulis(
                model.N_spin,
                model.name,
                {
                    "J": J,
                    "h": h,
                    "periodic": False,
                },
            )
        elif model.name == "TFXY":
            Jx = 2 * np.random.randn()
            Jy = Jx
            h = 2 * np.random.randn()
            model_paulis = model_to_paulis(
                model.N_spin,
                model.name,
                {
                    "Jx": Jx,
                    "Jy": Jy,
                    "h": h,
                    "periodic": False,
                },
            )
        elif model.name == "heisenberg":
            Jx = 2 * np.random.randn()
            Jy = Jx
            Jz = 2 * np.random.randn()
            h = 2 * np.random.randn()
            model_paulis = model_to_paulis(
                model.N_spin,
                model.name,
                {
                    "Jx": Jx,
                    "Jy": Jy,
                    "Jz": Jz,
                    "h": h,
                    "periodic": False,
                },
            )
        elif model.name == "fermi_hubbard":
            U = (5.0 - 1.0) * np.random.rand() + 1.0
            model_paulis = model_to_paulis(
                model.N_spin,
                model.name,
                {
                    "t": 2 * np.random.randn(),
                    "U": U,
                    "mu": U / 2
                },
            )
        elif model.name == "AIM":
            NI = 1
            NB = model.N_spin - NI
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
            model_paulis = model_to_paulis(
                model.N_spin,
                model.name,
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

        test_evals, test_evecs = model.solve(parameters)
        if abs(evals[0]) < 1e-12:
            errors.append(np.abs(evals[0] - test_evals[0]))
        else:
            errors.append(np.abs(evals[0] - test_evals[0]) / np.abs(evals[0]))

    return errors

MODEL_1 = "AIM"
N_1 = 4
PARAM_BOUNDS_1 = ((-5.0, 5.0), (-5.0, 5.0))
SELECTED_PARAMS_1 = ("vb", "eb")
INIT_THETA_1 = (PARAM_BOUNDS_1[0][0], PARAM_BOUNDS_1[1][0])

model_1 = SurrogateModel(
    MODEL_1,
    SELECTED_PARAMS_1,
    N_1,
    processes = 1
)

model_1.build_terms()

cfi_1_1 = VarianceCostFunction2(
    model_1,
    1e-9,
    INIT_THETA_1,
    PARAM_BOUNDS_1,
    1000,
    1
)

cfi_1_2 = ResidualCostFunction(
    model_1,
    1e-9,
    INIT_THETA_1,
    PARAM_BOUNDS_1,
    1000,
    1
)

model_1.iteration_costs = [[model_1.init_optimize(cfi_1_1, INIT_THETA_1)]]
model_1.set_optimal()

basis_sizes_1_1 = [model_1.basis.shape[1]]
max_errors_1_1 = [max(test_model(model_1))]

for i in range(model_1.max_it):
    model_1.log("Iteration {i + 1}")
    if model_1.optimize_step(cfi_1_1):
        break
    model_1.set_optimal()
    errors = test_model(model_1)
    max_error = max(errors)

    basis_sizes_1_1.append(model_1.basis.shape[1])
    max_errors_1_1.append(max_error)

model_1.reset()

model_1.iteration_costs = [[model_1.init_optimize(cfi_1_2, INIT_THETA_1)]]
model_1.set_optimal()

basis_sizes_1_2 = [model_1.basis.shape[1]]
max_errors_1_2 = [max(test_model(model_1))]

for i in range(model_1.max_it):
    model_1.log("Iteration {i + 1}")
    if model_1.optimize_step(cfi_1_2):
        break
    model_1.set_optimal()
    errors = test_model(model_1)
    max_error = max(errors)

    basis_sizes_1_2.append(model_1.basis.shape[1])
    max_errors_1_2.append(max_error)

plt.semilogy(basis_sizes_1_1, max_errors_1_1, label="Variance")
plt.semilogy(basis_sizes_1_2, max_errors_1_2, label="Residual")
plt.legend()
plt.show()
