import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import datetime

from advisor import SurrogateAdvisor
from examples import ResidualCostFunction, VarianceCostFunction
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

model.build_terms()
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

model.optimize(cfi, INIT_THETA)
gep_adv = GEPAdvisor(model, PARAM_BOUNDS)
gep_adv.sobol_sample()

errors = []
all_ps = []

xs = np.linspace(-5.0, 5.0, 100)
ys = np.linspace(-5.0, 5.0, 100)
grid = np.zeros((100, 100), dtype=float)
grid2 = np.zeros((100, 100), dtype=float)
tgrid = np.empty((100, 100), dtype=object)

for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        theta = (x, y)
        test_point = model.theta_to_training_point(theta)
        tgrid[i, j] = test_point

print("Start")
time1 = datetime.datetime.now()
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        evals, evecs = model.solve(tgrid[i, j])
        grid[i, j] = evals[0]

time2 = datetime.datetime.now()
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        theta = (x, y)
        grid2[i, j] = gep_adv.predict(theta)
time3 = datetime.datetime.now()

print(time2 - time1)
print(time3 - time2)

print("End")

plt.imshow(grid)
plt.colorbar()
plt.show()
plt.imshow(grid2)
plt.colorbar()
plt.show()
plt.imshow(np.abs(grid - grid2))
plt.colorbar()
plt.show()
