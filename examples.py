import numpy as np
import scipy as sp
import concurrent.futures

from costfunction import *
from pauli import *
from surrogate import SurrogateModel

def training_grid_generator(
    parameter_space,
    grid_size,
    model,
    processes,
    seed=None
):
    sobol_gen = sp.stats.qmc.Sobol(len(parameter_space),
        rng=np.random.default_rng(seed))
    # round up to the nearest power of two
    power = int(np.ceil(np.log2(grid_size)))
    points = np.array(sobol_gen.random_base2(power))
    for i, point in enumerate(points):
        for coord, param_range in enumerate(parameter_space):
            points[i][coord] = (
                (param_range[1] - param_range[0]) * points[i][coord]
                + param_range[0]
            )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=processes
    ) as pool:
        chuck_size = int(np.ceil(len(points) / processes))
        training_grid = list(pool.map(
            model.theta_to_training_point,
            points,
            chunksize=chuck_size
        ))
    return np.array(training_grid, dtype=dict)


class EnergyConvergenceCostFunction(
    CostFunctionInterface[float]
):
    model: SurrogateModel
    energy_convergence: float
    costs_storage: dict
    ec_energies: list

    def __init__(
        self,
        model: SurrogateModel,
        training_grid: np.ndarray[dict],
        energy_convergence: float
    ):
        self.model = model
        self.training_grid = training_grid
        self.energy_convergence = energy_convergence
        self.costs_storage = {}
        self.ec_energies = []
        self.not_chosen = list(range(len(self.training_grid)))

    def preiteration(self):
        if(self.model.overlap.shape == (1, 1)):
            self.ec_energies.append([])
            for training_point in self.training_grid:
                Hr = self.model.build_Hr(training_point)
                evals, evecs = sp.linalg.eigh(
                    Hr, self.model.overlap,
                    overwrite_a=True
                )

                self.ec_energies[-1].append(evals[0])

    def gen_training_points(self):
        return self.training_grid[self.not_chosen]
    
    def cost_function(
        self,
        training_point: dict
    ):
        Hr = self.model.build_Hr(training_point)
        evals, evecs = sp.linalg.eigh(
            Hr, self.model.overlap,
            overwrite_a=True
        )

        return evals[0]

    def cost_selector(
        self,
        training_points: np.ndarray,
        costs: np.ndarray
    ) -> dict | None:
        min_cost_idx = np.argmin(costs)
        training_point = training_points[min_cost_idx]
        for i, t in enumerate(self.training_grid[self.not_chosen]):
            if t == training_point:
                self.not_chosen.remove(self.not_chosen[i])
                return [min_cost_idx]
        
        return None

    def check_termination(
        self,
        iteration_costs: list[T]
    ) -> bool:
        self.ec_energies.append([])
        for training_point in self.training_grid:
            Hr = self.model.build_Hr(training_point)
            evals, evecs = sp.linalg.eigh(
                Hr, self.model.overlap,
                overwrite_a=True
            )
            self.ec_energies[-1].append(evals[0])

        for i in range(len(self.ec_energies[-1])):
            if(
                np.abs(self.ec_energies[-1][i] - self.ec_energies[-2][i])
                > self.energy_convergence
            ):
                return False

        return True

class VarianceCostFunction(CostFunctionInterface[float]):
    H2r_terms: dict
    terminate_before_basis_update = True

    def __init__(
        self,
        model,
        training_grid,
        var_threshold,
        degeneracy_truncation = 5,
    ):
        self.model = model
        self.training_grid = training_grid
        self.var_threshold = var_threshold
        self.degeneracy_truncation = degeneracy_truncation
        self.pauli2_strings = []
        self.pauli2_pairs = {}
        for h_i in self.model.pauli_strings:
            for h_j in self.model.pauli_strings:
                pauli2 = f"{h_i} @ {h_j}"
                self.pauli2_strings.append(pauli2)
                self.pauli2_pairs[pauli2] = (h_i, h_j)

        self.H2r_terms = {}
        self.not_chosen = list(range(len(self.training_grid)))

    def preiteration(self):
        # basis is fixed for the duration of this iteration's training-point
        # sweep, so build H2r terms once here instead of per training_point
        self.build_H2r_terms()

    def build_H2r_term(
        self,
        pauli_pair
    ):
        h_i = pauli_pair[0]
        h_j = pauli_pair[1]
        Y_i = self.model.get_H_term(h_i) @ self.model.basis
        Y_j = self.model.get_H_term(h_j) @ self.model.basis
        H2r_term = Y_i.conj().T @ Y_j
        H2r_term_T = H2r_term.conj().T

        if self.model.outdir:
            np.savez_compressed(
                self.model.outdir + f"/{h_i} @ {h_j}_r.npz",
                H2r_term
            )
            np.savez_compressed(
                self.model.outdir + f"/{h_j} @ {h_i}_r.npz",
                H2r_term_T
            )

        # when keep_on_disk is set, terms are written above and read
        # back one at a time by get_H2r_term, so they are not kept
        # resident in memory here
        if not self.model.keep_on_disk:
            self.H2r_terms[f"{h_i} @ {h_j}"] = H2r_term
            self.H2r_terms[f"{h_j} @ {h_i}"] = H2r_term_T

    def build_H2r_terms(self):
        # uses B^T H_i H_j B = (H_i B)^T (H_j B), so the full-space products
        # H_i H_j are never formed
        self.model.log("Building H2r terms...")
        paulis = self.model.pauli_strings
        if self.model.processes == 1:
            for i, h_i in enumerate(paulis):
                Y_i = self.model.get_H_term(h_i) @ self.model.basis
                for h_j in paulis[i:]:
                    Y_j = self.model.get_H_term(h_j) @ self.model.basis
                    H2r_term = Y_i.conj().T @ Y_j
                    H2r_term_T = H2r_term.conj().T

                    if self.model.outdir:
                        np.savez_compressed(
                            self.model.outdir + f"/{h_i} @ {h_j}_r.npz",
                            H2r_term
                        )
                        np.savez_compressed(
                            self.model.outdir + f"/{h_j} @ {h_i}_r.npz",
                            H2r_term_T
                        )

                    # when keep_on_disk is set, terms are written above and read
                    # back one at a time by get_H2r_term, so they are not kept
                    # resident in memory here
                    if not self.model.keep_on_disk:
                        self.H2r_terms[f"{h_i} @ {h_j}"] = H2r_term
                        self.H2r_terms[f"{h_j} @ {h_i}"] = H2r_term_T

        else:
            pauli_pairs = []
            for i, h_i in enumerate(paulis):
                for h_j in paulis[i:]:
                    pauli_pairs.append((h_i, h_j))

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.model.processes
            ) as pool:
                list(pool.map(
                    self.build_H2r_term,
                    pauli_pairs
                ))

        self.model.log("Built H2r terms")

    def get_H2r_term(
        self,
        pauli2
    ):
        if self.model.keep_on_disk:
            filename = self.model.outdir + f"/{pauli2}_r.npz"
            H2r_term = np.load(filename)["arr_0"]
            return H2r_term
        else:
            return self.H2r_terms[pauli2]

    def build_H2r(
        self,
        training_point2
    ):
        H2r = np.zeros((self.model.basis.shape[1], self.model.basis.shape[1]))

        for pauli2 in training_point2.keys():
            H2r += training_point2[pauli2] * self.get_H2r_term(pauli2)

        return H2r

    def gen_training_points(self):
        return self.training_grid[self.not_chosen]

    def cost_function(
        self,
        training_point: dict
    ):
        Hr = self.model.build_Hr(training_point)
        evals, evecs = sp.linalg.eigh(
            Hr, self.model.overlap,
            overwrite_a=True
        )

        # find degeneracy of the ground state
        degeneracy = 0
        eps = 1e-10 # for comparing floating points of GSE
        for e in evals:
            # absolute value is not needed here, e >= evals[0]
            if e - evals[0] < eps:
                degeneracy += 1
            else:
                break
            if degeneracy >= self.degeneracy_truncation:
                break

        # calculate residue using <psi|H^2|psi> = evec^T H2r evec with
        # psi = basis @ evec, so H2r (built once per basis in preiteration)
        # is reused across all training points instead of rebuilding
        # H_full and doing a full-space matvec for each one
        training_point2 = {
            pauli2: training_point[h_i] * training_point[h_j]
            for pauli2, (h_i, h_j) in self.pauli2_pairs.items()
        }
        H2r = self.build_H2r(training_point2)
        var = 0
        for k in range(degeneracy):
            evec = evecs[:, k]
            var += (
                evec.conj() @ (H2r @ evec)
                - (evals[k] * evals[k])
                * (evec.conj() @ (self.model.overlap @ evec))
            )

        return float(var.real)

    def cost_selector(
        self,
        training_points: np.ndarray,
        costs: np.ndarray
    ) -> int:
        max_cost_idx = np.argmax(costs)
        training_point = training_points[max_cost_idx]
        for i, t in enumerate(self.training_grid[self.not_chosen]):
            if t == training_point:
                self.not_chosen.remove(self.not_chosen[i])
                return [max_cost_idx]

        return None

    def check_termination(
        self,
        iteration_costs: list[T]
    ) -> bool:
        return iteration_costs[-1] < self.var_threshold

class ResidualCostFunction(CostFunctionInterface[float]):
    full_diag_per_point = True

    def __init__(
        self,
        model,
        res_threshold,
        param_space,
        num_points,
        points_per_iter,
        seed = None
    ):
        self.model = model
        self.res_threshold = res_threshold
        self.sobol_gen = sp.stats.qmc.Sobol(len(param_space),
            rng=np.random.default_rng(seed))
        # round up to the nearest power of two
        power = int(np.log2(num_points) + 0.5)
        self.points = np.array(self.sobol_gen.random_base2(power))
        for i, point in enumerate(self.points):
            for coord, param_range in enumerate(param_space):
                self.points[i][coord] = (
                    (param_range[1] - param_range[0]) * self.points[i][coord]
                    + param_range[0]
                )
        self.kd_tree = sp.spatial.KDTree(self.points)
        self.points_per_iter = points_per_iter
        self.search_points = None
 
    def preiteration(self):
        pass

    def gen_training_points(self):
        self.search_points = np.array(
            self.points[:self.points_per_iter]
        )
        training_points = []

        for point in self.search_points:
            training_points.append(self.model.theta_to_training_point(point))

        self.points = self.points[self.points_per_iter:]

        return np.array(training_points)

    def cost_function(
        self,
        training_point: dict
    ) -> T:
        evals, evecs = self.model.truth_solver(training_point)
        gs = evecs[:, 0]

        vec = (
            gs
            - self.model.basis @ np.linalg.solve(
                self.model.overlap, self.model.basis.conj().T @ gs
            )
        )

        return np.linalg.norm(vec)

    def cost_selector(
        self,
        training_points: np.ndarray,
        costs: np.ndarray
    ) -> int:
        if type(self.search_points) == type(None):
            # only for first iteration
            return 0
        max_cost_idxs = np.where(costs > self.res_threshold)[0]

        return max_cost_idxs

    def check_termination(
        self,
        iteration_costs: list[T]
    ) -> bool:
        if len(iteration_costs[-1]) == 0:
            return True
        else:
            return False

class VarianceCostFunction2(CostFunctionInterface[float]):
    H2r_terms: dict

    def __init__(
        self,
        model,
        var_threshold,
        param_space,
        num_points,
        points_per_iter,
        degeneracy_truncation = 5,
        seed = None
    ):
        self.model = model
        self.var_threshold = var_threshold
        self.degeneracy_truncation = degeneracy_truncation

        self.pauli2_strings = []
        self.pauli2_pairs = {}
        for h_i in self.model.pauli_strings:
            for h_j in self.model.pauli_strings:
                pauli2 = f"{h_i} @ {h_j}"
                self.pauli2_strings.append(pauli2)
                self.pauli2_pairs[pauli2] = (h_i, h_j)

        self.H2r_terms = {}

        self.sobol_gen = sp.stats.qmc.Sobol(len(param_space),
            rng=np.random.default_rng(seed))
        # round up to the nearest power of two
        power = int(np.log2(num_points) + 0.5)
        self.points = np.array(self.sobol_gen.random_base2(power))
        for i, point in enumerate(self.points):
            for coord, param_range in enumerate(param_space):
                self.points[i][coord] = (
                    (param_range[1] - param_range[0]) * self.points[i][coord]
                    + param_range[0]
                )
        self.kd_tree = sp.spatial.KDTree(self.points)
        self.points_per_iter = points_per_iter
        self.search_points = None

    def preiteration(self):
        # basis is fixed for the duration of this iteration's training-point
        # sweep, so build H2r terms once here instead of per training_point
        self.build_H2r_terms()

    def build_H2r_term(
        self,
        pauli_pair
    ):
        h_i = pauli_pair[0]
        h_j = pauli_pair[1]
        Y_i = self.model.get_H_term(h_i) @ self.model.basis
        Y_j = self.model.get_H_term(h_j) @ self.model.basis
        H2r_term = Y_i.conj().T @ Y_j
        H2r_term_T = H2r_term.conj().T

        if self.model.outdir:
            np.savez_compressed(
                self.model.outdir + f"/{h_i} @ {h_j}_r.npz",
                H2r_term
            )
            np.savez_compressed(
                self.model.outdir + f"/{h_j} @ {h_i}_r.npz",
                H2r_term_T
            )

        # when keep_on_disk is set, terms are written above and read
        # back one at a time by get_H2r_term, so they are not kept
        # resident in memory here
        if not self.model.keep_on_disk:
            self.H2r_terms[f"{h_i} @ {h_j}"] = H2r_term
            self.H2r_terms[f"{h_j} @ {h_i}"] = H2r_term_T

    def build_H2r_terms(self):
        # uses B^T H_i H_j B = (H_i B)^T (H_j B), so the full-space products
        # H_i H_j are never formed
        self.model.log("Building H2r terms...")
        paulis = self.model.pauli_strings
        if self.model.processes == 1:
            for i, h_i in enumerate(paulis):
                Y_i = self.model.get_H_term(h_i) @ self.model.basis
                for h_j in paulis[i:]:
                    Y_j = self.model.get_H_term(h_j) @ self.model.basis
                    H2r_term = Y_i.conj().T @ Y_j
                    H2r_term_T = H2r_term.conj().T

                    if self.model.outdir:
                        np.savez_compressed(
                            self.model.outdir + f"/{h_i} @ {h_j}_r.npz",
                            H2r_term
                        )
                        np.savez_compressed(
                            self.model.outdir + f"/{h_j} @ {h_i}_r.npz",
                            H2r_term_T
                        )

                    # when keep_on_disk is set, terms are written above and read
                    # back one at a time by get_H2r_term, so they are not kept
                    # resident in memory here
                    if not self.model.keep_on_disk:
                        self.H2r_terms[f"{h_i} @ {h_j}"] = H2r_term
                        self.H2r_terms[f"{h_j} @ {h_i}"] = H2r_term_T

        else:
            pauli_pairs = []
            for i, h_i in enumerate(paulis):
                for h_j in paulis[i:]:
                    pauli_pairs.append((h_i, h_j))

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.model.processes
            ) as pool:
                list(pool.map(
                    self.build_H2r_term,
                    pauli_pairs
                ))

        self.model.log("Built H2r terms")

    def get_H2r_term(
        self,
        pauli2
    ):
        if self.model.keep_on_disk:
            filename = self.model.outdir + f"/{pauli2}_r.npz"
            H2r_term = np.load(filename)["arr_0"]
            return H2r_term
        else:
            return self.H2r_terms[pauli2]

    def build_H2r(
        self,
        training_point2
    ):
        H2r = np.zeros((self.model.basis.shape[1], self.model.basis.shape[1]))

        for pauli2 in training_point2.keys():
            H2r += training_point2[pauli2] * self.get_H2r_term(pauli2)

        return H2r

    def gen_training_points(self):
        self.search_points = np.array(
            self.points[:self.points_per_iter]
        )
        training_points = []

        for point in self.search_points:
            training_points.append(self.model.theta_to_training_point(point))

        self.points = self.points[self.points_per_iter:]
        
        return np.array(training_points)

    def cost_function(
        self,
        training_point: dict
    ):
        Hr = self.model.build_Hr(training_point)
        evals, evecs = sp.linalg.eigh(
            Hr, self.model.overlap,
            overwrite_a=True
        )

        # find degeneracy of the ground state
        degeneracy = 0
        eps = 1e-10 # for comparing floating points of GSE
        for e in evals:
            # absolute value is not needed here, e >= evals[0]
            if e - evals[0] < eps:
                degeneracy += 1
            else:
                break
            if degeneracy >= self.degeneracy_truncation:
                break

        # calculate variance using <psi|H^2|psi> = evec^T H2r evec with
        # psi = basis @ evec, so H2r (built once per basis in preiteration)
        # is reused across all training points instead of rebuilding
        # H_full and doing a full-space matvec for each one
        training_point2 = {
            pauli2: training_point[h_i] * training_point[h_j]
            for pauli2, (h_i, h_j) in self.pauli2_pairs.items()
        }
        H2r = self.build_H2r(training_point2)
        var = 0
        for k in range(degeneracy):
            evec = evecs[:, k]
            var += (
                evec.conj() @ (H2r @ evec)
                - (evals[k] * evals[k])
                * (evec.conj() @ (self.model.overlap @ evec))
            )

        return float(var.real)

    def cost_selector(
        self,
        training_points: np.ndarray,
        costs: np.ndarray
    ) -> int:
        if type(self.search_points) == type(None):
            # only for first iteration
            return 0
        max_cost_idxs = np.where(costs > self.var_threshold)[0]

        return max_cost_idxs

    def check_termination(
        self,
        iteration_costs: list[T]
    ) -> bool:
        if len(iteration_costs[-1]) == 0:
            self.build_H2r_terms()
            return True
        else:
            return False

class NaiveMethod(CostFunctionInterface[None]):
    # set to True on subclasses whose cost_function diagonalizes the full
    # Hilbert-space Hamiltonian for every candidate point, so SurrogateModel
    # can count those toward n_full_diag
    full_diag_per_point: bool = False

    # set to True on subclasses whose check_termination only looks at the
    # cost values themselves (not self.model's basis/Hr state), so
    # SurrogateModel can check it before paying for find_basis_addition's
    # full-space diagonalization instead of after
    terminate_before_basis_update: bool = True

    def __init__(
        self,
        model,
        param_space,
        num_points,
        batch_size,
        degeneracy_truncation = 5,
        seed = None
    ):
        self.model = model
        self.batch_size = batch_size
        self.degeneracy_truncation = degeneracy_truncation
        self.search_points = None

        self.sobol_gen = sp.stats.qmc.Sobol(len(param_space),
            rng=np.random.default_rng(seed))

        # round up to the nearest power of two
        power = int(np.ceil(np.log2(num_points)))
        self.points = np.array(self.sobol_gen.random_base2(power))
        for i, point in enumerate(self.points):
            for coord, param_range in enumerate(param_space):
                self.points[i][coord] = (
                    (param_range[1] - param_range[0]) * self.points[i][coord]
                    + param_range[0]
                )

    def preiteration(self):
        pass
    
    def gen_training_points(self) -> np.ndarray:
        self.search_points = np.array(
            self.points[:self.batch_size]
        )
        training_points = []

        for point in self.search_points:
            training_points.append(self.model.theta_to_training_point(point))

        self.points = self.points[self.batch_size:]

        return np.array(training_points)


    def cost_function(
        self,
        training_point: dict
    ) -> None:
        return None

    def cost_selector(
        self,
        training_points: np.ndarray,
        costs: np.ndarray
    ) -> int:
        if type(self.search_points) == type(None):
            # only for first iteration
            return 0

        return np.arange(len(training_points))

    def check_termination(
        self,
        iteration_costs: list[T]
    ) -> bool:
        if self.model.compress_add == None:
            return False

        return (self.model.compress_add == 0)
