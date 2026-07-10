import copy
import numpy as np
import scipy as sp

from costfunction import *
from pauli import *
from surrogate import SurrogateModel

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
                Hr = np.zeros(
                    (self.model.basis.shape[1], self.model.basis.shape[1]),
                    dtype=complex
                )
                for pauli in self.model.Hr_terms.keys():
                    Hr += (
                        training_point[pauli] * self.model.Hr_terms[pauli]
                    )
                    evals, evecs = sp.linalg.eigh(Hr, self.model.overlap)
                self.ec_energies[-1].append(evals[0])

    def gen_training_points(self):
        return self.training_grid[self.not_chosen]
    
    def cost_function(
        self,
        training_point: dict
    ):
        Hr = np.zeros(
            (self.model.basis.shape[1], self.model.basis.shape[1]),
            dtype=complex
        )

        for pauli in self.model.Hr_terms.keys():
            Hr += (
                training_point[pauli] * self.model.Hr_terms[pauli]
            )

        evals, evecs = sp.linalg.eigh(Hr, self.model.overlap)

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
            Hr = np.zeros(
                (self.model.basis.shape[1], self.model.basis.shape[1]),
                dtype=complex
            )
            for pauli in self.model.Hr_terms.keys():
                Hr += (
                    training_point[pauli] * self.model.Hr_terms[pauli]
                )
                evals, evecs = sp.linalg.eigh(Hr, self.model.overlap)
            self.ec_energies[-1].append(evals[0])

        for i in range(len(self.ec_energies[-1])):
            if(
                np.abs(self.ec_energies[-1][i] - self.ec_energies[-2][i])
                > self.energy_convergence
            ):
                return False

        return True

class VarianceCostFunction(CostFunctionInterface[float]):
    H2_terms: dict
    training_grid2: list

    H2r_terms: dict

    def __init__(
        self,
        model,
        training_grid,
        res2_threshold,
        degeneracy_truncation = 5,
    ):
        self.model = model
        self.training_grid = training_grid
        self.res2_threshold = res2_threshold
        self.degeneracy_truncation = degeneracy_truncation
        self.pauli2_strings = []
        for h_i in self.model.pauli_strings:
            for h_j in self.model.pauli_strings:
                self.pauli2_strings.append(f"{h_i} @ {h_j}")

        self.H2_terms = {}
        needed_terms = []
        if self.model.save_folder:
            for h_ij in self.pauli2_strings:
                filename = self.model.save_folder + f"/{h_ij}.npz"
                try:
                    if self.model.keep_on_disk:
                        if not os.path.exists(filename):
                            self.model.log(f"Failed to find {filename}")
                            needed_terms.append(h_ij)
                        else:
                            self.model.log(f"Found {filename} on disk")
                    elif self.sparse:
                        self.H_terms[h_ij] = sp.sparse.load_npz(
                            filename)
                        self.model.log(f"Retrieved {filename}")
                    else:
                        self.H_terms[h_ij] = np.load(filename)["arr_0"]
                        self.model.log(f"Retrieved {filename}")
                except:
                    needed_terms.append(h_ij)
                    self.model.log(f"Failed to retrieve {filename}")
        else:
            needed_terms = copy.copy(self.pauli2_strings)
        for h_ij in needed_terms:
            hs = h_ij.split(" @ ")
            H2_term = (
                self.model.get_H_term(hs[0]) @ self.model.get_H_term(hs[1])
            )
            if self.model.save_folder:
                filename = self.model.save_folder + f"/{h_ij}.npz"
                if self.model.sparse:
                    sp.sparse.save_npz(filename, H2_term)
                else:
                    np.savez_compressed(filename, H2_term)
                self.model.log(f"Saving term {filename}")
            if not self.model.keep_on_disk:
                self.H2_terms[h_ij] = H2_term


        self.not_chosen = list(range(len(self.training_grid)))

    def get_H2_term(
        self,
        pauli2_string
    ):
        if self.model.keep_on_disk:
            filename = self.model.save_folder + f"/{pauli2_string}.npz"
            if self.model.sparse:
                H2_term = sp.sparse.load_npz(filename)
            else:
                H2_term = np.load(filename)["arr_0"]
            return H2_term
        else:
            return self.H2_terms[pauli2_string]

    def preiteration(self):
        self.H2r_terms = {}
        for pauli2 in self.pauli2_strings:
            self.H2r_terms[pauli2] = (
                self.model.basis.conj().T @ self.get_H2_term(pauli2) @
                self.model.basis
            )

    def gen_training_points(self):
        return self.training_grid[self.not_chosen]

    def cost_function(
        self,
        training_point: dict
    ):
        Hr = self.model.build_Hr(training_point)
        H2r = np.zeros(
            (self.model.basis.shape[1], self.model.basis.shape[1]),
            dtype=float
        )

        training_point2 = {}
        for mu_i in training_point.keys():
            for mu_j in training_point.keys():
                training_point2[mu_i  + " @ " +  mu_j] = (
                    training_point[mu_i] * training_point[mu_j]
                )

        for pauli in self.H2r_terms.keys():
            H2r += (
                training_point2[pauli] * self.H2r_terms[pauli]
            )

        evals, evecs = sp.linalg.eigh(Hr, self.model.overlap)

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

        # calculate residue
        res2 = 0
        for k in range(degeneracy):
            res2 += (
                evecs[:, k].conj().T
                @ (H2r - ((evals[k] * evals[k]) * self.model.overlap))
                @ evecs[:, k]
            )

        return float(res2.real)

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
        return iteration_costs[-1] < self.res2_threshold

class ResidualCostFunction(CostFunctionInterface[float]):
    def __init__(
        self,
        model,
        res_threshold,
        init_param_point,
        param_space,
        num_points,
        points_per_iter,
        num_to_exclude = 2, # exclude itself and its nearest neighbor
        wait_at_least = None,
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
        self.num_to_exclude = num_to_exclude
        self.chosen = np.array([init_param_point])
        self.search_points = None
        self.basis_size_history = []
        self.wait_at_least = int(self.model.size / 4 + 0.5)
 
    def preiteration(self):
        pass

    def gen_training_points(self):
        """
        _, bad_indices = self.kd_tree.query(self.chosen, k=self.num_to_exclude)
        self.search_points = (
            np.delete(self.points, bad_indices, axis = 0)[:self.points_per_iter]
        )
        """
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
        H_full = self.model.build_H_full(training_point)

        if self.model.sparse:
            evals, evecs = sps.linalg.eigsh(
                H_full.real,
                k=int(self.model.size * self.model.sparse_proportion)+1,
                which='SA'
            )
        else:
            evals, evecs = sp.linalg.eigh(H_full)

        gs = evecs[:, 0]

        vec = (
            gs
            - self.model.basis @ np.linalg.solve(
                self.model.overlap, self.model.basis.conj().T
            ) @ gs
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
        self.chosen = np.append(
            self.chosen,
            self.search_points[max_cost_idxs],
            axis = 0
        )

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
    H2_terms: dict
    training_grid2: list

    H2r_terms: dict

    def __init__(
        self,
        model,
        var_threshold,
        init_param_point,
        param_space,
        num_points,
        points_per_iter,
        num_to_exclude = 2, # exclude itself and its nearest neighbor
        wait_at_least = None,
        degeneracy_truncation = 5,
        seed = None
    ):
        self.model = model
        self.var_threshold = var_threshold
        self.degeneracy_truncation = degeneracy_truncation

        self.pauli2_strings = []
        for h_i in self.model.pauli_strings:
            for h_j in self.model.pauli_strings:
                self.pauli2_strings.append(f"{h_i} @ {h_j}")

        self.H2_terms = {}
        needed_terms = []
        if self.model.save_folder:
            for h_ij in self.pauli2_strings:
                filename = self.model.save_folder + f"/{h_ij}.npz"
                try:
                    if self.model.keep_on_disk:
                        if not os.path.exists(filename):
                            self.model.log(f"Failed to find {filename}")
                            needed_terms.append(h_ij)
                        else:
                            self.model.log(f"Found {filename} on disk")
                    elif self.sparse:
                        self.H_terms[h_ij] = sp.sparse.load_npz(
                            filename)
                        self.model.log(f"Retrieved {filename}")
                    else:
                        self.H_terms[h_ij] = np.load(filename)["arr_0"]
                        self.model.log(f"Retrieved {filename}")
                except:
                    needed_terms.append(h_ij)
                    self.model.log(f"Failed to retrieve {filename}")
        else:
            needed_terms = copy.copy(self.pauli2_strings)
        for h_ij in needed_terms:
            hs = h_ij.split(" @ ")
            H2_term = (
                self.model.get_H_term(hs[0]) @ self.model.get_H_term(hs[1])
            )
            if self.model.save_folder:
                filename = self.model.save_folder + f"/{h_ij}.npz"
                if self.model.sparse:
                    sp.sparse.save_npz(filename, H2_term)
                else:
                    np.savez_compressed(filename, H2_term)
                self.model.log(f"Saving term {filename}")
            if not self.model.keep_on_disk:
                self.H2_terms[h_ij] = H2_term

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
        self.num_to_exclude = num_to_exclude
        self.chosen = np.array([init_param_point])
        self.search_points = None
        self.basis_size_history = []
        self.wait_at_least = int(self.model.size / 4 + 0.5)

    def get_H2_term(
        self,
        pauli2_string
    ):
        if self.model.keep_on_disk:
            filename = self.model.save_folder + f"/{pauli2_string}.npz"
            if self.model.sparse:
                H2_term = sp.sparse.load_npz(filename)
            else:
                H2_term = np.load(filename)["arr_0"]
            return H2_term
        else:
            return self.H2_terms[pauli2_string]

    def preiteration(self):
        self.H2r_terms = {}
        for pauli2 in self.pauli2_strings:
            self.H2r_terms[pauli2] = (
                self.model.basis.conj().T @ self.get_H2_term(pauli2) @
                self.model.basis
            )

    def gen_training_points(self):
        """
        _, bad_indices = self.kd_tree.query(self.chosen, k=self.num_to_exclude)
        self.search_points = (
            np.delete(self.points, bad_indices, axis = 0)[:self.points_per_iter]
        )
        """
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
        H2r = np.zeros(
            (self.model.basis.shape[1], self.model.basis.shape[1]),
            dtype=complex
        )

        training_point2 = {}
        for mu_i in training_point.keys():
            for mu_j in training_point.keys():
                training_point2[mu_i  + " @ " +  mu_j] = (
                    training_point[mu_i] * training_point[mu_j]
                )

        for pauli in self.H2r_terms.keys():
            H2r += (
                training_point2[pauli] * self.H2r_terms[pauli]
            )

        evals, evecs = sp.linalg.eigh(Hr, self.model.overlap)

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

        # calculate variance
        var = 0
        for k in range(degeneracy):
            var += (
                evecs[:, k].conj().T
                @ (H2r - ((evals[k] * evals[k]) * self.model.overlap))
                @ evecs[:, k]
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
        self.chosen = np.append(
            self.chosen,
            self.search_points[max_cost_idxs],
            axis = 0
        )

        return max_cost_idxs

    def check_termination(
        self,
        iteration_costs: list[T]
    ) -> bool:
        if len(iteration_costs[-1]) == 0:
            return True
        else:
            return False
