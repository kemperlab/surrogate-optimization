import numpy as np
import scipy as sp

from costfunction import *

class EnergyConvergenceCostFunction(CostFunctionInterface[float]):
    energy_convergence: float

    def __init__(
        self,
        energy_convergence: float
    ):
        self.energy_convergence = energy_convergence

    def preiteration(
        self,
        Hr_terms,
        basis,
        overlap
    ):
        pass
    
    def cost_function(
        self,
        Hr_terms,
        basis,
        overlap,
        training_grid,
        grid_index
    ):
        Hr = np.zeros((basis.shape[1], basis.shape[1]), dtype=complex)

        for pauli in Hr_terms.keys():
            Hr += training_grid[grid_index][pauli] * Hr_terms[pauli]

        evals, evecs = sp.linalg.eigh(Hr, overlap)

        return evals[0]

    def cost_selector(
        self,
        costs: dict[int, float]
    ):
        return min(costs, key=costs.get)

    def check_termination(
        self,
        iteration_costs: list[T]
    ) -> bool:
        return (np.abs(
            (iteration_costs[-2] - iteration_costs[-1]))
            < self.energy_convergence
        )

class VarianceCostFunction(CostFunctionInterface[float]):
    H2_terms: dict
    training_grid2: list

    H2r_terms: dict

    def __init__(
        self,
        H_terms,
        training_grid,
        res2_threshold,
        degeneracy_truncation = 5,
    ):
        self.res2_threshold = res2_threshold
        self.degeneracy_truncation = degeneracy_truncation
        self.H2_terms = {}
        for h_i in H_terms.keys():
            for h_j in H_terms.keys():
                self.H2_terms[h_i + " * " + h_j] = (
                    H_terms[h_i] @ H_terms[h_j]
                )

        self.training_grid2 = []
        for mu in training_grid:
            bulk = {}
            for mu_i in mu.keys():
                for mu_j in mu.keys():
                    bulk[mu_i  + " * " +  mu_j] = (
                        mu[mu_i] * mu[mu_j]
                    )
            self.training_grid2.append(bulk)

    def preiteration(
        self,
        Hr_terms,
        basis,
        overlap
    ):
        self.H2r_terms = {}
        for pauli in self.H2_terms.keys():
            self.H2r_terms[pauli] = (
                basis.conj().T * self.H2_terms[pauli] @ basis
            )

    def cost_function(
        self,
        Hr_terms,
        basis,
        overlap,
        training_grid,
        grid_index
    ):
        Hr = np.zeros((basis.shape[1], basis.shape[1]), dtype=complex)
        H2r = np.zeros((basis.shape[1], basis.shape[1]), dtype=complex)

        for pauli in Hr_terms.keys():
            Hr += training_grid[grid_index][pauli] * Hr_terms[pauli]

        for pauli in self.H2r_terms.keys():
            H2r += (
                self.training_grid2[grid_index][pauli] * self.H2r_terms[pauli]
            )

        evals, evecs = sp.linalg.eigh(Hr, overlap)

        evals, evecs = sp.linalg.eigh(
            Hr,
            overlap
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

        # calculate residue
        res2 = 0
        for k in range(degeneracy):
            res2 += (
                evecs[:, k].conj().T
                @ (H2r - ((evals[k] * evals[k]) * overlap))
                @ evecs[:, k]
            )

        return res2

    def cost_selector(
        self,
        costs: dict[int, float]
    ):
        return max(costs, key=costs.get)

    def check_termination(
        self,
        iteration_costs: list[T]
    ) -> bool:
        return iteration_costs[-1] < self.res2_threshold

class ResidualCostFunction(CostFunctionInterface[float]):
    projection: np.ndarray
    H_terms: dict

    def __init__(
        self,
        H_terms
    ):
        self.H_terms = H_terms

    def preiteration(
        self,
        Hr_terms,
        basis,
        overlap
    ):
        self.projection = (
            np.eye(basis.shape[0], basis.shape[0])
            - basis @ np.linalg.inv(overlap) @ basis.conj().T
        )

    def cost_function(
        self,
        Hr_terms,
        basis,
        overlap,
        training_grid,
        grid_index
    ) -> T:
        H_full = np.zeros((basis.shape[0], basis.shape[0]), dtype=complex)
        for pauli in self.H_terms.keys():
            H_full += training_grid[grid_index][pauli] * self.H_terms[pauli]

        evals, evecs = sp.linalg.eigh(H_full)

        return np.linalg.norm(self.projection @ evecs[0])

    def cost_selector(
        self,
        costs: dict[int, T]
    ) -> int:
        return max(costs, key=costs.get)

    def check_termination(
        self,
        iteration_costs: list[T]
    ) -> bool:
        return False
