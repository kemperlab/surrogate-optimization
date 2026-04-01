import abc
import copy
import numpy as np
import scipy as sp

from costfunction import *
from functools import partial
from pauli import *
from typing import Callable

def create_variance_closure(H_terms, training_grid):
    degeneracy_truncation = 5
    H2_terms = {}
    for h_i in H_terms.keys():
        for h_j in H_terms.keys():
            H2_terms[h_i + " * " + h_j] = (
                H_terms[h_i] @ H_terms[h_j]
            )

    training_grid2 = []
    for mu in training_grid:
        bulk = {}
        for mu_i in mu.keys():
            for mu_j in mu.keys():
                bulk[mu_i  + " * " +  mu_j] = (
                    mu[mu_i] * mu[mu_j]
                )
        training_grid2.append(bulk)

    def variance_cf(Hr_terms, basis, overlap, training_point, training_idx):
        Hr = np.zeros((basis.shape[1], basis.shape[1]), dtype=complex)
        H2r = np.zeros((basis.shape[1], basis.shape[1]), dtype=complex)

        for pauli in Hr_terms.keys():
            Hr += training_point[pauli] * Hr_terms[pauli]
        for pauli in H2_terms.keys():
            H2r += (
                training_grid2[training_idx][pauli] * basis.conj().T
                @ H2_terms[pauli] @ basis
            )

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
            if degeneracy >= degeneracy_truncation:
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
    return variance_cf


def energy_cf(Hr_terms, basis, overlap, training_point, training_idx):
    Hr = np.zeros((basis.shape[1], basis.shape[1]), dtype=complex)
    for pauli in Hr_terms.keys():
        Hr += training_point[pauli] * Hr_terms[pauli]

    evals, evecs = sp.linalg.eigh(Hr, overlap)

    return evals[0];

class SurrogateModel:
    # Model Parameters
    model_name: str
    N: int
    pauli_strings: list[str]
    particle_selection: tuple[int, int] | int | None
    basis_ordering: str
    sparse: bool

    # Output
    overlap: np.ndarray | None
    opt_basis: np.ndarray | None
    reduced_terms: dict[str, np.ndarray] | None

    # Internal State
    H_terms: dict[str, np.ndarray] | None

    def __init__(
        self,
        model_name: str,
        N: int,
        pauli_strings: list[str],
        training_grid: np.ndarray,
        particle_selection: tuple[int, int] | int | None = None,
        basis_ordering: str = "uudd",
        sparse: bool = True
    ):
        self.model_name = model_name
        self.N = N
        self.pauli_strings = pauli_strings
        self.training_grid = training_grid
        self.particle_selection = particle_selection
        self.basis_ordering = basis_ordering
        self.sparse = sparse

        self.overlap = None
        self.opt_basis = None
        self.reduced_terms = None

        self.H_terms = None
        self.H_fulls = None
        
        if type(self.particle_selection) == type(None):
            self.size = 2**N
        elif type(self.particle_selection) == int:
            self.size = comb(N, self.particle_selection)
        elif type(self.particle_selection) == tuple:
            self.size = (
                comb(N // 2, self.particle_selection[0])
                * comb(N // 2, self.particle_selection[1])
            )
        else:
            raise Exception(
                "Particle selection should be an int, a tuple, or None"
            )

    def build_terms(
        self,
        pregenerate_fulls: bool = False,
        processes: int = 1
    ):
        self.H_terms = {}

        if processes == 1:
            for pauli_string in self.pauli_strings:
                self.H_terms[pauli_string] = gen_from_pauli_string(
                    self.N,
                    pauli_string,
                    self.particle_selection,
                    ordering=self.basis_ordering,
                    sparse=self.sparse
                )

        else:
            ppe = ProcessPoolExecutor(processes)
            batch_size = int(len(self.pauli_strings) / processes + 1)
            pauli_string_batches = [
                self.pauli_strings[j:j + batch_size]
                for j in range(0, len(self.pauli_strings), batch_size)
            ]
            H_terms_list = list(ppe.map(
                partial(
                    gen_from_pauli_string_batch,
                    N=self.N,
                    particle_selection=self.particle_selection,
                    ordering=self.basis_ordering,
                    sparse=self.sparse
                ),
                pauli_string_batches
            ))
            self.H_terms = {}
            for H_terms_element in H_terms_list:
                self.H_terms.update(H_terms_element)

    def optimize(
        self,
        cfi: CostFunctionInterface,
        max_condition: float = 1e10,
        svd_tolerance: float = 1e-8,
        sparse_proportion:  float = .20,
        degeneracy_truncation: int = 5,
        init_vec: np.ndarray | None = None,
        processes: int = 1
    ):
        if processes > 1:
            ppe = ProcessPoolExecutor(processes)

        # build terms if they are not already built
        if(
            type(self.H_terms) == type(None)
        ):
            self.build_terms()
        
        # list of indices into the training grid
        chosen = []

        # chosen cost for each iteration
        iteration_costs = []

        # list of remaining indices into the training grid
        not_chosen = list(range(len(self.training_grid)))

        # initial vector is not provided, so we choose from the training grid
        if type(init_vec) == type(None):
            H_full = self._build_H_full(0)
            if self.sparse:
                evals, evecs = sps.linalg.eigsh(
                    H_full.real,
                    k=int(self.size*sparse_proportion),
                    which='SA'
                )
            else:
                evals, evecs = sp.linalg.eigh(H_full)
            init_vec = evecs[:, 0]
            chosen.append(0)
            not_chosen.remove(0)

        basis_list = [init_vec]
        basis = np.array(basis_list).T
        overlap = (basis.conj().T @ basis).real
        Hr_terms = {}
        for pauli in self.H_terms.keys():
            Hr_terms[pauli] = basis.conj().T @ self.H_terms[pauli] @ basis
        cfi.preiteration(Hr_terms, basis, overlap)
        iteration_costs.append(
            cfi.cost_function(
                Hr_terms,
                basis,
                overlap,
                self.training_grid,
                0
            )
        )

        num_iterations = len(not_chosen)
        for i in range(num_iterations):
            overlap = (basis.conj().T @ basis).real

            print(np.linalg.cond(overlap))
            if(np.linalg.cond(overlap) > max_condition):
                break

            cfi.preiteration(Hr_terms, basis, overlap)

            next_choice = None
            costs = {}

            Hr_terms = {}
            for pauli in self.H_terms.keys():
                Hr_terms[pauli] = basis.conj().T @ self.H_terms[pauli] @ basis
            for j in not_chosen:
                costs[j] = cfi.cost_function(
                    Hr_terms, basis, overlap, self.training_grid, j
                )
            next_choice = cfi.cost_selector(costs)

            if type(self.H_fulls) == type(None):
                chosen_H_full = self._build_H_full(next_choice)

            if self.sparse:
                evals, evecs = sps.linalg.eigsh(
                    chosen_H_full.real,
                    k=int(self.size*sparse_proportion),
                    which='SA'
                )
            else:
                evals, evecs = sp.linalg.eigh(chosen_H_full)

            # find degeneracy of the ground state
            eps = 1e-10 # for comparing floating points of GSE
            degeneracy = 0
            for e in evals:
                if e - evals[0] < eps:
                    degeneracy += 1
                else:
                    break
                if degeneracy >= degeneracy_truncation:
                    break

            basis_addition = evecs[:, 0:degeneracy]

            # compress the basis
            projection = basis_addition - basis @ sp.linalg.solve(
                overlap, basis.conj().T @ basis_addition
            )

            U, sigmas, Vdagger = np.linalg.svd(projection)
            compress_add = 0
            for s in sigmas:
                if s > svd_tolerance:
                    compress_add += 1
                else:
                    break

            for j in range(compress_add):
                basis_list += [U[:, j]]

            basis_reduced = np.array(basis_list).T
            if basis_reduced.shape[1] <= basis.shape[1]:
                print(
                    "Warning: Basis did not increase in size after compression."
                )
                break
            else:
                basis = copy.copy(basis_reduced)

            not_chosen.remove(next_choice)
            chosen.append(next_choice)
            iteration_costs.append(costs[next_choice])

            if cfi.check_termination(iteration_costs):
                break

        self.opt_basis = basis
        self.overlap = basis.conj().T @ basis
        self.reduced_terms = None

        return self.opt_basis

    def solve(
        self,
        parameters: list[complex],
    ) -> complex:
        """
        Approximate the eigenvalues and eigenvectors for a given set of
        parameters

        Parameters
        ----------
        parameters : `list[complex]`
            The parameters to approximate eigenvalues and eigenvectors for
        
        Returns
        -------
        evals : `np.ndarray`
            A list of the eigenvalues
        evecs : `np.ndarray`
            A matrix of the eigenvectors, with each column representing each
            eigenvector
        """
        if (
            type(self.opt_basis) == type(None)
            or type(self.overlap) == type(None)
        ):
            self.optimize()

        if type(self.reduced_terms) == type(None):
            self.reduced_terms = {}
            for pauli in self.H_terms.keys():
                self.reduced_terms[pauli] = (
                    self.opt_basis.conj().T @ self.H_terms[pauli] @ self.opt_basis
                )

        Hr = np.zeros(
            (self.opt_basis.shape[1], self.opt_basis.shape[1]),
            dtype=complex
        )

        for pauli in self.reduced_terms.keys():
            Hr += parameters[pauli] * self.reduced_terms[pauli]

        evals, evecs = sp.linalg.eigh(Hr, self.overlap)

        return evals, evecs

    def _build_H_full(
        self,
        parameter_idx: int
    ) -> np.ndarray:
        """
        Builds the full Hamiltonian for a given paremeter index

        Parameters
        ----------
        parameter_idx : `int`
            the parameter index to build the full Hamiltonian for

        Returns
        -------
        H_full : `np.ndarray`
            The matrix in the full Hilbert space
        """

        H_full = np.zeros((self.size, self.size), dtype=complex)
        for pauli in self.pauli_strings:
            H_full += (
                self.training_grid[parameter_idx][pauli] * self.H_terms[pauli]
            )

        return H_full
