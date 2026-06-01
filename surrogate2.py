import abc
import copy
import numpy as np
import scipy as sp

from pathos.multiprocessing import ProcessPool
from costfunction import *
from functools import partial
from pauli import *

class SurrogateModel:
    # Model Parameters
    model_name: str
    model_params: list[str]
    N: int
    pauli_strings: list[str]
    particle_selection: tuple[int, int] | int | None
    basis_ordering: str
    sparse: bool
    max_it: int
    processes: int

    # Output
    opt_overlap: np.ndarray | None
    opt_basis: np.ndarray | None
    opt_Hr_terms: dict[str, np.ndarray] | None

    # Internal State
    size: int
    H_terms: dict[str, np.ndarray] | None
    H_fulls: dict[str, np.ndarray] | None
    overlap: np.ndarray | None
    basis: np.ndarray | None
    Hr_terms: dict[str, np.ndarray] | None
    pp: ProcessPool | None

    def __init__(
        self,
        model_name: str,
        model_params: list[str],
        N: int,
        pauli_strings: list[str],
        particle_selection: tuple[int, int] | int | None = None,
        basis_ordering: str = "uudd",
        sparse: bool = True,
        max_it: int | None = None,
        processes: int = 1
    ):
        self.model_name = model_name
        self.model_params = model_params
        self.N = N
        self.pauli_strings = pauli_strings
        self.particle_selection = particle_selection
        self.basis_ordering = basis_ordering
        self.sparse = sparse
        self.processes = processes

        self.opt_overlap = None
        self.opt_basis = None
        self.opt_Hr_terms = None

        self.H_terms = None
        self.H_fulls = None
        self.overlap = None
        self.basis = None
        self.Hr_terms = None
        
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

        if type(max_it) == type(None):
            self.max_it = self.size
        else:
            self.max_it = max_it

        if self.processes == 1:
            self.pp = None
        elif self.processes > 1:
            self.pp = ProcessPool(nodes=processes)
        else:
            raise Exception(
                "Number of processes should be an integer greater than or equal"
                + " to one"
            )

    def build_terms(
        self,
        pregenerate_fulls: bool = False,
    ):
        self.H_terms = {}

        if self.processes == 1:
            for pauli_string in self.pauli_strings:
                self.H_terms[pauli_string] = gen_from_pauli_string(
                    pauli_string,
                    self.N,
                    self.particle_selection,
                    ordering=self.basis_ordering,
                    sparse=self.sparse
                )

        else:
            batch_size = int(np.ceil(len(self.pauli_strings) / self.processes))
            H_terms_list = list(self.pp.map(
                partial(
                    gen_from_pauli_string,
                    N=self.N,
                    particle_selection=self.particle_selection,
                    ordering=self.basis_ordering,
                    sparse=self.sparse
                ),
                self.pauli_strings,
                chunksize=batch_size
            ))
            self.H_terms = {}
            for pauli_string, H_term in zip(self.pauli_strings, H_terms_list):
                self.H_terms[pauli_string] = H_term

    def optimize(
        self,
        cfi: CostFunctionInterface,
        init_training_point: dict,
        max_condition: float = 1e10,
        svd_tolerance: float = 1e-8,
        sparse_proportion:  float = .20,
        degeneracy_truncation: int = 5,
    ):
        # build terms if they are not already built
        if(type(self.H_terms) == type(None)):
            self.build_terms()
        
        # cost for each iteration
        iteration_costs = []
        self.basis = np.zeros((self.size, 0), dtype=complex)
        self.overlap = np.zeros((0, 0), dtype=complex)

        H_full = self._build_H_full(init_training_point)
        if self.sparse:
            evals, evecs = sps.linalg.eigsh(
                H_full.real,
                k=int(self.size*sparse_proportion),
                which='SA'
            )
        else:
            evals, evecs = sp.linalg.eigh(H_full)

        init_vec = evecs[:, 0]

        basis_list = [init_vec]
        self.basis = np.array(basis_list).T
        self.overlap = (self.basis.conj().T @ self.basis).real
        self.Hr_terms = {}
        for pauli in self.H_terms.keys():
            self.Hr_terms[pauli] = (
                self.basis.conj().T @ self.H_terms[pauli] @ self.basis
            )

        # initial iteration preiteration
        cfi.preiteration()

        init_cost = cfi.cost_function(init_training_point)
        costs = np.array([[init_cost]])
        training_points = np.array([init_training_point])
        print(f"Training point: {init_training_point}")
        print(f"Cost: {init_cost}")

        # for constitency, this must be run as in some cases it changes the
        # state of the cost function interface, even if we don't use the output
        cfi.cost_selector(training_points, costs)

        iteration_costs.append(costs)

        for i in range(self.max_it):
            if(np.linalg.cond(self.overlap) > max_condition):
                print("Condition number is to large")
                break

            cfi.preiteration()

            training_points = cfi.gen_training_points()
            if len(training_points) == 0:
                # no more training points
                print("No more training points")
                break

            print(f"Generated {len(training_points)} training points")

            if self.processes == 1:
                costs = np.zeros(len(training_points), dtype=float)
                for j, training_point in enumerate(training_points):
                    costs[j] = cfi.cost_function(training_point)
            else:
                batch_size = int(np.ceil(len(training_points) / self.processes))
                costs = np.array(list(self.pp.map(
                    cfi.cost_function,
                    training_points,
                    chunksize = batch_size
                )))

            training_point_idxs = cfi.cost_selector(training_points, costs)
            next_training_points = training_points[training_point_idxs]
            basis_addition = None

            if len(next_training_points) == 0:
                print("No viable training points found")
            else:
                for training_point in next_training_points:
                    H_full = self._build_H_full(training_point)

                    if self.sparse:
                        evals, evecs = sps.linalg.eigsh(
                            H_full.real,
                            k=int(self.size*sparse_proportion),
                            which='SA'
                        )
                    else:
                        evals, evecs = sp.linalg.eigh(H_full)

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

                    if type(basis_addition) == type(None):
                        basis_addition = evecs[:, 0:degeneracy]
                    else:
                        basis_addition = np.append(
                            basis_addition,
                            evecs[:, 0:degeneracy],
                            axis = 1
                        )

                    if type(basis_addition) == type(None):
                        raise Exception("Failed to create any basis addition")

                # compress the basis
                projection = basis_addition - self.basis @ sp.linalg.solve(
                    self.overlap, self.basis.conj().T @ basis_addition
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
                if basis_reduced.shape[1] <= self.basis.shape[1]:
                    print(
                        "Warning: Basis did not increase in size after "
                        + "compression."
                    )
                else:
                    self.basis = copy.copy(basis_reduced)

                self.overlap = (self.basis.conj().T @ self.basis).real
                self.Hr_terms = {}
                for pauli in self.H_terms.keys():
                    self.Hr_terms[pauli] = (
                        self.basis.conj().T @ self.H_terms[pauli] @ self.basis
                    )

            iteration_costs.append(costs[training_point_idxs])
            print(f"Training point: {next_training_points}")
            print(f"Cost: {iteration_costs[-1]}")

            if cfi.check_termination(iteration_costs):
                print("Termination condition met")
                break

        self.opt_basis = self.basis
        self.opt_overlap = self.basis.conj().T @ self.basis
        self.opt_Hr_terms = self.Hr_terms

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
            or type(self.opt_overlap) == type(None)
        ):
            self.optimize()

        if type(self.opt_Hr_terms) == type(None):
            self.opt_Hr_terms = {}
            for pauli in self.H_terms.keys():
                self.opt_Hr_terms[pauli] = (
                    self.opt_basis.conj().T @ self.H_terms[pauli] @ self.opt_basis
                )

        Hr = np.zeros(
            (self.opt_basis.shape[1], self.opt_basis.shape[1]),
            dtype=complex
        )

        for pauli in self.opt_Hr_terms.keys():
            Hr += parameters[pauli] * self.opt_Hr_terms[pauli]

        evals, evecs = sp.linalg.eigh(Hr, self.opt_overlap)

        return evals, evecs

    def _build_H_full(
        self,
        training_point: dict,
    ) -> np.ndarray:
        """
        Builds the full Hamiltonian for a given training point

        Parameters
        ----------
        training_point : `dict`
            the training point to build the full Hamiltonian for

        Returns
        -------
        H_full : `np.ndarray`
            The matrix in the full Hilbert space
        """

        H_full = np.zeros((self.size, self.size), dtype=complex)
        for pauli in self.pauli_strings:
            H_full += (
                training_point[pauli] * self.H_terms[pauli]
            )

        return H_full
