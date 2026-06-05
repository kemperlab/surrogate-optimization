import abc
import copy
import datetime
import io
import numpy as np
import scipy as sp
import sys

from pathos.multiprocessing import ProcessPool
from costfunction import *
from functools import partial
from pauli import *

class SurrogateModel:
    # Model Parameters
    name: str
    params: list[str]
    N: int
    pauli_strings: list[str]
    particle_selection: tuple[int, int] | int | None
    basis_ordering: str
    sparse: bool
    max_it: int
    processes: int
    max_condition: float
    svd_tolerance: float
    sparse_proportion:  float
    degeneracy_truncation: int

    # Output
    opt_overlap: np.ndarray | None
    opt_basis: np.ndarray | None
    opt_Hr_terms: dict[str, np.ndarray] | None

    # Internal State
    size: int
    H_terms: dict[str, np.ndarray] | None
    H_fulls: dict[str, np.ndarray] | None
    overlap: np.ndarray | None
    basis_list: list | None
    basis: np.ndarray | None
    Hr_terms: dict[str, np.ndarray] | None
    iteration_costs: list
    pp: ProcessPool | None

    def __init__(
        self,
        name: str,
        params: list[str],
        N: int,
        particle_selection: tuple[int, int] | int | None = None,
        basis_ordering: str = "uudd",
        max_it: int | None = None,
        sparse: bool = True,
        max_condition: float = 1e10,
        svd_tolerance: float = 1e-8,
        sparse_proportion:  float = .20,
        degeneracy_truncation: int = 5,
        processes: int = 1,
        output_stream: io.IOBase = sys.stdout,
        error_stream: io.IOBase = sys.stderr,
    ):
        self.name = name
        self.params = params
        self.N = N
        self.pauli_strings = get_model_paulis(self.name, self.N)
        self.particle_selection = particle_selection
        self.basis_ordering = basis_ordering
        self.sparse = sparse
        self.max_condition = max_condition
        self.svd_tolerance = svd_tolerance
        self.sparse_proportion = sparse_proportion
        self.degeneracy_truncation = degeneracy_truncation
        self.processes = processes
        self.output_stream = output_stream
        self.error_stream = error_stream

        self.opt_overlap = None
        self.opt_basis = None
        self.opt_Hr_terms = None

        self.H_terms = None
        self.H_fulls = None
        self.overlap = None
        self.basis_list = None
        self.basis = None
        self.Hr_terms = None
        self.iteration_costs = None
        
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

    def reset(self):
        self.opt_overlap = None
        self.opt_basis = None
        self.opt_Hr_terms = None
        self.overlap = None
        self.basis_list = None
        self.basis = None
        self.Hr_terms = None
        self.iteration_costs = None

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

    def init_optimize(
        self,
        cfi: CostFunctionInterface,
        init_param_point: tuple
    ):
        init_training_point = param_to_paulis(
            init_param_point,
            self.params,
            self.name,
            self.N
        )
        # build terms if they are not already built
        if(type(self.H_terms) == type(None)):
            self.build_terms()
        
        # cost for each iteration
        self.basis = np.zeros((self.size, 0), dtype=complex)
        self.overlap = np.zeros((0, 0), dtype=complex)

        H_full = self.build_H_full(init_training_point)
        if self.sparse:
            evals, evecs = sps.linalg.eigsh(
                H_full.real,
                k=int(self.size * self.sparse_proportion),
                which='SA'
            )
        else:
            evals, evecs = sp.linalg.eigh(H_full)

        init_vec = evecs[:, 0]

        self.basis_list = [init_vec]
        self.basis = np.array(self.basis_list).T
        self.overlap = (self.basis.conj().T @ self.basis).real
        self.make_Hr_terms()

        # initial iteration preiteration
        cfi.preiteration()

        init_cost = cfi.cost_function(init_training_point)
        costs = np.array([[init_cost]])
        training_points = np.array([init_training_point])

        # for constitency, this must be run as in some cases it changes the
        # state of the cost function interface, even if we don't use the output
        cfi.cost_selector(training_points, costs)

        self.log(f"Training point: {init_training_point}\n")
        self.log(f"Cost: {init_cost}\n")

        return costs

    def optimize(
        self,
        cfi: CostFunctionInterface,
        init_param_point: dict,
    ):
        self.iteration_costs = []

        if type(self.basis) == type(None):
            # no basis yet, we need to initialize
            costs = self.init_optimize(
                cfi,
                init_param_point,
            )
            self.iteration_costs.append(costs)

        for i in range(self.max_it):
            if(np.linalg.cond(self.overlap) > self.max_condition):
                self.log("Condition number is to large\n")
                break

            cfi.preiteration()

            training_points = cfi.gen_training_points()
            if len(training_points) == 0:
                # no more training points
                self.log("No more training points\n")
                break

            self.log(f"Generated {len(training_points)} training points\n")

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
            next_costs = costs[training_point_idxs]

            if len(next_training_points) == 0:
                self.log("No viable training points found\n")
                # no training points found, no point in continuing
                break
            else:
                basis_addition = self.find_basis_addition(
                    next_costs,
                    next_training_points
                )

                self.compress_basis(basis_addition)
                self.make_Hr_terms()

            self.iteration_costs.append(next_costs)

            if cfi.check_termination(self.iteration_costs):
                self.log("Termination condition met\n")
                break

        self.set_optimal()

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

    def build_H_full(
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

    def find_basis_addition(
        self,
        next_costs: list,
        next_training_points: list
    ):
        basis_addition = None

        for cost, training_point in zip(
            next_costs,
            next_training_points
        ):
            H_full = self.build_H_full(training_point)

            if self.sparse:
                evals, evecs = sps.linalg.eigsh(
                    H_full.real,
                    k=int(self.size * self.sparse_proportion),
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
                if degeneracy >= self.degeneracy_truncation:
                    break

            if type(basis_addition) == type(None):
                basis_addition = evecs[:, 0:degeneracy]
            else:
                basis_addition = np.append(
                    basis_addition,
                    evecs[:, 0:degeneracy],
                    axis = 1
                )

            self.log(f"Training point: {training_point}\n")
            self.log(f"Cost: {cost}\n")

        return basis_addition

    def compress_basis(
        self,
        basis_addition: np.ndarray
    ):
        # compress the basis
        projection = basis_addition - self.basis @ sp.linalg.solve(
            self.overlap, self.basis.conj().T @ basis_addition
        )

        U, sigmas, Vdagger = np.linalg.svd(projection)
        compress_add = 0
        for s in sigmas:
            if s > self.svd_tolerance:
                compress_add += 1
            else:
                break

        for j in range(compress_add):
            self.basis_list += [U[:, j]]

        basis_reduced = np.array(self.basis_list).T
        if basis_reduced.shape[1] <= self.basis.shape[1]:
            self.log(
                "Warning: Basis did not increase in size after "
                + "compression.\n"
            )
        else:
            self.basis = copy.copy(basis_reduced)

        self.overlap = (self.basis.conj().T @ self.basis).real

    def make_Hr_terms(self):
        self.Hr_terms = {}
        for pauli in self.H_terms.keys():
            self.Hr_terms[pauli] = (
                self.basis.conj().T @ self.H_terms[pauli] @ self.basis
            )

    def set_optimal(self):
        self.opt_basis = self.basis
        self.opt_overlap = self.basis.conj().T @ self.basis
        self.make_Hr_terms()
        self.opt_Hr_terms = self.Hr_terms

    def log(
        self,
        text: str
    ):
        self.output_stream.write(str(datetime.datetime.now()) + ": " + text)

    def log_error(
        self,
        text: str
    ):
        self.error_stream.write(str(datetime.datetime.now()) + ": " + text)
