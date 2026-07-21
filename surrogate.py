import abc
import concurrent.futures
import copy
import datetime
import io
import numpy as np
import scipy as sp
import sys

from costfunction import *
from functools import partial
from pauli import *

def gen_Hr_and_save(
    pauli_string,
    model
):
    Hr_term = model.build_Hr_term(pauli_string)

    if pauli_string == "":
        filename = model.save_folder + "/I_r.npz"
    else:
        filename = model.save_folder + f"/{pauli_string}_r.npz"
    np.savez_compressed(filename, Hr_term)

def gen_and_save(
    pauli_string,
    N,
    particle_selection,
    ordering,
    sparse,
    save_folder,
):
    H_term = gen_from_pauli_string(
        pauli_string,
        N,
        particle_selection,
        ordering,
        sparse
    )

    if pauli_string == "":
        filename = save_folder + "/I.npz"
    else:
        filename = save_folder + f"/{pauli_string}.npz"
    if sparse:
        sp.sparse.save_npz(filename, H_term)
    else:
        np.savez_compressed(filename, H_term)

class SurrogateModel:
    # Model Parameters
    name: str
    params: list[str]
    selected_params: tuple[str]
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
    overlap: np.ndarray | None
    basis: np.ndarray | None
    Hr_terms: dict[str, np.ndarray] | None
    iteration_costs: list
    save_folder: str | None
    keep_on_disk: bool

    def __init__(
        self,
        name: str,
        selected_params: tuple[str],
        N: int,
        particle_selection: tuple[int, int] | int | None = None,
        max_it: int | None = None,
        sparse: bool = True,
        max_condition: float = 1e10,
        svd_tolerance: float = 1e-8,
        sparse_proportion:  float = .20,
        degeneracy_truncation: int = 5,
        processes: int = 1,
        save_folder: str | None = ".",
        keep_on_disk = False
    ):
        self.name = name
        self.params = get_model_parameters(self.name)
        self.selected_params = selected_params
        self.N_spin = N
        self.N = get_model_N(self.name, N)
        self.pauli_strings = get_model_paulis(self.name, self.N_spin)
        self.particle_selection = particle_selection
        self.sparse = sparse
        self.max_condition = max_condition
        self.svd_tolerance = svd_tolerance
        self.sparse_proportion = sparse_proportion
        self.degeneracy_truncation = degeneracy_truncation
        self.processes = processes

        self.opt_overlap = None
        self.opt_basis = None
        self.opt_Hr_terms = None

        self.H_terms = None
        self.overlap = None
        self.basis = None
        self.Hr_terms = None
        self.iteration_costs = None
        self.basis_growth = None

        if keep_on_disk and not save_folder:
            raise Exception(
                "A save folder must to specfied to keep terms on disk"
            )
        else:
            self.keep_on_disk = keep_on_disk

        if save_folder:
            self.save_folder = save_folder + f"/{self.name}_{self.N_spin}"
            
            if self.sparse:
                self.save_folder += "_sparse"

            if not os.path.isdir(self.save_folder):
                os.mkdir(self.save_folder)
        else:
            self.save_folder = None
        
        if type(self.particle_selection) == type(None):
            self.size = 2**self.N
        elif type(self.particle_selection) == int:
            self.size = comb(self.N, self.particle_selection)
        elif type(self.particle_selection) == tuple:
            self.size = (
                comb(self.N // 2, self.particle_selection[0])
                * comb(self.N // 2, self.particle_selection[1])
            )
        else:
            raise Exception(
                "Particle selection should be an int, a tuple, or None"
            )

        if type(max_it) == type(None):
            self.max_it = self.size
        else:
            self.max_it = max_it

        if self.processes < 1:
            raise Exception(
                "Number of processes should be an integer greater than or equal"
                + " to one"
            )

        if self.name == "AIM" or self.name == "fermi_hubbard":
            self.basis_ordering = "udud"
        else:
            self.basis_ordering = "uudd"

        self.log(f"Initializing surrogate model for {self.name}")
        self.log(f"Paulis: {self.pauli_strings}")
        self.log(f"N: {self.N_spin}")
        self.log(f"Selected Parameters: {self.selected_params}")
        self.log(f"Particle Selection: {self.particle_selection}")
        self.log(f"Hilbert Space Size: {self.size}")

    def reset(self):
        self.log("Resetting model...")
        self.opt_overlap = None
        self.opt_basis = None
        self.opt_Hr_terms = None
        self.overlap = None
        self.basis = None
        self.Hr_terms = None
        self.iteration_costs = None
        self.basis_growth = None

    def build_terms(
        self
    ):
        self.H_terms = {}

        if self.save_folder:
            self.log(f"Retrieving terms from {self.save_folder}")

            if self.keep_on_disk:
                self.log("Terms will be kept on disk")

            needed_terms = []

            for pauli_string in self.pauli_strings:
                if pauli_string == "":
                    filename = self.save_folder + "/I.npz"
                else:
                    filename = self.save_folder + f"/{pauli_string}.npz"
                try:
                    if self.keep_on_disk:
                        if not os.path.exists(filename):
                            #self.log(f"Failed to locate {filename}")
                            needed_terms.append(pauli_string)
                        #else:
                            #self.log(f"Found {filename} on disk")
                    elif self.sparse:
                        self.H_terms[pauli_string] = sp.sparse.load_npz(
                            filename)
                        #self.log(f"Retrieved {filename}")
                    else:
                        self.H_terms[pauli_string] = np.load(filename)["arr_0"]
                        #self.log(f"Retrieved {filename}")
                except:
                    needed_terms.append(pauli_string)
                    #self.log(f"Failed to retrieved {filename}")
        else:
            needed_terms = copy.copy(self.pauli_strings)

        if len(needed_terms) != 0:
            self.log(f"Building terms: {needed_terms}")

        if self.processes == 1:
            for pauli_string in needed_terms:
                H_term = gen_from_pauli_string(
                    pauli_string,
                    self.N,
                    self.particle_selection,
                    ordering=self.basis_ordering,
                    sparse=self.sparse
                )
                if self.keep_on_disk:
                    if pauli_string == "":
                        filename = self.save_folder + "/I.npz"
                    else:
                        filename = self.save_folder + f"/{pauli_string}.npz"
                    if self.sparse:
                        sp.sparse.save_npz(filename, H_term)
                    else:
                        np.savez_compressed(filename, H_term)
                else:
                    self.H_terms[pauli_string] = H_term

        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.processes
            ) as pool:
                batch_size = int(np.ceil(len(needed_terms) / self.processes))

                if self.keep_on_disk and batch_size != 0:
                    list(pool.map(
                        partial(
                            gen_and_save,
                            N=self.N,
                            particle_selection=self.particle_selection,
                            ordering=self.basis_ordering,
                            sparse=self.sparse,
                            save_folder=self.save_folder
                        ),
                        needed_terms,
                        chunksize=batch_size
                    ))
                elif batch_size != 0:
                    H_terms_list = list(pool.map(
                        partial(
                            gen_from_pauli_string,
                            N=self.N,
                            particle_selection=self.particle_selection,
                            ordering=self.basis_ordering,
                            sparse=self.sparse
                        ),
                        needed_terms,
                        chunksize=batch_size
                    ))
                    for pauli_string, H_term in zip(needed_terms, H_terms_list):
                        self.H_terms[pauli_string] = H_term

        self.log("Built terms")

        if (
            self.save_folder and len(needed_terms) != 0
            and not self.keep_on_disk
        ):
            for pauli_string in needed_terms:
                if pauli_string == "":
                    filename = self.save_folder + "/I.npz"
                else:
                    filename = self.save_folder + f"/{pauli_string}.npz"
                if self.sparse:
                    sp.sparse.save_npz(filename, self.H_terms[pauli_string])
                elif not self.keep_on_disk:
                    np.savez_compressed(filename, self.H_terms[pauli_string])
            self.log(f"Saved {needed_terms}")
        elif self.keep_on_disk and len(needed_terms) != 0:
            self.log(f"Saved {needed_terms}")

    def init_optimize(
        self,
        cfi: CostFunctionInterface,
        init_theta: tuple
    ):
        self.log(f"Initializing optimization with parameter point {init_theta}")
        init_training_point = self.theta_to_training_point(init_theta)

        # build terms if they are not already built
        if(type(self.H_terms) == type(None)):
            self.build_terms()
        
        # cost for each iteration
        self.basis = np.zeros((self.size, 0), dtype=float)
        self.overlap = np.zeros((0, 0), dtype=float)
        
        self.log("Diagonalizing H...")

        H_full = self.build_H_full(init_training_point)
        v0 = np.ones(H_full.shape[0]) / np.sqrt(H_full.shape[0])
        if self.sparse:
            evals, evecs = sps.linalg.eigsh(
                H_full.real,
                k=min(int(self.size * self.sparse_proportion) + 1, 4),
                v0=v0,
                which='SA'
            )
        else:
            evals, evecs = sp.linalg.eigh(H_full)

        init_vec = evecs[:, 0]

        self.basis = init_vec.reshape(-1, 1)
        self.overlap = (self.basis.conj().T @ self.basis).real
        self.build_Hr_terms()
        self.basis_growth.append(1)


        # initial iteration preiteration
        self.log("Running preiteration...")
        cfi.preiteration()

        self.log("Calculating costs...")
        init_cost = cfi.cost_function(init_training_point)
        costs = np.array([[init_cost]])
        training_points = np.array([init_training_point])

        # for constitency, this must be run as in some cases it changes the
        # state of the cost function interface, even if we don't use the output
        cfi.cost_selector(training_points, costs)
        self.log("Adding 1 point(s)")

        self.log("Adding point...")
        self.log(f"Training point: {init_training_point}")
        self.log(f"Cost: {init_cost}")

        return costs

    def optimize(
        self,
        cfi: CostFunctionInterface,
        init_param_point: dict,
        results_name = "results"
    ):
        if self.save_folder:
            results_folder = f"{self.save_folder}/{results_name}"
            if not os.path.isdir(results_folder):
                os.mkdir(results_folder)
            filename_basis = f"{results_folder}/opt_basis.npz"
            filename_growth = f"{results_folder}/basis_growth.npz"
            filename_costs = f"{results_folder}/iteration_costs"

            if os.path.exists(filename_basis):
                self.log("Loading previously created basis...")
                self.basis = np.load(filename_basis)["arr_0"]
                self.basis_growth = np.load(filename_growth)["arr_0"]
                self.iteration_costs = []
                for i in range(len(self.basis_growth)):
                    self.iteration_costs.append(
                        np.load(f"{filename_costs}{i}.npz")["arr_0"]
                    )
                self.set_optimal(results_name)

                return self.opt_basis

        self.iteration_costs = []
        self.basis_growth = []

        if type(self.basis) == type(None):
            # no basis yet, we need to initialize
            costs = self.init_optimize(
                cfi,
                init_param_point,
            )
            self.iteration_costs.append(costs)

        self.log("Beginning optimization")
        for i in range(self.max_it):
            self.log(f"Iteration {i + 1}")
            self.log(f"Current basis size: {self.basis.shape[1]}")
            if self.optimize_step(cfi):
                break

        self.set_optimal(results_name)

        return self.opt_basis

    def solve(
        self,
        training_point: dict
    ) -> float:
        """
        Approximate the eigenvalues and eigenvectors for a given set of
        parameters

        Parameters
        ----------
        parameters : `list[float]`
            The parameters to approximate eigenvalues and eigenvectors for
        
        Returns
        -------
        evals : `np.ndarray`
            A list of the eigenvalues
        evecs : `np.ndarray`
            A matrix of the eigenvectors, with each column representing each
            eigenvector
        """
        #if (
        #    type(self.opt_basis) == type(None)
        #    or type(self.opt_overlap) == type(None)
        #):
        #    self.optimize()

        if type(self.opt_Hr_terms) == type(None):
            self.build_Hr_terms()
            self.opt_Hr_terms = self.Hr_terms
        
        Hr = self.build_Hr(training_point)

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

        H_full = sum(
            training_point[pauli] * self.get_H_term(pauli)
            for pauli in training_point.keys()
        )

        return H_full

    def build_Hr(
        self,
        training_point: dict,
    ) -> np.ndarray:
        Hr = np.zeros((self.basis.shape[1], self.basis.shape[1]), dtype=float)
        for pauli in self.pauli_strings:
            Hr += training_point[pauli] * self.get_Hr_term(pauli)

        return Hr

    def optimize_step(
        self,
        cfi: CostFunctionInterface
    ) -> bool:
        if(np.linalg.cond(self.overlap) > self.max_condition):
            self.log("Condition number is to large")
            return True

        self.log("Running preiteration...")
        cfi.preiteration()

        self.log("Generating training points for current iteration...")
        training_points = cfi.gen_training_points()
        if len(training_points) == 0:
            # no more training points
            self.log("No more training points")
            return True

        self.log(f"Generated {len(training_points)} training points")
        self.log("Calculating costs...")

        if self.processes == 1:
            costs = np.zeros(len(training_points), dtype=float)
            for j, training_point in enumerate(training_points):
                costs[j] = cfi.cost_function(training_point)
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.processes
            ) as pool:
                batch_size = int(np.ceil(len(training_points) / self.processes))
                costs = np.array(list(pool.map(
                    cfi.cost_function,
                    training_points,
                    chunksize = batch_size
                )))

        training_point_idxs = cfi.cost_selector(training_points, costs)
        self.log(f"Adding {training_point_idx} point(s)")
        next_training_points = training_points[training_point_idxs]
        next_costs = costs[training_point_idxs]

        if len(next_training_points) == 0:
            self.log("No viable training points found")
            # no training points found, no point in continuing
            return True
        else:
            self.log("Diagonalizing Hs...")

            basis_addition = self.find_basis_addition(
                next_costs,
                next_training_points
            )

            self.compress_basis(basis_addition)
            self.build_Hr_terms()

        self.iteration_costs.append(next_costs)

        if cfi.check_termination(self.iteration_costs):
            self.log("Termination condition met")
            return True
        
        return False

    def get_H_full_ground_state(
        self,
        training_point,
    ):
        H_full = self.build_H_full(training_point)
        v0 = np.ones(H_full.shape[0]) / np.sqrt(H_full.shape[0])

        if self.sparse:
            evals, evecs = sps.linalg.eigsh(
                H_full.real,
                k=min(int(self.size * self.sparse_proportion)+1, 4),
                v0=v0,
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

        return evecs[:, 0:degeneracy]


    def find_basis_addition(
        self,
        next_costs: list,
        next_training_points: list
    ):
        basis_addition = None
        if self.processes == 1:
            for cost, training_point in zip(
                next_costs,
                next_training_points
            ):
                new_vecs = self.get_H_full_ground_state(training_point)

                if type(basis_addition) == type(None):
                    basis_addition = new_vecs
                else:
                    basis_addition = np.append(
                        basis_addition,
                        new_vecs,
                        axis = 1
                    )

                self.log("Adding point...")
                self.log(f"Training point: {training_point}")
                self.log(f"Cost: {cost}")
        else:
            batch_size = int(
                np.ceil(len(next_training_points) / self.processes)
            )

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.processes
            ) as pool:
                vecs_list = list(pool.map(
                    self.get_H_full_ground_state,
                    next_training_points,
                    chunksize = batch_size
                ))

                for vecs, training_point, cost in zip(
                    vecs_list, next_training_points, next_costs
                ):
                    if type(basis_addition) == type(None):
                        basis_addition = vecs
                    else:
                        basis_addition = np.append(
                            basis_addition,
                            vecs,
                            axis = 1
                        )

                    self.log("Adding point...")
                    self.log(f"Training point: {training_point}")
                    self.log(f"Cost: {cost}")

        return basis_addition

    def compress_basis(
        self,
        basis_addition: np.ndarray
    ):
        # compress the basis
        projection = basis_addition - self.basis @ sp.linalg.solve(
            self.overlap, self.basis.conj().T @ basis_addition
        )

        U, sigmas, Vdagger = np.linalg.svd(projection, full_matrices=False)
        compress_add = 0
        for s in sigmas:
            if s > self.svd_tolerance:
                compress_add += 1
            else:
                break

        self.basis_growth.append(compress_add)

        self.log(f"Adding {compress_add} vector(s)")

        if compress_add == 0:
            self.log(
                "Warning: Basis did not increase in size after compression."
            )
        else:
            self.basis = np.hstack([self.basis, U[:, :compress_add]])

        self.overlap = (self.basis.conj().T @ self.basis).real

    def build_Hr_terms(self):
        self.log("Building Hr terms...")
        self.Hr_terms = {}

        if self.processes == 1:
            for pauli_string in self.pauli_strings:
                Hr_term = self.build_Hr_term(pauli_string)

                if self.keep_on_disk:
                    if pauli_string == "":
                        filename = self.save_folder + "/I_r.npz"
                    else:
                        filename = self.save_folder + f"/{pauli_string}_r.npz"
                    np.savez_compressed(filename, Hr_term)
                else:
                    self.Hr_terms[pauli_string] = Hr_term

        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.processes
            ) as pool:
                batch_size = int(np.ceil(len(self.pauli_strings) / self.processes))

                if self.keep_on_disk:
                    list(pool.map(
                        partial(
                            gen_Hr_and_save,
                            model=self
                        ),
                        self.pauli_strings,
                        chunksize=batch_size
                    ))
                else:
                    Hr_terms_list = list(pool.map(
                        self.build_Hr_term,
                        self.pauli_strings,
                        chunksize=batch_size
                    ))
                    for pauli_string, Hr_term in zip(self.pauli_strings, Hr_terms_list):
                        self.Hr_terms[pauli_string] = Hr_term

        self.log("Built Hr terms")

    def get_Hr_term(
        self,
        pauli
    ):
        if self.keep_on_disk:
            if pauli == "":
                filename = self.save_folder + "/I_r.npz"
            else:
                filename = self.save_folder + f"/{pauli}_r.npz"
            Hr_term = np.load(filename)["arr_0"]
            return Hr_term
        else:
            return self.Hr_terms[pauli]

    def theta_to_training_point(
        self,
        theta
    ):
        param = theta_to_param(
            theta,
            self.selected_params,
            self.name,
            self.N_spin
        )
        training_point = param_to_paulis(
            param,
            self.params,
            self.name,
            self.N_spin
        )

        return training_point

    def set_optimal(self, results_location="results"):
        self.opt_basis = self.basis
        self.opt_overlap = self.basis.conj().T @ self.basis
        self.build_Hr_terms()
        self.opt_Hr_terms = self.Hr_terms

        self.log(
            f"Using basis found with size {self.opt_basis.shape[1]}"
            + f" reduced from full Hilbert size of {self.size}"
        )

        if self.save_folder:
            results_folder = f"{self.save_folder}/{results_location}"
            if not os.path.isdir(results_folder):
                os.mkdir(results_folder)
            filename_basis = f"{results_folder}/opt_basis.npz"
            filename_growth = f"{results_folder}/basis_growth.npz"
            filename_costs = f"{results_folder}/iteration_costs"

            np.savez_compressed(filename_basis, self.opt_basis)
            basis_growth = np.array(self.basis_growth)
            np.savez_compressed(filename_growth, basis_growth)

            for i, iteration in enumerate(self.iteration_costs):
                iteration_arr = np.array(iteration)
                np.savez_compressed(f"{filename_costs}{i}.npz", iteration)

    def get_H_term(
        self,
        pauli_string
    ):
        if self.keep_on_disk:
            if pauli_string == "":
                filename = self.save_folder + "/I.npz"
            else:
                filename = self.save_folder + f"/{pauli_string}.npz"
            if self.sparse:
                H_term = sp.sparse.load_npz(filename)
            else:
                H_term = np.load(filename)["arr_0"]
            return H_term
        else:
            return self.H_terms[pauli_string]

    def build_Hr_term(
        self,
        pauli
    ):
        return self.basis.conj().T @ (self.get_H_term(pauli) @ self.basis)

    def log(
        self,
        text: str
    ):
        sys.stdout.write(
            str(datetime.datetime.now()) + ": " + text + "\n"
        )
