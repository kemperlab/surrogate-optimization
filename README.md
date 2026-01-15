# Surrogate Optimizer
A surrogate optimizer for fermionic Hamiltonians given their Jordan-Wigner decomposition.

Easy-setup code is provided for the following models:
+ TFIM
+ TFXY
+ Heisenberg
+ Fermi-Hubbard
+ Anderson Impurity

## Walk-Through
The main part of the code is the `SurrogateModel` class. This class does the actual surrogate
optimization and can be used to approximate the eigenvalues/eigenvectors of the provided Hamiltonian.

### Initializing the Surrogate Model
In the constructor for `SurrogateModel` a list of the Pauli strings for each term in the Hamiltonian
is provided as well as a training grid. The training grid is a list of dictionaries, each dictionary
mapping the provided Pauli strings to a coefficient for the corresponding term in the Hamiltonian.
The model name and the number of particles, N, are also provided. The model name is used for saving
results and logging. Note N is the number of particles and not the number of orbitals.

If particle selection is too be used, this can also be specified. Default is no particle-selection.
An integer between 0 and N forces a partricle selection of that amount, or a tuple specified as
(up, down) specifies the particle selection for up and down spins.

The basis ordering dertermines the order of up and down spins with the basis vectors. "uudd" means that
all the up spins are first, followed by the down spins. "udud" means that up and down spins alternate.

If log is true, logs are saved in a surrogate.log file.

### Building Hamiltonian Terms
Each term in the Hamiltonian only needs to be built once. The method `build_terms`, when called, will
build the terms of the Hamiltonian and the Hamiltonian squared. If not called manually, the `optimize`
method will call this function automatically. Optionally, if the `save` parameter is true, this will
either read the Hamiltonian terms from a file, or will generate and save the Hamiltonian terms to a
file. This function can also pregenerate the Hamiltonian in the full Hilbert space for each set of
parameters in the training grid. This requires more memory to store each Hamiltonian and more upfront,
cost in terms of time, but should speed up runtime over the whole optimization process.

### Optimization
The `optimization` method runs the actual optimization part of the surrogate optimization. If `build_terms` have not been called before, `optimize` will call it first. To change how the optimization is run, the following parameters are available to be set:
+ `residue_threshold`: The value of maximum residue for which to cut off the optimization
+ `init_vec`: An initial vector to use for the optimization. If not set, the algorithm will use the first element in the training grid.
+ `solution_grid`: For testing only. A 2D array of actual ground state energies. Supposing only two parameters are varied, this will display the difference between the current ground state energies and the actual ground state energy.
+ `svd_tolerance`: The highest value to not count a singular value as zero during an SVD.
+ `degeneracy_truncation`: The maximum number of degeneracies in the ground state.
+ `save`: Attempts to load the optimal basis from a file. If this fails, the optimization is run and the optimal basis is saved to a file.
+ `residue_graphing`: Graphs residues from each iteration
+ `processes`: How many processes to use for the optimization.

### Solving
The `solve` method inputs a list of parameters for the Hamiltonian, and using the optimal basis, finds the eigenvalues and eigenvectors.

## Test Run
The following shows results from running the Anderson Impurity Model with 1 impurity orbital and 3 bath orbitals.
<img width="768" height="686" alt="image" src="https://github.com/user-attachments/assets/197d926c-a678-48c4-83d8-fc61f064b369" />
