import numpy as np
from math import comb
from itertools import combinations
import os
from openfermion import (
    jordan_wigner,
    fermi_hubbard,
    get_fermion_operator,
    generate_hamiltonian,
    QubitOperator,
)
from openfermion.linalg import get_sparse_operator
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from functions_for_liam import *

class Pauli:
    I = np.array([
        [1, 0],
        [0, 1]
    ], dtype=complex)

    X = np.array([
        [0, 1],
        [1, 0]
    ], dtype=complex)

    Y = np.array([
        [0, -1j],
        [1j, 0]
    ], dtype=complex)

    Z = np.array([
        [1, 0],
        [0, -1]
    ], dtype=complex)

def get_ps_basis(
    s: int | tuple[int],
    N: int,
    ordering="uudd"
) -> np.ndarray:
    """
    Get the particle-number (and possibly spin) sector basis for a system of N
    sites. If the basis file does not exist, it will be calculated and saved.

    Args:
        s (int | tuple[int]): Number of particles (or tuple of spin-up and
            spin-down particles).
        N (int): Total number of sites.
    Returns:
        np.ndarray: Array of basis states in integer representation.
    """

    spin_protected = isinstance(s, tuple)
    basis_dir = "basis_ixs_udud"
    os.makedirs(basis_dir, exist_ok=True)
    if spin_protected:
        basis_file = os.path.join(basis_dir, f"basis_{N}_{s[0]}_{s[1]}.txt")
    else:
        basis_file = os.path.join(basis_dir, f"basis_{N}_{s}.txt")
    try:
        basis = np.loadtxt(basis_file).astype(np.int64)
        return basis

    except FileNotFoundError:
        print("No basis file, calculating now...")
        if spin_protected:
            if ordering == "udud":
                if N % 2 != 0:
                    raise ValueError("N must be even for spin-protected basis.")
                if (s[0] > N // 2) or (s[1] > N // 2):
                    raise ValueError(
                        "Number of spin-up or spin-down particles exceeds half"
                         + "the system size."
                    )

                half = N // 2
                # Generate all bit patterns for each half
                # Build patterns on even and odd site indices (0-based)
                even_positions = list(range(0, N, 2))
                odd_positions = list(range(1, N, 2))

                first_half = np.zeros(comb(len(even_positions), s[0]),
                    dtype=np.int64)
                for i, ones_idx in enumerate(
                    combinations(range(len(even_positions)), s[0])
                ):
                    n = 0
                    for idx in ones_idx:
                        pos = even_positions[idx]
                        n |= 1 << (N - 1 - pos)
                    first_half[i] = n

                second_half = np.zeros(comb(len(odd_positions), s[1]),
                    dtype=np.int64)
                for i, ones_idx in enumerate(
                    combinations(range(len(odd_positions)), s[1])
                ):
                    n = 0
                    for idx in ones_idx:
                        pos = odd_positions[idx]
                        n |= 1 << (N - 1 - pos)
                    second_half[i] = n

                # Combine even- and odd-site patterns
                basis = np.zeros(len(first_half) * len(second_half),
                    dtype=np.int64)
                idx = 0
                for fh in first_half:
                    for sh in second_half:
                        basis[idx] = fh | sh
                        idx += 1

                basis = np.sort(basis)
                np.savetxt(basis_file, basis, fmt="%d")
                return basis

            if ordering == "uudd":

                if N % 2 != 0:
                    raise ValueError("N must be even for spin-protected basis.")
                if (s[0] > N // 2) or (s[1] > N // 2):
                    raise ValueError(
                        "Number of spin-up or spin-down particles exceeds half "
                        + "the system size."
                    )

                half = N // 2
                # Generate all bit patterns for each half
                first_half = np.zeros(comb(half, s[0]), dtype=np.int64)
                for i, ones_pos in enumerate(combinations(range(half), s[0])):
                    n = 0
                    for pos in ones_pos:
                        n |= 1 << (half - 1 - pos)
                    first_half[i] = n

                second_half = np.zeros(comb(half, s[1]), dtype=np.int64)
                for i, ones_pos in enumerate(combinations(range(half), s[1])):
                    n = 0
                    for pos in ones_pos:
                        n |= 1 << (half - 1 - pos)
                    second_half[i] = n

                # Combine both halves
                basis = np.zeros(len(first_half) * len(second_half),
                    dtype=np.int64)
                idx = 0
                for fh in first_half:
                    for sh in second_half:
                        combined = (fh << half) | sh
                        basis[idx] = combined
                        idx += 1

                basis = np.sort(basis)
                np.savetxt(basis_file, basis, fmt="%d")
                return basis

            else:
                raise NotImplementedError(
                    f"Ordering '{ordering}' not implemented for spin-protected "
                    + "basis."
                )

        else:

            basis = np.zeros(comb(N, s), dtype=np.int64)
            for i, ones_pos in enumerate(combinations(range(N), s)):
                n = 0
                for pos in ones_pos:
                    n |= 1 << (N - 1 - pos)
                basis[i] = n
            basis = np.sort(basis)
            np.savetxt(basis_file, basis, fmt="%d")
            return basis

def of_operator_to_pauli_and_coeff(N, of_operator) -> list[tuple[str, complex]]:
    pauli_list = []
    for term in of_operator.terms:
        pauli_string = ""
        for qubit_index in range(N):
            if qubit_index in [idx for idx, _ in term]:
                pauli_type = [
                    ptype for idx, ptype in term if idx == qubit_index
                ][0]
                pauli_string += f"{pauli_type}{qubit_index} "
        pauli_string = pauli_string.strip()  # Remove trailing space
        coeff = of_operator.terms[term]
        pauli_list.append((pauli_string, coeff))
    return pauli_list

def parse_openfermion_term(
    term: tuple[int, str]
) -> str:
    string = ""
    for factor in term:
        string += factor[1] + str(factor[0])

    return pauli_strings

def gen_from_pauli_string(
    pauli_string: str,
    N: int,
    particle_selection: tuple[int, int] | int = None,
    ordering="uudd",
    sparse=False,
) -> np.ndarray | sps.csc_matrix:
    if pauli_string == "":
        if sparse:
            mat = sps.eye(2**N, format="csc", dtype=float)
            if particle_selection is not None:
                basis = get_ps_basis(particle_selection, N, ordering=ordering)
                mat = mat[:, basis][basis]
            return mat
        else:
            mat = np.eye(2**N, dtype=float)
            if particle_selection is not None:
                basis = get_ps_basis(particle_selection, N, ordering=ordering)
                mat = mat[:, basis][basis]
            return mat
    else:
        op = QubitOperator(pauli_string)
        if sparse:
            mat = get_sparse_operator(op, N).tocsc().astype(float)
            if particle_selection is not None:
                basis = get_ps_basis(particle_selection, N, ordering=ordering)
                mat = mat[:, basis][basis]
            return mat
        else:
            mat = get_sparse_operator(op, N).toarray().astype(float)

            if particle_selection is not None:
                basis = get_ps_basis(particle_selection, N, ordering=ordering)
                mat = mat[:, basis][basis]
            return mat

def model_to_paulis(
    N: int, model: str, model_parameters: dict
) -> list[tuple[str, complex]]:

    if model == "fermi_hubbard":
        t = model_parameters.get("t", 1.0)
        U = model_parameters.get("U", 0.0)
        mu = model_parameters.get("mu", 0.0)
        of_hamiltonian = fermi_hubbard(
            N,
            1,
            t,
            U,
            chemical_potential=mu,
            periodic=False,
        )
        jw_hamiltonian = jordan_wigner(of_hamiltonian)
        pauli_list = of_operator_to_pauli_and_coeff(N * 2, jw_hamiltonian)
        return pauli_list

    elif model == "heisenberg":
        Jx = model_parameters.get("Jx", 1.0)
        Jy = model_parameters.get("Jy", 1.0)
        Jz = model_parameters.get("Jz", 1.0)
        h = model_parameters.get("h", 0.0)
        periodic = model_parameters.get("periodic", False)

        operator = QubitOperator()
        for i in range(N - 1):
            operator += QubitOperator(f"X{i} X{i+1}", Jx)
            operator += QubitOperator(f"Y{i} Y{i+1}", Jy)
            operator += QubitOperator(f"Z{i} Z{i+1}", Jz)
        if periodic:
            operator += QubitOperator(f"X{N-1} X0", Jx)
            operator += QubitOperator(f"Y{N-1} Y0", Jy)
            operator += QubitOperator(f"Z{N-1} Z0", Jz)

        for i in range(N):
            operator += QubitOperator(f"Z{i}", h)

        pauli_list = of_operator_to_pauli_and_coeff(N, operator)
        return pauli_list

    elif model == "TFIM":
        operator = QubitOperator()
        J = model_parameters.get("J", -1.0)
        h = model_parameters.get("h", 1.0)
        periodic = model_parameters.get("periodic", False)
        for i in range(N):
            operator += QubitOperator(f"X{i}", h)
        for i in range(N - 1):
            operator += QubitOperator(f"Z{i} Z{i+1}", J)
        if periodic:
            operator += QubitOperator(f"Z{N-1} Z0", J)

        pauli_list = of_operator_to_pauli_and_coeff(N, operator)
        return pauli_list

    elif model == "TFXY":
        operator = QubitOperator()
        Jx = model_parameters.get("Jx", 1.0)
        Jy = model_parameters.get("Jy", 1.0)
        h = model_parameters.get("h", 1.0)
        periodic = model_parameters.get("periodic", False)
        for i in range(N):
            operator += QubitOperator(f"X{i}", h)
        for i in range(N - 1):
            operator += QubitOperator(f"X{i} X{i+1}", Jx)
            operator += QubitOperator(f"Y{i} Y{i+1}", Jy)
        if periodic:
            operator += QubitOperator(f"X{N-1} X0", Jx)
            operator += QubitOperator(f"Y{N-1} Y0", Jy)

        pauli_list = of_operator_to_pauli_and_coeff(N, operator)
        return pauli_list

    elif model == "AIM":
        model_parameters["eb"] = np.array([model_parameters["eb"]])
        model_parameters["vb"] = np.array([model_parameters["vb"]])
        jw_hamiltonian = AIM_hamiltonian_JW(model_parameters)

        pauli_list = of_operator_to_pauli_and_coeff(N * 2, jw_hamiltonian)
        return pauli_list

    else:
        raise ValueError(f"Model '{model}' not recognized.")

def param_to_paulis(
    theta: tuple,
    params: tuple[str],
    model: str,
    N: int
) -> dict[str, float]:
    """
    Convert a parameter point to the coefficients of Pauli matrices

    Parameter
    ---------
    theta : `tuple`
        A parameter point in the training grid
    params : `tuple[str]`
        The parameters being referenced
    model : `str`
        The model thats being used
    N: number of particles
    """
    paulis = model_to_paulis(N, model, {
        label : theta_i for label, theta_i in zip(params, theta)
    })

    return{p: c.real for p, c in paulis}

def get_model_paulis(model_type, N):
    if model_type == "TFIM":
        model_parameters = {
            "J": 1,
            "h": 1,
            "periodic": False,
        }
    elif model_type == "TFXY":
        model_parameters = {
            "Jx": 1,
            "Jy": 1,
            "h": 1,
            "periodic": False,
        }
    elif model_type == "heisenberg":
        model_parameters = {
            "Jx": 1,
            "Jy": 1,
            "Jz": 1,
            "h": 1,
            "periodic": False,
        }
    elif model_type == "fermi_hubbard":
        model_parameters = {
            "t": 1.0,
            "U": 1.0,
            "mu": 0.5,
            "periodic": False,
        }
    elif model_type == "AIM":
        U = 4.0
        NI = 1
        NB = N - NI
        model_parameters = {
            "NI": NI,
            "NB": NB,
            "U": U,
            "ei": [0.0] * NI,
            "vb": np.array([0.01] * ((NB) % 2) + [1.0] * (NB - (NB) % 2)),
            "eb": np.array(
                [0.0] * ((NB) % 2)
                + [1.0] * ((NB - (NB) % 2) // 2)
                + [-1.0] * ((NB - (NB) % 2) // 2)
            ),
            "mu": U / 2,
            "periodic": False,
        }

    model_paulis = model_to_paulis(N, model_type, model_parameters)
    H_paulis = [t[0] for t in model_paulis]

    return H_paulis

def get_model_base_parameters(model_type, N):
    if model_type == "TFIM":
        model_parameters = {
            "J": 1,
            "h": 1,
            "periodic": False,
        }
    elif model_type == "TFXY":
        model_parameters = {
            "Jx": 1,
            "Jy": 1,
            "h": 1,
            "periodic": False,
        }
    elif model_type == "heisenberg":
        model_parameters = {
            "Jx": 1,
            "Jy": 1,
            "Jz": 1,
            "h": 1,
            "periodic": False,
        }
    elif model_type == "fermi_hubbard":
        model_parameters = {
            "t": 1.0,
            "U": 1.0,
            "mu": 0.5,
            "periodic": False,
        }
    elif model_type == "AIM":
        U = 4.0
        NI = 1
        NB = N - NI
        model_parameters = {
            "NI": NI,
            "NB": NB,
            "U": U,
            "ei": [0.0] * NI,
            "vb": np.array([0.01] * ((NB) % 2) + [1.0] * (NB - (NB) % 2)),
            "eb": np.array(
                [0.0] * ((NB) % 2)
                + [1.0] * ((NB - (NB) % 2) // 2)
                + [-1.0] * ((NB - (NB) % 2) // 2)
            ),
            "mu": U / 2,
            "periodic": False,
        }

    return model_parameters

def get_model_parameters(model_type):
    if model_type == "TFIM":
        model_parameters = ("J", "h")
    elif model_type == "TFXY":
        model_parameters = ("Jx", "Jy", "h")
    elif model_type == "heisenberg":
        model_parameters = ("Jx", "Jy", "Jz", "h")
    elif model_type == "fermi_hubbard":
        model_parameters = ("t", "U", "mu")
    elif model_type == "AIM":
        model_parameters = ("NI", "NB", "U", "ei", "vb", "eb", "mu")

    return model_parameters

def get_model_N(model_type, N):
    if model_type == "fermi_hubbard" or model_type == "AIM":
        return 2 * N
    else:
        return N

def theta_to_param(theta, selected_params, model_type, N):
    if model_type == "TFIM":
        param = []
        if "J" in selected_params:
            idx = selected_params.index("J")
            param.append(theta[idx])
        else:
            param.append(1.0)

        if "h" in selected_params:
            idx = selected_params.index("h")
            param.append(theta[idx])
        else:
            param.append(1.0)

        return tuple(param)

    elif model_type == "TFXY":
        param = []
        if "Jx" in selected_params:
            idx = selected_params.index("Jx")
            param.append(theta[idx])
        else:
            param.append(1.0)

        if "Jy" in selected_params:
            idx = selected_params.index("Jy")
            param.append(theta[idx])
        else:
            param.append(1.0)

        if "h" in selected_params:
            idx = selected_params.index("h")
            param.append(theta[idx])
        else:
            param.append(1.0)

        return tuple(param)

    elif model_type == "fermi_hubbard":
        params = []
        if "t" in selected_params:
            idx = selected_params.index("t")
            params.append(theta[idx])
        else:
            params.append(1.0)

        if "U" in selected_params:
            idx = selected_params.index("U")
            params.append(theta[idx])
        else:
            params.append(4.0)

        if "mu" in selected_params:
            idx = selected_params.index("mu")
            params.append(theta[idx])
        else:
            params.append(theta[1] / 2)

        return tuple(params)

    elif model_type == "AIM":
        NI = 1
        NB = N - NI
        param = [NI, NB]
        
        # manual override for test cases
        hopend = (NB // 2) + (NB % 2) + 1
        param = [
            NI, NB,
            theta[0],
            [0.0] * NI,
            [theta[i] for i in range(1, hopend)]
            + [theta[i] for i in range(1 + (NB % 2), hopend)],
            [0] * (NB % 2)
            + [theta[i] for i in range(hopend, hopend + (NB // 2))]
            + [-theta[i] for i in range(hopend, hopend + (NB // 2))],
            theta[0] / 2
        ]

        return tuple(param)

        ###########################

        if len(theta) == 6: #(theta == ("U", "vb1", "vb2", "vb3", "eb2", "eb3").all()):
            param = [
                NI, NB,
                theta[0], [0.0],
                [theta[1], theta[2], theta[3], theta[2], theta[3]],
                [0, theta[4], theta[5], -theta[4], -theta[5]],
                theta[0] / 2
            ]

            return tuple(param)

        if len(theta) == 10:
            param = [
                NI, NB,
                theta[0], [0.0],
                [
                    theta[1],
                    theta[2], theta[3], theta[4], theta[5],
                    theta[2], theta[3], theta[4], theta[5]
                ],
                [
                    0,
                    theta[6], theta[7], theta[8], theta[9],
                    -theta[6], -theta[7], -theta[8], -theta[9]
                ],
                theta[0] / 2
            ]

            return tuple(param)

        if "U" in selected_params:
            idx = selected_params.index("U")
            param.append(theta[idx])
        else:
            param.append(4.0)

        if "ei" in selected_params:
            idx = selected_params.index("ei")
            param.append([theta[idx]] * NI)
        else:
            param.append([0.0] * NI)

        if "vb" in selected_params:
            idx = selected_params.index("vb")
            param.append(
                np.array([0.01] * ((NB) % 2) + [theta[idx]] * (NB - (NB) % 2))
            )
        else:
            param.append(
                np.array([0.01] * ((NB) % 2) + [1.0] * (NB - (NB) % 2))
            )
        
        if "eb" in selected_params:
            idx = selected_params.index("eb")
            param.append(np.array(
                [0.0] * ((NB) % 2)
                + [theta[idx]] * ((NB - (NB) % 2) // 2)
                + [-theta[idx]] * ((NB - (NB) % 2) // 2)
            ))
        else:
            param.append(np.array(
                [0.0] * ((NB) % 2)
                + [1.0] * ((NB - (NB) % 2) // 2)
                + [-1.0] * ((NB - (NB) % 2) // 2)
            ))

        if "mu" in selected_params:
            idx = selected_params.index("mu")
            param.append(theta[idx])
        else:
            param.append(param[2] / 2)

        return tuple(param)

def get_op_dict(jw_hamiltonian, N, sparse=True, make_ops=True):
    op_dict = {}
    for term in jw_hamiltonian.terms:
        pauli_string = ""
        for qubit_index in range(N):
            if qubit_index in [idx for idx, _ in term]:
                pauli_type = [ptype for idx, ptype in term if idx == qubit_index][0]
                pauli_string += f"{pauli_type}{qubit_index} "
        pauli_string = pauli_string.strip()  # Remove trailing space
        coeff = jw_hamiltonian.terms[term]
        if make_ops:
            if sparse:
                op = get_sparse_operator(QubitOperator(pauli_string, 1.0), N)
            else:
                op = get_sparse_operator(QubitOperator(pauli_string, 1.0), N).toarray()
            op_dict[pauli_string] = (coeff.real, op)
        else:
            op = pauli_string
            op_dict[pauli_string] = coeff.real
    return op_dict

def get_particle_selected_basis(
    s: int | tuple[int], N: int, ordering="udud"
) -> np.ndarray:
    """
    Get the particle-number (and possibly spin) sector basis for a system of N
    sites. If the basis file does not exist, it will be calculated and saved.
    Args:
        s (int | tuple[int]): Number of particles (or tuple of spin-up and
            spin-down particles).
        N (int): Total number of sites.
    Returns:
        np.ndarray: Array of basis states in integer representation.
    """

    spin_protected = isinstance(s, tuple)
    basis_dir = "basis_ixs_udud"
    os.makedirs(basis_dir, exist_ok=True)
    if spin_protected:
        basis_file = os.path.join(basis_dir, f"basis_{N}_{s[0]}_{s[1]}.txt")
    else:
        basis_file = os.path.join(basis_dir, f"basis_{N}_{s}.txt")
    try:
        basis = np.loadtxt(basis_file).astype(np.int64)
        return basis
    except FileNotFoundError:
        print("No basis file, calculating now...")
        if spin_protected:
            if ordering == "udud":
                if N % 2 != 0:
                    raise ValueError("N must be even for spin-protected basis.")
                if (s[0] > N // 2) or (s[1] > N // 2):
                    raise ValueError(
                        "Number of spin-up or spin-down particles exceeds half the system size."
                    )

                half = N // 2
                # Generate all bit patterns for each half
                # Build patterns on even and odd site indices (0-based)
                even_positions = list(range(0, N, 2))
                odd_positions = list(range(1, N, 2))

                first_half = np.zeros(comb(len(even_positions), s[0]), dtype=np.int64)
                for i, ones_idx in enumerate(
                    combinations(range(len(even_positions)), s[0])
                ):
                    n = 0
                    for idx in ones_idx:
                        pos = even_positions[idx]
                        n |= 1 << (N - 1 - pos)
                    first_half[i] = n

                second_half = np.zeros(
                    comb(len(odd_positions), s[1]), dtype=np.int64
                )
                for i, ones_idx in enumerate(
                    combinations(range(len(odd_positions)), s[1])
                ):
                    n = 0
                    for idx in ones_idx:
                        pos = odd_positions[idx]
                        n |= 1 << (N - 1 - pos)
                    second_half[i] = n

                # Combine even- and odd-site patterns
                basis = np.zeros(len(first_half) * len(second_half),
                    dtype=np.int64)
                idx = 0
                for fh in first_half:
                    for sh in second_half:
                        basis[idx] = fh | sh
                        idx += 1

                basis = np.sort(basis)
                np.savetxt(basis_file, basis, fmt="%d")
                return basis

            if ordering == "uudd":

                if N % 2 != 0:
                    raise ValueError("N must be even for spin-protected basis.")
                if (s[0] > N // 2) or (s[1] > N // 2):
                    raise ValueError(
                        "Number of spin-up or spin-down particles exceeds half "
                        + "the system size."
                    )

                half = N // 2
                # Generate all bit patterns for each half
                first_half = np.zeros(comb(half, s[0]), dtype=np.int64)
                for i, ones_pos in enumerate(combinations(range(half), s[0])):
                    n = 0
                    for pos in ones_pos:
                        n |= 1 << (half - 1 - pos)
                    first_half[i] = n

                second_half = np.zeros(comb(half, s[1]), dtype=np.int64)
                for i, ones_pos in enumerate(combinations(range(half), s[1])):
                    n = 0
                    for pos in ones_pos:
                        n |= 1 << (half - 1 - pos)
                    second_half[i] = n

                # Combine both halves
                basis = np.zeros(len(first_half) * len(second_half), dtype=np.int64)
                idx = 0
                for fh in first_half:
                    for sh in second_half:
                        combined = (fh << half) | sh
                        basis[idx] = combined
                        idx += 1

                basis = np.sort(basis)
                np.savetxt(basis_file, basis, fmt="%d")
                return basis

            else:
                raise NotImplementedError(
                    f"Ordering '{ordering}' not implemented for spin-protected basis."
                )

        else:

            basis = np.zeros(comb(N, s), dtype=np.int64)
            for i, ones_pos in enumerate(combinations(range(N), s)):  # type: ignore
                n = 0
                for pos in ones_pos:
                    n |= 1 << (N - 1 - pos)
                basis[i] = n
            basis = np.sort(basis)
            np.savetxt(basis_file, basis, fmt="%d")
            return basis
