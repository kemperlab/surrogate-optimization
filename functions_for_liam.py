import numpy as np
from scipy import sparse as sps
from openfermion import (
    jordan_wigner,
    get_fermion_operator,
    generate_hamiltonian,
    QubitOperator,
    get_sparse_operator,
    fermi_hubbard,
    FermionOperator,
)
import os
from math import comb
from itertools import combinations
from typing import Any

def build_kanamori_integrals_spinorb(nbands, U, J, mu=0.0, verbose=False):
    """
    Build one-body and two-body integrals for the Kanamori model in SPIN-ORBITAL basis.

    This is compatible with OpenFermion's generate_hamiltonian() which expects
    spin-orbital indices using 'udud' ordering: up_index(i)=2*i, down_index(i)=2*i+1.

    The Kanamori Hamiltonian is:
        H = U Σ_i n_{i↑}n_{i↓}                                    [intra-orbital]
          + (U-2J) Σ_{i<j,σσ'} n_{iσ}n_{jσ'}                     [inter-orbital density]
          + J Σ_{i<j,σ} (c†_{iσ}c†_{j,-σ}c_{i,-σ}c_{jσ})        [spin-flip]
          + J Σ_{i<j} (c†_{i↑}c†_{i↓}c_{j↓}c_{j↑} + h.c.)       [pair-hopping]

    Parameters:
    -----------
    nbands : int
        Number of spatial orbitals (impurity sites)
    U : float
        Intra-orbital Coulomb repulsion
    J : float
        Hund's coupling
    mu : float
        Chemical potential (only used for 3-band case)

    Returns:
    --------
    t_spinorb : np.ndarray
        One-body integrals in spin-orbital basis, shape (2*nbands, 2*nbands)
    u_spinorb : np.ndarray
        Two-body integrals in spin-orbital basis, shape (2*nbands, 2*nbands, 2*nbands, 2*nbands)
        Convention: u[p,q,r,s] corresponds to c†_p c†_q c_s c_r in OpenFermion
    """
    from openfermion.utils import up_index, down_index

    if nbands not in [2, 3]:
        raise NotImplementedError(f"Only 2 and 3 band cases implemented, got {nbands}")

    n_spinorb = 2 * nbands
    t = np.zeros((n_spinorb, n_spinorb))
    u = np.zeros((n_spinorb, n_spinorb, n_spinorb, n_spinorb))

    # U' = U - 2J for inter-orbital density-density
    Up = U - 2.0 * J

    if nbands == 2:
        # ========== 2-BAND CASE ==========
        # Matches original build_kanamori_integrals for nbands==2

        # === Intra-orbital Hubbard U: U n_{i↑} n_{i↓} ===
        if verbose:
            print(
                f"[2-band] Adding intra-orbital Hubbard U: U={U}\n================================================================"
            )
        for i in range(nbands):
            iu = up_index(i)
            id = down_index(i)
            if verbose:
                print(
                    f"[2-band] Adding intra-orbital U for orbital {i}: spin orbitals: [{iu}†, {id}†, {id}, {iu}] = U"
                )
                print(
                    f"[2-band] Adding intra-orbital U for orbital {i}: spin orbitals: [{id}†, {iu}†, {iu}, {id}] = U"
                )
            u[iu, id, id, iu] = U
            u[id, iu, iu, id] = U

        # === Inter-orbital density-density: (U-2J) for i<j only ===
        for i in range(nbands):
            for j in range(i + 1, nbands):
                iu, id = up_index(i), down_index(i)
                ju, jd = up_index(j), down_index(j)

                if verbose:
                    print(
                        f"\n[2-band] Adding inter-orbital density-density for orbitals {i}<{j}: U'= U - 2J = {Up}\n================================================================"
                    )

                # n_{i↑} n_{j↑}
                if verbose:
                    print("n_{i↑} n_{j↑}")
                    print(
                        f"[2-band] Adding n_{{i↑}} n_{{j↑}} for orbitals {i}<{j}: spin orbitals: [{iu}†, {ju}†, {ju}, {iu}] = U' = {Up}"
                    )
                    print(
                        f"[2-band] Adding n_{{i↑}} n_{{j↑}} for orbitals {i}<{j}: spin orbitals: [{ju}†, {iu}†, {iu}, {ju}] = U' = {Up}"
                    )
                u[iu, ju, ju, iu] = Up
                u[ju, iu, iu, ju] = Up

                # n_{i↑} n_{j↓}
                if verbose:
                    print("\nn_{i↑} n_{j↓}")
                    print(
                        f"[2-band] Adding n_{{i↑}} n_{{j↓}} for orbitals {i}<{j}: spin orbitals: [{iu}†, {jd}†, {jd}, {iu}] = U' = {Up}"
                    )
                    print(
                        f"[2-band] Adding n_{{i↑}} n_{{j↓}} for orbitals {i}<{j}: spin orbitals: [{jd}†, {iu}†, {iu}, {jd}] = U' = {Up}"
                    )
                u[iu, jd, jd, iu] = Up
                u[jd, iu, iu, jd] = Up

                # n_{i↓} n_{j↑}
                if verbose:
                    print("\nn_{i↓} n_{j↑}")
                    print(
                        f"[2-band] Adding n_{{i↓}} n_{{j↑}} for orbitals {i}<{j}: spin orbitals: [{id}†, {ju}†, {ju}, {id}] = U' = {Up}"
                    )
                    print(
                        f"[2-band] Adding n_{{i↓}} n_{{j↑}} for orbitals {i}<{j}: spin orbitals: [{ju}†, {id}†, {id}, {ju}] = U' = {Up}"
                    )
                u[id, ju, ju, id] = Up
                u[ju, id, id, ju] = Up

                # n_{i↓} n_{j↓}
                if verbose:
                    print("\nn_{i↓} n_{j↓}")
                    print(
                        f"[2-band] Adding n_{{i↓}} n_{{j↓}} for orbitals {i}<{j}: spin orbitals: [{id}†, {jd}†, {jd}, {id}] = U' = {Up}"
                    )
                    print(
                        f"[2-band] Adding n_{{i↓}} n_{{j↓}} for orbitals {i}<{j}: spin orbitals: [{jd}†, {id}†, {id}, {jd}] = U' = {Up}"
                    )
                u[id, jd, jd, id] = Up
                u[jd, id, id, jd] = Up

        # === Exchange terms (i<j only): u[i,j,j,i] and u[i,j,i,j] ===
        for i in range(nbands):
            for j in range(i + 1, nbands):
                iu, id = up_index(i), down_index(i)
                ju, jd = up_index(j), down_index(j)

                if verbose:
                    print(
                        f"\n[2-band] Adding exchange terms for orbitals {i}<{j}: J={J}\n================================================================"
                    )

                # Spin-flip: c†_{i↑} c†_{j↓} c_{i↓} c_{j↑} (and variants)
                if verbose:
                    print("Spin-flip: c†_{i↑} c†_{j↓} c_{i↓} c_{j↑} (and variants)")
                    print(
                        f"[2-band] Adding spin-flip terms for orbitals {i}<{j}: spin orbitals: [{iu}†, {jd}†, {ju}, {id}] = J = {J}"
                    )
                    print(
                        f"[2-band] Adding spin-flip terms for orbitals {i}<{j}: spin orbitals: [{jd}†, {iu}†, {id}, {ju}] = J = {J}"
                    )
                    print(
                        f"[2-band] Adding spin-flip terms for orbitals {i}<{j}: spin orbitals: [{id}†, {ju}†, {jd}, {iu}] = J = {J}"
                    )
                    print(
                        f"[2-band] Adding spin-flip terms for orbitals {i}<{j}: spin orbitals: [{ju}†, {id}†, {iu}, {jd}] = J = {J}"
                    )
                u[iu, jd, ju, id] = J
                u[jd, iu, id, ju] = J
                u[id, ju, jd, iu] = J
                u[ju, id, iu, jd] = J

                # Pair-hopping: c†_{i↑} c†_{i↓} c_{j↓} c_{j↑} (and h.c.)
                if verbose:
                    print("\nPair-hopping: c†_{i↑} c†_{i↓} c_{j↓} c_{j↑} (and h.c.)")
                    print(
                        f"[2-band] Adding pair-hopping terms for orbitals {i}<{j}: spin orbitals: [{iu}†, {id}†, {ju}, {jd}] = J = {J}"
                    )
                    print(
                        f"[2-band] Adding pair-hopping terms for orbitals {i}<{j}: spin orbitals: [{id}†, {iu}†, {jd}, {ju}] = J = {J}"
                    )
                    print(
                        f"[2-band] Adding pair-hopping terms for orbitals {i}<{j}: spin orbitals: [{ju}†, {jd}†, {iu}, {id}] = J = {J}"
                    )
                    print(
                        f"[2-band] Adding pair-hopping terms for orbitals {i}<{j}: spin orbitals: [{jd}†, {ju}†, {id}, {iu}] = J = {J}"
                    )
                u[iu, id, ju, jd] = J
                u[id, iu, jd, ju] = J
                u[ju, jd, iu, id] = J
                u[jd, ju, id, iu] = J

        # === One-body chemical potential for half-filling WITH particle-hole symmetry ===
        # For PH symmetry: need one-body = -3U/2 + 2J = 0.5*(4J - 3U)
        # Note: The old formula 0.5*(5J - 3U) ensures half-filling but breaks PH symmetry
        if verbose:
            print(
                "\n[2-band] Adding one-body chemical potential shift for half-filling + PH symmetry: 0.5 * (4*J - 3*U)\n================================================================"
            )
        # # t[i,i] += 0.5 * (4*J - 3*U) for PH symmetry
        # one_body_shift = 0.5 * (4.0 * J - 3.0 * U)
        # for i in range(nbands):
        #     iu = up_index(i)
        #     id = down_index(i)
        #     if verbose:
        #         print(
        #             f"[2-band] Adding half-filling potential for orbital {i}: spin orbitals: [{iu}†, {iu}] and [{id}†, {id}] = 0.5 * ({4}*J - {3}*U) = {one_body_shift}"
        #         )
        #     t[iu, iu] = one_body_shift
        #     t[id, id] = one_body_shift

    elif nbands == 3:
        # ========== 3-BAND CASE ==========
        # Matches original build_kanamori_integrals_old for nbands==3

        # === Chemical potential: t[i,i] -= mu ===
        for i in range(nbands):
            iu = up_index(i)
            id = down_index(i)
            if verbose:
                print(
                    f"[3-band] Adding -mu for orbital {i}: spin orbitals: [{iu}†, {iu}] and [{id}†, {id}] = -mu"
                )
            t[iu, iu] -= mu
            t[id, id] -= mu

        if verbose:
            print(
                f"[3-band] Adding intra-orbital Hubbard U: U={U}\n================================================================"
            )
        for i in range(nbands):
            iu = up_index(i)
            id = down_index(i)
            if verbose:
                print(
                    f"[3-band] Adding intra-orbital U for orbital {i}: spin orbitals: [{iu}†, {id}†, {id}, {iu}] = U"
                )
                print(
                    f"[3-band] Adding intra-orbital U for orbital {i}: spin orbitals: [{id}†, {iu}†, {iu}, {id}] = U"
                )
            u[iu, id, id, iu] = U / 2
            u[id, iu, iu, id] = U / 2

        # === Inter-orbital density-density: (U-2J) for ALL i,j pairs ===
        # This includes i==j which gives the intra-orbital term!
        for i in range(nbands):
            for j in range(nbands):
                iu, id = up_index(i), down_index(i)
                ju, jd = up_index(j), down_index(j)

                if verbose:
                    print(
                        f"[3-band] Adding density-density for orbitals {i},{j}: spin orbitals: [{iu}†, {iu}] and [{id}†, {id}] and [{ju}†, {ju}] and [{jd}†, {jd}] = U'={Up}"
                    )
                    print("====================================================")

                # For i==j, this gives intra-orbital with coefficient (U-2J)
                # The additional 2J to get U comes from exchange terms below

                # n_{i↑} n_{j↓} (this is the key term for both inter and intra)
                if verbose:
                    print("n_{i↑} n_{j↓} (inter-orbital and intra-orbital)")
                    print(
                        f"[3-band] Adding n_{{i↑}} n_{{j↓}} for orbitals {i},{j}: spin orbitals: [{iu}†, {jd}†, {jd}, {iu}] = U'={Up}"
                    )
                    print(
                        f"[3-band] Adding n_{{i↑}} n_{{j↓}} for orbitals {i},{j}: spin orbitals: [{jd}†, {iu}†, {iu}, {jd}] = U'={Up}"
                    )
                u[iu, jd, jd, iu] += Up
                u[jd, iu, iu, jd] += Up

                # Same-spin inter-orbital (only for i!=j)
                if verbose:
                    print("\nSame-spin inter-orbital (only for i!=j)")
                    print(
                        f"[3-band] Adding n_{{i↑}} n_{{j↑}} for orbitals {i},{j}: U'= U - 2J = {Up} (only if i!=j)"
                    )
                    print("====================================================")
                if i != j:
                    if verbose:
                        print(
                            f"Adding same-spin density-density for spin orbitals [{iu}†, {ju}†, {ju}, {iu}] = U'={Up}"
                        )
                        print(
                            f"Adding same-spin density-density for spin orbitals [{ju}†, {iu}†, {iu}, {ju}] = U'={Up}"
                        )
                        print(
                            f"Adding same-spin density-density for spin orbitals [{id}†, {jd}†, {jd}, {id}] = U'={Up}"
                        )
                        print(
                            f"Adding same-spin density-density for spin orbitals [{jd}†, {id}†, {id}, {jd}] = U'={Up}"
                        )
                    u[iu, ju, ju, iu] += Up
                    u[ju, iu, iu, ju] += Up
                    u[id, jd, jd, id] += Up
                    u[jd, id, id, jd] += Up
                else:
                    if verbose:
                        print(
                            f"[3-band] Skipping same-spin density-density for i==j={i} since it's already included in the n_{{i↑}} n_{{i↓}} term with coefficient (U-2J)={Up}"
                        )

        # === Exchange: u[i,j,j,i] for ALL i,j ===
        for i in range(nbands):
            for j in range(nbands):
                iu, id = up_index(i), down_index(i)
                ju, jd = up_index(j), down_index(j)

                if verbose:
                    print(f"[3-band] Adding exchange for orbitals {i},{j}: J={J}")
                    print("====================================================")

                # Spin-flip terms
                if verbose:
                    print(
                        f"Spin-flip: spin orbitals: [{iu}†, {jd}†, {ju}, {id}] = J = {J}"
                    )
                    print(
                        f"Spin-flip: spin orbitals: [{jd}†, {iu}†, {id}, {ju}] = J = {J}"
                    )
                    print(
                        f"Spin-flip: spin orbitals: [{id}†, {ju}†, {jd}, {iu}] = J = {J}"
                    )
                    print(
                        f"Spin-flip: spin orbitals: [{ju}†, {id}†, {iu}, {jd}] = J = {J}"
                    )
                u[iu, jd, ju, id] += J
                u[jd, iu, id, ju] += J
                u[id, ju, jd, iu] += J
                u[ju, id, iu, jd] += J
        if verbose:
            print()

        # === Pair-hopping: u[i,j,i,j] for ALL i,j ===
        for i in range(nbands):
            for j in range(nbands):
                iu, id = up_index(i), down_index(i)
                ju, jd = up_index(j), down_index(j)

                if verbose:
                    print(f"[3-band] Adding pair-hopping for orbitals {i},{j}: J={J}")
                    print("====================================================")

                    print(
                        f"Pair-hopping: spin orbitals: [{iu}†, {id}†, {ju}, {jd}] = J = {J}"
                    )
                    print(
                        f"Pair-hopping: spin orbitals: [{id}†, {iu}†, {jd}, {ju}] = J = {J}"
                    )
                u[iu, id, ju, jd] += J
                u[id, iu, jd, ju] += J

        if verbose:
            print()

        # === One-body term for half-filling ===
        # t[i,i] += (U - 8*J) / 2
        if verbose:
            print(
                "\n[3-band] Adding one-body chemical potential shift for half-filling: (U - 8*J) / 2\n================================================================"
            )
        one_body_shift = (U - 8.0 * J) / 2.0
        for i in range(nbands):
            iu = up_index(i)
            id = down_index(i)
            if verbose:
                print(
                    f"Adding half-filling potential for spin orbitals [{iu}†, {iu}] and [{id}†, {id}] = {one_body_shift}"
                )
            t[iu, iu] += one_body_shift
            t[id, id] += one_body_shift

    return t, u


def spinorb_integrals_to_fermion_operator(one_body, two_body):
    """
    Convert one-body and two-body integrals in SPIN-ORBITAL basis to FermionOperator.

    This bypasses generate_hamiltonian() which expects spatial orbitals.

    Parameters:
    -----------
    one_body : np.ndarray
        One-body integrals, shape (n_spinorb, n_spinorb)
        h[p,q] corresponds to c†_p c_q
    two_body : np.ndarray
        Two-body integrals, shape (n_spinorb, n_spinorb, n_spinorb, n_spinorb)
        Convention: two_body[p,q,r,s] corresponds to c†_p c†_q c_r c_s
        (physicists' notation)

    Returns:
    --------
    FermionOperator
    """
    n_spinorb = one_body.shape[0]
    fermion_op = FermionOperator()

    # One-body terms: Σ_{p,q} h[p,q] c†_p c_q
    for p in range(n_spinorb):
        for q in range(n_spinorb):
            coeff = one_body[p, q]
            if abs(coeff) > 1e-12:
                fermion_op += FermionOperator(((p, 1), (q, 0)), coeff)

    # Two-body terms: (1/2) Σ_{p,q,r,s} g[p,q,r,s] c†_p c†_q c_r c_s
    # Factor of 1/2 accounts for double-counting in antisymmetric tensor
    for p in range(n_spinorb):
        for q in range(n_spinorb):
            for r in range(n_spinorb):
                for s in range(n_spinorb):
                    coeff = two_body[p, q, r, s]
                    if abs(coeff) > 1e-12:
                        # c†_p c†_q c_r c_s
                        fermion_op += FermionOperator(
                            ((p, 1), (q, 1), (r, 0), (s, 0)), 0.5 * coeff
                        )

    return fermion_op

def AIM_hamiltonian_FO(parameters):
    """
    Build the AIM Hamiltonian as a QubitOperator (Jordan-Wigner form).
    This is purely symbolic and doesn't create any matrices.

    Returns the combined QubitOperator that can be converted to a sparse matrix
    with a single get_sparse_operator() call.
    """
    from openfermion.utils import up_index, down_index

    N = parameters["NI"] + parameters["NB"]  # N spatial orbitals
    n_spinorb = 2 * N  # 2N spin-orbitals

    one_body = np.zeros((n_spinorb, n_spinorb))
    two_body = np.zeros((n_spinorb, n_spinorb, n_spinorb, n_spinorb))

    NI = parameters["NI"]
    NB = N - NI
    ei = parameters["ei"]
    ebs = parameters["eb"]
    vbs = parameters["vb"]
    if NI > 1:
        vis = parameters["vi"]
    U = parameters["U"]
    mu = parameters["mu"]

    # Impurity on-site energies (both spins)
    for i in range(NI):
        iu = up_index(i)
        id = down_index(i)
        one_body[iu, iu] = ei[i] - mu
        one_body[id, id] = ei[i] - mu

    if NI > 1:
        J = parameters["J"]
        # Get Kanamori integrals in spin-orbital form for the impurity sites
        t_kan, u_kan = build_kanamori_integrals_spinorb(NI, U, J, mu=mu, verbose=False)
        # Add to the impurity block (first 2*NI spin-orbitals)
        one_body[: 2 * NI, : 2 * NI] += t_kan
        two_body[: 2 * NI, : 2 * NI, : 2 * NI, : 2 * NI] += u_kan
    else:
        # Single impurity: just Hubbard U term
        # U n_↑ n_↓ -> u[0, 1, 1, 0] in spin-orbital basis
        iu = up_index(0)
        id = down_index(0)
        two_body[iu, id, id, iu] = U
        two_body[id, iu, iu, id] = U

    # Inter-impurity hopping (vi) - both spins
    if NI > 1:
        for i in range(NI):
            for j in range(NI):
                if i != j:
                    for spin_func in [up_index, down_index]:
                        si = spin_func(i)
                        sj = spin_func(j)
                        one_body[si, sj] += vis[i, j]

    # Bath site energies and impurity-bath hybridization
    for i in range(NI):
        for j in range(NB // NI):
            bath_spatial_idx = NI + i * (NB // NI) + j
            bu = up_index(bath_spatial_idx)
            bd = down_index(bath_spatial_idx)
            iu = up_index(i)
            id = down_index(i)

            # Bath on-site energy (both spins)
            one_body[bu, bu] = ebs[i, j]
            one_body[bd, bd] = ebs[i, j]

            # Impurity-bath hybridization (both spins)
            one_body[iu, bu] = vbs[i, j]
            one_body[bu, iu] = vbs[i, j]
            one_body[id, bd] = vbs[i, j]
            one_body[bd, id] = vbs[i, j]

    # Use direct spin-orbital to FermionOperator conversion
    # (generate_hamiltonian expects spatial orbitals and would double the count)
    fermion_op_hamiltonian = spinorb_integrals_to_fermion_operator(one_body, two_body)

    return fermion_op_hamiltonian

def AIM_hamiltonian_JW(parameters):
    """
    Build the AIM Hamiltonian as a QubitOperator (Jordan-Wigner form).
    This is purely symbolic and doesn't create any matrices.

    Returns the combined QubitOperator that can be converted to a sparse matrix
    with a single get_sparse_operator() call.
    """
    fermion_op_hamiltonian = AIM_hamiltonian_FO(parameters)
    jw_hamiltonian = jordan_wigner(fermion_op_hamiltonian)
    return jw_hamiltonian

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
    Get the particle-number (and possibly spin) sector basis for a system of N sites.
    If the basis file does not exist, it will be calculated and saved.
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

                second_half = np.zeros(comb(len(odd_positions), s[1]), dtype=np.int64)
                for i, ones_idx in enumerate(
                    combinations(range(len(odd_positions)), s[1])
                ):
                    n = 0
                    for idx in ones_idx:
                        pos = odd_positions[idx]
                        n |= 1 << (N - 1 - pos)
                    second_half[i] = n

                # Combine even- and odd-site patterns
                basis = np.zeros(len(first_half) * len(second_half), dtype=np.int64)
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
                        "Number of spin-up or spin-down particles exceeds half the system size."
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
