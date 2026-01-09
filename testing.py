from surrogate import *
from openfermion.hamiltonians import fermi_hubbard
from openfermion.transforms import jordan_wigner

if __name__ == "__main__":
    ###############################################################
    # Residue threshold for terminating optimization (lower means more accurate,
    # at the cost of more basis vectors)
    res_thresh = 1e-3

    # For removing linear dependence in basis vectors
    svd_tol = 1e-8

    # Parameter grid values for training grid
    mu = np.linspace(-5.0, 5.0, 20)

    # Number of sites (total for TFIM/TFXY/Heisenberg, per spin for fermi_hubbard, AIM)
    N = 4

    # None or Between 0 and N (2*N for AIM, fermi_hubbard), N for
    # TFIM/TFXY/Heisenberg. Can be tuple for (n_up, n_down) for AIM,
    # fermi_hubbard
    ps = None

    # AIM = Single Impurity Anderson Model, fermi_hubbard, TFIM, TFXY,
    # heisenberg
    model_type = "fermi_hubbard"

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
        U = 1.0
        model_parameters = {
            "t": 1.0,
            "mu": 1.0,
            "U": U,
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
    H_paulis_order = {}

    for i, pauli in enumerate(H_paulis):
        H_paulis_order[pauli] = i

    ### Any training grid can be used here, this is an example of the model
    # being parameterized over two parameters
    training_grid = []
    for m1 in mu:
        for m2 in mu:
            if model_type == "TFIM":
                model_parameters["J"] = m1
                model_parameters["h"] = m2
            elif model_type == "TFXY":
                model_parameters["Jx"] = m1
                model_parameters["Jy"] = m1
                model_parameters["h"] = m2
            elif model_type == "heisenberg":
                model_parameters["Jx"] = m1
                model_parameters["Jy"] = m1
                model_parameters["Jz"] = m2
                model_parameters["h"] = 0.1
            elif model_type == "fermi_hubbard":
                model_parameters["t"] = m1
                model_parameters["mu"] = 0.5
                model_parameters["U"] = m2
            elif model_type == "AIM":
                model_parameters["vb"] = np.array(
                    [0.01] * ((NB) % 2) + [m1] * (NB - (NB) % 2)
                )

                model_parameters["eb"] = np.array(
                    [0.0] * ((NB) % 2)
                    + [m2] * ((NB - (NB) % 2) // 2)
                    + [-m2] * ((NB - (NB) % 2) // 2)
                )
            model_paulis = model_to_paulis(N, model_type, model_parameters)
            params = np.zeros(len(H_paulis), dtype=tuple)

            for t in model_paulis:
                try:
                    params[H_paulis_order[t[0]]] = t[1]
                except:
                    raise Exception("Failed to generate all terms in model")

            model_paulis = list(zip(H_paulis, params))
            paulis_dict = {}
            for t in model_paulis:
                paulis_dict[t[0]] = t[1]
            training_grid.append(paulis_dict)

    if model_type == "AIM" or model_type == "fermi_hubbard":
        surrogate_N = 2 * N
        surrogate_ord = "udud"
    else:
        surrogate_N = N
        surrogate_ord = "uudd"
    model = SurrogateModel(
        model_type,
        surrogate_N,
        H_paulis,
        training_grid,
        particle_selection=ps,
        basis_ordering=surrogate_ord,
        log=False
    )

    model.build_terms(processes=4)
    print("Done building terms")

    # Calculate the real solutions for testing (only for 2D parameter grids)
    solution_grid = np.zeros((len(mu), len(mu)), dtype=complex)
    for i in range(len(mu)):
        for j in range(len(mu)):
            H_full = np.zeros((model.size, model.size), dtype=complex)
            parameters = training_grid[i * len(mu) + j]
            for pauli in model.H_terms.keys():
                H_full += parameters[pauli] * model.H_terms[pauli]
            if model.sparse:
                evals, evecs = sps.linalg.eigsh(H_full.real)
            else:
                evals, evecs = np.linalg.eigh(H_full)
            solution_grid[i, j] = evals[0]

    basis = model.optimize(
        solution_grid=(
            solution_grid,
            mu,
        ),  # Comment this out if no solution grid is desired
        svd_tolerance=svd_tol,
        residue_threshold=res_thresh,
        processes=1
    )
    print("Basis Size", basis.shape[1])

    for i in range(basis.shape[1]):
        bv = basis[:, i]
        print(f"Basis vector {i}")
        occ = 0.0
        for j in range(2*N):
            for k in range(2**(2*N)):
                if(k & (0x1 << j)):
                    occ += bv[k] * bv[k].conj()
        print(f"occ = {occ}")

    ### Testing the surrogate model against random parameters
    errors = []
    all_ps = []
    for i in range(200):
        H_full = np.zeros((model.size, model.size), dtype=complex)

        if model_type == "TFIM":
            J = 2 * np.random.randn()
            h = 2 * np.random.randn()
            model_paulis = model_to_paulis(
                N,
                model_type,
                {
                    "J": J,
                    "h": h,
                    "periodic": False,
                },
            )
        elif model_type == "TFXY":
            Jx = 2 * np.random.randn()
            Jy = Jx
            h = 2 * np.random.randn()
            model_paulis = model_to_paulis(
                N,
                model_type,
                {
                    "Jx": Jx,
                    "Jy": Jy,
                    "h": h,
                    "periodic": False,
                },
            )
        elif model_type == "heisenberg":
            Jx = 2 * np.random.randn()
            Jy = Jx
            Jz = 2 * np.random.randn()
            h = 2 * np.random.randn()
            model_paulis = model_to_paulis(
                N,
                model_type,
                {
                    "Jx": Jx,
                    "Jy": Jy,
                    "Jz": Jz,
                    "h": h,
                    "periodic": False,
                },
            )
        elif model_type == "fermi_hubbard":
            model_paulis = model_to_paulis(
                N,
                model_type,
                {
                    "t": 2 * np.random.randn(),
                    "mu": 2 * np.random.randn(),
                    "U": U,
                },
            )
        elif model_type == "AIM":
            vb_test = np.array(
                [0.01] * ((NB) % 2) + [2 * np.random.randn()] * (NB - (NB) % 2)
            )
            eb_r = 2.0 * np.random.randn()
            eb_test = np.array(
                [0.0] * ((NB) % 2)
                + [eb_r] * ((NB - (NB) % 2) // 2)
                + [-eb_r] * ((NB - (NB) % 2) // 2)
            )
            print("vb_test", vb_test)
            print("eb_test", eb_test)
            model_paulis = model_to_paulis(
                N,
                model_type,
                {
                    "NI": NI,
                    "NB": NB,
                    "U": U,
                    "ei": [0.0] * NI,
                    "vb": vb_test,
                    "eb": eb_test,
                    "mu": U / 2,
                    "periodic": False,
                },
            )

        paulis_dict = {}
        for t in model_paulis:
            paulis_dict[t[0]] = t[1]
        parameters = paulis_dict

        for pauli in model.H_terms.keys():
            H_full += parameters[pauli] * model.H_terms[pauli]

        if model.sparse:
            evals, evecs = sps.linalg.eigsh(H_full.real)
        else:
            evals, evecs = np.linalg.eigh(H_full)

        # print(parameters)
        print("Real", evals[0])
        print("Approx", model.solve(parameters))
        if abs(evals[0]) < 1e-12:
            errors.append(np.abs(evals[0] - model.solve(parameters)))
        else:
            errors.append(np.abs(evals[0] - model.solve(parameters)) / np.abs(evals[0]))
        print(
            "Relative Error",
            errors[-1],
        )
        print()
        all_ps.append(parameters)
    print("Basis Size:", basis.shape[1])
    print("Full Hilbert Size:", model.size)
    plt.plot(errors, "o-")
    plt.xlabel("Test Case")
    plt.ylabel("Relative Error")
    plt.title("Surrogate Model Relative Errors")
    plt.yscale("log")
    plt.ylim(1e-20, 1)
    # plt.xticks(
    #     range(len(errors)),
    #     [f"({p[0]:.2f}, {p[n2b_terms + 1]:.2f})" for p in all_ps],
    #     rotation=90,
    # )
    plt.show()
