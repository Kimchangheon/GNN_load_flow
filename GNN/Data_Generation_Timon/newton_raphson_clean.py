import numpy as np

# ------------------------------------------------------------------
# helper: build “reduced” mismatch vector and Jacobian
# ------------------------------------------------------------------
def build_mismatch_and_jacobian(bus_type, Y, V, P_spec, Q_spec):
    """
    bus_type : 1-slack, 2-PV, 3-PQ   (length N)
    Y        : full complex Y-bus    (N×N)
    V        : complex voltages      (N,)
    P_spec,Q_spec : target active / reactive injections (N,)

    Returns
    --------
    F      : concatenated mismatch [ΔP ; ΔQ] with slack rows dropped and
             PV-Q rows dropped             (length n_red)
    J_red  : correspondingly reduced Jacobian               (n_red × n_red)
    pv_mag_mask : Boolean mask pointing to the |V|-columns of PV buses
                  inside the FULL Jacobian (used later to zero Δ|V|).
    """
    N = len(V)
    # ---- calculated powers ------------------------------------------------
    I = Y @ V
    S_calc = V * np.conj(I)
    P_calc =  S_calc.real
    Q_calc =  S_calc.imag

    dP = P_spec - P_calc                 # ΔP, ΔQ (length N)
    dQ = Q_spec - Q_calc

    # ---- full 2N × 2N Jacobian -------------------------------------------
    J = full_jacobian(Y, V)              # user-supplied or from your J1..J4

    # ---- masks for rows / cols -------------------------------------------
    slack = (bus_type == 1)
    pv    = (bus_type == 2)
    pq    = (bus_type == 3)

    # indices for angle part 0..N-1 ,  magnitude part N..2N-1
    ang_idx = np.arange(N)
    mag_idx = np.arange(N) + N

    rows_keep = np.r_[                     # keep angle rows of PV,PQ
        ang_idx[~slack],                   #   (all but slack)
        mag_idx[pq]                        # keep |V| rows of PQ only
    ]

    # same for columns
    cols_keep = np.r_[ang_idx[~slack], mag_idx[pq]]

    # mask that lives in the full (2N) space → True where |V| of a PV bus
    pv_mag_mask_full = np.zeros(2*N, dtype=bool)
    pv_mag_mask_full[mag_idx[pv]] = True

    # build reduced objects
    F  = np.concatenate([dP[~slack], dQ[pq]])
    Jr = J[np.ix_(rows_keep, cols_keep)]

    return F, Jr, pv_mag_mask_full

# ------------------------------------------------------------------
# main NR solver (flat start, per-unit)
# ------------------------------------------------------------------
def newton_power_flow(bus_type, Y, P_spec, Q_spec,
                      V0=None, tol=1e-6, max_iter=20):
    """
    Returns converged complex voltages V  (length N)
    """
    N = len(bus_type)
    if V0 is None:
        V0 = np.ones(N, dtype=complex)
        V0[bus_type == 1] = 1.06 + 0j   # slack magnitude 1.06 ∠0°

    V = V0.copy()

    for k in range(max_iter):
        F, Jr, pv_mag_mask_full = build_mismatch_and_jacobian(
                                      bus_type, Y, V, P_spec, Q_spec)

        # solve reduced system
        delta_red = np.linalg.solve(Jr, -F)

        # embed back into full Δx (angle followed by |V|)
        delta_full = np.zeros(2*N)
        rows_keep = np.flatnonzero(~(bus_type == 1))          # angle rows kept
        delta_full[rows_keep]      = delta_red[:len(rows_keep)]
        delta_full[np.concatenate([np.flatnonzero(bus_type==3)+N])] \
             = delta_red[len(rows_keep):]

        # ---- lock PV magnitudes -----------------------------------------
        delta_full[pv_mag_mask_full] = 0.0        # <-- the key line

        # ---- update voltages -------------------------------------------
        dtheta = delta_full[:N]
        dVm    = delta_full[N:]

        V = (np.abs(V) + dVm) * np.exp(1j*(np.angle(V) + dtheta))

        # convergence
        if np.max(np.abs(delta_full)) < tol:
            print(f"converged in {k+1} iters")
            return V

    raise RuntimeError("NR did not converge")

# ------------------------------------------------------------------
# > your full_jacobian(Y,V) goes here  (combine J1..J4 or use existing lib)
# ------------------------------------------------------------------
def full_jacobian(Y, V):
    # stub: call your JacobianMatrix3p or a library routine, return (2N×2N)
    J, _, _, _, _ = JacobianMatrix3p(Y, V)
    return J

def JacobianMatrix3p(Y_admittance, U):
    N, M = Y_admittance.shape

    # dP/dU bezeichnet als J2
    J2 = np.zeros((N, M))

    for n in range(
            N):  # Zeilen; Eine Zeile ist weggelassen - ein Knoten entfällt, weil er ein Kompensationsknoten ist (Slack)
        for m in range(
                M):  # Spalten eine Zeile ist weggelassen- ein Knoten entfällt, weil er ein Kompensationsknoten ist (Slack).
            if m == n:  # für die diagonalen Komponenten
                J2[n, m] = 2 * np.abs(U[n]) * np.abs(Y_admittance[n, m]) * np.cos(np.angle(Y_admittance[n, m]))

                for k in range(M):
                    if k != n:
                        J2[n, m] = J2[n, m] + np.abs(Y_admittance[n, k]) * np.abs(U[k]) * np.cos(
                            np.angle(U[n]) - np.angle(U[k]) - np.angle(Y_admittance[n, k]))

            if m != n:  # für undiagonalen Komponenten
                J2[n, m] = np.abs(Y_admittance[n, m]) * np.abs(U[n]) * np.cos(
                    np.angle(U[n]) - np.angle(U[m]) - np.angle(Y_admittance[n, m]))

    # dP/dfi bezeichnet als J1
    J1 = np.zeros((N, M))

    for n in range(N):
        for m in range(M):
            if m == n:
                for k in range(M):
                    if k != n:
                        J1[n, m] = J1[n, m] - np.abs(Y_admittance[n, k]) * np.abs(U[k]) * np.abs(U[n]) * np.sin(
                            np.angle(U[n]) - np.angle(U[k]) - np.angle(Y_admittance[n, k]))

            if n != m:  # für undiagonalen Komponenten
                J1[n, m] = np.abs(Y_admittance[n, m]) * np.abs(U[m]) * np.abs(U[n]) * np.sin(
                    np.angle(U[n]) - np.angle(U[m]) - np.angle(Y_admittance[n, m]))

    # dQ/dU bezeichnet als J4
    J4 = np.zeros((N, M))

    for n in range(N):
        for m in range(M):
            if m == n:
                J4[n, m] = -2 * np.abs(U[m]) * np.abs(Y_admittance[n, m]) * np.sin(np.angle(Y_admittance[n, m]))

                for k in range(M):
                    if k != n:
                        J4[n, m] = J4[n, m] + np.abs(Y_admittance[n, k]) * np.abs(U[k]) * np.sin(
                            np.angle(U[n]) - np.angle(U[k]) - np.angle(Y_admittance[n, k]))

            if m != n:  # für undiagonalen Komponenten
                J4[n, m] = np.abs(Y_admittance[n, m]) * np.abs(U[n]) * np.sin(
                    np.angle(U[n]) - np.angle(U[m]) - np.angle(Y_admittance[n, m]))

    # dQ/df
    # dQ/dfi bezeichnet als J3
    J3 = np.zeros((N, M))

    for n in range(N):
        for m in range(M):
            if m == n:
                for k in range(M):
                    if k != n:
                        J3[n, m] = J3[n, m] + np.abs(Y_admittance[n, k]) * np.abs(U[k]) * np.abs(U[n]) * np.cos(
                            np.angle(U[n]) - np.angle(U[k]) - np.angle(Y_admittance[n, k]))

            if m != n:  # für undiagonalen Komponenten
                J3[n, m] = -np.abs(Y_admittance[n, m]) * np.abs(U[m]) * np.abs(U[n]) * np.cos(
                    np.angle(U[n]) - np.angle(U[m]) - np.angle(Y_admittance[n, m]))

    # J = np.block([[J1[1:N, 1:M], J2[1:N, 1:M]], [J3[1:N, 1:M], J4[1:N, 1:M]]])  # ganze Jacobi - Matrix
    J = np.block([[J1, J2], [J3, J4]])  # ganze Jacobi - Matrix

    # J = np.block([[J2[1:N, 1:M], J1[1:N, 1:M]], [J4[1:N, 1:M], J3[1:N, 1:M]]])  # ganze Jacobi - Matrix
    return J, J1, J2, J3, J4
