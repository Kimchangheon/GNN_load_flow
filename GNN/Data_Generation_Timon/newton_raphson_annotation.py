import numpy as np
import traceback

# ------------------------------------------------------------
# VoltageCalculation3p
#   • builds reduced mismatch vector ΔP‖ΔQ (slack & PV-Q rows dropped)
#   • solves J ΔU = ΔS
#   • inserts zeros back for the deleted rows / columns
#   • updates the complex-voltage vector U (1 Newton step)
# ------------------------------------------------------------
def VoltageCalculation3p(bus_typ, JMatrix, Y_admittance, U, P, Q):
    N, _ = Y_admittance.shape

    # ---------- 1. currents and complex powers at current guess ----------
    I  = Y_admittance @ U                    # I = Y·V
    Ss = np.diag(U) @ np.conj(I)             # S = V I*

    # vector  [P_calc ; Q_calc]  length = 2N
    PQs = np.concatenate([np.real(Ss), np.imag(Ss)])

    # ---------- 2. drop SLACK rows/cols ----------------------------------
    slack_idx = np.where(bus_typ == 1)[0]  # e.g. array([0])
    if slack_idx.size:  # enter only if slack exists
        # ---------- MISMATCH VECTOR ΔS = [ΔP  ΔQ] ---------------------------
        # Row order in ΔS is  [ΔP1 … ΔPN  ΔQ1 … ΔQN]

        PQs = np.delete(PQs, slack_idx + N)  # ► delete ΔQ_slack   row
        PQs = np.delete(PQs, slack_idx)  # ► delete ΔP_slack   row
        # P_spec, Q_spec hold the targets. Remove slack entries there too:
        P = np.delete(P, slack_idx)  # ► drop P_spec for slack
        Q = np.delete(Q, slack_idx)  # ► drop Q_spec for slack

        # ---------- JACOBIAN  J  (2N × 2N) ----------------------------------
        # Column order: first all ∂/∂δ_i  (angles)  then all ∂/∂|V|_i  (mags)

        # 1) Remove the two **rows** that correspond to the slack equations
        JMatrix = np.delete(JMatrix,slack_idx + N, axis=0)  # ► remove ΔQ_slack  row
        JMatrix = np.delete(JMatrix,slack_idx, axis=0)  # ► remove ΔP_slack  row

        # 2) Remove the two columns that hold the slack variable
        JMatrix = np.delete(JMatrix, slack_idx + N, axis=1)  # ► eliminate |V|_slack column
        JMatrix = np.delete(JMatrix, slack_idx, axis=1)  # ► eliminate δ_slack   column
        # After these deletions the slack bus contributes no equations and no unknowns.

        #       |   dP/dδ  | dP/d|V| |
        # J  =  |----------+---------|
        #       |   dQ/dδ  | dQ/d|V| |
        # row/col order:  P1…PN | Q1…QN   and   δ1…δN | |V|1…|V|N


    # ---------- 3. drop PV bus' Q rows (Q unknown) -----------------------------
    pv_idx = np.where(bus_typ == 2)[0]
    if pv_idx.size:
        # remove Q-mismatch rows for PV buses

        # drop Q : PV-bus Q is unknown and should be solved "after" convergence
        PQs = np.delete(PQs, pv_idx - 2*len(slack_idx) + N) # points to the Q-row of each PV bus after the slack removal.
        Q   = np.delete(Q,   pv_idx - len(slack_idx)) # Deleting those entries means we no longer force “ΔQ=0” for PV buses while iterating.

        #Delete the row of the Jacobian that corresponds to the Q-mismatch equation of each PV bus.
        JMatrix = np.delete(JMatrix, pv_idx + N - 2*len(slack_idx), axis=0)   # remove ROW  ∂Q_PV / ∂(•)

        #Why delete that column? :PV-bus voltage magnitude is fixed at its set-point, so it is not an unknown.
        #By deleting the column we guarantee Newton-Raphson will never propose a correction for |V|PV.
        JMatrix = np.delete(JMatrix, pv_idx + N - 2*len(slack_idx), axis=1)   # remove COLUMN  Δ|V|_PV

    # ---------- 4. build ΔS and solve for ΔU ------------------------------
    deltaS = np.concatenate([P, Q]) - PQs                # mismatch vector  # ΔS = S_spec − S_calc
    deltaU_red = np.linalg.solve(JMatrix, deltaS)        # reduced system, #np.linalg.inv(JMatrix) @ delta

    # ---------- 5. re-insert zeros for the lack of the magnitude slot of each PV bus.--------------------------
    # [ Δδ0 … ΔδN-1 | Δ|V|0 … Δ|V|N-1 ] : inserting a zero magnitude-correction slot so that the reduced solution vector once again has the full 2 N entries
    for pv in pv_idx:
        deltaU_red = np.insert(deltaU_red, pv + N - 2, 0.0) # -2?  because two rows/columns (the slack-bus P and Q) were removed earlier
                        # ▲                     ▲      ▲
                        # |                     |      └── value 0 (no Δ|V|)
                        # |                     └ index where |V|_PV would sit
                        # └ work on the reduced vector

    # ---------- 6. 🔒  ZERO magnitude corrections of PV buses ------------
    #     angle slot = pv,  magnitude slot = pv+N-1 (because one slack row gone)
    for pv in pv_idx:
        mag_slot = pv + N - 1
        deltaU_red[mag_slot] = 0.0           # 🔒 PV-MAG LOCK

    # ---------- 7. update complex voltages --------------------------------
    for k in range(N-1):
        mag   = np.abs(U[k+1]) + deltaU_red[k+N-1]       # |V| + Δ|V|
        angle = np.angle(U[k+1]) + deltaU_red[k]         # θ + Δθ
        U[k+1] = mag * np.exp(1j * angle)

    return U.copy(), Ss.copy()


# ------------------------------------------------------------
# JacobianMatrix3p (unchanged – long but standard)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# newtonrapson : orchestrates the iterations
# ------------------------------------------------------------
def newtonrapson(bus_typ, Y_system, S_start, U_start):
    try:
        # s_L given (specified) complex injections
        P_start, Q_start = np.real(S_start), np.imag(S_start)
        U = U_start.copy()

        traj = [U.copy()]                      # to monitor convergence

        for it in range(40):
            J, *_ = JacobianMatrix3p(Y_system, U)

            # U : updated bus-voltage phasors after one NR correction step
            U, _  = VoltageCalculation3p(bus_typ, J, Y_system, U,
                                         P_start.copy(), Q_start.copy())
            traj.append(U.copy())

            # simple stopping test (ΔV in volts) --------------------------
            if it >= 2:
                if np.max(np.abs(traj[-1]-traj[-2])) < 5e-4 \
                and np.max(np.abs(traj[-2]-traj[-3])) < 5e-4:
                    print("converged")
                    break
        else:
            print("did not converge")

        I = Y_system @ U
        S = np.diag(U) @ np.conj(I) #complex power injections S = V I^{*}
        return U, I, S

    except Exception as e:
        print("NR error:", e)
        traceback.print_exc()
        return [], [], []