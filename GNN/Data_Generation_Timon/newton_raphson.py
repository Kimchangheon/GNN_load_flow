import numpy as np
import traceback
def VoltageCalculation3p(bus_typ, JMatrix, Y_admittance, U, P, Q):
    N, M = Y_admittance.shape

    I = Y_admittance @ U  # Currents
    Ss = np.diag(U) @ np.conj(I)  # Complex powers

    PQs = np.concatenate((np.real(Ss), np.imag(Ss)))  # Aufteilung des Vektors auf Wirkleistungen und Blindleistungen

    # Slack erkennen
    Slack_indices = np.where(bus_typ == 1)[0]

    # Slack entfernen
    if np.any(bus_typ == 1):
        # Stellen im anderen Array löschen
        PQs = np.delete(PQs, Slack_indices + N)
        PQs = np.delete(PQs, Slack_indices)
        P = np.delete(P, Slack_indices)
        Q = np.delete(Q, Slack_indices)
        JMatrix = np.delete(JMatrix, Slack_indices + N, axis=1)
        JMatrix = np.delete(JMatrix, Slack_indices + N, axis=0)
        JMatrix = np.delete(JMatrix, Slack_indices, axis=1)
        JMatrix = np.delete(JMatrix, Slack_indices, axis=0)

    # PQs_Test = np.concatenate((np.real(Ss[1:N]), np.imag(Ss[1:N])))
    # PV Bus erkennen
    PV_indices = np.where(bus_typ == 2)[0]
    #print("PV_indices",PV_indices)
    # Q_save = Q[indices]
    # print(Q_save)

    #print("JMatrix",JMatrix)
    #print("PQs",PQs)
    #print("P",P)
    #print("Q",Q)
    if np.any(bus_typ == 2):
        PQs = np.delete(PQs, PV_indices - 2 * len(Slack_indices) + N) # Q can be solved after convergence, so we drop it even if it's unkown)
        JMatrix = np.delete(JMatrix, PV_indices + N - 2 * len(Slack_indices), axis=1)
        JMatrix = np.delete(JMatrix, PV_indices + N - 2 * len(Slack_indices), axis=0)
        Q = np.delete(Q, PV_indices - len(Slack_indices))

    #print("JMatrix", JMatrix)
    #print("PQs", PQs)
    #print("P", P)
    #print("Q", Q)

    delta = (np.concatenate((P, Q)) - PQs)  #Difference between the given outputs and the calculated outputs – interpreted as a step of the algorithm.

    # print(delta)
    try:
        np.linalg.inv(JMatrix)
    except np.linalg.LinAlgError:
        # deltaU = np.full(((N-1)*2), 1) # diskutieren
        # print(deltaU)
        print("Fehler")
        # return None
    else:
        deltaU = np.linalg.inv(JMatrix) @ delta
        # print(deltaU)
    # Formel 3.31


    #deltaU = np.insert(deltaU, PV_indices + N - 2, 0)

    for position in PV_indices:
       deltaU = np.insert(deltaU, position+ N-2, 0)

    # # 🔒 PV-MAG LOCK
    # for position in PV_indices:
    #     mag_slot = position + N - 1
    #     deltaU[mag_slot] = 0.0

    #print("deltaU",deltaU)
    for k in range(N - 1): # suppose first bus is slack
        U[k + 1] = (np.abs(U[k + 1]) + deltaU[k + N - 1]) * np.exp(1j * np.angle(U[k + 1]) + 1j * deltaU[k])
        #test = deltaU[k + N - 1]
        # Änderung zu Script von czarek, Taylorreihe

    # Ss=np.insert(deltaU, indices, Q_save)
    Us = U.copy()
    S = Ss.copy()

    # print(abs(Us))
    return Us, S  # Berechnung der neuen Spannungen


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


# Parameter für die Lastflussberechnung
def newtonrapson(bus_typ, Y_system, s_L, U):

    try:
        #These are the specified (target) powers for every bus.
        S = s_L # given (specified) complex injections # S_start
        P = np.real(S) # active-power set-points
        Q = np.imag(S) # reactive-power set-points

        N, M = Y_system.shape

        # save the privous u_jacobi to compare the difference
        u_jacobi = np.empty([N, ], dtype=complex)
        print(f'{N}- Bus', end='', flush=True)
        for iter in range(40):  # Schrittverfahren
            J, J1, J2, J3, J4 = JacobianMatrix3p(Y_system, U)
            print('.', end='', flush=True)

            #Newton–Raphson corrects the voltages
            # Us : updated bus-voltage phasors after one NR correction step
            #S1 the apparent power injections calculated with the previous voltage guess
            Us, S1 = VoltageCalculation3p(bus_typ, J, Y_system, U, P, Q)
            U = Us
            u_jacobi = np.vstack((u_jacobi, U))

            if len(u_jacobi) >= 3:
                if max(abs(u_jacobi[-1] - u_jacobi[-2])) < 0.0005:
                    if max(abs(u_jacobi[-2] - u_jacobi[-3])) < 0.0005:
                        print("|convergence successful|")
                        break
        else:

            u_jacobi[-1, :] = u_jacobi[-2, :] * 0
            #print(u_jacobi[-1, :])
            print("non-convergence")



        I = Y_system @ U  # Berechnete Ströme in Knoten bus-current phasors I = YV
        S = np.diag(U) @ np.conj(I) #complex power injections S = V I^{*}
        #print(S/ 1e6)
        #print(I)
        u_final = u_jacobi[-1, :] # u_final so it can be returned. (Same as U)
        #print("u_jacobi", u_jacobi[-1, :])
    except Exception as e:
        print(f"Fehler in newtonrapson: {str(e)}")
        traceback.print_exc()

        u_final =[]
        I = []
        S = []

    return u_final, I, S
