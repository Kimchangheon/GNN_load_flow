import matplotlib.pyplot as plt
import numpy as np
import math
import time
def plot_random_connections(bus_number, lines_connected):
    # Create coordinates for the buses (points)

    angles = np.linspace(0, 2 * np.pi, bus_number, endpoint=False)

    x = np.cos(angles)

    y = np.sin(angles)

    # Create a plot

    plt.figure(figsize=(8, 8))

    plt.scatter(x, y, color='blue')  # Plot the points

    # Label each bus

    for i in range(bus_number):
        plt.text(x[i], y[i], f'Bus {i + 1}', horizontalalignment='right', verticalalignment='bottom')

    # Draw lines for the connections and label them

    connection_index = 0

    for i in range(bus_number):

        for j in range(i + 1, bus_number):

            if lines_connected[connection_index] == 1:
                plt.plot([x[i], x[j]], [y[i], y[j]], color='gray')

                mid_x, mid_y = (x[i] + x[j]) / 2, (y[i] + y[j]) / 2

                plt.text(mid_x, mid_y, f'L {i + 1}-{j + 1}', color='red')

            connection_index += 1

    # Improve plot appearance

    plt.gca().set_aspect('equal', adjustable='box')

    plt.title(f'Zufällige Verbindungen zwischen {bus_number} Bussen')

    plt.xticks([])

    plt.yticks([])

    plt.show()


def is_bus_one_connected_to_all_others(bus_number, lines_connected):

    # Create an adjacency matrix to represent connections

    adjacency_matrix = np.zeros((bus_number, bus_number), dtype=int)



    # Fill the adjacency matrix based on the lines_connected array

    connection_index = 0

    for i in range(bus_number):

        for j in range(i+1, bus_number):

            if lines_connected[connection_index] == 1:

                adjacency_matrix[i][j] = adjacency_matrix[j][i] = 1

            connection_index += 1



    # Perform a Breadth-First Search (BFS) to check connectivity from Bus 1 to all other buses

    visited = [False] * bus_number

    queue = [0]  # Start from Bus 1 (index 0)



    while queue:

        current_bus = queue.pop(0)

        visited[current_bus] = True

        for i in range(bus_number):

            if adjacency_matrix[current_bus][i] == 1 and not visited[i]:

                queue.append(i)



    # Check if all buses have been visited

    return all(visited)

def create_adjacency_matrix(bus_number, lines_connected):
    # Create an adjacency matrix to represent connections
    adjacency_matrix = np.zeros((bus_number, bus_number), dtype=int)

    # Fill the adjacency matrix based on the lines_connected array
    connection_index = 0
    for i in range(bus_number):
        for j in range(i+1, bus_number):
            if lines_connected[connection_index] == 1:
                adjacency_matrix[i][j] = adjacency_matrix[j][i] = 1
            connection_index += 1

    return adjacency_matrix

def insert_values_in_matrix(matrix, connections, values):
    # Create a copy of the matrix to avoid modifying the original
    matrix_with_values = np.array(matrix, dtype=float)
    new_values = connections * values
    value_index = 0
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):
            if matrix[i][j] == 1:
                # Insert the value regardless of the connection array (since the matrix itself defines the connections)
                matrix_with_values[i][j] = matrix_with_values[j][i] = values[value_index]
            value_index += 1

    return matrix_with_values

import numpy as np

def insert_values_in_matrix_komplex(matrix, connections, values):
    # Create a copy of the matrix to avoid modifying the original
    matrix_with_values = np.array(matrix, dtype=complex)
    new_values = connections * values
    value_index = 0
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):
            if matrix[i][j] == 1:
                # Insert the complex value regardless of the connection array (since the matrix itself defines the connections)
                matrix_with_values[i][j] = matrix_with_values[j][i] =values[value_index]
            value_index += 1

    return matrix_with_values


def build_Y_matrix(matrix, Y_C_Bus, Line_matrix):
    # Create a copy of the matrix to avoid modifying the original
    Y_matrix = np.zeros_like(matrix, dtype=complex)
    value_index = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j:
                Y_matrix[i][i] = Y_matrix[i][i] + 1j * Y_C_Bus[i]
            else:
                if matrix[i][j] == 1:
                    Y_matrix[i][j] = Y_matrix[j][i] = -Line_matrix[i][j]
                    Y_matrix[i][i] = Y_matrix[i][i] + Line_matrix[i][j]
            value_index += 1

    return Y_matrix

def create_bus_typ(busnummer, fixed):
    bus_typ = np.zeros(busnummer, dtype=int)
    bus_typ[0] = 1  # Die erste Stelle ist immer 1 (Slack)

    if busnummer > 1:
        for i in range(1, busnummer):
            if fixed:
                np.random.seed(busnummer*i)

            bus_typ[i] = np.random.choice([2, 3])



    return bus_typ

def generate_PQ(bus_typ):
    num_buses = len(bus_typ)
    P = np.zeros(num_buses)
    Q = np.zeros(num_buses)
    np.random.seed(time.time_ns() % (2 ** 32))
    for i in range(num_buses):
        if bus_typ[i] == 1:  # Slack
            P[i] = 0
            Q[i] = 0
        elif bus_typ[i] == 2:  # PV
            P[i] = - np.round(np.random.uniform(0, 50, 1)).astype(int)
            Q[i] = 0
        elif bus_typ[i] == 3:  # PQ
            P[i] = - np.round(np.random.uniform(0, 50, 1)).astype(int)
            Q[i] = - np.round(np.random.uniform(0, 50, 1)).astype(int)

    # Skalieren Sie die Werte auf Millionen (1e6)

    return P, Q

def generate_u_start_for_buses(bustype):
    u_start = np.zeros_like(bustype, dtype=complex)  # Initialisiert u_start mit Nullen und dem gleichen Datentyp wie bustype
    np.random.seed(time.time_ns() % (2 ** 32))
    for i, bus_typ in enumerate(bustype):
        if bus_typ == 1:
            u_start[i] =  np.round(np.random.uniform(0.9, 1.1), 2) + 0j
        elif bus_typ == 2:
            u_start[i] = np.round(np.random.uniform(0.9, 1.1), 2) + 0j
        elif bus_typ == 3:
            # Für Bus-Typ 3: Wert ist immer 1.0
            u_start[i] = 1.0 ++ 0j
        else:
            raise ValueError(f"Ungültiger Bus-Typ für Bus {i + 1}. Die Bus-Typen sind 1, 2 oder 3.")

    return u_start


def case_generation(Bus_number, fixed, debugging, pic, U_Grid, S_Base):
    ######
    if debugging:
        Bus_number = 5



    Z_Base = U_Grid ** 2 / (S_Base)
    Y_Base = 1 / Z_Base

    # Example usage
    num_connections = math.comb(Bus_number, 2)

    # Definieren Sie den Bereich für die Real- und Imaginärteile
    min_real = 0.01
    max_real = 0.3

    if debugging:
        Z_Lines = np.array(
            [0.02 + 0.06 * 1j, 0.08 + 0.24 * 1j, 0, 0, 0.06 + 0.18 * 1j, 0.06 + 0.18 * 1j, 0.04 + 0.12 * 1j,
             0.01 + 0.03 * 1j, 0, 0.08 + 0.24 * 1j], dtype=complex)

    else:
        if fixed: # want to keep the impedance values consistent for a given network size (say 16 buses), you set fixed=True.
            np.random.seed(num_connections)
        Z_Lines = np.random.uniform(min_real, max_real, num_connections) + 1j * np.random.uniform(min_real * 3,
                                                                                                  max_real * 3,
                                                                                                  num_connections)

    Z_Lines = Z_Lines * Z_Base

    Y_Lines = np.zeros_like(Z_Lines, dtype=complex)  # Erzeugen eines leeren Arrays mit derselben Form wie Z_Lines

    for i in range(len(Z_Lines)):
        if Z_Lines[i] != 0:
            Y_Lines[i] = 1 / (Z_Lines[i])
        else:
            Y_Lines[i] = 0
    np.random.seed(time.time_ns() % (2 ** 32))
    if debugging:
        Lines_connected = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 1])
        print("Lines_conected ", Lines_connected)

    else:
        Lines_connected = np.random.randint(2, size=num_connections)

    if debugging:
        plot_random_connections(Bus_number, Lines_connected)

    is_connected = is_bus_one_connected_to_all_others(Bus_number, Lines_connected) #Bus 1 (index 0) can reach every other bus

    Conection_matrix = create_adjacency_matrix(Bus_number, Lines_connected)

    #print(Conection_matrix)

    if debugging:
        Y_C_Lines = np.array([0.06, 0.05, 0, 0, 0.04, 0.04, 0.03, 0.02, 0, 0.05], dtype=float) / 1 / Z_Base
        print("Lines_conected ", Lines_connected)

    else:
        if fixed:
            np.random.seed(num_connections)

        Y_C_Lines = np.random.uniform(0.01, 0.10, size=num_connections) / 1 / Z_Base

    Y_C_Bus = np.sum(insert_values_in_matrix(Conection_matrix, Lines_connected, Y_C_Lines), axis=0)

    Line_matrix = insert_values_in_matrix_komplex(Conection_matrix, Lines_connected, Y_Lines)

    Y_matrix = build_Y_matrix(Conection_matrix, Y_C_Bus, Line_matrix)
    #print(np.round(Y_matrix, 3))
    if debugging:
        bus_typ = np.array([
            1,  # Slack
            2,  # PV
            3,  # PQ
            3,
            3
        ])
        """bus_typ = np.array([
            1,  # Slack
            2,  # PV
            3,  # PQ
            3,
            3
        ]) 
    """
    else:
        bus_typ = create_bus_typ(Bus_number, fixed)

    if debugging:
        P = np.array([0, -50, -65, -35, -45])
        Q = np.array([0, 0, -10, -10, -15])
    else:
        P, Q = generate_PQ(bus_typ)

    P, Q = P * 1e6, Q * 1e6

    if debugging:
        u_start = np.array([
            1.1 + 0j,
            1.0 + 0j,
            1.0 + 0j,
            1.0 + 0j,
            1.0 + 0j
        ])
    else:
        u_start = generate_u_start_for_buses(bus_typ)

    u_start = u_start * U_Grid

    s_multi = P + 1j * Q

    if pic:
        plot_random_connections(Bus_number, Lines_connected)

    return bus_typ, s_multi, u_start, Y_matrix, is_connected, Y_Lines, Y_C_Lines, Lines_connected


