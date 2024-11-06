import numpy as np
from barbell import barbell
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def initialize_conductances(knn_distances):
    k_0 = 0.0005  # base thermal conductivity
    conductances = k_0 / knn_distances
    return conductances

def melting(points_per_second=1000, pixels_per_m=2000, k=4):
    dt = 1 / points_per_second
    T_0 = 300  #System temperature

    knn_indices, knn_distances, pixel_coords, point_labels, T = melting_setup(k, T_0, pixels_per_m)
    num_nodes = len(knn_indices)
    conductances = initialize_conductances(knn_distances)
    
    laplacian = construct_laplacian_matrix(knn_indices, conductances)

    #Radiation loss over all nodes
    epsilon = 0.5  #Emissivity
    sigma = 5.67e-8  #Stefan-Boltzmann constant
    h_rad = 4 * sigma * epsilon * T_0 ** 3

    #Conductive loss at edges
    edge_nodes = (point_labels == 2).astype(float)
    internal_nodes = np.where(point_labels != 2)[0]
    h_edge = 0.2 #heat transfer coefficient
    h_conv = h_edge * edge_nodes

    h_total = h_rad + h_conv
    b = h_total * T_0
    heat_loss_matrix = diags(h_total)

    #Combined matrix
    A = laplacian + heat_loss_matrix

    #Crank-Nicholson matrices
    I = identity(num_nodes)
    lhs_matrix = I + (dt / 2) * A
    rhs_matrix = I - (dt / 2) * A

    Q_melt = 100
    melting_nodes = np.random.choice(internal_nodes, size=num_nodes, replace=True)
    T_max = []
    variance_list = []
    simulation_steps = int(1 * num_nodes)

    for n in range(simulation_steps):
        S = np.zeros(num_nodes)
        #melt_node = internal_nodes[n]
        melt_node = melting_nodes[n]
        S[melt_node] = Q_melt
        S_total = S + b

        # Right-hand side for Crank-Nicolson
        rhs = rhs_matrix.dot(T) + dt * S_total

        # Solve the linear system
        T = spsolve(lhs_matrix, rhs)

        T_max.append(np.max(T))
        variance_list.append(np.var(T))
    
    visualize_T_over_time(simulation_steps, T_max, variance_list)
    visualize_temperature_2D(pixel_coords, T)
    return T

def construct_adjacency_matrix(knn_indices, conductances):
    row_indices = []
    col_indices = []
    data = []
    num_nodes = len(knn_indices)
    for i in range(num_nodes):
        neighbors = knn_indices[i]
        conductance_values = conductances[i]
        for idx, j in enumerate(neighbors):
            row_indices.append(i)
            col_indices.append(j)
            data.append(conductance_values[idx])
            row_indices.append(j)
            col_indices.append(i)
            data.append(conductance_values[idx])

    adjacency_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))
    return adjacency_matrix

def construct_degree_matrix(conductances):
    degrees = [np.sum(conductance_row) for conductance_row in conductances]
    degree_matrix = diags(degrees, format="csr")
    return degree_matrix

def construct_laplacian_matrix(knn_indices, conductances):
    degree_matrix = construct_degree_matrix(conductances)
    adjacency_matrix = construct_adjacency_matrix(knn_indices, conductances)
    return degree_matrix - adjacency_matrix

def melting_setup(k, T_0, pixels_per_m):
    knn_indices, knn_distances, pixel_coords, point_labels = barbell.build_knn_graph(k, pixels_per_m)
    T = np.zeros(len(pixel_coords)) + T_0
    return knn_indices, knn_distances, pixel_coords, point_labels, T

def visualize_temperature_2D(pixel_coords, temperatures):
    x_coords = pixel_coords[:, 0]
    y_coords = pixel_coords[:, 1]
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(x_coords, y_coords, c=temperatures, cmap='hot', s=1)
    plt.colorbar(scatter, label='Temperature')
    plt.axis('equal')
    plt.show()

def visualize_T_over_time(simulation_steps, T_max, variance_list):
    # Plotting T_max and variance_list on the same plot with different y-axes
    fig, ax1 = plt.subplots()

    # Primary y-axis (for T_max)
    ax1.plot(range(simulation_steps), T_max, color='b', label='Max Temperature')
    ax1.set_xlabel('Simulation Step')
    ax1.set_ylabel('Max Temperature', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Secondary y-axis (for variance)
    ax2 = ax1.twinx()
    ax2.plot(range(simulation_steps), variance_list, color='r', label='Temperature Variance')
    ax2.set_ylabel('Temperature Variance', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Adding legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title("Max Temperature and Temperature Variance Over Time")
    plt.show()
