import numpy as np
from barbell import barbell
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting toolkit
from matplotlib import cm  # Import colormap
from scipy.interpolate import griddata

def initialize_conductances(knn_distances):
    k_0 = 0.0001  # base thermal conductivity
    conductances = k_0 / knn_distances
    return conductances

def melting(points_per_second=1000, pixels_per_m=2000, k=2):
    dt = 1 / points_per_second
    h = 10000
    T_0 = 300

    knn_indices, knn_distances, pixel_coords, point_labels, T = melting_setup(k, T_0, pixels_per_m)
    num_nodes = len(knn_indices)
    conductances = initialize_conductances(knn_distances)
    
    laplacian = construct_laplacian_matrix(knn_indices, conductances)

    epsilon = 0.1
    b_radiation = np.ones(num_nodes)
    sigma = 5.67e-8  # Stefan-Boltzmann constant
    linearized_radiative_coeff = 4 * sigma * epsilon * T_0 ** 3
    radiative_matrix = diags(linearized_radiative_coeff * b_radiation)

    edge_nodes = (point_labels == 2)
    b = edge_nodes.astype(float)

    internal_nodes = np.where(~edge_nodes)[0]
    melting_nodes = np.random.choice(internal_nodes, size=num_nodes, replace=True)

    Q_melt = 0
    T_max = []
    variance_list = []
    simulation_steps = int(1 * num_nodes)

    # Construct Crank-Nicolson matrices
    I = identity(num_nodes)
    boundary_matrix = diags(h * b)
    A = laplacian + boundary_matrix + radiative_matrix

    # Crank-Nicolson matrices: (I - dt/2 * A) and (I + dt/2 * A)
    lhs_matrix = I - (dt / 2) * A  # Left-hand side matrix
    rhs_matrix = I + (dt / 2) * A  # Right-hand side matrix

    for n in range(simulation_steps):
        S = np.zeros(num_nodes)
        melt_node = melting_nodes[n]
        S[melt_node] = Q_melt

        # Right-hand side for Crank-Nicolson
        rhs = rhs_matrix.dot(T) + dt * S

        # Solve the linear system (I - dt/2 * A) * T_new = rhs
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
