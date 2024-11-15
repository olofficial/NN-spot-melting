import numpy as np
from barbell import barbell
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import bicgstab, spilu, LinearOperator
import matplotlib.pyplot as plt
import pyamg

def initialize_conductances(knn_distances, pixels_per_m):
    k_0 = 170 # base thermal conductivity
    conductances = k_0 / knn_distances
    return conductances

def melting_setup(pixels_per_m, points_per_second=1000, k=4, T_init = None, melting_sequence = None):
    dt = 1 / points_per_second
    T_0 = 300  #System temperature
    knn_indices, knn_distances, pixel_coords, point_labels = barbell.build_knn_graph(k, pixels_per_m)
    if T_init == None:
        T_init = np.zeros(len(knn_indices)) + T_0
    num_nodes = len(knn_indices)
    conductances = initialize_conductances(knn_distances, pixels_per_m)
    
    laplacian = construct_laplacian_matrix(knn_indices, conductances)

    #Radiation loss over all nodes
    epsilon = 0.3  #Emissivity
    sigma = 5.67e-8  #Stefan-Boltzmann constant
    h_rad = 4 * sigma * epsilon * T_0 ** 3

    #Conductive loss at edges
    edge_nodes = (point_labels == 2).astype(float)
    internal_nodes = np.where(point_labels != 2)[0]
    h_edge = 200 #heat transfer coefficient
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

    Q_melt = 2e13 / pixels_per_m ** 2
    if melting_sequence == None:
        melting_sequence = np.random.choice(internal_nodes, size=num_nodes, replace=True)
    simulation_steps = int(1 * len(internal_nodes))

    simulation_setup = {
        'lhs_matrix': lhs_matrix,
        'rhs_matrix': rhs_matrix,
        'T_0': T_0,
        'dt': dt,
        'num_nodes': num_nodes,
        'internal_nodes': internal_nodes,
        'Q_melt': Q_melt,
        'b': b,
        'knn_indices': knn_indices,
        'pixel_coords': pixel_coords,
        'heat_loss_matrix': heat_loss_matrix,
        'simulation_steps': simulation_steps,
        'T_init' : T_init,
        'melting_sequence': melting_sequence
    }
    
    return simulation_setup

def compute_melting(simulation_setup, melting_sequence = None, plot_melting = False, log_interval = 1):
    simulation_steps = simulation_setup['simulation_steps']
    num_nodes = simulation_setup['num_nodes']
    if melting_sequence.any() == 0:
        melting_sequence = simulation_setup['melting_sequence']
    rhs_matrix = simulation_setup['rhs_matrix']
    lhs_matrix = simulation_setup['lhs_matrix']
    b = simulation_setup['b']
    dt = simulation_setup['dt']
    internal_nodes = simulation_setup['internal_nodes']
    Q_melt = simulation_setup['Q_melt']
    T_init = simulation_setup['T_init']
    pixel_coords = simulation_setup['pixel_coords']

    T = T_init.copy()
    
    T_max = np.zeros(simulation_steps)
    variance_list = np.zeros(simulation_steps)

    ilu = spilu(lhs_matrix.tocsc(), fill_factor=1)  # Minimal fill-in for near-tridiagonal structure
    M_ilu = LinearOperator(lhs_matrix.shape, ilu.solve)
    
    for n in range(simulation_steps):
        S = np.zeros(num_nodes)
        #melt_node = internal_nodes[n]
        melt_node = melting_sequence[n]
        S[melt_node] = Q_melt
        S_total = S + b

        # Right-hand side for Crank-Nicolson
        rhs = rhs_matrix.dot(T) + dt * S_total

        # Solve the linear system
        T, _ = bicgstab(lhs_matrix, rhs, x0 = T, tol = 1e-4, maxiter = 100, M = M_ilu)
        T_max[n] = np.max(T[internal_nodes])
        variance_list[n] = np.var(T[internal_nodes])
    variance_list = np.log(variance_list)
    variance_of_variances = np.mean(variance_list)
    if plot_melting:
        visualize_T_over_time(simulation_steps, T_max, variance_list)
        visualize_temperature_2D(pixel_coords, T)
    
    return variance_of_variances, variance_list

def construct_adjacency_matrix(knn_indices, conductances):
    num_nodes = len(knn_indices)
    row_indices = np.repeat(np.arange(num_nodes), knn_indices.shape[1])
    col_indices = knn_indices.flatten()
    data = conductances.flatten()
    adjacency_matrix = csr_matrix((data, (row_indices, col_indices)), shape = (num_nodes, num_nodes))
    return adjacency_matrix

def construct_degree_matrix(adjacency_matrix):
    degrees = np.array(adjacency_matrix.sum(axis = 1)).flatten()
    degree_matrix = diags(degrees, format='csr')
    return degree_matrix

def construct_laplacian_matrix(knn_indices, conductances):
    adjacency_matrix = construct_adjacency_matrix(knn_indices, conductances)
    degree_matrix = construct_degree_matrix(adjacency_matrix)
    return degree_matrix - adjacency_matrix

def visualize_temperature_2D(pixel_coords, temperatures):
    x_coords = pixel_coords[:, 0]
    y_coords = pixel_coords[:, 1]
    plt.figure(figsize=(8, 8))
    #plt.gca().set_facecolor('black']
    scatter = plt.scatter(x_coords, y_coords, c=temperatures, cmap='hot', s=16, marker='s')
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
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.show()
