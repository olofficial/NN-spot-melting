import numpy as np
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import bicgstab, spilu, LinearOperator
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.ndimage import binary_dilation
from sklearn.neighbors import KDTree as SKKDTree

def grid_barbell(spots_per_m):
    # Define geometry
    ball_radius = 0.01
    handle_length = 0.01
    handle_thickness = 0.002
    ball_centers = np.array([-(ball_radius + handle_length / 2), (ball_radius + handle_length / 2)])

    # spotation
    total_width = ball_radius * 4 + handle_length
    total_height = ball_radius * 2
    spot_width = 1 / spots_per_m
    spots_x = int(total_width * spots_per_m + 2)
    spots_y = int(total_height * spots_per_m + 2)

    # Grid
    x_values = np.linspace(-(total_width / 2 + spot_width), (total_width / 2 + spot_width), spots_x)
    y_values = np.linspace(-(total_height / 2 + spot_width), (total_height / 2 + spot_width), spots_y)
    X, Y = np.meshgrid(x_values, y_values)

    # Barbell shape
    barbell_shape = ((X - ball_centers[0])**2 + Y**2 < ball_radius**2) | \
                    ((X - ball_centers[1])**2 + Y**2 < ball_radius**2) | \
                    ((np.abs(X) <= handle_length / 2) & (np.abs(Y) <= handle_thickness / 2))

    barbell = np.zeros_like(X, dtype=np.uint8)
    barbell[barbell_shape] = 1

    # Edge points
    expanded_barbell = binary_dilation(barbell_shape)
    edge_points = expanded_barbell & (~barbell_shape)
    barbell[edge_points] = 2

    return barbell, X, Y

def build_kdtree(spot_list):
    # Using Scikit-learn's KDTree for efficiency
    tree = SKKDTree(spot_list, leaf_size=30, metric='euclidean')
    return tree

def build_knn_graph(k, spots_per_m):
    # Build barbell shape
    barbell, X, Y = grid_barbell(spots_per_m)
    barbell_indices = np.where(barbell > 0)
    x_coords = X[barbell_indices]
    y_coords = Y[barbell_indices]
    spot_coords = np.column_stack((x_coords, y_coords))
    point_labels = barbell[barbell_indices]

    # Build KNN graph using Scikit-learn's KDTree
    tree = build_kdtree(spot_coords)
    distances, indices = tree.query(spot_coords, k=k + 1)  # k+1 because the first neighbor is itself

    knn_indices = indices[:, 1:]
    knn_distances = distances[:, 1:]

    return knn_indices, knn_distances, spot_coords, point_labels


def initialize_conductances(knn_distances):
    k_0 = 170 
    conductances = k_0 / knn_distances
    return conductances

def melting_setup(spots_per_m, points_per_second=1000, k=4, T_init = None, melting_sequence = None):
    dt = 1 / points_per_second
    T_0 = 300  
    knn_indices, knn_distances, spot_coords, point_labels = build_knn_graph(k, spots_per_m)
    if T_init == None:
        T_init = np.zeros(len(knn_indices)) + T_0
    num_nodes = len(knn_indices)
    conductances = initialize_conductances(knn_distances)
    
    laplacian = construct_laplacian_matrix(knn_indices, conductances)

    epsilon = 0.3 
    sigma = 5.67e-8 
    h_rad = 4 * sigma * epsilon * T_0 ** 3

    edge_nodes = (point_labels == 2).astype(float)
    internal_nodes = np.where(point_labels != 2)[0]
    h_edge = 200 
    h_conv = h_edge * edge_nodes

    h_total = h_rad + h_conv
    b = h_total * T_0
    heat_loss_matrix = diags(h_total)

    A = laplacian + heat_loss_matrix

    I = identity(num_nodes)
    lhs_matrix = I + (dt / 2) * A
    rhs_matrix = I - (dt / 2) * A

    Q_melt = 2e13 / spots_per_m ** 2
    if melting_sequence == None:
        melting_sequence = np.random.choice(internal_nodes, size=num_nodes, replace=True)
        simulation_steps = int(1 * len(internal_nodes))
    else:
        simulation_steps = len(melting_sequence)

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
        'spot_coords': spot_coords,
        'heat_loss_matrix': heat_loss_matrix,
        'simulation_steps': simulation_steps,
        'T_init' : T_init,
        'melting_sequence': melting_sequence,
        'spots_per_m': spots_per_m
    }
    
    return simulation_setup

def compute_melting(simulation_setup, melting_sequence = None, plot_melting = False):
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
    spot_coords = simulation_setup['spot_coords']

    T = T_init.copy()
    
    T_max = np.zeros(simulation_steps)
    variance_list = np.zeros(simulation_steps)

    ilu = spilu(lhs_matrix.tocsc(), fill_factor=1)  
    M_ilu = LinearOperator(lhs_matrix.shape, ilu.solve)
    
    for n in range(simulation_steps):
        S = np.zeros(num_nodes)
        melt_node = melting_sequence[n]
        S[melt_node] = Q_melt
        S_total = S + b
       
        rhs = rhs_matrix.dot(T) + dt * S_total

        T, info = bicgstab(lhs_matrix, rhs, x0 = T, rtol = 1e-4, maxiter = 100, M = M_ilu)
        if info != 0:
            return True
        T_max[n] = np.max(T[internal_nodes])
        variance_list[n] = np.var(T[internal_nodes])
    variance_list = np.log(variance_list)
    variance_of_variances = np.mean(variance_list)
    if plot_melting:
        visualize_T_over_time(simulation_steps, T_max, variance_list)
        visualize_temperature_2D(spot_coords, T)
    
    return False

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

def visualize_temperature_2D(spot_coords, temperatures):
    x_coords = spot_coords[:, 0]
    y_coords = spot_coords[:, 1]
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(x_coords, y_coords, c=temperatures, cmap='hot', s=16, marker='s')
    plt.colorbar(scatter, label='Temperature')
    plt.axis('equal')
    plt.show()

def visualize_T_over_time(simulation_steps, T_max, variance_list):
    fig, ax1 = plt.subplots()

    ax1.plot(range(simulation_steps), T_max, color='b', label='Max Temperature')
    ax1.set_xlabel('Simulation Step')
    ax1.set_ylabel('Max Temperature', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(range(simulation_steps), variance_list, color='r', label='Temperature Variance')
    ax2.set_ylabel('Temperature Variance', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.show()

def main():
    # Debugging parameters
    spots_per_m = 600  # Resolution
    points_per_second = 1000
    k = 4

    simulation_setup = melting_setup(
        spots_per_m=spots_per_m,
        points_per_second=points_per_second,
        k=k
    )

    n_iters = 100
    counter = 0
    for i in range(n_iters):
        # Generate a random melting sequence
        random_sequence = np.random.permutation(simulation_setup['internal_nodes'])

        counter_bool = compute_melting(
            simulation_setup=simulation_setup,
            melting_sequence=random_sequence,
            plot_melting=False
        )
        if counter_bool:
            counter += 1
    print(counter / n_iters)

    # Log results
    #print(f"Variance metric (variance of variances): {variance_metric}")
    #print(f"Variance list (log variance over time): {variance_list}")

if __name__ == "__main__":
    main()
