import numpy as np
from barbell import barbell
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import bicgstab, spilu, LinearOperator
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from barbell.barbell import grid_barbell

class ConductanceCalculator:
    def initialize_conductances(knn_distances, k_0 = 170.0):
        conductances = k_0 / knn_distances
        return conductances

class MatrixBuilder:
    def construct_adjacency_matrix(knn_indices, conductances):
        num_nodes = len(knn_indices)
        row_indices = np.repeat(np.arange(num_nodes), knn_indices.shape[1])
        col_indices = knn_indices.flatten()
        data = conductances.flatten()
        adjacency_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))
        return adjacency_matrix
    
    def construct_degree_matrix(adjacency_matrix):
        degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
        degree_matrix = diags(degrees, format='csr')
        return degree_matrix

    def construct_laplacian_matrix(knn_indices, conductances):
        adjacency_matrix = MatrixBuilder.construct_adjacency_matrix(knn_indices, conductances)
        degree_matrix = MatrixBuilder.construct_degree_matrix(adjacency_matrix)
        laplacian_matrix = degree_matrix - adjacency_matrix
        return laplacian_matrix

class SimulationSetup:
    def __init__(self, spots_per_m, points_per_second = 1000, k = 4, T_init = None, melting_sequence = None):
        self.spots_per_m = spots_per_m
        self.points_per_second = points_per_second
        self.k = k
        self.T_init = T_init
        self.melting_sequence = melting_sequence
        self.simulation_setup = self.setup_simulation()

    def setup_simulation(self):
        dt = 1 / self.points_per_second
        T_0 = 300
        knn_indices, knn_distances, spot_coords, point_labels = barbell.build_knn_graph(self.k, self.spots_per_m)
        
        if self.T_init is None:
            T_init = np.full(len(knn_indices), T_0)
        else:
            T_init = self.T_init

        num_nodes = len(knn_indices)
        conductances = ConductanceCalculator.initialize_conductances(knn_distances)
        
        laplacian = MatrixBuilder.construct_laplacian_matrix(knn_indices, conductances)

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

        Q_melt = 3e13 / self.spots_per_m ** 2
        if self.melting_sequence is None:
            melting_sequence = np.random.choice(internal_nodes, size=num_nodes, replace=True)
            simulation_steps = int(1 * len(internal_nodes))
        else:
            melting_sequence = self.melting_sequence
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
            'spots_per_m': self.spots_per_m
        }
        
        return simulation_setup


class MeltingSimulator:
    def __init__(self, simulation_setup):
        self.simulation_setup = simulation_setup

    def compute_melting(self, plot_melting = False):
        setup = self.simulation_setup
        simulation_steps = setup['simulation_steps']
        num_nodes = setup['num_nodes']
        melting_sequence = setup['melting_sequence']
        internal_nodes = setup['internal_nodes']
        Q_melt = setup['Q_melt']
        T_init = setup['T_init']
        spot_coords = setup['spot_coords']
    
        T = T_init.copy()
        
        T_max = np.zeros(simulation_steps)
        variance_list = np.zeros(simulation_steps)
    
        ilu = spilu(setup['lhs_matrix'].tocsc(), fill_factor=1)  
        M_ilu = LinearOperator(setup['lhs_matrix'].shape, ilu.solve)
        
        for n in range(simulation_steps):
            S = np.zeros(num_nodes)
            melt_node = melting_sequence[n]
            S[melt_node] = Q_melt
            S_total = S + setup['b']
           
            rhs = setup['rhs_matrix'].dot(T) + setup['dt'] * S_total

            #solve the linear system
            T, info = bicgstab(setup['lhs_matrix'], rhs, x0=T, tol=1e-2, maxiter=100, M=M_ilu)
            
            T_max[n] = np.max(T[internal_nodes])
            variance_list[n] = np.var(T[internal_nodes])
        
        variance_list = np.log(variance_list)
        variance_of_variances = np.mean(variance_list)
        
        if plot_melting:
            visualizer = Visualizer()
            visualizer.visualize_temperature_2D(spot_coords, T)
            visualizer.visualize_T_over_time(simulation_steps, T_max, variance_list)
            
        return variance_of_variances, variance_list


class Visualizer:
    def visualize_temperature_2D(self, spot_coords, temperatures):
        spots_per_m = 600
        barbell_mask, X, Y = grid_barbell(spots_per_m)

        temp_map = np.full_like(barbell_mask, np.nan, dtype=float)
        active = barbell_mask > 0
        coord_map = np.column_stack((X[active], Y[active]))

        from scipy.spatial import cKDTree
        tree = cKDTree(spot_coords)
        _, idx = tree.query(coord_map)
        temp_map[active] = temperatures[idx]

        cmap = plt.cm.hot
        cmap.set_bad(color='white')

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(np.ma.masked_invalid(temp_map), cmap=cmap, origin='lower',
                       extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='equal')
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label('Temperature')
        plt.tight_layout()
        plt.show()

    def visualize_T_over_time(self, simulation_steps, T_max, variance_list):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Simulation Step')
        ax1.set_ylabel('Max Temperature', color=color)
        ax1.plot(range(simulation_steps), T_max, color=color, label='Max Temperature')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()

        color = 'tab:red'
        ax2.set_ylabel('Temperature Variance', color=color)
        ax2.plot(range(simulation_steps), variance_list, color=color, label='Temperature Variance')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.title('Temperature Over Time')
        plt.show()