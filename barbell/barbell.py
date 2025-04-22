# barbell.py

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
