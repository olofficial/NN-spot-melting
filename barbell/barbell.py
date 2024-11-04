import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation

def pixelated_barbell():
    #defining geometry of the barbell
    ball_radius = 1
    handle_length = 1
    handle_thickness = 0.1
    ball_centers = np.array([-(ball_radius + handle_length / 2), (ball_radius + handle_length / 2)])

    #defining pixelation of the barbell
    total_width = ball_radius * 4 + handle_length
    total_height = ball_radius * 2
    pixels_per_cm = 20
    pixel_width = 1 / pixels_per_cm
    pixels_x = int(total_width * pixels_per_cm + 2)
    pixels_y = int(total_height * pixels_per_cm + 2)

    #initializing the grid
    x_values = np.linspace(-(total_width / 2 + pixel_width), (total_width / 2  + pixel_width), pixels_x)
    y_values = np.linspace(-(total_height / 2 + pixel_width), (total_height / 2 + pixel_width), pixels_y)
    X, Y = np.meshgrid(x_values, y_values)

    #creating true/false grid for the points of the barbell
    barbell = np.zeros_like(X, dtype=np.uint8)
    barbell_shape = np.zeros_like(X, dtype=bool)
    edge_points = np.zeros_like(X, dtype=bool)
    #checking the points in each circle
    for center in ball_centers:
        barbell_shape |= ((X - center) ** 2 + Y ** 2) < ball_radius ** 2

    #checking for the handle
    barbell_shape |= (np.abs(X) <= handle_length / 2) & (np.abs(Y) <= handle_thickness / 2)

    barbell[barbell_shape] = 1

    expanded_barbell = binary_dilation(barbell_shape)

    edge_points = expanded_barbell & (~barbell_shape)

    barbell[edge_points] = 2

    # Plot the barbell shape
    plt.figure(figsize=(6, 6))
    plt.imshow(barbell, cmap = "gray")
    plt.colorbar()
    plt.axis('equal')
    plt.show()

    return barbell, X, Y

def build_kdtree(pixel_list):
    tree = cKDTree(pixel_list)
    return tree

def build_knn_graph(k = 4):
    # Call the function
    barbell, X, Y = pixelated_barbell()
    barbell_indices = np.where(barbell > 0)
    x_coords = X[barbell_indices]
    y_coords = Y[barbell_indices]
    pixel_coords = np.column_stack((x_coords, y_coords))

    tree = build_kdtree(pixel_coords)
    distances, indices = tree.query(pixel_coords, k = k + 1)
    knn_indices = indices[:, 1]
    knn_distances = distances[:, 1]
    return knn_indices, knn_distances


