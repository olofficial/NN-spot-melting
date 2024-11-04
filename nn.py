import numpy as np
import torch as to
import matplotlib.pyplot as plt
from scipy.sparse import diags, identity, csc_matrix
from scipy.sparse.linalg import spsolve

#square grid
L_x = 1
L_y = 1
N_x = 20
N_y = N_x
dx = L_x / N_x
dy = L_y / N_y

melt_rate = 100 #pixels per second
dt = 1 / melt_rate #second per pixel
N_t = N_x * N_y / melt_rate #second per layer

u = np.ones((N_x, N_y))
u_new = np.ones((N_x, N_y))

def construct_A(N, d):
    main_diagonal = -2 * np.ones(N - 2)
    off_diagonal = 1 * to.ones(N - 3)
    diagonals = [main_diagonal, off_diagonal, off_diagonal]
    positions = [0, -1, 1]
    A = diags(diagonals, positions, shape=(N, N))
    return A / d ** 2

#construct 
A_x = construct_A(N_x, dx)
A_y = construct_A(N_y, dy)

eye_x = identity(N_x - 2)
eye_y = identity(N_y - 2)

#construct left hand side matrix
LHS_x = A_x - dt * eye_x / 2
LHS_y = A_y - dt * eye_y / 2

RHS_x = A_x + dt * eye_x / 2
RHS_y = A_y + dt * eye_y / 2