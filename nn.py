import numpy as np
import torch as to
import matplotlib.pyplot as plt

#square grid
L_x = 1
L_y = 1
N_x = 20
N_y = N_x

melt_rate = 100 #pixels per second
dt = 1 / melt_rate #second per pixel
N_t = N_x * N_y / melt_rate #second per layer

