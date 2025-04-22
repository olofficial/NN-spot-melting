# Finding and learning optimal melting sequences in E-PBF 3D printing

Electron beam Powder Bed Fusion (E-PBF) is a manufacturing technique where a beam of electrons heat a metal powder bed, melting it into a desired shape. New powder is added and melted, which builds a component layer by layer. One notable advantage of this technique is that the electron beam can traverse extremely quickly, making it possible to divide the build area into a grid of points, and melt each point separately (called spot melting in this project). Spot melting makes it possible, but not easy, to fine-tune the build process and achieve desired material properties. 

This project is aimed at a specific problem in E-PBF: **how to maximize thermal evenness during the build process**. Uneven heating, for instance in thin bridges between larger structures, can lead to warping and internal stress. This effect is more pronounced when these thin parts are connected to larger thermal sinks, which can absorb or emit heat and destabilize the build.

I formulated the task as an optimization problem over melting sequences: the order in which points in the 2D cross-section are heated. The goal is to **predict a sequence that keeps the temperature field as uniform as possible throughout the melting of the layer**.

---

## Physical model
### Heat equation
The heat equation is a common parabolic Partial Differential Equation (PDE):
$$
\frac{\partial T}{\partial t} = \Delta u = \nabla \cdot (k \nabla T) - h(T - T_0).
$$

where
- $k$ is a position-dependent conductivity derived from a KNN graph,
- $h$ models radiative and convective losses,
- $T_0$ is the ambient temperature,
- Melting input is modeled as point-wise heat injection $S$ for each time step (delta function)

This PDE can be solved easily by implementing an explicit solver (for instance Euler forward). However, for guaranteed stability and overall efficiency, implicit solvers are preferable. This could be the Euler backward method, but here the Crank-Nicholson method will be used. This method is second-order accurate and unconditionally stable for linear systems, making it suitable for our ends.

### KNN Graph
Dividing a set of points into a graph is a hard problem. One naïve way of doing it is by connecting each point to its $k$ nearest neighbors, as given by the euclidian distance $\sqrt{x^2 + y^2}$. If the points are (locally) given as a 2D square grid, then $k = 4$ produces a (locally) square grid graph, which we recognize as a common discretization for 2D numeric solvers. This makes it easy to translate a complex shape into a grid, which we need for our thermal simulation. The thermal conductivity between nodes is inversely proportional to their Euclidean distance.

The geometry used is a **barbell**: two large disks connected by a thin neck. The disks act as heat reservoirs, and the thin neck is the subcomponent at risk.

### Numerical Solver
The temporal discretization follows the Crank–Nicolson method:
$$
(I + \frac{\Delta t}{2} A) T^{n+1} = (I - \frac{\Delta t}{2} A) T^n + \Delta t S,
$$
where
- $A$ is the Laplacian matrix + boundary heat loss term,
- $S$ is the source term for the melting point at the current step,
- We solve this system iteratively for each time step using BiCGSTAB with ILU preconditioning.

To ensure an even temperature distribution, we want to minimize the **log variance of the internal temperatures over time**:
$$
\frac{1}{N} \sum_{n=1}^N \log(\mathrm{Var}[T^n_{{internal}}])
$$
Henceforth, this log-variance metric will be abbreviated as LVM.


## Data collection

I generate melting sequences in two ways:
- **Random**: Uniformly shuffled sequences. These are very quick to generate and evaluate, making them the bulk of my training sequences (around 70%).
- **Genetic Algorithm**: Evolves sequences based on fitness (low LVM). This algorithm starts from a population of random sequences and evolves them over several generations. In each generation:

    1. A fitness score is assigned to each sequence by simulating its thermal evolution and computing the log-variance metric.

    2. Sequences are selected using tournament selection.

    3. New sequences are created by order crossover and mutated via random swaps.

    4. Elites (best-performing individuals) are retained each generation.

    This gives us (hopefully) good melting sequences without evaluating every possible combination (which is infeasible). The best sequences found by the genetic algorithm are used as training data and as performance comparison for the learning model.

## Machine learning
UNDER CONSTRUCTION