# Physical model
## Heat equation
The heat equation is a common parabolic Partial Differential Equation (PDE):
$$
\frac{\partial T}{\partial t} = \Delta u.
$$
It can be solved easily by implementing an explicit solver (for instance Euler forward). However, for guaranteed stability and overall efficiency, implicit solvers are preferable. This could be the Euler backward method, but here the Crank-Nicholson method will be used.  
## KNN Graph
Dividing a set of points into a graph is a hard problem. One naïve way of doing it is by connecting each point to its $k$ nearest neighbors, as given by the euclidian distance $\sqrt{x^2 + y^2}$. If the points are (locally) given as a 2D square grid, then $k = 4$ produces a (locally) square grid graph, which we recognize as a common discretization for 2D numeric solvers.