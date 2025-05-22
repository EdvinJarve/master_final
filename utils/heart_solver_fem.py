import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector
from dolfinx.fem import Function, assemble_scalar
from petsc4py import PETSc
from mpi4py import MPI
from scipy.integrate import solve_ivp
import time

class FEMBase:
    """
    A generic base class that sets up:
      - The spatial mesh
      - Total time, time step, and number of steps
      - A scalar function space (self.V) for PDE unknown(s)
      - Basic routines for initializing fields

    This version automatically handles 1D, 2D, or 3D meshes.
    """
    def __init__(self, mesh, T, dt):
        self.mesh = mesh
        self.T = T
        self.dt = dt
        self.num_steps = int(T / dt)

        # Topological dimension (number of spatial coords in the PDE)
        self.topo_dim  = mesh.topology.dim
        # Embedding dimension (columns in geometry.x, may exceed topo_dim)
        self.embed_dim = mesh.geometry.x.shape[1]

        # Create a scalar function space; adjust if you need vector spaces.
        self.V = fem.functionspace(mesh, ("Lagrange", 1))

        # Create placeholders for PDE solutions in this base class.
        self.v_n = fem.Function(self.V)
        self.v_h = fem.Function(self.V)

    def initialize_function(self, initial_data, func):
        """
        Initialize a dolfinx Function object with either a scalar or a spatially varying function.
        """
        if callable(initial_data):
            func.interpolate(initial_data)
        else:
            func.interpolate(lambda x: np.full_like(x[0], initial_data))

    def _get_dof_coordinates(self):
        """
        Returns the degrees-of-freedom coordinates as an array of shape (num_nodes, embed_dim).
        """
        coords = self.V.tabulate_dof_coordinates()
        return coords.reshape(-1, self.embed_dim)


class MonodomainSolverFEM(FEMBase):
    """
    A monodomain PDE-ODE solver that inherits from FEMBase.
    Uses operator splitting (theta-scheme) and can handle multiple ODE variables.
    Supports 1D, 2D, or 3D problems.
    """
    def __init__(self, mesh, T, dt, M_i,
                 source_term_func=None,
                 ode_system=None,
                 ode_initial_conditions=None,
                 initial_v=0.0,
                 theta=0.5):
        super().__init__(mesh, T, dt)
        self.M_i = M_i
        self.source_term_func = source_term_func
        self.ode_system = ode_system
        self.theta = theta

        # Initialize PDE variable (v)
        self.initialize_function(initial_v, self.v_n)
        self.initialize_function(initial_v, self.v_h)

        # Time constant for UFL
        self.time_ufl = fem.Constant(self.mesh, PETSc.ScalarType(0.0))

        # Handle ODE variables
        self.ode_initial_conditions = ode_initial_conditions or []
        self.num_ode_vars = len(self.ode_initial_conditions)
        self.ode_vars = None
        if self.ode_system and self.num_ode_vars > 0:
            self._initialize_ode_variables()

        # Setup PDE
        self._create_variational_problem()
        self._create_solver()

    def _initialize_ode_variables(self):
        num_nodes = self.V.dofmap.index_map.size_local
        self.ode_vars = np.zeros((num_nodes, self.num_ode_vars), dtype=np.float64)
        coords = self._get_dof_coordinates()
        for j, init_cond in enumerate(self.ode_initial_conditions):
            if callable(init_cond):
                self.ode_vars[:, j] = init_cond(coords)
            else:
                self.ode_vars[:, j] = init_cond

    def _create_variational_problem(self):
        self.u_v = ufl.TrialFunction(self.V)
        self.v_v = ufl.TestFunction(self.V)
        dx = ufl.Measure("dx", domain=self.mesh)

        grad_u = ufl.grad(self.u_v)
        grad_v = ufl.grad(self.v_v)

        # Handle scalar M_i by converting to isotropic tensor
        if isinstance(self.M_i, (float, int)):
            M_expr = self.M_i * ufl.Identity(self.mesh.topology.dim)
        else:
            M_expr = self.M_i  # assume it's already a valid UFL tensor

        diffusion = ufl.dot(ufl.dot(M_expr, grad_u), grad_v)

        a = (self.u_v * self.v_v + self.dt * diffusion) * dx

        self.A_v = fem.petsc.assemble_matrix(fem.form(a))
        self.A_v.assemble()
        self.b_v = fem.petsc.create_vector(fem.form(a))



    def _create_solver(self):
        self.solver_v = PETSc.KSP().create(self.mesh.comm)
        self.solver_v.setOperators(self.A_v)
        self.solver_v.setType(PETSc.KSP.Type.PREONLY)
        self.solver_v.getPC().setType(PETSc.PC.Type.LU)

    def _setup_rhs_pde(self):
        dx = ufl.Measure("dx", domain=self.mesh)
        if self.source_term_func:
            X = ufl.SpatialCoordinate(self.mesh)
            coords = tuple(X[i] for i in range(self.topo_dim))
            I_app = self.source_term_func(*coords, self.time_ufl)
            L = (self.v_n + self.dt*I_app) * self.v_v * dx
        else:
            L = self.v_n * self.v_v * dx

        self.b_v.zeroEntries()
        fem.petsc.assemble_vector(self.b_v, fem.form(L))

    def _solve_pde(self):
        self.solver_v.solve(self.b_v, self.v_h.vector)
        self.v_h.x.scatter_forward()

    def _solve_ode(self, t0, t1, v_vals, ode_vals):
        if not self.ode_system:
            return v_vals, ode_vals
        num = len(v_vals)
        y0 = np.concatenate([v_vals] + [ode_vals[:, j] for j in range(self.num_ode_vars)])
        sol = solve_ivp(self.ode_system, (t0, t1), y0,
                        method="RK45", vectorized=False, rtol=1e-6, atol=1e-9)
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")
        yend = sol.y[:, -1]
        v_new = yend[:num]
        ode_new = None
        if self.num_ode_vars > 0:
            ode_new = np.column_stack([yend[num + j*num : num + (j+1)*num]
                                      for j in range(self.num_ode_vars)])
        return v_new, ode_new

    def run(self, analytical_solution_v=None, time_points=None):
        """
        Run the monodomain simulation with operator splitting.
        If time_points is not None, store solutions at those times.
        """
        errors_v = []
        start = time.time()
        coords = self._get_dof_coordinates()
        solutions = {} if time_points is not None else None

        v_vals = self.v_n.x.array.copy()
        ode_vals = self.ode_vars.copy() if self.ode_vars is not None else None

        for i in range(self.num_steps + 1):
            t = i * self.dt
            self.time_ufl.value = t

            # store solution if requested
            if (time_points is not None) and any(abs(t - tp) < 1e-14 for tp in time_points):
                solutions[t] = v_vals.copy()

            if i == self.num_steps:
                break

            # half-step ODE
            if self.ode_system and (self.theta > 0):
                v_vals, ode_vals = self._solve_ode(t, t + self.theta*self.dt, v_vals, ode_vals)

            # full PDE step
            self.v_n.x.array[:] = v_vals
            self.v_n.x.scatter_forward()
            self._setup_rhs_pde()
            self._solve_pde()
            v_pde = self.v_h.x.array.copy()

            # second half-step ODE
            if self.ode_system:
                if abs(self.theta - 1.0) > 1e-14:
                    v_vals, ode_vals = self._solve_ode(t + self.theta*self.dt,
                                                       t + self.dt,
                                                       v_pde, ode_vals)
                else:
                    v_vals = v_pde
            else:
                v_vals = v_pde

            # compute analytical error if requested
            if analytical_solution_v is not None:
                if self.topo_dim == 1:
                    ana = analytical_solution_v(coords[:,0], t+self.dt)
                elif self.topo_dim == 2:
                    ana = analytical_solution_v(coords[:,0], coords[:,1], t+self.dt)
                else:
                    ana = analytical_solution_v(coords[:,0], coords[:,1], coords[:,2], t+self.dt)
                errors_v.append(np.sqrt(np.mean((v_vals - ana)**2)))

        total_time = time.time() - start
        if time_points is not None:
            return errors_v, total_time, solutions
        else:
            return errors_v, total_time



"""
# =============================================================================
# Example usage (FitzHugh-Nagumo)
# =============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Example FitzHugh-Nagumo ODE system with both v, w in the flatten
    def ode_system_fhn(t, y_flat):
        # Suppose we unify [v, w] in one array
        num_nodes = y_flat.size // 2
        v = y_flat[:num_nodes]
        w = y_flat[num_nodes:]

        # FHN parameters
        a = 0.13
        b = 0.013
        c1 = 0.26
        c2 = 0.1
        c3 = 1.0

        dv_dt = c1 * v * (v - a) * (1.0 - v) - c2 * v * w
        dw_dt = b * (v - c3 * w)

        return np.concatenate([dv_dt, dw_dt])

    # Non-uniform initial conditions
    def initial_v_function(x):
        v_init = np.zeros_like(x[0])
        # Example: v=0.8 where x <= 0.5
        left_mask = x[0] <= 0.2
        v_init[left_mask] = 1.0
        return v_init

    def initial_w_function(x):
        return np.zeros_like(x[0])  # w=0

    # Build mesh
    Nx, Ny = 200, 200
    domain_mesh = mesh.create_unit_square(MPI.COMM_WORLD, Nx, Ny)
    
    # Simulation parameters
    T = 0.05
    Nt = 100
    dt = T / Nt
    M = 1.0  # diffusion
    theta = 0.5  # Strang splitting
    from matplotlib.tri import Triangulation


    full_time_steps = np.linspace(0, T, Nt + 1)  # Includes all time steps, including t=0 and t=T

    # Initialize and run the solver
    solver = MonodomainSolverFEM(
        mesh=domain_mesh,
        T=T,
        dt=dt,
        M_i=M,
        source_term_func=None,
        ode_system=ode_system_fhn,
        ode_initial_conditions=[initial_w_function],
        initial_v=initial_v_function,
        theta=theta
        )

    # Run the simulation
    errors, comp_time, solutions = solver.run(time_points=full_time_steps)
    print("Simulation complete.")
    print("Errors:", errors)
    print("Computation time:", comp_time)

    # Get the mesh coordinates (for plotting)
    dof_coords = solver.V.tabulate_dof_coordinates()  # (num_nodes, dim)
    x_coords = dof_coords[:, 0]  # x-coordinates
    y_coords = dof_coords[:, 1]  # y-coordinates

    # Create a range of time steps based on Nt
    full_time_steps = np.linspace(0, T, Nt + 1)  # Includes all time steps, including t=0 and t=T
    total_gradients = []

    for i, t in enumerate(full_time_steps):
        if t in solutions:
            v_vals = solutions[t]

            # Create a FEniCSx function for the solution
            v_func = Function(solver.V)
            v_func.vector.setArray(v_vals)

            # Compute the gradient norm |∇v|^2 and integrate over the domain
            grad_norm = np.abs(ufl.inner(ufl.grad(v_func), ufl.grad(v_func)))
            total_gradient = assemble_scalar(fem.form(grad_norm * ufl.dx))
            
            # Append the total gradient
            total_gradients.append(total_gradient)
        else:
            print(f"[INFO] Skipping time step t={t:.2f} as no solution data is available.")

    from scipy.integrate import simpson  # Importing Simpson's rule for integration

    # Ensure all time steps have corresponding gradient values
    assert len(full_time_steps) == len(total_gradients), "Mismatch in time steps and gradients length!"

    # Ensure total_gradients and full_time_steps are numpy arrays
    total_gradients = np.array(total_gradients)

    # Compute normalized gradients
    area_under_curve = simpson(y = total_gradients, x = full_time_steps)  # Compute the total area under the curve
    normalized_gradients = total_gradients / area_under_curve  # Normalize so area = 1

    # Define the intervals and colors for plotting
    intervals = [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.0)]
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(full_time_steps, normalized_gradients, label='Normalized Total Gradient (PDF)', color='black', linewidth=1.5)

    # Calculate and annotate areas under the curve for each interval
    for i, (start, end) in enumerate(intervals):
        # Find indices corresponding to the interval
        mask = (full_time_steps >= start) & (full_time_steps <= end)
        time_interval = full_time_steps[mask]
        gradient_interval = normalized_gradients[mask]

        # Compute area under the curve for this interval
        interval_area = simpson(y = gradient_interval, x = time_interval)
        percentage = interval_area * 100  # Convert to percentage

        # Plot the interval with a different color
        plt.fill_between(time_interval, gradient_interval, color=colors[i], alpha=0.5, label=f'{start:.2f}-{end:.2f} ({percentage:.2f}%)')

        # Annotate the percentage on the plot
        mid_time = (start + end) / 2
        mid_gradient = np.max(gradient_interval) * 0.5
        plt.text(mid_time, mid_gradient, f'{percentage:.2f}%', fontsize=10, ha='center', color=colors[i])

    # Finalize plot
    plt.title('Normalized Gradient Distribution with Area Percentages')
    plt.xlabel('Time (t)')
    plt.ylabel('Normalized Total Gradient')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


"""




class BidomainSolverFEM(FEMBase):
    """
    FEM solver for the bidomain model.

    Attributes:
        M_i (fem.Constant or ufl.Expr): Intracellular conductivity tensor.
        M_e (fem.Constant or ufl.Expr): Extracellular conductivity tensor.
    """
    def __init__(self, N_x, N_y, T, stimulus_expression, M_i, M_e, dt, initial_v=0.0, initial_u_e=0.0):
        """
        Initialize the bidomain solver.

        Parameters:
            N_x (int): Number of elements in the x-direction.
            N_y (int): Number of elements in the y-direction.
            T (float): Total time for the simulation.
            stimulus_expression (ufl.Expr): Expression for the stimulus term.
            M_i (float or ufl.Expr): Intracellular conductivity tensor.
            M_e (float or ufl.Expr): Extracellular conductivity tensor.
            dt (float): Time step size.
            initial_v (float): Initial value for the solution.
            initial_u_e (float): Initial value for the extracellular potential.
        """
        super().__init__(N_x, N_y, T, dt, stimulus_expression, initial_v, initial_u_e)

        if isinstance(M_i, (int, float)):
            self.M_i = fem.Constant(self.domain, PETSc.ScalarType(M_i))
        else:
            self.M_i = ufl.as_tensor(M_i)

        if isinstance(M_e, (int, float)):
            self.M_e = fem.Constant(self.domain, PETSc.ScalarType(M_e))
        else:
            self.M_e = ufl.as_tensor(M_e)

        self.create_variational_problems()
        self.create_solvers()

    def create_variational_problems(self):
        """
        Create variational problems for the bidomain model.
        """
        # Define trial and test functions
        self.u_v, self.v_v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        self.u_u_e, self.v_u_e = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)

        # Define the source term
        x, y = ufl.SpatialCoordinate(self.domain)
        self.I_app = self.stimulus_expression(x, y, 0)

        # Define measures
        dx = ufl.Measure("dx", domain=self.domain)

        # Define variational problem for v
        self.a_v = (self.u_v * self.v_v + self.dt * ufl.dot(self.M_i * ufl.grad(self.u_v), ufl.grad(self.v_v))) * dx
        self.L_v = (self.v_n + self.dt * self.I_app) * self.v_v * dx

        # Define variational problem for u_e
        self.a_u_e = (self.u_u_e * self.v_u_e + self.dt * ufl.dot((self.M_i + self.M_e) * ufl.grad(self.u_u_e), ufl.grad(self.v_u_e))) * dx
        self.L_u_e = (self.u_e_n - self.dt * self.I_app) * self.v_u_e * dx

        # Assemble forms
        self.bilinear_form_v = fem.form(self.a_v)
        self.linear_form_v = fem.form(self.L_v)
        self.bilinear_form_u_e = fem.form(self.a_u_e)
        self.linear_form_u_e = fem.form(self.L_u_e)

        # Assemble matrices
        self.A_v = fem.petsc.assemble_matrix(self.bilinear_form_v)
        self.A_v.assemble()
        self.A_u_e = fem.petsc.assemble_matrix(self.bilinear_form_u_e)
        self.A_u_e.assemble()

        # Create vectors
        self.b_v = fem.petsc.create_vector(self.linear_form_v)
        self.b_u_e = fem.petsc.create_vector(self.linear_form_u_e)

    def create_solvers(self):
        """
        Create solvers for the bidomain model.
        """
        # Create solvers for v and u_e
        self.solver_v = PETSc.KSP().create(self.domain.comm)
        self.solver_v.setOperators(self.A_v)
        self.solver_v.setType(PETSc.KSP.Type.PREONLY)
        self.solver_v.getPC().setType(PETSc.PC.Type.LU)

        self.solver_u_e = PETSc.KSP().create(self.domain.comm)
        self.solver_u_e.setOperators(self.A_u_e)
        self.solver_u_e.setType(PETSc.KSP.Type.PREONLY)
        self.solver_u_e.getPC().setType(PETSc.PC.Type.LU)

    def run(self, analytical_solution_v=None, analytical_solution_u_e=None, time_points=None):
        """
        Run the bidomain simulation.

        Parameters:
            analytical_solution_v (callable, optional): Analytical solution for validation of v.
            analytical_solution_u_e (callable, optional): Analytical solution for validation of u_e.
            time_points (list of float, optional): Time points to store the solution.

        Returns:
            tuple: Errors for v (list of float), errors for u_e (list of float), computation time (float), and optionally solutions at specified time points.
        """
        errors_v = []
        errors_u_e = []
        start_time = time.time()
        dof_coords = self.V.tabulate_dof_coordinates()
        x_coords = dof_coords[:, 0]
        y_coords = dof_coords[:, 1]

        # Prepare dictionary to store solutions at specified time points
        if time_points is not None:
            solutions_v = {t: None for t in time_points}
            solutions_u_e = {t: None for t in time_points}

        for i in range(self.num_steps + 1):
            t = i * self.dt  # Correctly set t for the current step

            # Update source term
            x, y = ufl.SpatialCoordinate(self.domain)
            self.I_app = self.stimulus_expression(x, y, t)
            
            # Update variational problems for v
            self.L_v = (self.v_n + self.dt * self.I_app) * self.v_v * ufl.dx
            self.linear_form_v = fem.form(self.L_v)
            with self.b_v.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(self.b_v, self.linear_form_v)
            self.solver_v.solve(self.b_v, self.v_h.vector)
            self.v_h.x.scatter_forward()
            self.v_n.x.array[:] = self.v_h.x.array

            # Update variational problems for u_e
            self.L_u_e = (self.u_e_n - self.dt * self.I_app) * self.v_u_e * ufl.dx
            self.linear_form_u_e = fem.form(self.L_u_e)
            with self.b_u_e.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(self.b_u_e, self.linear_form_u_e)
            self.solver_u_e.solve(self.b_u_e, self.u_e_h.vector)
            self.u_e_h.x.scatter_forward()
            self.u_e_n.x.array[:] = self.u_e_h.x.array

            # Store the solution if the current time is one of the specified time points
            if time_points is not None:
                for tp in time_points:
                    if np.isclose(t, tp, atol=1e-5):  # Increased tolerance
                        solutions_v[tp] = self.v_h.x.array.copy()
                        solutions_u_e[tp] = self.u_e_h.x.array.copy()

            # Calculate errors for v
            if analytical_solution_v is not None:
                analytical_values_v = analytical_solution_v(x_coords, y_coords, t)
                numerical_values_v = self.v_h.x.array
                error_v = np.sqrt(np.sum((numerical_values_v - analytical_values_v) ** 2) / len(analytical_values_v))
                errors_v.append(error_v)

            # Calculate errors for u_e
            if analytical_solution_u_e is not None:
                analytical_values_u_e = analytical_solution_u_e(x_coords, y_coords, t)
                numerical_values_u_e = self.u_e_h.x.array
                error_u_e = np.linalg.norm(numerical_values_u_e - analytical_values_u_e) / np.linalg.norm(analytical_values_u_e)
                errors_u_e.append(error_u_e)

        computation_time = time.time() - start_time

        # Return errors and solutions
        if time_points is not None:
            return errors_v, errors_u_e, computation_time, solutions_v, solutions_u_e
        else:
            return errors_v, errors_u_e, computation_time