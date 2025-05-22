#!/usr/bin/env python
import sys, os, time, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from sklearn.metrics import mean_squared_error

# -----------------------------------------------------------------------------
# Project Setup
# -----------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)
base_results_dir = os.path.join(project_root,
    'monodomain_results', 'forward_problems', 'enhanced_pinns', 'analytical_case')
os.makedirs(base_results_dir, exist_ok=True)

# Make sure we use the enhanced solver
from utils.heart_solver_pinns import EnhancedMonodomainSolverPINNs

torch.manual_seed(42)
np.random.seed(42)
device = 'cuda'

# ------------------------------------
# Load FEM data
# ------------------------------------
fem_data_file = os.path.join(base_results_dir, 'fem_data.npz')
if not os.path.isfile(fem_data_file):
    raise FileNotFoundError(f"{fem_data_file} not found. Please ensure fem_data is already exported.")
fem_data = np.load(fem_data_file)
x_coords = fem_data['x_coords']
y_coords = fem_data['y_coords']
time_points = fem_data['time_points']
fem_solutions = fem_data['fem_solutions']  # shape: (N_dofs, N_times)

solutions_fem = {t: fem_solutions[:, i] for i, t in enumerate(time_points)}
triang = Triangulation(x_coords, y_coords)
print(f"FEM data loaded from {fem_data_file}.")

# -----------------------------------------------------------------------------
# Domain and Problem Setup
# -----------------------------------------------------------------------------
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
T = 1.0
t_min, t_max = 0.0, T
M_val = 1.0  # Diffusion parameter

def analytical_solution_v(x, y, t):
    return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.sin(t)

def source_term_func_pinns(x_spatial, t):
    pi = torch.pi
    x = x_spatial[:, 0:1]
    y = x_spatial[:, 1:2]
    return (8 * pi**2 * torch.cos(2*pi*x) * torch.cos(2*pi*y) * torch.sin(t) +
            torch.cos(2*pi*x) * torch.cos(2*pi*y) * torch.cos(t))

# -----------------------------------------------------------------------------
# Helper Functions for Random Sampling
# -----------------------------------------------------------------------------
def sample_collocation_points(N, device='cpu'):
    X = torch.empty((N, 3), device=device)
    X[:, 0] = x_min + (x_max - x_min) * torch.rand(N, device=device)
    X[:, 1] = y_min + (y_max - y_min) * torch.rand(N, device=device)
    X[:, 2] = t_min + (t_max - t_min) * torch.rand(N, device=device)
    return X

def sample_IC_points(N, device='cpu'):
    X = torch.empty((N, 3), device=device)
    X[:, 0] = x_min + (x_max - x_min) * torch.rand(N, device=device)
    X[:, 1] = y_min + (y_max - y_min) * torch.rand(N, device=device)
    X[:, 2] = t_min
    return X

def sample_BC_points_with_normals(N, device='cpu'):
    N_per = N // 4
    # Left
    X_left = torch.empty((N_per, 3), device=device)
    X_left[:, 0] = x_min
    X_left[:, 1] = y_min + (y_max - y_min) * torch.rand(N_per, device=device)
    X_left[:, 2] = t_min + (t_max - t_min) * torch.rand(N_per, device=device)
    normals_left = torch.tensor([[-1.0, 0.0]], device=device).repeat(N_per, 1)
    # Right
    X_right = X_left.clone()
    X_right[:, 0] = x_max
    normals_right = torch.tensor([[1.0, 0.0]], device=device).repeat(N_per, 1)
    # Bottom
    X_bottom = torch.empty((N_per, 3), device=device)
    X_bottom[:, 0] = x_min + (x_max - x_min) * torch.rand(N_per, device=device)
    X_bottom[:, 1] = y_min
    X_bottom[:, 2] = t_min + (t_max - t_min) * torch.rand(N_per, device=device)
    normals_bottom = torch.tensor([[0.0, -1.0]], device=device).repeat(N_per, 1)
    # Top
    X_top = X_bottom.clone()
    X_top[:, 1] = y_max
    normals_top = torch.tensor([[0.0, 1.0]], device=device).repeat(N_per, 1)

    Xb = torch.cat([X_left, X_right, X_bottom, X_top], dim=0)
    Nb = torch.cat([normals_left, normals_right, normals_bottom, normals_top], dim=0)
    return Xb, Nb

# -----------------------------------------------------------------------------
# Pre-generate Fixed Validation Sets
# -----------------------------------------------------------------------------
X_collocation_val_fixed = sample_collocation_points(10000, device=device)
X_ic_val_fixed          = sample_IC_points(100, device=device)
X_boundary_val_fixed, normal_vectors_val_fixed = sample_BC_points_with_normals(100, device=device)

# Convert to tensors for model
X_collocation_val_tensor = X_collocation_val_fixed
X_ic_val_tensor          = X_ic_val_fixed
expected_u0_val_fixed    = torch.tensor(
    analytical_solution_v(
        X_ic_val_fixed[:,0].cpu().numpy(),
        X_ic_val_fixed[:,1].cpu().numpy(),
        X_ic_val_fixed[:,2].cpu().numpy()
    ).reshape(-1,1),
    dtype=torch.float32, device=device
)
X_boundary_val_tensor    = X_boundary_val_fixed
normal_vectors_val_tensor= normal_vectors_val_fixed

# -----------------------------------------------------------------------------
# RUN SIMULATION
# -----------------------------------------------------------------------------
def run_pinns_simulation(num_layers, num_neurons, scaling_par=None):
    run_dir = os.path.join(
        base_results_dir,
        f"results_layers={num_layers}_depth={num_neurons}"
    )
    os.makedirs(run_dir, exist_ok=True)
    params_file = os.path.join(run_dir, 'model_parameters.json')

    # Model parameters
    model_params = {
        'num_inputs':    3,
        'num_layers':    num_layers,
        'num_neurons':   num_neurons,
        'use_ode':       False,
        'n_state_vars':  0,
        'loss_function': 'L2',
        'weight_strategy':'dynamic',
        'alpha':         0.9,
    }

    # Instantiate the enhanced PINN
    model = EnhancedMonodomainSolverPINNs(
        num_inputs        = model_params['num_inputs'],
        num_layers        = model_params['num_layers'],
        num_neurons       = model_params['num_neurons'],
        device            = device,
        source_term_func  = source_term_func_pinns,
        M                 = M_val,
        use_ode           = model_params['use_ode'],
        ode_func          = None,
        n_state_vars      = model_params['n_state_vars'],
        loss_function     = model_params['loss_function'],
        weight_strategy   = model_params['weight_strategy'],
        alpha             = model_params['alpha'],
        use_fourier       = True,
        fourier_dim       = 256,
        use_rwf           = True,
        M_segments        = 5,
        t_min             = t_min,
        t_max             = t_max,
        epsilon           = 1.0,
        f                 = 1000,
        curriculum_enabled= True
    )

    # Generate training sets
    Xc, Xi = sample_collocation_points(20000, device=device), sample_IC_points(200, device=device)
    Xb, Nb = sample_BC_points_with_normals(2000, device=device)
    u0      = torch.tensor(
        analytical_solution_v(Xi[:,0].cpu(), Xi[:,1].cpu(), Xi[:,2].cpu()).reshape(-1,1),
        dtype=torch.float32, device=device
    )

    # Attach data
    model.X_collocation  = Xc
    model.X_ic           = Xi
    model.expected_u0    = u0
    model.X_boundary     = Xb
    model.normal_vectors = Nb
    model.X_data         = None
    model.expected_data  = None

    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs, batch_size, patience = 20000, 2048, 1000
    best_val_loss = float('inf')
    no_improve    = 0

    # History storage
    epoch_list      = []
    train_loss_list = []
    val_loss_list   = []
    pde_list        = []
    ic_list         = []
    bc_list         = []

    start_time = time.time()
    for epoch in range(epochs+1):
        # train_step now returns exactly 5 values
        pde_l, ic_l, bc_l, ode_l, tot_l = model.train_step(
            optimizer, Xc, Xi, u0, Xb, Nb, epoch
        )

        # validation
        val_l = model.validate(
            X_collocation_val_tensor,
            X_ic_val_tensor,
            expected_u0_val_fixed,
            X_boundary_val_tensor,
            normal_vectors_val_tensor
        )

        if epoch % 100 == 0:
            epoch_list.append(epoch)
            train_loss_list.append(tot_l)
            val_loss_list.append(val_l)
            pde_list.append(pde_l)
            ic_list.append(ic_l)
            bc_list.append(bc_l)

            print(f"Epoch {epoch:5d} | PDE={pde_l:.2e} IC={ic_l:.2e} BC={bc_l:.2e} Total={tot_l:.2e} Val={val_l:.2e}")

            if val_l < best_val_loss:
                best_val_loss = val_l
                no_improve = 0
                # model.save_model(best_model_path)
                print(f"  → New best (val={val_l:.2e})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("  → Early stopping.")
                    break

    end_time = time.time()

    # Save training history
    np.savez_compressed(
        os.path.join(run_dir, 'training_losses_analytical.npz'),
        epochs      = np.array(epoch_list),
        train_loss  = np.array(train_loss_list),
        val_loss    = np.array(val_loss_list),
        pde_loss    = np.array(pde_list),
        ic_loss     = np.array(ic_list),
        bc_loss     = np.array(bc_list),
    )

    # Plotting
    plt.figure(figsize=(10,6))
    plt.plot(epoch_list, train_loss_list, label='Train')
    plt.plot(epoch_list, val_loss_list,   label='Val')
    plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(run_dir,'loss_plot.png'), dpi=300); plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(epoch_list, pde_list, label=r'$\mathcal{L}_{PDE}$')
    plt.plot(epoch_list, ic_list,  label=r'$\mathcal{L}_{IC}$')
    plt.plot(epoch_list, bc_list,  label=r'$\mathcal{L}_{BC}$')
    plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(run_dir,'loss_components_plot.png'), dpi=300); plt.close()

    # Final evaluation
    X_test = np.column_stack((x_coords, y_coords, np.full_like(x_coords, T)))
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    model.load_model(best_model_path)
    t0 = time.time()
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy().reshape(-1)
    validation_time = time.time() - t0

    u_true = analytical_solution_v(x_coords, y_coords, T)
    mse = mean_squared_error(u_true, y_pred)
    rmse = np.sqrt(mse)
    rel_mse = np.sum((y_pred - u_true)**2) / np.sum(u_true**2)
    relative_rmse = np.sqrt(rel_mse)

    # Save parameters
    model_params.update({
        'computation_time': end_time - start_time,
        'validation_time': validation_time,
        'rmse': rmse,
        'relative_rmse': relative_rmse
    })
    with open(params_file, 'w') as f:
        json.dump(model_params, f, indent=4)

    return rmse, relative_rmse

# =============================================================================
# Sweep architectures
# =============================================================================
num_layers_list  = [3, 4, 5]
num_neurons_list = [128, 256, 512]
test_rmse_dict          = {}
test_relative_rmse_dict = {}

for depth in num_neurons_list:
    for layers in num_layers_list:
        rmse_val, rel_val = run_pinns_simulation(layers, depth)
        test_rmse_dict[(layers, depth)]          = rmse_val
        test_relative_rmse_dict[(layers, depth)] = rel_val

# Save final tables
with open(os.path.join(base_results_dir, 'test_rmse.txt'), 'w') as f:
    for depth in num_neurons_list:
        f.write(" ".join(f"{test_rmse_dict[(l, depth)]:.6e}" for l in num_layers_list) + "\n")

with open(os.path.join(base_results_dir, 'test_relative_rmse.txt'), 'w') as f:
    for depth in num_neurons_list:
        f.write(" ".join(f"{test_relative_rmse_dict[(l, depth)]:.6e}" for l in num_layers_list) + "\n")

print("All enhanced-PINN simulations complete. Results saved.")
