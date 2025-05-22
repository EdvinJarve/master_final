# =============================================================================
# Imports
# =============================================================================

import sys
import os
import numpy as np
import time
import json
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.tri import Triangulation
from sklearn.metrics import mean_squared_error
from scipy.stats import qmc

# Project setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)

# Import PINNs class
from utils.heart_solver_pinns import MonodomainSolverPINNs

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Select device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device, 
      " CUDA version:", torch.version.cuda, 
      " CUDA available:", torch.cuda.is_available())

# =============================================================================
# Results Directory
# =============================================================================

base_results_dir = os.path.join(project_root, 'monodomain_results', 'forward_problems', 'regular_pinns', 'corner_case_smooth')
os.makedirs(base_results_dir, exist_ok=True)

# =============================================================================
# Gaussian Stimulus Source Term (PINNs Implementation)
# =============================================================================
def source_term_func_pinns(x_spatial, t):
    """
    Gaussian stimulus in the upper region, with a time window (ramp-up, constant, ramp-down).
    x_spatial: (N, 2) => (x, y)
    t: (N, 1) => time
    Returns: Tensor of shape (N, 1)
    """
    x0 = 0.3
    y0 = 0.7
    sigma = 0.05

    t_on_start = 0.05
    t_on_end   = 0.3
    t_off_start= 0.4
    t_off_end  = 0.7

    x = x_spatial[:, 0:1]
    y = x_spatial[:, 1:2]

    gaussian = 50.0 * torch.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    # Piecewise time window
    time_window = torch.zeros_like(t)

    ramp_up_mask = (t >= t_on_start) & (t < t_on_end)
    ramp_up_val  = (t - t_on_start) / (t_on_end - t_on_start)

    constant_mask = (t >= t_on_end) & (t < t_off_start)
    constant_val  = torch.ones_like(t)

    ramp_down_mask = (t >= t_off_start) & (t <= t_off_end)
    ramp_down_val  = 1.0 - (t - t_off_start) / (t_off_end - t_off_start)

    time_window = torch.where(ramp_up_mask,  ramp_up_val,  time_window)
    time_window = torch.where(constant_mask, constant_val, time_window)
    time_window = torch.where(ramp_down_mask, ramp_down_val, time_window)

    return gaussian * time_window

# =============================================================================
# Load Precomputed FEM Data
# =============================================================================

fem_data_file = os.path.join(base_results_dir, 'fem_data.npz')
if not os.path.isfile(fem_data_file):
    raise FileNotFoundError(f"FEM data file not found: {fem_data_file}")

fem_data = np.load(fem_data_file)
x_coords = fem_data['x_coords']           # shape (N_points,)
y_coords = fem_data['y_coords']           # shape (N_points,)
time_points = fem_data['time_points']     # shape (N_times,)
fem_solutions = fem_data['fem_solutions']   # shape (N_points, N_times)

# Create a dictionary for FEM solutions at each time point
solutions_fem = {}
for i, t_val in enumerate(time_points):
    solutions_fem[t_val] = fem_solutions[:, i]

triang = Triangulation(x_coords, y_coords)
print(f"FEM data loaded from {fem_data_file}.")

# =============================================================================
# Collocation / IC / BC Generation
# =============================================================================

x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
T = 1.0
t_min, t_max = 0.0, T

N_collocation       = 20000
N_ic               = 4000
N_bc               = 4000
N_collocation_val  = 4000
N_ic_val           = 800
N_bc_val           = 800

# Collocation
sampler = qmc.LatinHypercube(d=3)
sample = sampler.random(n=N_collocation)
X_collocation = sample.copy()
X_collocation[:, 0] = x_min + (x_max - x_min) * sample[:, 0]
X_collocation[:, 1] = y_min + (y_max - y_min) * sample[:, 1]
X_collocation[:, 2] = t_min + (t_max - t_min) * sample[:, 2]

sampler_val_collocation = qmc.LatinHypercube(d=3)
sample_val_collocation = sampler_val_collocation.random(n=N_collocation_val)
X_collocation_val = sample_val_collocation.copy()
X_collocation_val[:, 0] = x_min + (x_max - x_min) * sample_val_collocation[:, 0]
X_collocation_val[:, 1] = y_min + (y_max - y_min) * sample_val_collocation[:, 1]
X_collocation_val[:, 2] = t_min + (t_max - t_min) * sample_val_collocation[:, 2]

# Initial Condition: v=0 at t=0
sampler_ic = qmc.LatinHypercube(d=2)
sample_ic = sampler_ic.random(n=N_ic)
X_ic = sample_ic.copy()
X_ic[:, 0] = x_min + (x_max - x_min) * sample_ic[:, 0]
X_ic[:, 1] = y_min + (y_max - y_min) * sample_ic[:, 1]
X_ic = np.hstack((X_ic, np.zeros((N_ic, 1))))
expected_u0 = np.zeros((N_ic, 1), dtype=np.float32)

sampler_ic_val = qmc.LatinHypercube(d=2)
sample_ic_val = sampler_ic_val.random(n=N_ic_val)
X_ic_val = sample_ic_val.copy()
X_ic_val[:, 0] = x_min + (x_max - x_min) * sample_ic_val[:, 0]
X_ic_val[:, 1] = y_min + (y_max - y_min) * sample_ic_val[:, 1]
X_ic_val = np.hstack((X_ic_val, np.zeros((N_ic_val, 1))))
expected_u0_val = np.zeros((N_ic_val, 1), dtype=np.float32)

# Neumann BC => dv/dn=0 on all edges
N_per_boundary = N_bc // 4
def generate_boundary_points(N_per_boundary, x_fixed=None, y_fixed=None):
    sampler_boundary = qmc.LatinHypercube(d=2)
    sample_boundary = sampler_boundary.random(n=N_per_boundary)
    X_boundary = np.zeros((N_per_boundary, 3), dtype=np.float32)
    if x_fixed is not None:
        X_boundary[:, 0] = x_fixed
        X_boundary[:, 1] = y_min + (y_max - y_min) * sample_boundary[:, 0]
    elif y_fixed is not None:
        X_boundary[:, 0] = x_min + (x_max - x_min) * sample_boundary[:, 0]
        X_boundary[:, 1] = y_fixed
    X_boundary[:, 2] = t_min + (t_max - t_min) * sample_boundary[:, 1]
    return X_boundary

X_left   = generate_boundary_points(N_per_boundary, x_fixed=x_min)
X_right  = generate_boundary_points(N_per_boundary, x_fixed=x_max)
X_bottom = generate_boundary_points(N_per_boundary, y_fixed=y_min)
X_top    = generate_boundary_points(N_per_boundary, y_fixed=y_max)
X_boundary = np.vstack([X_left, X_right, X_bottom, X_top])

normal_vectors_left   = np.tile(np.array([[-1.0, 0.0]]), (N_per_boundary, 1))
normal_vectors_right  = np.tile(np.array([[ 1.0, 0.0]]), (N_per_boundary, 1))
normal_vectors_bottom = np.tile(np.array([[ 0.0,-1.0]]), (N_per_boundary, 1))
normal_vectors_top    = np.tile(np.array([[ 0.0, 1.0]]), (N_per_boundary, 1))
normal_vectors = np.vstack([normal_vectors_left, normal_vectors_right,
                            normal_vectors_bottom, normal_vectors_top])

# Validation BC
N_per_boundary_val = N_bc_val // 4
X_left_val   = generate_boundary_points(N_per_boundary_val, x_fixed=x_min)
X_right_val  = generate_boundary_points(N_per_boundary_val, x_fixed=x_max)
X_bottom_val = generate_boundary_points(N_per_boundary_val, y_fixed=y_min)
X_top_val    = generate_boundary_points(N_per_boundary_val, y_fixed=y_max)
X_boundary_val = np.vstack([X_left_val, X_right_val, X_bottom_val, X_top_val])

normal_vectors_left_val   = np.tile(np.array([[-1.0,  0.0]]), (N_per_boundary_val, 1))
normal_vectors_right_val  = np.tile(np.array([[ 1.0,  0.0]]), (N_per_boundary_val, 1))
normal_vectors_bottom_val = np.tile(np.array([[ 0.0, -1.0]]), (N_per_boundary_val, 1))
normal_vectors_top_val    = np.tile(np.array([[ 0.0,  1.0]]), (N_per_boundary_val, 1))
normal_vectors_val        = np.vstack([normal_vectors_left_val, normal_vectors_right_val,
                                       normal_vectors_bottom_val, normal_vectors_top_val])

# Convert to torch
X_collocation_tensor       = torch.tensor(X_collocation,       dtype=torch.float32, device=device)
X_ic_tensor                = torch.tensor(X_ic,                dtype=torch.float32, device=device)
expected_u0_tensor         = torch.tensor(expected_u0,         dtype=torch.float32, device=device)
X_boundary_tensor          = torch.tensor(X_boundary,          dtype=torch.float32, device=device)
normal_vectors_tensor      = torch.tensor(normal_vectors,      dtype=torch.float32, device=device)

X_collocation_val_tensor   = torch.tensor(X_collocation_val,   dtype=torch.float32, device=device)
X_ic_val_tensor            = torch.tensor(X_ic_val,            dtype=torch.float32, device=device)
expected_u0_val_tensor     = torch.tensor(expected_u0_val,     dtype=torch.float32, device=device)
X_boundary_val_tensor      = torch.tensor(X_boundary_val,      dtype=torch.float32, device=device)
normal_vectors_val_tensor  = torch.tensor(normal_vectors_val,  dtype=torch.float32, device=device)

# Optional scaling function (unused here)
def scale_to_m1p1(X):
    return 2.0 * X - 1.0

# =============================================================================
# Main PINNs Training Function
# =============================================================================

def run_pinns_simulation(num_layers, num_neurons, scaling_func=None):
    """
    Train and evaluate a MonodomainSolverPINNs model using a Gaussian stimulus source term.
    Returns global MSE and global relative MSE across all spatiotemporal FEM points.
    """
    run_dir = os.path.join(base_results_dir, f"results_layers={num_layers}_depth={num_neurons}")
    os.makedirs(run_dir, exist_ok=True)

    model_params = {
        'num_inputs': 3,
        'num_layers': num_layers,
        'num_neurons': num_neurons,
        'use_ode': False,
        'n_state_vars': 0,
        'loss_function': 'L2',
        'weight_strategy': 'manual',
        'alpha': 1.0
    }

    # Instantiate PINN
    model = MonodomainSolverPINNs(
        num_inputs=model_params['num_inputs'],
        num_layers=model_params['num_layers'],
        num_neurons=model_params['num_neurons'],
        device=device,
        source_term_func=source_term_func_pinns,
        M=1.0,
        use_ode=model_params['use_ode'],
        ode_func=None,
        n_state_vars=model_params['n_state_vars'],
        loss_function=model_params['loss_function'],
        weight_strategy=model_params['weight_strategy'],
        alpha=model_params['alpha'],
        scaling_func=scaling_func
    )

    # Attach data
    model.X_collocation  = X_collocation_tensor
    model.X_ic           = X_ic_tensor
    model.expected_u0    = expected_u0_tensor
    model.X_boundary     = X_boundary_tensor
    model.normal_vectors = normal_vectors_tensor
    model.X_data         = None
    model.expected_data  = None

    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 35000
    batch_size = 8192
    patience = 1000
    best_val_loss = float('inf')
    no_improve_counter = 0
    best_model_path = os.path.join(run_dir, 'best_model.pth')

    # Loss storage
    loss_list = []
    val_loss_list = []
    pde_loss_list = []
    ic_loss_list = []
    bc_loss_list = []
    epoch_list = []

    start_time_pinns = time.time()
    for epoch in range(epochs + 1):
        pde_loss, ic_loss, bc_loss, data_loss_val, _, total_loss = model.train_step(optimizer, batch_size)

        # Validation
        model.eval()
        total_val_loss = model.validate(
            X_collocation_val=X_collocation_val_tensor,
            X_ic_val=X_ic_val_tensor,
            expected_u0_val=expected_u0_val_tensor,
            X_boundary_val=X_boundary_val_tensor,
            normal_vectors_val=normal_vectors_val_tensor
        )

        if epoch % 100 == 0:
            loss_list.append(total_loss)
            val_loss_list.append(total_val_loss)
            pde_loss_list.append(pde_loss)
            ic_loss_list.append(ic_loss)
            bc_loss_list.append(bc_loss)
            epoch_list.append(epoch)

            print(f'[Epoch {epoch}] PDE: {pde_loss:.4e}, IC: {ic_loss:.4e}, '
                  f'BC: {bc_loss:.4e}, Total: {total_loss:.4e}, Val: {total_val_loss:.4e}')

            # Save best model
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                no_improve_counter = 0
                model.save_model(best_model_path)
                print(f"New best model saved with validation loss {best_val_loss:.4e}")
            else:
                no_improve_counter += 1
                if no_improve_counter >= patience:
                    print("Early stopping triggered.")
                    break

    end_time_pinns = time.time()
    computation_time_pinns = end_time_pinns - start_time_pinns
    print(f"Training complete in {computation_time_pinns:.2f} s")

    # Load best model for final comparison
    model.load_model(best_model_path)

    # --- Save losses for replotting ---
    loss_file = os.path.join(run_dir, 'training_losses.npz')
    np.savez_compressed(
        loss_file,
        epochs=np.array(epoch_list),
        total_loss=np.array(loss_list),
        val_loss=np.array(val_loss_list),
        pde_loss=np.array(pde_loss_list),
        ic_loss=np.array(ic_loss_list),
        bc_loss=np.array(bc_loss_list)
    )
    print(f"Losses saved to {loss_file}")

    # --- Plot training vs. validation loss ---
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, loss_list, label=r'$\mathcal{L}_{\text{train}}$')
    plt.plot(epoch_list, val_loss_list, label=r'$\mathcal{L}_{\text{val}}$')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'loss_plot.png'), dpi=300)
    plt.close()

    # --- Plot individual loss components ---
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, pde_loss_list, label=r'$\mathcal{L}_{PDE}$')
    plt.plot(epoch_list, ic_loss_list, label=r'$\mathcal{L}_{IC}$')
    plt.plot(epoch_list, bc_loss_list, label=r'$\mathcal{L}_{BC}$')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training Loss Components')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'loss_components_plot.png'), dpi=300)
    plt.close()

    # =============================================================================
    # Compute Global RMSE and Relative RMSE Over All Time Points
    # =============================================================================

    N_points = x_coords.shape[0]
    N_times = time_points.shape[0]
    pinns_solutions = np.zeros((N_points, N_times), dtype=np.float32)

    val_start = time.time()
    # Evaluate model at all time steps and FEM points
    for i, t_val in enumerate(time_points):
        X_eval = np.column_stack((x_coords, y_coords, np.full_like(x_coords, t_val)))
        X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32, device=device)
        with torch.no_grad():
            y_pred = model(X_eval_tensor).cpu().numpy().ravel()
        pinns_solutions[:, i] = y_pred
    val_end = time.time()

    validation_time = val_end - val_start

    # Compute Global RMSE and Relative RMSE
    fem_all = fem_solutions.reshape(-1)
    pinns_all = pinns_solutions.reshape(-1)

    global_rmse = np.sqrt(mean_squared_error(fem_all, pinns_all))
    global_relative_rmse = np.sqrt(np.sum((fem_all - pinns_all)**2) / (np.sum(fem_all**2) + 1e-16))

    print(f"Global RMSE across all times and points: {global_rmse:.6e}")
    print(f"Global Relative RMSE across all times and points: {global_relative_rmse:.6e}")

    # =============================================================================
    # Save Model Parameters & Evaluation Metrics to JSON
    # =============================================================================

    params_file = os.path.join(run_dir, 'model_parameters.json')
    model_params['training_time'] = computation_time_pinns
    model_params['validation_time'] = validation_time
    model_params['global_rmse'] = global_rmse
    model_params['global_relative_rmse'] = global_relative_rmse

    with open(params_file, 'w') as f:
        json.dump(model_params, f, indent=4)

    print(f"Model parameters and evaluation metrics saved to {params_file}")

    # =============================================================================
    # Generate Comparison Plots for Selected Times
    # =============================================================================

    comparison_times = [0.0, 0.1, 0.2, 0.4, 0.8, 1.0]
    for tt in comparison_times:
        if tt not in solutions_fem:
            print(f"No FEM data for t={tt}, skipping.")
            continue

        fem_v = solutions_fem[tt]
        X_pinns_eval = np.column_stack((x_coords, y_coords, np.full_like(x_coords, tt)))
        X_pinns_eval_tensor = torch.tensor(X_pinns_eval, dtype=torch.float32, device=device)

        with torch.no_grad():
            y_pred_tensor = model.evaluate(X_pinns_eval_tensor)
            y_pinns_pred_np = y_pred_tensor.cpu().numpy().reshape(-1)

        abs_error = np.abs(y_pinns_pred_np - fem_v)
        rel_error = abs_error / (np.abs(fem_v) + 1e-8)

        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'PINNs vs FEM at t = {tt}', fontsize=16)

        # FEM solution
        cs0 = axs[0, 0].tricontourf(triang, fem_v, levels=50, cmap='viridis')
        fig.colorbar(cs0, ax=axs[0, 0]).set_label('FEM')
        axs[0, 0].set_title('FEM')

        # PINNs prediction
        cs1 = axs[0, 1].tricontourf(triang, y_pinns_pred_np, levels=50, cmap='viridis')
        fig.colorbar(cs1, ax=axs[0, 1]).set_label('PINNs')
        axs[0, 1].set_title('PINNs')

        # Absolute error
        cs2 = axs[1, 0].tricontourf(triang, abs_error, levels=50, cmap='viridis')
        fig.colorbar(cs2, ax=axs[1, 0]).set_label('|PINNs - FEM|')
        axs[1, 0].set_title('Absolute Error')

        # Relative error
        cs3 = axs[1, 1].tricontourf(triang, rel_error, levels=50, cmap='viridis')
        fig.colorbar(cs3, ax=axs[1, 1]).set_label('Relative Error')
        axs[1, 1].set_title('Relative Error')

        for ax_row in axs:
            for ax in ax_row:
                ax.set_xlabel('x')
                ax.set_ylabel('y')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = f'comparison_t_{tt}.png'
        plt.savefig(os.path.join(run_dir, plot_filename), dpi=300)
        plt.close()
        print(f"Comparison plot saved for time t = {tt} -> {plot_filename}")

    # =============================================================================
    # Return Global RMSE Metrics
    # =============================================================================

    return global_rmse, global_relative_rmse

# =============================================================================
# Loop Over Architectures
# =============================================================================

num_layers_list  = [3, 4, 5]
num_neurons_list = [128, 256, 512]

test_rmse_dict = {}
test_relative_rmse_dict = {}

for depth in num_neurons_list:
    for layers in num_layers_list:
        test_rmse_val, test_rel_rmse_val = run_pinns_simulation(layers, depth, scaling_func=None)
        test_rmse_dict[(layers, depth)] = test_rmse_val
        test_relative_rmse_dict[(layers, depth)] = test_rel_rmse_val

# Write final global RMSE to file
test_rmse_file = os.path.join(base_results_dir, 'test_rmse.txt')
with open(test_rmse_file, 'w') as f:
    for depth in num_neurons_list:
        rmse_values = [f"{test_rmse_dict[(layers, depth)]:.6e}" for layers in num_layers_list]
        f.write(" ".join(rmse_values) + "\n")

# Write final global relative RMSE to file
test_relative_rmse_file = os.path.join(base_results_dir, 'test_relative_rmse.txt')
with open(test_relative_rmse_file, 'w') as f:
    for depth in num_neurons_list:
        rel_rmse_values = [f"{test_relative_rmse_dict[(layers, depth)]:.6e}" for layers in num_layers_list]
        f.write(" ".join(rel_rmse_values) + "\n")

print("All simulations complete. Global RMSE and relative RMSE saved.")

# =============================================================================
# Loop Over Architectures
# =============================================================================

num_layers_list   = [3, 4, 5]
num_neurons_list  = [128, 256, 512]

test_mse_dict          = {}
test_relative_mse_dict = {}

for depth in num_neurons_list:
    for layers in num_layers_list:
        test_mse_val, test_rel_mse_val = run_pinns_simulation(layers, depth, scaling_func=None)
        test_mse_dict[(layers, depth)] = test_mse_val
        test_relative_mse_dict[(layers, depth)] = test_rel_mse_val

# Write final global MSE to file
test_mse_file = os.path.join(base_results_dir, 'test_mse.txt')
with open(test_mse_file, 'w') as f:
    for depth in num_neurons_list:
        mse_values = [f"{test_mse_dict[(layers, depth)]:.6e}" for layers in num_layers_list]
        f.write(" ".join(mse_values) + "\n")

# Write final global relative MSE to file
test_relative_mse_file = os.path.join(base_results_dir, 'test_relative_mse.txt')
with open(test_relative_mse_file, 'w') as f:
    for depth in num_neurons_list:
        rel_mse_values = [f"{test_relative_mse_dict[(layers, depth)]:.6e}" for layers in num_layers_list]
        f.write(" ".join(rel_mse_values) + "\n")

print("All simulations complete. Global MSE and relative MSE saved.")
