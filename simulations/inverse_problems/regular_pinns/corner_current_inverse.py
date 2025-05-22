#!/usr/bin/env python3
import sys
import os
import time
import json
import numpy as np
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
sys.path.insert(0, project_root)
from utils.heart_solver_pinns import InverseMonodomainSolverPINNs

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Select device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# =============================================================================
# Results Directory
# =============================================================================
base_results_dir = os.path.join(
    project_root,
    'monodomain_results',
    'inverse_problems',
    'corner_case_smooth'
)
os.makedirs(base_results_dir, exist_ok=True)

# =============================================================================
# Gaussian Stimulus Source Term
# =============================================================================
def source_term_func_pinns(x_spatial, t):
    x0, y0, sigma = 0.3, 0.7, 0.05
    t_on_start, t_on_end = 0.05, 0.3
    t_off_start, t_off_end = 0.4, 0.7

    x = x_spatial[:, 0:1]
    y = x_spatial[:, 1:2]
    gaussian = 50.0 * torch.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    ramp_up_mask    = (t >= t_on_start) & (t < t_on_end)
    constant_mask   = (t >= t_on_end)   & (t < t_off_start)
    ramp_down_mask  = (t >= t_off_start)& (t <= t_off_end)

    time_window = torch.zeros_like(t)
    time_window = torch.where(ramp_up_mask,
                              (t - t_on_start)/(t_on_end - t_on_start),
                              time_window)
    time_window = torch.where(constant_mask,
                              torch.ones_like(t),
                              time_window)
    time_window = torch.where(ramp_down_mask,
                              1.0 - (t - t_off_start)/(t_off_end - t_off_start),
                              time_window)

    return gaussian * time_window

# =============================================================================
# Load Precomputed FEM Data
# =============================================================================
fem_data_file = os.path.join(base_results_dir, 'fem_data_smooth_corner.npz')
if not os.path.isfile(fem_data_file):
    raise FileNotFoundError(f"FEM data file not found: {fem_data_file}")

fem_data      = np.load(fem_data_file)
x_coords      = fem_data['x_coords']
y_coords      = fem_data['y_coords']
time_points   = fem_data['time_points']
fem_solutions = fem_data['fem_solutions']

# transpose exactly as in GIF‐script
if fem_solutions.shape[0] == time_points.size and fem_solutions.shape[1] == x_coords.size:
    fem_solutions = fem_solutions.T

solutions_fem = {t: fem_solutions[:, i] for i, t in enumerate(time_points)}
triang = Triangulation(x_coords, y_coords)
print(f"FEM data loaded from {fem_data_file}.")

# =============================================================================
# Domain and Sampling Params
# =============================================================================
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
T = 1.0

N_collocation      = 20000
N_ic               = 4000
N_bc               = 4000
N_collocation_val  = 1000
N_ic_val           = 100
N_bc_val           = 100

# ----------------------------------------------------------------------------
# Generate Collocation Points
# ----------------------------------------------------------------------------
sampler = qmc.LatinHypercube(d=3)
sample = sampler.random(n=N_collocation)
X_collocation = np.zeros_like(sample)
X_collocation[:, 0] = x_min + (x_max - x_min) * sample[:, 0]
X_collocation[:, 1] = y_min + (y_max - y_min) * sample[:, 1]
X_collocation[:, 2] = 0.0 + (T - 0.0) * sample[:, 2]

sampler_val = qmc.LatinHypercube(d=3)
sample_val = sampler_val.random(n=N_collocation_val)
X_collocation_val = np.zeros_like(sample_val)
X_collocation_val[:, 0] = x_min + (x_max - x_min) * sample_val[:, 0]
X_collocation_val[:, 1] = y_min + (y_max - y_min) * sample_val[:, 1]
X_collocation_val[:, 2] = 0.0 + (T - 0.0) * sample_val[:, 2]

# =============================================================================
# Generate IC Points
# =============================================================================
sampler_ic = qmc.LatinHypercube(d=2)
sample_ic = sampler_ic.random(n=N_ic)
X_ic = np.zeros((N_ic, 3))
X_ic[:, 0] = x_min + (x_max - x_min) * sample_ic[:, 0]
X_ic[:, 1] = y_min + (y_max - y_min) * sample_ic[:, 1]
X_ic[:, 2] = 0.0
expected_u0 = np.zeros((N_ic, 1), dtype=np.float32)

sampler_ic_val = qmc.LatinHypercube(d=2)
sample_ic_val = sampler_ic_val.random(n=N_ic_val)
X_ic_val = np.zeros((N_ic_val, 3))
X_ic_val[:, 0] = x_min + (x_max - x_min) * sample_ic_val[:, 0]
X_ic_val[:, 1] = y_min + (y_max - y_min) * sample_ic_val[:, 1]
X_ic_val[:, 2] = 0.0
expected_u0_val = np.zeros((N_ic_val, 1), dtype=np.float32)

# =============================================================================
# Generate BC Points (Neumann)
# =============================================================================
def generate_boundary_points(Nb, x_fixed=None, y_fixed=None):
    sm = qmc.LatinHypercube(d=2).random(n=Nb)
    Xb = np.zeros((Nb, 3))
    if x_fixed is not None:
        Xb[:, 0] = x_fixed
        Xb[:, 1] = y_min + (y_max - y_min) * sm[:, 0]
    else:
        Xb[:, 0] = x_min + (x_max - x_min) * sm[:, 0]
        Xb[:, 1] = y_fixed
    Xb[:, 2] = 0.0 + (T - 0.0) * sm[:, 1]
    return Xb

Nb = N_bc // 4
X_left   = generate_boundary_points(Nb, x_fixed=x_min)
X_right  = generate_boundary_points(Nb, x_fixed=x_max)
X_bottom = generate_boundary_points(Nb, y_fixed=y_min)
X_top    = generate_boundary_points(Nb, y_fixed=y_max)
X_boundary = np.vstack([X_left, X_right, X_bottom, X_top])

n_left   = np.tile([[-1.0, 0.0]], (Nb, 1))
n_right  = np.tile([[ 1.0, 0.0]], (Nb, 1))
n_bottom = np.tile([[ 0.0,-1.0]], (Nb, 1))
n_top    = np.tile([[ 0.0, 1.0]], (Nb, 1))
normal_vectors = np.vstack([n_left, n_right, n_bottom, n_top])

Nb_val = N_bc_val // 4
X_left_val   = generate_boundary_points(Nb_val, x_fixed=x_min)
X_right_val  = generate_boundary_points(Nb_val, x_fixed=x_max)
X_bottom_val = generate_boundary_points(Nb_val, y_fixed=y_min)
X_top_val    = generate_boundary_points(Nb_val, y_fixed=y_max)
X_boundary_val = np.vstack([X_left_val, X_right_val, X_bottom_val, X_top_val])

n_left_val   = np.tile([[-1.0, 0.0]], (Nb_val,1))
n_right_val  = np.tile([[ 1.0, 0.0]], (Nb_val,1))
n_bottom_val = np.tile([[ 0.0,-1.0]], (Nb_val,1))
n_top_val    = np.tile([[ 0.0, 1.0]], (Nb_val,1))
normal_vectors_val = np.vstack([n_left_val, n_right_val, n_bottom_val, n_top_val])

# =============================================================================
# Convert to Tensors
# =============================================================================
X_collocation_tensor      = torch.tensor(X_collocation, dtype=torch.float32, device=device)
X_ic_tensor               = torch.tensor(X_ic, dtype=torch.float32, device=device)
expected_u0_tensor        = torch.tensor(expected_u0, dtype=torch.float32, device=device)
X_boundary_tensor         = torch.tensor(X_boundary, dtype=torch.float32, device=device)
normal_vectors_tensor     = torch.tensor(normal_vectors, dtype=torch.float32, device=device)

X_collocation_val_tensor  = torch.tensor(X_collocation_val, dtype=torch.float32, device=device)
X_ic_val_tensor           = torch.tensor(X_ic_val, dtype=torch.float32, device=device)
expected_u0_val_tensor    = torch.tensor(expected_u0_val, dtype=torch.float32, device=device)
X_boundary_val_tensor     = torch.tensor(X_boundary_val, dtype=torch.float32, device=device)
normal_vectors_val_tensor = torch.tensor(normal_vectors_val, dtype=torch.float32, device=device)

# =============================================================================
# Main Training & Eval
# =============================================================================
def run_pinns_simulation(num_layers, num_neurons, N_data, noise_level, scaling_func=None):
    run_dir = os.path.join(
        base_results_dir,
        f"results_layers={num_layers}_depth={num_neurons}_Ndata={N_data}_noise={noise_level}"
    )
    os.makedirs(run_dir, exist_ok=True)

    model = InverseMonodomainSolverPINNs(
        num_inputs       = 3,
        num_layers       = num_layers,
        num_neurons      = num_neurons,
        device           = device,
        source_term_func = source_term_func_pinns,
        initial_M        = 0.1,
        use_ode          = False,
        ode_func         = None,
        n_state_vars     = 0,
        loss_function    = 'L2',
        weight_strategy  = 'manual',
        alpha            = 0.9,
        scaling_func     = scaling_func
    )

    # attach data
    model.X_collocation  = X_collocation_tensor
    model.X_ic           = X_ic_tensor
    model.expected_u0    = expected_u0_tensor
    model.X_boundary     = X_boundary_tensor
    model.normal_vectors = normal_vectors_tensor

    # prepare noisy samples
    X_data_full, y_full = [], []
    for i, t_val in enumerate(time_points):
        t_arr = np.full_like(x_coords, t_val)
        X_data_full.append(np.column_stack((x_coords, y_coords, t_arr)))
        y_full.append(fem_solutions[:, i:i+1])
    X_data_full = np.vstack(X_data_full)
    y_full = np.vstack(y_full)

    idx = np.random.choice(X_data_full.shape[0], N_data, replace=False)
    X_sample, y_sample = X_data_full[idx], y_full[idx]
    if noise_level > 0:
        y_sample += noise_level * np.random.randn(*y_sample.shape)

    model.X_data        = torch.tensor(X_sample, dtype=torch.float32, device=device)
    model.expected_data = torch.tensor(y_sample, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    epochs, batch_size, patience = 35000, 4096, 1000
    best_val_loss, no_improve = float('inf'), 0
    best_model_path = os.path.join(run_dir, 'best_model.pth')

    # history lists
    epoch_list, loss_list = [], []
    pde_list, ic_list, bc_list, data_list = [], [], [], []
    val_list = []
    M_history = []

    start_t = time.time()
    for epoch in range(epochs+1):
        pde_l, ic_l, bc_l, data_l, _, tot_l = model.train_step(optimizer, batch_size)
        M_history.append(model.M.detach().cpu().item())

        if epoch % 100 == 0:
            val_l = model.validate(
                X_collocation_val=X_collocation_val_tensor,
                X_ic_val=X_ic_val_tensor,
                expected_u0_val=expected_u0_val_tensor,
                X_boundary_val=X_boundary_val_tensor,
                normal_vectors_val=normal_vectors_val_tensor
            )

            epoch_list.append(epoch)
            loss_list.append(tot_l)
            pde_list.append(pde_l)
            ic_list.append(ic_l)
            bc_list.append(bc_l)
            data_list.append(data_l)
            val_list.append(val_l)

            print(f"[Epoch {epoch}] PDE={pde_l:.2e} IC={ic_l:.2e} BC={bc_l:.2e} "
                  f"Data={data_l:.2e} Tot={tot_l:.2e} Val={val_l:.2e}")

            if val_l < best_val_loss:
                best_val_loss, no_improve = val_l, 0
                model.save_model(best_model_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping.")
                    break
    end_t = time.time()

    # save loss histories
    np.savez_compressed(
        os.path.join(run_dir, 'training_losses.npz'),
        epochs     = np.array(epoch_list),
        total_loss = np.array(loss_list),
        val_loss   = np.array(val_list),
        pde_loss   = np.array(pde_list),
        ic_loss    = np.array(ic_list),
        bc_loss    = np.array(bc_list),
        data_loss  = np.array(data_list),
        M_history  = np.array(M_history)
    )

    # plot train vs val loss
    plt.figure(figsize=(8,5))
    plt.plot(epoch_list, loss_list, label='Train')
    plt.plot(epoch_list, val_list,   label='Val')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'train_val_loss.png'), dpi=300)
    plt.close()

    # plot loss components (incl. data loss)
    plt.figure(figsize=(8,5))
    plt.plot(epoch_list, pde_list,  label=r'$\mathcal{L}_{PDE}$')
    plt.plot(epoch_list, ic_list,   label=r'$\mathcal{L}_{IC}$')
    plt.plot(epoch_list, bc_list,   label=r'$\mathcal{L}_{BC}$')
    plt.plot(epoch_list, data_list, label=r'$\mathcal{L}_{Data}$')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'loss_components.png'), dpi=300)
    plt.close()

    # plot M evolution
    plt.figure(figsize=(8,5))
    plt.plot(np.arange(len(M_history)), M_history, label='M')
    plt.xlabel('Epoch')
    plt.ylabel('M')
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'M_evolution.png'), dpi=300)
    plt.close()

    # reload best and evaluate
    model.load_model(best_model_path)

    # final at T
    X_test = np.column_stack((x_coords, y_coords, np.full_like(x_coords, T)))
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        y_pred_T = model(X_test_t).cpu().numpy().ravel()
    true_T = solutions_fem[T]
    mse_T = mean_squared_error(true_T, y_pred_T)
    rel_mse_T = np.sum((y_pred_T - true_T)**2) / np.sum(true_T**2)

    # global eval
    all_fem, pinns_all = [], []
    for i, t_val in enumerate(time_points):
        all_fem.append(fem_solutions[:, i])
        coords = np.column_stack((x_coords, y_coords, np.full_like(x_coords, t_val)))
        with torch.no_grad():
            pinns_all.append(model(torch.tensor(coords, dtype=torch.float32, device=device))
                              .cpu().numpy().ravel())
    all_fem   = np.hstack(all_fem)
    pinns_all = np.hstack(pinns_all)

    global_rmse     = np.sqrt(np.mean((pinns_all - all_fem)**2))
    global_rel_rmse = np.sqrt(np.sum((pinns_all - all_fem)**2) / np.sum(all_fem**2))

    params = {
        'layers': num_layers,
        'neurons': num_neurons,
        'N_data': N_data,
        'noise_level': noise_level,
        'mse_T': mse_T,
        'rel_mse_T': rel_mse_T,
        'global_rmse': global_rmse,
        'global_rel_rmse': global_rel_rmse,
        'predicted_M': model.M.detach().cpu().item(),
        'training_time_s': end_t - start_t
    }
    with open(os.path.join(run_dir, 'model_parameters.json'), 'w') as f:
        json.dump(params, f, indent=4)

    return global_rmse, global_rel_rmse, model.M.detach().cpu().item()

# =============================================================================
# Run for all data/noise settings
# =============================================================================
data_points_list = [50, 100, 300]
noise_levels     = [0, 0.01, 0.05]

results = {'rmse': {}, 'rel_rmse': {}, 'predicted_M': {}}
for N_data in data_points_list:
    for noise in noise_levels:
        rmse, rel_rmse, M_val = run_pinns_simulation(
            num_layers=3,
            num_neurons=512,
            N_data=N_data,
            noise_level=noise
        )
        results['rmse'][(N_data,noise)]       = rmse
        results['rel_rmse'][(N_data,noise)]   = rel_rmse
        results['predicted_M'][(N_data,noise)] = M_val

# Save summary tables
with open(os.path.join(base_results_dir, 'test_rmse.txt'), 'w') as f:
    for N_data in data_points_list:
        line = " ".join(f"{results['rmse'][(N_data,noise)]:.6e}" for noise in noise_levels)
        f.write(line + "\n")

with open(os.path.join(base_results_dir, 'test_relative_rmse.txt'), 'w') as f:
    for N_data in data_points_list:
        line = " ".join(f"{results['rel_rmse'][(N_data,noise)]:.6e}" for noise in noise_levels)
        f.write(line + "\n")

with open(os.path.join(base_results_dir, 'predicted_M.txt'), 'w') as f:
    for N_data in data_points_list:
        line = " ".join(f"{results['predicted_M'][(N_data,noise)]:.6e}" for noise in noise_levels)
        f.write(line + "\n")

print("All simulations complete. Metrics saved.")
