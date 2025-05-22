#!/usr/bin/env python3

# =============================================================================
# Imports
# =============================================================================
import sys
import os
import time
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.stats import qmc
from sklearn.metrics import mean_squared_error

# =============================================================================
# Project Setup
# =============================================================================
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../..')
)
sys.path.insert(0, project_root)

# Import the PINNs solver
from utils.heart_solver_pinns import InverseMonodomainSolverPINNs

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Select device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device,
      "CUDA version:", torch.version.cuda,
      "CUDA available:", torch.cuda.is_available())

# =============================================================================
# Results Directory
# =============================================================================
base_results_dir = os.path.join(
    project_root,
    'monodomain_results',
    'inverse_problems',
    'corner_case_smooth_tensor'
)
os.makedirs(base_results_dir, exist_ok=True)

# =============================================================================
# Gaussian Stimulus Source Term (PINNs)
# =============================================================================
def source_term_func_pinns(x_spatial, t):
    x0, y0, sigma = 0.3, 0.7, 0.05
    t_on_start, t_on_end = 0.05, 0.3
    t_off_start, t_off_end = 0.4, 0.7

    x = x_spatial[:, 0:1]
    y = x_spatial[:, 1:2]
    gaussian = 50.0 * torch.exp(
        -((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)
    )

    # Time windows: ramp up, constant, ramp down
    ramp_up   = (t >= t_on_start)   & (t <  t_on_end)
    constant  = (t >= t_on_end)     & (t <  t_off_start)
    ramp_down = (t >= t_off_start)  & (t <= t_off_end)

    time_window = torch.zeros_like(t)
    time_window = torch.where(
        ramp_up,
        (t - t_on_start) / (t_on_end - t_on_start),
        time_window
    )
    time_window = torch.where(
        constant,
        torch.ones_like(t),
        time_window
    )
    time_window = torch.where(
        ramp_down,
        1.0 - (t - t_off_start) / (t_off_end - t_off_start),
        time_window
    )

    return gaussian * time_window

# =============================================================================
# Load Precomputed FEM Data
# =============================================================================
fem_data_file = os.path.join(
    base_results_dir,
    'fem_data_smooth_corner_tensor.npz'
)

if not os.path.isfile(fem_data_file):
    raise FileNotFoundError(f"FEM data file not found: {fem_data_file}")

fem_data      = np.load(fem_data_file)
x_coords      = fem_data['x_coords']
y_coords      = fem_data['y_coords']
time_points   = fem_data['time_points']      # shape (N_times,)
fem_solutions = fem_data['fem_solutions']    # shape (N_points, N_times)

# Filter to the desired time window
# (we only use 0.05 <= t <= 0.8 for training/validation)
t_low, t_high = 0.05, 0.8
mask = (time_points >= t_low) & (time_points <= t_high)
time_points   = time_points[mask]
fem_solutions = fem_solutions[:, mask]

# Build lookup dict and triangulation
solutions_fem = {t: fem_solutions[:, i] for i, t in enumerate(time_points)}
triang        = Triangulation(x_coords, y_coords)
print(
    f"FEM data loaded from {fem_data_file}. "
    f"Kept {mask.sum()} timesteps between {t_low} and {t_high}."
)

# =============================================================================
# Domain and Sampling Parameters
# =============================================================================
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
T = 1.0
t_min, t_max = 0.0, T

N_collocation     = 20000
N_ic              = 4000
N_bc              = 4000
N_collocation_val = 1000
N_ic_val          = 100
N_bc_val          = 100

# Generate Collocation Points (PDE)
sampler = qmc.LatinHypercube(d=3)
sample = sampler.random(n=N_collocation)
X_collocation = np.zeros_like(sample)
X_collocation[:, 0] = x_min + (x_max - x_min) * sample[:, 0]
X_collocation[:, 1] = y_min + (y_max - y_min) * sample[:, 1]
X_collocation[:, 2] = t_min + (t_max - t_min) * sample[:, 2]

sampler_val = qmc.LatinHypercube(d=3)
sample_val = sampler_val.random(n=N_collocation_val)
X_collocation_val = np.zeros_like(sample_val)
X_collocation_val[:, 0] = x_min + (x_max - x_min) * sample_val[:, 0]
X_collocation_val[:, 1] = y_min + (y_max - y_min) * sample_val[:, 1]
X_collocation_val[:, 2] = t_min + (t_max - t_min) * sample_val[:, 2]

# Generate Initial Condition Points (IC)
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

# Generate Boundary Points (Neumann BC)
def generate_boundary_points(Nb, x_fixed=None, y_fixed=None):
    sampler_b = qmc.LatinHypercube(d=2)
    sample_b  = sampler_b.random(n=Nb)
    Xb = np.zeros((Nb, 3))

    if x_fixed is not None:
        Xb[:, 0] = x_fixed
        Xb[:, 1] = y_min + (y_max - y_min) * sample_b[:, 0]
    else:
        Xb[:, 0] = x_min + (x_max - x_min) * sample_b[:, 0]
        Xb[:, 1] = y_fixed

    Xb[:, 2] = t_min + (t_max - t_min) * sample_b[:, 1]
    return Xb

Nb = N_bc // 4
X_left   = generate_boundary_points(Nb, x_fixed=x_min)
X_right  = generate_boundary_points(Nb, x_fixed=x_max)
X_bottom = generate_boundary_points(Nb, y_fixed=y_min)
X_top    = generate_boundary_points(Nb, y_fixed=y_max)
X_boundary = np.vstack([X_left, X_right, X_bottom, X_top])

# Normals for Neumann BC
normals = np.vstack([
    np.tile([[-1.0,  0.0]], (Nb, 1)),
    np.tile([[ 1.0,  0.0]], (Nb, 1)),
    np.tile([[ 0.0, -1.0]], (Nb, 1)),
    np.tile([[ 0.0,  1.0]], (Nb, 1))
])

Nb_val = N_bc_val // 4
X_left_val   = generate_boundary_points(Nb_val, x_fixed=x_min)
X_right_val  = generate_boundary_points(Nb_val, x_fixed=x_max)
X_bottom_val = generate_boundary_points(Nb_val, y_fixed=y_min)
X_top_val    = generate_boundary_points(Nb_val, y_fixed=y_max)
X_boundary_val = np.vstack([X_left_val, X_right_val, X_bottom_val, X_top_val])

normals_val = np.vstack([
    np.tile([[-1.0,  0.0]], (Nb_val, 1)),
    np.tile([[ 1.0,  0.0]], (Nb_val, 1)),
    np.tile([[ 0.0, -1.0]], (Nb_val, 1)),
    np.tile([[ 0.0,  1.0]], (Nb_val, 1))
])

# Convert All Data to Torch Tensors
X_coll_t   = torch.tensor(X_collocation,     dtype=torch.float32, device=device)
X_ic_t     = torch.tensor(X_ic,              dtype=torch.float32, device=device)
X_bc_t     = torch.tensor(X_boundary,        dtype=torch.float32, device=device)
norm_t     = torch.tensor(normals,           dtype=torch.float32, device=device)

X_coll_v_t = torch.tensor(X_collocation_val, dtype=torch.float32, device=device)
X_ic_v_t   = torch.tensor(X_ic_val,          dtype=torch.float32, device=device)
X_bc_v_t   = torch.tensor(X_boundary_val,    dtype=torch.float32, device=device)
norm_v_t   = torch.tensor(normals_val,       dtype=torch.float32, device=device)

u0_t       = torch.tensor(expected_u0,       dtype=torch.float32, device=device)
u0_v_t     = torch.tensor(expected_u0_val,   dtype=torch.float32, device=device)

# =============================================================================
# Main Training & Evaluation Function
# =============================================================================
def run_pinns_simulation(
    num_layers,
    num_neurons,
    N_data,
    noise_level,
    scaling_func=None
):
    # Create run directory
    run_dir = os.path.join(
        base_results_dir,
        f"results_layers={num_layers}_neurons={num_neurons}_Ndata={N_data}_noise={noise_level}"
    )
    os.makedirs(run_dir, exist_ok=True)

    # Initialize M as a 4-vector tensor
    M0 = torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.float32, device=device)

    # Instantiate the PINNs model
    model = InverseMonodomainSolverPINNs(
        num_inputs=3,
        num_layers=num_layers,
        num_neurons=num_neurons,
        device=device,
        source_term_func=source_term_func_pinns,
        initial_M=M0,
        use_ode=False,
        ode_func=None,
        n_state_vars=0,
        loss_function='L2',
        weight_strategy='manual',
        alpha=0.9,
        scaling_func=scaling_func
    )

    # Attach collocation, IC, and BC data
    model.X_collocation  = X_coll_t
    model.X_ic           = X_ic_t
    model.expected_u0    = u0_t
    model.X_boundary     = X_bc_t
    model.normal_vectors = norm_t

    # Prepare noisy data samples
    X_all, y_all = [], []
    for i, t_val in enumerate(time_points):
        t_arr = np.full_like(x_coords, t_val)
        X_all.append(np.column_stack((x_coords, y_coords, t_arr)))
        y_all.append(fem_solutions[:, i:i+1])
    X_all = np.vstack(X_all)
    y_all = np.vstack(y_all)

    idx = np.random.choice(X_all.shape[0], N_data, replace=False)
    X_sample = X_all[idx]
    y_sample = y_all[idx]
    if noise_level > 0:
        y_sample += noise_level * np.random.randn(*y_sample.shape)

    model.X_data        = torch.tensor(X_sample, dtype=torch.float32, device=device)
    model.expected_data = torch.tensor(y_sample, dtype=torch.float32, device=device)

    # Optimizer and training settings
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs     = 50000
    batch_size = 4096
    patience   = 1000

    best_val_loss = float('inf')
    no_improve    = 0
    best_model_fp = os.path.join(run_dir, 'best_model.pth')

    # Histories
    epochs_hist = []
    loss_hist   = []
    pde_hist    = []
    ic_hist     = []
    bc_hist     = []
    data_hist   = []
    val_hist    = []
    M_hist      = []

    start_time = time.time()
    for epoch in range(epochs+1):
        pde_l, ic_l, bc_l, data_l, _, tot_l = model.train_step(optimizer, batch_size)
        M_hist.append(model.M.detach().cpu().numpy().copy())

        # Validation every 100 epochs
        if epoch % 100 == 0:
            val_l = model.validate(
                X_collocation_val=X_coll_v_t,
                X_ic_val=X_ic_v_t,
                expected_u0_val=u0_v_t,
                X_boundary_val=X_bc_v_t,
                normal_vectors_val=norm_v_t
            )

            epochs_hist.append(epoch)
            # Convert losses/tensors to Python floats
            loss_hist.append(float(tot_l))
            pde_hist.append(float(pde_l))
            ic_hist.append(float(ic_l))
            bc_hist.append(float(bc_l))
            data_hist.append(float(data_l))
            val_hist.append(float(val_l))

            print(f"[Epoch {epoch}] Tot={tot_l:.2e} PDE={pde_l:.2e} IC={ic_l:.2e} BC={bc_l:.2e} Data={data_l:.2e} Val={val_l:.2e}")

            # Early stopping
            if val_l < best_val_loss:
                best_val_loss, no_improve = val_l, 0
                model.save_model(best_model_fp)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping triggered.")
                    break

    end_time = time.time()

    # Save training histories including data and validation losses
    np.savez_compressed(
        os.path.join(run_dir, 'training_history.npz'),
        epochs=np.array(epochs_hist),
        total_loss=np.array(loss_hist),
        pde_loss=np.array(pde_hist),
        ic_loss=np.array(ic_hist),
        bc_loss=np.array(bc_hist),
        data_loss=np.array(data_hist),
        val_loss=np.array(val_hist),
        M_history=np.stack(M_hist)
    )

    # Plot total train vs val loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_hist, loss_hist, label=r'$\mathcal{L}_{\mathrm{Train}}$')
    plt.plot(epochs_hist, val_hist,  label=r'$\mathcal{L}_{\mathrm{Val}}$')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'train_val_loss.png'), dpi=300)
    plt.close()

    # Plot loss components
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_hist, pde_hist,  label=r'$\mathcal{L}_{PDE}$')
    plt.plot(epochs_hist, ic_hist,   label=r'$\mathcal{L}_{IC}$')
    plt.plot(epochs_hist, bc_hist,   label=r'$\mathcal{L}_{BC}$')
    plt.plot(epochs_hist, data_hist, label=r'$\mathcal{L}_{Data}$')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Component')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'loss_components.png'), dpi=300)
    plt.close()

    # Plot M evolution with labels M_xx, M_xy, M_yx, M_yy
    # Align M_history to recorded epochs
    M_arr = np.stack([M_hist[i] for i in epochs_hist])
    plt.figure(figsize=(8, 5))
    labels = [r'$M_{xx}$', r'$M_{xy}$', r'$M_{yx}$', r'$M_{yy}$']
    for i in range(M_arr.shape[1]):
        plt.plot(epochs_hist, M_arr[:, i], label=labels[i])
    plt.xlabel('Epoch')
    plt.ylabel('Conductivity Tensor Components')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'M_evolution.png'), dpi=300)
    plt.close()

    # Reload best model for final evaluation
    model.load_model(best_model_fp)

    # Final evaluation at last time T
    X_test = np.column_stack((x_coords, y_coords, np.full_like(x_coords, T)))
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        y_pred_T = model(X_test_t).cpu().numpy().ravel()
    true_T = solutions_fem[time_points[-1]]  # use last available FEM timestep
    rmse_T = np.sqrt(mean_squared_error(true_T, y_pred_T))
    rel_rmse_T = np.sqrt(np.sum((y_pred_T - true_T)**2) / np.sum(true_T**2))

    # Global evaluation over all times
    all_true, all_pred = [], []
    for i, t_val in enumerate(time_points):
        all_true.append(fem_solutions[:, i])
        coords = np.column_stack((x_coords, y_coords, np.full_like(x_coords, t_val)))
        with torch.no_grad():
            all_pred.append(model(torch.tensor(coords, dtype=torch.float32, device=device)).cpu().numpy().ravel())
    all_true = np.hstack(all_true)
    all_pred = np.hstack(all_pred)
    global_rmse     = np.sqrt(np.mean((all_pred - all_true)**2))
    global_rel_rmse = np.sqrt(np.sum((all_pred - all_true)**2) / np.sum(all_true**2))

    # Extract predicted M (4 values)
    predicted_M = model.M.detach().cpu().numpy().tolist()

    # Save parameters
    params = {
        'rmse_T': rmse_T,
        'rel_rmse_T': rel_rmse_T,
        'global_rmse': global_rmse,
        'global_rel_rmse': global_rel_rmse,
        'predicted_M': predicted_M,
        'training_time_s': end_time - start_time
    }
    with open(os.path.join(run_dir, 'model_parameters.json'), 'w') as f:
        json.dump(params, f, indent=4)

    return global_rmse, global_rel_rmse, predicted_M


# =============================================================================
# Execute for Various Data/Noise Configurations
# =============================================================================
data_points_list = [100, 500, 1000]
noise_levels     = [0.00, 0.01, 0.05]

results = {'rmse': {}, 'rel_rmse': {}, 'predicted_M': {}}
for N_data in data_points_list:
    for noise in noise_levels:
        grmse, grel, Mvals = run_pinns_simulation(
            num_layers=3,
            num_neurons=512,
            N_data=N_data,
            noise_level=noise
        )
        results['rmse'][(N_data, noise)]        = grmse
        results['rel_rmse'][(N_data, noise)]    = grel
        results['predicted_M'][(N_data, noise)] = Mvals

# Save summary tables
with open(os.path.join(base_results_dir, 'test_rmse.txt'), 'w') as f:
    for N_data in data_points_list:
        line = " ".join(
            f"{results['rmse'][(N_data, noise)]:.6e}" for noise in noise_levels
        )
        f.write(line + "\n")

with open(os.path.join(base_results_dir, 'test_relative_rmse.txt'), 'w') as f:
    for N_data in data_points_list:
        line = " ".join(
            f"{results['rel_rmse'][(N_data, noise)]:.6e}" for noise in noise_levels
        )
        f.write(line + "\n")

with open(os.path.join(base_results_dir, 'predicted_M.txt'), 'w') as f:
    for N_data in data_points_list:
        line = " ".join(
            f"{val:.6e}" for val in results['predicted_M'][(N_data, noise)]
        )
        f.write(line + "\n")

print("All simulations complete. Metrics saved.")
