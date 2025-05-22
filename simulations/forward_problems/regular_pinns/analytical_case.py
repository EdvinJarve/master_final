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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import qmc

# =============================================================================
# Project Setup (Paths, Directories)
# =============================================================================

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)

base_results_dir = os.path.join(
    project_root,
    'monodomain_results',
    'forward_problems',
    'regular_pinns',
    'analytical_case'
)
os.makedirs(base_results_dir, exist_ok=True)

utils_path = os.path.join(project_root, 'utils')
sys.path.append(utils_path)
from utils.heart_solver_pinns import MonodomainSolverPINNs

torch.manual_seed(42)
np.random.seed(42)

device = 'cuda'
print(torch.version.cuda)
print(torch.cuda.is_available())

# =============================================================================
# Problem Setup
# =============================================================================

x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
T = 1.0
t_min, t_max = 0.0, T

Nx, Ny, Nt = 100, 100, 100
dt = T / Nt
M = 1.0

def analytical_solution_v(x, y, t):
    return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.sin(t)

def source_term_func_pinns(x_spatial, t):
    pi = torch.pi
    x = x_spatial[:, 0:1]
    y = x_spatial[:, 1:2]
    return (
        8 * pi**2 * torch.cos(2 * pi * x) * torch.cos(2 * pi * y) * torch.sin(t)
        + torch.cos(2 * pi * x) * torch.cos(2 * pi * y) * torch.cos(t)
    )

# =============================================================================
# Load FEM Data
# =============================================================================

fem_data_file = os.path.join(base_results_dir, 'fem_data.npz')
if not os.path.isfile(fem_data_file):
    raise FileNotFoundError(f"{fem_data_file} not found. Please ensure fem_data is already exported.")

fem_data = np.load(fem_data_file)
x_coords = fem_data['x_coords']
y_coords = fem_data['y_coords']
time_points = fem_data['time_points']
fem_solutions = fem_data['fem_solutions']

solutions_fem = {t: fem_solutions[:, i] for i, t in enumerate(time_points)}
triang = Triangulation(x_coords, y_coords)
print(f"FEM data loaded from {fem_data_file}.")

# =============================================================================
# PINNs Data Generation Using LHS
# =============================================================================

N_collocation = 20000
N_ic          = 4000
N_bc          = 4000
N_val         = 2000
N_test        = 2000
N_collocation_val = 4000
N_ic_val      = 100
N_bc_val      = 100

# collocation
sampler = qmc.LatinHypercube(d=3)
sample = sampler.random(n=N_collocation)
X_collocation = np.empty_like(sample)
X_collocation[:,0] = x_min + (x_max-x_min)*sample[:,0]
X_collocation[:,1] = y_min + (y_max-y_min)*sample[:,1]
X_collocation[:,2] = t_min + (t_max-t_min)*sample[:,2]

# validation collocation
sampler_val = qmc.LatinHypercube(d=3)
sv = sampler_val.random(n=N_collocation_val)
X_collocation_val = np.empty_like(sv)
X_collocation_val[:,0] = x_min + (x_max-x_min)*sv[:,0]
X_collocation_val[:,1] = y_min + (y_max-y_min)*sv[:,1]
X_collocation_val[:,2] = t_min + (t_max-t_min)*sv[:,2]

# IC
sampler_ic = qmc.LatinHypercube(d=2)
s_ic = sampler_ic.random(n=N_ic)
X_ic = np.empty((N_ic,3))
X_ic[:,0] = x_min + (x_max-x_min)*s_ic[:,0]
X_ic[:,1] = y_min + (y_max-y_min)*s_ic[:,1]
X_ic[:,2] = 0.0
expected_u0 = analytical_solution_v(X_ic[:,0], X_ic[:,1], X_ic[:,2]).reshape(-1,1)

# IC val
sampler_ic_val = qmc.LatinHypercube(d=2)
s_icv = sampler_ic_val.random(n=N_ic_val)
X_ic_val = np.empty((N_ic_val,3))
X_ic_val[:,0] = x_min + (x_max-x_min)*s_icv[:,0]
X_ic_val[:,1] = y_min + (y_max-y_min)*s_icv[:,1]
X_ic_val[:,2] = 0.0
expected_u0_val = analytical_solution_v(X_ic_val[:,0], X_ic_val[:,1], X_ic_val[:,2]).reshape(-1,1)

# BC helper
def generate_boundary(Np, x_fixed=None, y_fixed=None):
    sm = qmc.LatinHypercube(d=2).random(n=Np)
    Xb = np.zeros((Np,3))
    if x_fixed is not None:
        Xb[:,0] = x_fixed
        Xb[:,1] = y_min + (y_max-y_min)*sm[:,0]
    else:
        Xb[:,0] = x_min + (x_max-x_min)*sm[:,0]
        Xb[:,1] = y_fixed
    Xb[:,2] = t_min + (t_max-t_min)*sm[:,1]
    return Xb

Np_b = N_bc//4
X_left   = generate_boundary(Np_b, x_fixed=x_min)
X_right  = generate_boundary(Np_b, x_fixed=x_max)
X_bottom = generate_boundary(Np_b, y_fixed=y_min)
X_top    = generate_boundary(Np_b, y_fixed=y_max)
X_boundary = np.vstack([X_left, X_right, X_bottom, X_top])

norms = np.vstack([
    np.tile([-1.0,0.0],(Np_b,1)),
    np.tile([ 1.0,0.0],(Np_b,1)),
    np.tile([ 0.0,-1.0],(Np_b,1)),
    np.tile([ 0.0, 1.0],(Np_b,1))
])

# val BC
Np_bv = N_bc_val//4
X_left_val   = generate_boundary(Np_bv, x_fixed=x_min)
X_right_val  = generate_boundary(Np_bv, x_fixed=x_max)
X_bottom_val = generate_boundary(Np_bv, y_fixed=y_min)
X_top_val    = generate_boundary(Np_bv, y_fixed=y_max)
X_boundary_val = np.vstack([X_left_val, X_right_val, X_bottom_val, X_top_val])

norms_val = np.vstack([
    np.tile([-1.0,0.0],(Np_bv,1)),
    np.tile([ 1.0,0.0],(Np_bv,1)),
    np.tile([ 0.0,-1.0],(Np_bv,1)),
    np.tile([ 0.0, 1.0],(Np_bv,1))
])

# to tensors
X_collocation_tensor    = torch.tensor(X_collocation,    dtype=torch.float32).to(device)
X_ic_tensor             = torch.tensor(X_ic,             dtype=torch.float32).to(device)
expected_u0_tensor      = torch.tensor(expected_u0,      dtype=torch.float32).to(device)
X_boundary_tensor       = torch.tensor(X_boundary,       dtype=torch.float32).to(device)
normal_vectors_tensor   = torch.tensor(norms,            dtype=torch.float32).to(device)

X_collocation_val_tensor = torch.tensor(X_collocation_val, dtype=torch.float32).to(device)
X_ic_val_tensor          = torch.tensor(X_ic_val,          dtype=torch.float32).to(device)
expected_u0_val_tensor   = torch.tensor(expected_u0_val,   dtype=torch.float32).to(device)
X_boundary_val_tensor    = torch.tensor(X_boundary_val,    dtype=torch.float32).to(device)
normal_vectors_val_tensor= torch.tensor(norms_val,         dtype=torch.float32).to(device)

def run_pinns_simulation(num_layers, num_neurons, scaling_par=None):
    run_dir = os.path.join(
        base_results_dir,
        f"results_layers={num_layers}_depth={num_neurons}"
    )
    os.makedirs(run_dir, exist_ok=True)

    model_params = {
        'num_inputs': 3,
        'num_layers': num_layers,
        'num_neurons': num_neurons,
        'use_ode': False,
        'n_state_vars': 0,
        'loss_function': 'L2',
        'weight_strategy': 'dynamic',
        'alpha': 0.9,
    }
    params_file = os.path.join(run_dir, 'model_parameters.json')

    model = MonodomainSolverPINNs(
        num_inputs     = model_params['num_inputs'],
        num_layers     = model_params['num_layers'],
        num_neurons    = model_params['num_neurons'],
        device         = device,
        source_term_func = source_term_func_pinns,
        M              = M,
        use_ode        = model_params['use_ode'],
        ode_func       = None,
        n_state_vars   = model_params['n_state_vars'],
        loss_function  = model_params['loss_function'],
        weight_strategy= model_params['weight_strategy'],
        scaling_func   = scaling_par
    )

    print("Starting simulation...")

    model.X_collocation = X_collocation_tensor
    model.X_ic          = X_ic_tensor
    model.expected_u0   = expected_u0_tensor
    model.X_boundary    = X_boundary_tensor
    model.normal_vectors= normal_vectors_tensor
    model.X_data        = None
    model.expected_data = None

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    epochs = 20000
    batch_size = 2048
    patience = 1000
    best_val_loss = float('inf')
    no_improve = 0
    best_model_path = os.path.join(run_dir, 'best_model.pth')

    loss_hist = []
    val_loss_hist = []
    pde_hist = []
    ic_hist = []
    bc_hist = []
    epochs_list = []

    t_start = time.time()
    for epoch in range(epochs+1):
        pde_loss, ic_loss, bc_loss, data_loss_val, ode_loss, tot_loss = \
            model.train_step(optimizer, batch_size)

        model.eval()
        val_loss = model.validate(
            X_collocation_val = X_collocation_val_tensor,
            X_ic_val          = X_ic_val_tensor,
            expected_u0_val   = expected_u0_val_tensor,
            X_boundary_val    = X_boundary_val_tensor,
            normal_vectors_val= normal_vectors_val_tensor
        )

        if epoch % 100 == 0:
            loss_hist.append(tot_loss)
            val_loss_hist.append(val_loss)
            pde_hist.append(pde_loss)
            ic_hist.append(ic_loss)
            bc_hist.append(bc_loss)
            epochs_list.append(epoch)
            print(
                f"Epoch {epoch}, PDE: {pde_loss:.4e}, IC: {ic_loss:.4e}, "
                f"BC: {bc_loss:.4e}, ODE: {ode_loss:.4e}, "
                f"Train: {tot_loss:.4e}, Val: {val_loss:.4e}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                model.save_model(best_model_path)
                print(f"  Saved new best model (val {best_val_loss:.4e})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("  Early stopping.")
                    break

    t_end = time.time()
    comp_time = t_end - t_start

    # save loss history
    np.savez_compressed(
        os.path.join(run_dir, 'training_losses_analytical.npz'),
        epochs=np.array(epochs_list),
        total_loss=np.array(loss_hist),
        val_loss=np.array(val_loss_hist),
        pde_loss=np.array(pde_hist),
        ic_loss=np.array(ic_hist),
        bc_loss=np.array(bc_hist)
    )

    model.load_model(best_model_path)

    # plots
    plt.figure(figsize=(10,6))
    plt.plot(epochs_list, loss_hist, label='Train')
    plt.plot(epochs_list, val_loss_hist, label='Val')
    plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(run_dir,'loss_plot.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(epochs_list, pde_hist, label=r'$\mathcal{L}_{PDE}$')
    plt.plot(epochs_list, ic_hist, label=r'$\mathcal{L}_{IC}$')
    plt.plot(epochs_list, bc_hist, label=r'$\mathcal{L}_{BC}$')
    plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(run_dir,'loss_components_plot.png'), dpi=300)
    plt.close()

    # final evaluation
    X_test = np.column_stack((x_coords, y_coords, np.full_like(x_coords, T)))
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    t_val_start = time.time()
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy().reshape(-1)
    val_time = time.time() - t_val_start

    u_true = analytical_solution_v(x_coords, y_coords, T)
    mse = mean_squared_error(u_true, y_pred)
    rmse = np.sqrt(mse)
    rel_mse = np.sum((y_pred - u_true)**2) / np.sum(u_true**2)
    rel_rmse = np.sqrt(rel_mse)

    model_params.update({
        'validation_time': val_time,
        'computation_time': comp_time,
        'rmse': rmse,
        'relative_rmse': rel_rmse
    })
    with open(params_file, 'w') as f:
        json.dump(model_params, f, indent=4)

    # comparison plots
    for tt in [0.0, 0.1, 0.2, 0.4, 0.8, 1.0]:
        t_key = round(tt,2)
        if t_key not in solutions_fem:
            continue
        fem_v = solutions_fem[t_key]
        X_ev = np.column_stack((x_coords, y_coords, np.full_like(x_coords, tt)))
        X_ev_t = torch.tensor(X_ev, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_p = model.evaluate(X_ev_t).cpu().numpy().reshape(-1)
        u_t = analytical_solution_v(x_coords, y_coords, tt)
        err_p = np.abs(y_p - u_t)
        err_f = np.abs(fem_v - u_t)

        fig, axs = plt.subplots(2,2,figsize=(16,12))
        fig.suptitle(f'Comparisons at t = {tt}')
        cs = axs[0,0].tricontourf(triang, y_p,   levels=50, cmap='viridis')
        fig.colorbar(cs,ax=axs[0,0]).set_label('PINN Pred')
        axs[0,0].set_title('PINN Prediction')
        cs = axs[0,1].tricontourf(triang, err_p, levels=50, cmap='viridis')
        fig.colorbar(cs,ax=axs[0,1]).set_label('PINN Abs Error')
        axs[0,1].set_title('PINN Error')
        cs = axs[1,0].tricontourf(triang, fem_v, levels=50, cmap='viridis')
        fig.colorbar(cs,ax=axs[1,0]).set_label('FEM Pred')
        axs[1,0].set_title('FEM Prediction')
        cs = axs[1,1].tricontourf(triang, err_f, levels=50, cmap='viridis')
        fig.colorbar(cs,ax=axs[1,1]).set_label('FEM Error')
        axs[1,1].set_title('FEM Error')
        for ax in axs.flatten():
            ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.tight_layout(rect=[0,0.03,1,0.95])
        plt.savefig(os.path.join(run_dir,f'comparison_t_{tt}.png'), dpi=300)
        plt.close()
        print(f"Plots saved for t = {tt}")

    return rmse, rel_rmse

# =============================================================================
# Main Loop Over Architectures
# =============================================================================

num_layers_list   = [3,4,5]
num_neurons_list  = [128,256,512]
test_rmse_dict    = {}
test_rel_rmse_dict= {}

for depth in num_neurons_list:
    for layers in num_layers_list:
        r, rr = run_pinns_simulation(layers, depth, scaling_par=None)
        test_rmse_dict[(layers,depth)]     = r
        test_rel_rmse_dict[(layers,depth)] = rr

# Write RMSE
test_rmse_file = os.path.join(base_results_dir,'test_rmse.txt')
with open(test_rmse_file,'w') as f:
    for depth in num_neurons_list:
        row = [f"{test_rmse_dict[(layers,depth)]:.6e}" for layers in num_layers_list]
        f.write(" ".join(row)+"\n")

# Write relative RMSE
test_rel_rmse_file = os.path.join(base_results_dir,'test_relative_rmse.txt')
with open(test_rel_rmse_file,'w') as f:
    for depth in num_neurons_list:
        row = [f"{test_rel_rmse_dict[(layers,depth)]:.6e}" for layers in num_layers_list]
        f.write(" ".join(row)+"\n")

print("All simulations complete. Test RMSE and relative RMSE values saved.")
