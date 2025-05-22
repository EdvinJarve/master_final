#!/usr/bin/env python
import sys, os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from sklearn.metrics import mean_squared_error
import torch.optim as optim

# -----------------------------------------------------------------------------
# Project Setup
# -----------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)
base_results_dir = os.path.join(
    project_root,
    'monodomain_results', 'forward_problems', 'enhanced_pinns', 'corner_case_smooth'
)
os.makedirs(base_results_dir, exist_ok=True)
utils_path = os.path.join(project_root, 'utils')
sys.path.append(utils_path)

from utils.heart_solver_pinns import EnhancedMonodomainSolverPINNs

torch.manual_seed(42)
np.random.seed(42)
device = 'cuda'

# -----------------------------------------------------------------------------
# Load FEM data
# -----------------------------------------------------------------------------
fem_data_file = os.path.join(base_results_dir, 'fem_data.npz')
if not os.path.isfile(fem_data_file):
    raise FileNotFoundError(f"{fem_data_file} not found. Please ensure fem_data is already exported.")

fem_data      = np.load(fem_data_file)
x_coords      = fem_data['x_coords']
y_coords      = fem_data['y_coords']
time_points   = fem_data['time_points']
fem_solutions = fem_data['fem_solutions']  # shape: (N_dofs, N_times)

solutions_fem = {t: fem_solutions[:, i] for i, t in enumerate(time_points)}
triang        = Triangulation(x_coords, y_coords)
print(f"FEM data loaded from {fem_data_file}.")

# -----------------------------------------------------------------------------
# Domain and Problem Setup
# -----------------------------------------------------------------------------
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
T            = 1.0
t_min, t_max = 0.0, T
M_val        = 1.0

# -----------------------------------------------------------------------------
# Source Term
# -----------------------------------------------------------------------------
def source_term_func_pinns(x_spatial, t):
    x0, y0, sigma = 0.3, 0.7, 0.05
    t_on_start, t_on_end = 0.05, 0.3
    t_off_start, t_off_end = 0.4, 0.7

    x       = x_spatial[:, 0:1]
    y       = x_spatial[:, 1:2]
    gaussian = 50.0 * torch.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    time_window = torch.zeros_like(t)
    ramp_up     = (t - t_on_start) / (t_on_end - t_on_start)
    ramp_down   = 1.0 - (t - t_off_start) / (t_off_end - t_off_start)

    time_window = torch.where((t >= t_on_start)&(t < t_on_end), ramp_up, time_window)
    time_window = torch.where((t >= t_on_end)&(t < t_off_start), torch.ones_like(t), time_window)
    time_window = torch.where((t >= t_off_start)&(t <= t_off_end), ramp_down, time_window)

    return gaussian * time_window

# -----------------------------------------------------------------------------
# Sampling Helpers
# -----------------------------------------------------------------------------
def sample_collocation_points(N, device='cpu'):
    X = torch.empty((N, 3), device=device)
    X[:,0] = x_min + (x_max - x_min)*torch.rand(N, device=device)
    X[:,1] = y_min + (y_max - y_min)*torch.rand(N, device=device)
    X[:,2] = t_min + (t_max - t_min)*torch.rand(N, device=device)
    return X

def sample_IC_points(N, device='cpu'):
    X = torch.empty((N, 3), device=device)
    X[:,0] = x_min + (x_max - x_min)*torch.rand(N, device=device)
    X[:,1] = y_min + (y_max - y_min)*torch.rand(N, device=device)
    X[:,2] = t_min
    return X

def sample_BC_points_with_normals(N, device='cpu'):
    Np = N // 4
    # left
    Xl = torch.empty((Np,3), device=device); Xl[:,0]=x_min
    Xl[:,1]= y_min + (y_max - y_min)*torch.rand(Np, device=device)
    Xl[:,2]= t_min + (t_max - t_min)*torch.rand(Np, device=device)
    nl = torch.tensor([[-1.,0.]], device=device).repeat(Np,1)
    # right
    Xr = torch.empty((Np,3), device=device); Xr[:,0]=x_max
    Xr[:,1]= y_min + (y_max - y_min)*torch.rand(Np, device=device)
    Xr[:,2]= t_min + (t_max - t_min)*torch.rand(Np, device=device)
    nr = torch.tensor([[1.,0.]], device=device).repeat(Np,1)
    # bottom
    Xb = torch.empty((Np,3), device=device); Xb[:,1]=y_min
    Xb[:,0]= x_min + (x_max - x_min)*torch.rand(Np, device=device)
    Xb[:,2]= t_min + (t_max - t_min)*torch.rand(Np, device=device)
    nb = torch.tensor([[0.,-1.]], device=device).repeat(Np,1)
    # top
    Xt = torch.empty((Np,3), device=device); Xt[:,1]=y_max
    Xt[:,0]= x_min + (x_max - x_min)*torch.rand(Np, device=device)
    Xt[:,2]= t_min + (t_max - t_min)*torch.rand(Np, device=device)
    nt = torch.tensor([[0.,1.]], device=device).repeat(Np,1)
    Xb_all = torch.cat([Xl,Xr,Xb,Xt], dim=0)
    nb_all = torch.cat([nl,nr,nb,nt], dim=0)
    return Xb_all, nb_all

# -----------------------------------------------------------------------------
# Fixed Validation Sets
# -----------------------------------------------------------------------------
Xc_val, Xi_val = sample_collocation_points(10000,device=device), sample_IC_points(100,device=device)
Xb_val, nb_val = sample_BC_points_with_normals(100, device=device)

# -----------------------------------------------------------------------------
# Main Training & Evaluation
# -----------------------------------------------------------------------------
def run_pinns_simulation(num_layers, num_neurons):
    run_dir     = os.path.join(base_results_dir, f"layers={num_layers}_neurons={num_neurons}")
    os.makedirs(run_dir, exist_ok=True)
    params_file = os.path.join(run_dir, 'model_params.json')

    # instantiate model
    model = EnhancedMonodomainSolverPINNs(
        num_inputs=3, num_layers=num_layers, num_neurons=num_neurons,
        device=device, source_term_func=source_term_func_pinns,
        M=M_val, use_ode=False, ode_func=None, n_state_vars=0,
        loss_function='L2', weight_strategy='dynamic', alpha=0.9,
        use_fourier=True, fourier_dim=256, sigma=1.0,
        use_rwf=True, mu=1.0, sigma_rwf=0.1,
        M_segments=4, t_min=t_min, t_max=T,
        epsilon=1.0, f=1000, curriculum_enabled=True
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs     = 35000
    batch_size = 8192
    patience   = 1000
    best_val   = float('inf')
    wait       = 0
    best_path  = os.path.join(run_dir, 'best.pth')

    # storage for losses
    epoch_list            = []
    weighted_total_list   = []
    unweighted_total_list = []
    val_list              = []
    pde_raw_list          = []
    ic_list               = []
    bc_list               = []

    start = time.time()
    for epoch in range(epochs+1):
        # sample batches
        Xc = sample_collocation_points(batch_size, device=device)
        Xi = sample_IC_points(batch_size//3,      device=device)
        u0 = torch.zeros((Xi.shape[0],1),device=device)
        Xb, nb = sample_BC_points_with_normals(batch_size//3, device=device)

        # raw losses
        pde_raw, _ = model.pde(Xc)
        ic_raw     = model.IC(Xi, u0).item()
        bc_raw     = model.BC_neumann(Xb, nb).item()

        # train step returns (weighted_pde, ic_raw, bc_raw, data_loss, total_weighted)
        _, _, _, _, total_w = model.train_step(
            optimizer, Xc, Xi, u0, Xb, nb, epoch
        )

        # validation weighted
        vpde, _     = model.pde(Xc_val)
        vic         = model.IC(Xi_val, torch.zeros((Xi_val.shape[0],1),device=device))
        vbc         = model.BC_neumann(Xb_val, nb_val)
        val_w       = (model.lambda_ic*vic + model.lambda_bc*vbc + model.lambda_r*vpde).item()

        if epoch % 100 == 0:
            epoch_list.append(epoch)
            weighted_total_list.append(total_w)
            unweighted_total_list.append(pde_raw.item() + ic_raw + bc_raw)
            val_list.append(val_w)
            pde_raw_list.append(pde_raw.item())
            ic_list.append(ic_raw)
            bc_list.append(bc_raw)

            print(f"Epoch {epoch:5d} | W_total: {total_w:.3e} | U_total: {(pde_raw+ic_raw+bc_raw):.3e} | Val_w: {val_w:.3e}")

            if val_w < best_val:
                best_val = val_w; wait = 0
                model.save_model(best_path)
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping"); break

    end = time.time()
    training_time = end - start
    model.load_model(best_path)

    # save history
    np.savez_compressed(
        os.path.join(run_dir, 'loss_history.npz'),
        epoch=np.array(epoch_list),
        weighted_total=np.array(weighted_total_list),
        unweighted_total=np.array(unweighted_total_list),
        val_weighted=np.array(val_list),
        pde_raw=np.array(pde_raw_list),
        ic=np.array(ic_list),
        bc=np.array(bc_list)
    )

    # 1) plot weighted vs unweighted total loss
    plt.figure(figsize=(10,6))
    plt.plot(epoch_list, weighted_total_list,   label="Train Weighted Total")
    plt.plot(epoch_list, unweighted_total_list, label="Train Unweighted Total")
    plt.yscale("log")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Weighted vs Unweighted Total Loss")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(run_dir, "total_loss_compare.png"), dpi=300)
    plt.close()

    # 2) plot raw components
    plt.figure(figsize=(10,6))
    plt.plot(epoch_list, pde_raw_list, label="PDE (raw)")
    plt.plot(epoch_list, ic_list,      label="IC  (raw)")
    plt.plot(epoch_list, bc_list,      label="BC  (raw)")
    plt.yscale("log")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Unweighted Loss Components")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(run_dir, "components_loss.png"), dpi=300)
    plt.close()

    # ====================== evaluation metrics ======================
    Np, Nt = x_coords.shape[0], time_points.shape[0]
    preds = np.zeros((Np,Nt),dtype=np.float32)
    for i,tv in enumerate(time_points):
        Xe   = np.column_stack((x_coords, y_coords, np.full_like(x_coords, tv)))
        Xt   = torch.tensor(Xe, dtype=torch.float32, device=device)
        with torch.no_grad():
            preds[:,i] = model(Xt).cpu().numpy().ravel()

    fem_all   = fem_solutions.reshape(-1)
    pinns_all = preds.reshape(-1)
    rmse      = np.sqrt(mean_squared_error(fem_all, pinns_all))
    relrmse   = np.sqrt(np.sum((fem_all-pinns_all)**2) / (np.sum(fem_all**2)+1e-16))
    print(f"Global RMSE: {rmse:.3e}, Rel RMSE: {relrmse:.3e}")

    # save params + metrics
    params = {
        "layers": num_layers,
        "neurons": num_neurons,
        "training_time": training_time,
        "global_rmse": rmse, 
        "global_relative_rmse": relrmse
    }
    with open(params_file, "w") as f:
        json.dump(params, f, indent=4)

    return rmse, relrmse

if __name__=="__main__":
    for nl in [3]:
        for nn in [256]:
            run_pinns_simulation(nl, nn)
    print("All simulations complete.")
