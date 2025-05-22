#!/usr/bin/env python3
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
from sklearn.metrics import mean_squared_error
from scipy.stats import qmc

# Project setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)
from utils.heart_solver_pinns import EnhancedMonodomainSolverPINNs

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# Results directory
base_results_dir = os.path.join(
    project_root,
    'monodomain_results',
    'forward_problems',
    'enhanced_pinns',
    'corner_case_3d_fhn'
)
os.makedirs(base_results_dir, exist_ok=True)

# ----------------------------------------------------------------------------
# FitzHugh–Nagumo reaction & ODE
# ----------------------------------------------------------------------------
def reaction_fhn(v, w):
    return v - v**3/3 - w

def ode_func_fhn(v, w, X):
    eps, a, b = 0.01, 0.2, 0.5
    dv = v - v**3/3 - w
    dw = eps*(v + a - b*w)
    return torch.cat([dv, dw], dim=1)

# ----------------------------------------------------------------------------
# Load FEM data
# ----------------------------------------------------------------------------
fem_data_file = os.path.join(base_results_dir, 'fem_data_3d_fhn.npz')
if not os.path.isfile(fem_data_file):
    raise FileNotFoundError(f"{fem_data_file} not found. Please export FEM data first.")

data = np.load(fem_data_file)
x_coords      = data['x_coords']
y_coords      = data['y_coords']
z_coords      = data['z_coords']
time_points   = data['time_points']
fem_solutions = data['fem_solutions']
if fem_solutions.shape[0] == len(time_points):
    fem_solutions = fem_solutions.T
solutions_fem = {float(t): fem_solutions[:,i] for i,t in enumerate(time_points)}
print(f"Loaded FEM data: {x_coords.size} points, {len(time_points)} time steps")

# Domain/time bounds
x_min,x_max = float(x_coords.min()), float(x_coords.max())
y_min,y_max = float(y_coords.min()), float(y_coords.max())
z_min,z_max = float(z_coords.min()), float(z_coords.max())
t_min,t_max = float(time_points.min()), float(time_points.max())
M_val        = 0.1  # diffusion coefficient

# Precompute for BC sampling
mins = [x_min, y_min, z_min]
maxs = [x_max, y_max, z_max]

# ----------------------------------------------------------------------------
# Sampling helpers
# ----------------------------------------------------------------------------
def lhs_sample(N, bounds):
    sampler = qmc.LatinHypercube(d=len(bounds))
    pts = sampler.random(n=N)
    arr = np.zeros((N,len(bounds)),dtype=np.float32)
    for i,(lo,hi) in enumerate(bounds):
        arr[:,i] = lo + (hi-lo)*pts[:,i]
    return arr

def sample_BC(N):
    """Generate Neumann BC points on all 6 faces, plus normals."""
    Np = N // 6
    Xb_list, nb_list = [], []
    faces = [
        (0, mins[0], [-1,0,0]), (0, maxs[0], [1,0,0]),
        (1, mins[1], [0,-1,0]), (1, maxs[1], [0,1,0]),
        (2, mins[2], [0,0,-1]), (2, maxs[2], [0,0,1]),
    ]
    for dim, val, nvec in faces:
        sampler = qmc.LatinHypercube(d=3)
        pts = sampler.random(n=Np)
        Xb = np.zeros((Np,4),dtype=np.float32)
        for i in range(3):
            if i==dim:
                Xb[:,i] = val
            else:
                # choose the correct column of the LHS sample
                idx = 0 if i<dim else 1
                Xb[:,i] = mins[i] + (maxs[i]-mins[i]) * pts[:,idx]
        # time coordinate
        Xb[:,3] = t_min + (t_max-t_min)*pts[:,2]
        Xb_list.append(Xb)
        nb_list.append(np.tile(nvec,(Np,1)))
    Xb  = np.vstack(Xb_list)
    Nb  = np.vstack(nb_list)
    return (
        torch.tensor(Xb, dtype=torch.float32, device=device),
        torch.tensor(Nb, dtype=torch.float32, device=device)
    )

# ----------------------------------------------------------------------------
# Fixed validation sets
# ----------------------------------------------------------------------------
Xcv, Xicv = lhs_sample(12000, [(x_min,x_max),(y_min,y_max),(z_min,z_max),(t_min,t_max)]), \
            lhs_sample(500,  [(x_min,x_max),(y_min,y_max),(z_min,z_max),(t_min,t_min)])
Xcv_t     = torch.tensor(Xcv, dtype=torch.float32, device=device)
Xicv_t    = torch.tensor(Xicv, dtype=torch.float32, device=device)
u0_icv    = np.hstack([
    (np.linalg.norm(Xicv[:,:3]-np.array([0.5,0.5,0.5]),axis=1)<0.33).astype(np.float32).reshape(-1,1),
    np.zeros((Xicv.shape[0],1),dtype=np.float32)
])
u0_icv_t  = torch.tensor(u0_icv, dtype=torch.float32, device=device)
Xbv_t, nbv_t = sample_BC(1000)

# ----------------------------------------------------------------------------
# Main enhanced-PINN run
# ----------------------------------------------------------------------------
def run_enhanced(num_layers, num_neurons, scaling_par=None):
    run_dir    = os.path.join(base_results_dir, f"layers={num_layers}_neurons={num_neurons}")
    os.makedirs(run_dir, exist_ok=True)
    params_file= os.path.join(run_dir,'model_parameters.json')

    # instantiate model
    model = EnhancedMonodomainSolverPINNs(
        num_inputs        = 4,
        num_layers        = num_layers,
        num_neurons       = num_neurons,
        device            = device,
        source_term_func  = reaction_fhn,
        M                 = M_val,
        use_ode           = True,
        ode_func          = ode_func_fhn,
        n_state_vars      = 1,
        loss_function     = 'L2',
        weight_strategy   = 'dynamic',
        alpha             = 0.9,
        use_fourier       = True,
        fourier_dim       = 256,
        use_rwf           = True,
        mu                = 2.0,
        sigma_rwf         = 0.1,
        M_segments        = 4,
        t_min             = t_min,
        t_max             = t_max,
        epsilon           = 1.0,
        f                 = 1000,
        curriculum_enabled= True,
        weight_update_batch_size=2048
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # generate training sets once
    Xc = lhs_sample(40000, [(x_min,x_max),(y_min,y_max),(z_min,z_max),(t_min,t_max)])
    Xi = lhs_sample(2000,  [(x_min,x_max),(y_min,y_max),(z_min,z_max),(t_min,t_min)])
    Xb, Nb = sample_BC(6000)
    Xc_t = torch.tensor(Xc, dtype=torch.float32, device=device)
    Xi_t = torch.tensor(Xi, dtype=torch.float32, device=device)
    u0_np= np.hstack([
        (np.linalg.norm(Xi[:,:3]-np.array([0.5,0.5,0.5]),axis=1)<0.33).astype(np.float32).reshape(-1,1),
        np.zeros((Xi.shape[0],1),dtype=np.float32)
    ])
    u0_t = torch.tensor(u0_np, dtype=torch.float32, device=device)

    # attach
    model.X_collocation = Xc_t
    model.X_ic          = Xi_t
    model.expected_u0   = u0_t
    model.X_boundary    = Xb
    model.normal_vectors= Nb

    # training loop
    epochs, patience = 1, 1000
    best_val, wait   = float('inf'), 0
    history = {'epoch':[], 'pde':[], 'ic':[], 'bc':[], 'ode':[], 'tot':[], 'val':[]}
    t0 = time.time()
    for ep in range(epochs+1):
        pde_l, ic_l, bc_l, ode_l, tot_l = model.train_step(
            optimizer,
            Xc_t, Xi_t, u0_t, Xb, Nb, ep
        )
        if ep % 100 == 0:
            val_l = model.validate(Xcv_t, Xicv_t, u0_icv_t, Xbv_t, nbv_t)
            history['epoch'].append(ep)
            history['pde'].append(pde_l)
            history['ic' ].append(ic_l)
            history['bc' ].append(bc_l)
            history['ode'].append(ode_l)
            history['tot'].append(tot_l)
            history['val'].append(val_l)
            print(f"[{ep:5d}] PDE={pde_l:.2e} IC={ic_l:.2e} BC={bc_l:.2e} ODE={ode_l:.2e} TOT={tot_l:.2e} VAL={val_l:.2e}")
            if val_l < best_val:
                best_val, wait = val_l, 0
                model.save_model(os.path.join(run_dir,'best.pth'))
            else:
                wait += 1
                if wait >= patience:
                    print("→ Early stopping")
                    break
    comp_time = time.time()-t0

    # save history & plots
    np.savez_compressed(
        os.path.join(run_dir,'loss_history.npz'),
        epoch=np.array(history['epoch']),
        pde=np.array(history['pde']),
        ic =np.array(history['ic']),
        bc =np.array(history['bc']),
        ode=np.array(history['ode']),
        tot=np.array(history['tot']),
        val=np.array(history['val'])
    )
    plt.figure(figsize=(10,6))
    plt.plot(history['epoch'],history['tot'],label='Train Total')
    plt.plot(history['epoch'],history['val'],label='Val Total')
    plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(run_dir,'loss_plot.png'),dpi=300); plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(history['epoch'],history['pde'], label='PDE')
    plt.plot(history['epoch'],history['ic'],  label='IC')
    plt.plot(history['epoch'],history['bc'],  label='BC')
    plt.plot(history['epoch'],history['ode'], label='ODE')
    plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Loss Components')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(run_dir,'loss_components_plot.png'),dpi=300); plt.close()

    # final evaluation over the whole FEM grid
    model.load_model(os.path.join(run_dir,'best.pth'))
    Np, Nt = x_coords.size, time_points.size
    preds = np.zeros((Np,Nt),dtype=np.float32)
    for i,tv in enumerate(time_points):
        Xe = np.column_stack([x_coords,y_coords,z_coords,np.full_like(x_coords,tv)])
        with torch.no_grad():
            out = model(torch.tensor(Xe,dtype=torch.float32,device=device))
        preds[:,i] = out[:,0].cpu().numpy()

    fem_all   = fem_solutions.flatten()
    pinns_all = preds.flatten()
    rmse      = np.sqrt(mean_squared_error(fem_all,pinns_all))
    rrmse     = np.sqrt(((fem_all-pinns_all)**2).sum()/((fem_all**2).sum()+1e-16))

    # save parameters
    params = {
        'layers':        num_layers,
        'neurons':       num_neurons,
        'training_time': comp_time,
        'global_rmse':   float(rmse),
        'global_rel_rmse':float(rrmse)
    }
    with open(params_file,'w') as fp:
        json.dump(params, fp, indent=2)

    # slice-plane plots at z≈0.5
    mask = np.abs(z_coords-0.5)<1e-2
    tri  = Triangulation(x_coords[mask], y_coords[mask])
    for tt in [0.1,0.3,0.6,0.9]:
        idx = int(np.argmin(np.abs(time_points-tt)))
        fem_v  = fem_solutions[mask,idx]
        pinn_v = preds[mask,idx]
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.tricontourf(tri,fem_v,levels=50); plt.title(f"FEM t={tt}")
        plt.colorbar(shrink=0.6)
        plt.subplot(1,2,2)
        plt.tricontourf(tri,pinn_v,levels=50); plt.title(f"PINN t={tt}")
        plt.colorbar(shrink=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir,f"slice_t{tt:.2f}.png"),dpi=300)
        plt.close()

    print(f"→ layers={num_layers}, neurons={num_neurons}, RMSE={rmse:.3e}, RRMSE={rrmse:.3e}")
    return rmse, rrmse

# =============================================================================
# Sweep architectures
# =============================================================================
test_rmse, test_rrmse = {}, {}
for L in (4,5):
    for N in (256,512):
        r, rr = run_enhanced(L,N)
        test_rmse[(L,N)]    = r
        test_rrmse[(L,N)]   = rr

# save summaries
with open(os.path.join(base_results_dir,'test_rmse.txt'),'w') as f:
    for N in (256, 512):
        f.write(" ".join(f"{test_rmse[(L,N)]:.6e}" for L in (4,5))+"\n")
with open(os.path.join(base_results_dir,'test_rel_rmse.txt'),'w') as f:
    for N in (256,512):
        f.write(" ".join(f"{test_rrmse[(L,N)]:.6e}" for L in (4,5))+"\n")

print("All enhanced-PINN simulations complete.")
