#!/usr/bin/env python3
import sys, os, time, json
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
sys.path.append(project_root)
from utils.heart_solver_pinns import MonodomainSolverPINNs

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# Results dir
base_results_dir = os.path.join(
    project_root,
    'monodomain_results',
    'forward_problems',
    'regular_pinns',
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
data = np.load(fem_data_file)
x_coords      = data['x_coords']    # (N,)
y_coords      = data['y_coords']
z_coords      = data['z_coords']
time_points   = data['time_points']  # (T,)
fem_solutions = data['fem_solutions']  # (N, T)
# transpose if needed
if fem_solutions.shape[0] == len(time_points):
    fem_solutions = fem_solutions.T
solutions_fem = {float(t): fem_solutions[:,i] for i,t in enumerate(time_points)}
print(f"Loaded FEM data: {x_coords.size} points, {len(time_points)} time steps")

# Domain/time bounds
x_min,x_max = 0.0,1.0
y_min,y_max = 0.0,1.0
z_min,z_max = 0.0,1.0
t_min,t_max = float(time_points.min()), float(time_points.max())
M = 0.1

# ----------------------------------------------------------------------------
# Latin Hypercube sampling helper
# ----------------------------------------------------------------------------
def lhs_sample(N, bounds):
    sampler = qmc.LatinHypercube(d=len(bounds))
    pts = sampler.random(n=N)
    arr = np.zeros((N,len(bounds)),dtype=np.float32)
    for i,(low,high) in enumerate(bounds):
        arr[:,i] = low + (high-low)*pts[:,i]
    return arr

# ----------------------------------------------------------------------------
# Generate collocation, IC, BC
# ----------------------------------------------------------------------------
# Collocation
N_coll, N_coll_val = 120_000, 12000
bounds_coll = [(x_min,x_max),(y_min,y_max),(z_min,z_max),(t_min,t_max)]
X_coll     = lhs_sample(N_coll,     bounds_coll)
X_coll_val = lhs_sample(N_coll_val, bounds_coll)

# Initial Conditions at t=0
N_ic, N_ic_val = 5000, 500
bounds_ic      = [(x_min,x_max),(y_min,y_max),(z_min,z_max),(0.0,0.0)]
X_ic     = lhs_sample(N_ic,     bounds_ic)
X_ic_val = lhs_sample(N_ic_val, bounds_ic)

def initial_v_ball_np(X):
    pts = X[:,:3]
    center = np.array([0.5,0.5,0.5],dtype=np.float32)
    r = np.linalg.norm(pts - center[None,:],axis=1)
    v0 = np.zeros_like(r,dtype=np.float32)
    v0[r<0.33] = 1.0
    return v0.reshape(-1,1)

u0_ic     = np.hstack([ initial_v_ball_np(X_ic),     np.zeros((N_ic,1),dtype=np.float32) ])
u0_ic_val = np.hstack([ initial_v_ball_np(X_ic_val), np.zeros((N_ic_val,1),dtype=np.float32) ])

# Boundary Conditions on 6 faces
def gen_bc(N, dim, val):
    sampler = qmc.LatinHypercube(d=3)
    pts = sampler.random(n=N)
    Xb = np.zeros((N,4),dtype=np.float32)
    if dim==0:
        Xb[:,0]=val
        Xb[:,1]=y_min+(y_max-y_min)*pts[:,0]
        Xb[:,2]=z_min+(z_max-z_min)*pts[:,1]
    elif dim==1:
        Xb[:,0]=x_min+(x_max-x_min)*pts[:,0]
        Xb[:,1]=val
        Xb[:,2]=z_min+(z_max-z_min)*pts[:,1]
    else:
        Xb[:,0]=x_min+(x_max-x_min)*pts[:,0]
        Xb[:,1]=y_min+(y_max-y_min)*pts[:,1]
        Xb[:,2]=val
    Xb[:,3] = t_min + (t_max-t_min)*pts[:,2]
    return Xb

N_bc = 6000
Np = N_bc//6
faces = []
norms = []
for dim,val,nvec in [
    (0,x_min,[-1,0,0]),(0,x_max,[1,0,0]),
    (1,y_min,[0,-1,0]),(1,y_max,[0,1,0]),
    (2,z_min,[0,0,-1]),(2,z_max,[0,0,1])
]:
    faces.append(gen_bc(Np,dim,val))
    norms.append(np.tile(nvec,(Np,1)))
X_bc        = np.vstack(faces)
normal_vecs = np.vstack(norms)

# to torch
X_coll_t     = torch.tensor(X_coll,     dtype=torch.float32,device=device)
X_coll_val_t = torch.tensor(X_coll_val, dtype=torch.float32,device=device)
X_ic_t       = torch.tensor(X_ic,       dtype=torch.float32,device=device)
X_ic_val_t   = torch.tensor(X_ic_val,   dtype=torch.float32,device=device)
u0_ic_t      = torch.tensor(u0_ic,      dtype=torch.float32,device=device)
u0_ic_val_t  = torch.tensor(u0_ic_val,  dtype=torch.float32,device=device)
X_bc_t       = torch.tensor(X_bc,       dtype=torch.float32,device=device)
normals_t    = torch.tensor(normal_vecs,dtype=torch.float32,device=device)

# ----------------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------------

def scaling_fn(X):
    # X[:,0:3] in [0,1] → [-1,1]
    xyz = 2.0*(X[:,0:3] - 0.5)
    # X[:,3] in [t_min,t_max] → [-1,1]
    t   = 2.0*(X[:,3] - t_min)/(t_max-t_min) - 1.0
    return torch.cat([xyz, t], dim=1)

# pick a few random points
v = torch.tensor([[0.0],[0.5],[1.0]], requires_grad=True)
w = torch.tensor([[0.0],[0.2],[0.8]], requires_grad=True)
X_dummy = torch.zeros((3,4))  # not used by ode_func_fhn

# analytical dv and dw
dv_true = v - v**3/3 - w
dw_true = 0.01*(v + 0.2 - 0.5*w)

# your implementations
dv_fnw, dw_fnw = ode_func_fhn(v,w,X_dummy).chunk(2,dim=1)
rch = reaction_fhn(v,w)

assert torch.allclose(dv_fnw, dv_true, atol=1e-6)
assert torch.allclose(dw_fnw, dw_true, atol=1e-6)
assert torch.allclose(rch, dv_true,   atol=1e-6)
print("✅ reaction_fhn and ode_func_fhn match the analytical formulas.")

def run_pinns_simulation(num_layers, num_neurons, scaling_par=None):
    run_dir = os.path.join(base_results_dir,f"layers={num_layers}_neurons={num_neurons}")
    os.makedirs(run_dir,exist_ok=True)
    params_file = os.path.join(run_dir,"model_parameters.json")

    model_params = {
        "num_inputs":    4,
        "num_layers":    num_layers,
        "num_neurons":   num_neurons,
        "use_ode":       True,
        "n_state_vars":  1,
        "loss_function": "L2",
        "weight_strategy":"dynamic",
        "alpha":         0.9,
        "M":             M,
    }

    model = MonodomainSolverPINNs(
        num_inputs       = 4,
        num_layers       = num_layers,
        num_neurons      = num_neurons,
        device           = device,
        source_term_func = reaction_fhn,
        M                = M,
        use_ode          = True,
        ode_func         = ode_func_fhn,
        n_state_vars     = 1,
        loss_function    = model_params["loss_function"],
        weight_strategy  = model_params["weight_strategy"],
        alpha            = model_params["alpha"],
        scaling_func     = scaling_par
    )

    # attach data
    model.X_collocation = X_coll_t
    model.X_ic          = X_ic_t
    model.expected_u0   = u0_ic_t
    model.X_boundary    = X_bc_t
    model.normal_vectors= normals_t

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs, batch_size, patience = 35000, 4096, 1000
    best_val, wait = float("inf"), 0
    best_path = os.path.join(run_dir,"best.pth")
    history = {k:[] for k in ("epoch","train","val","pde","ic","bc","ode")}

    t0 = time.time()
    for ep in range(epochs+1):
        pde_l, ic_l, bc_l, data_l, ode_l, tot_l = model.train_step(optimizer,batch_size)
        if ep % 100 == 0:
            model.eval()
            val_l = model.validate(
                X_collocation_val = X_coll_val_t,
                X_ic_val          = X_ic_val_t,
                expected_u0_val   = u0_ic_val_t,
                X_boundary_val    = X_bc_t,
                normal_vectors_val= normals_t
            )
            history["epoch"].append(ep)
            history["train"].append(float(tot_l))
            history["val"].append(float(val_l))
            history["pde"].append(float(pde_l))
            history["ic"].append(float(ic_l))
            history["bc"].append(float(bc_l))
            history["ode"].append(float(ode_l))
            # after model.train_step(...)
            print(f"[{ep:5d}]  PDE={pde_l:.2e}  ODE={ode_l:.2e}  IC={ic_l:.2e}  BC={bc_l:.2e}  total={tot_l:.2e}")

            if val_l < best_val:
                best_val, wait = val_l, 0
                #model.save_model(best_path)
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping.")
                    break
    comp_time = time.time()-t0

    # save & plot losses
    np.savez_compressed(
        os.path.join(run_dir,"loss_history.npz"),
        epoch=np.array(history["epoch"]),
        train=np.array(history["train"]),
        val=np.array(history["val"]),
        pde=np.array(history["pde"]),
        ic=np.array(history["ic"]),
        bc=np.array(history["bc"]),
        ode=np.array(history["ode"])
    )
    plt.figure(); plt.plot(history["epoch"],history["train"],label="train")
    plt.plot(history["epoch"],history["val"],label="val")
    plt.yscale("log"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(run_dir,"loss.png"),dpi=300); plt.close()
    plt.figure()
    plt.plot(history["epoch"],history["pde"],label="PDE")
    plt.plot(history["epoch"],history["ic"], label="IC")
    plt.plot(history["epoch"],history["bc"], label="BC")
    plt.plot(history["epoch"],history["ode"],label="ODE")
    plt.yscale("log"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(run_dir,"loss_components.png"),dpi=300); plt.close()

    # final evaluation: **only take v ([:,0])**
    model.load_model(best_path)
    Np,Nt = x_coords.size, time_points.size
    preds = np.zeros((Np,Nt),dtype=np.float32)
    for i,tv in enumerate(time_points):
        Xe = np.column_stack([x_coords,y_coords,z_coords,np.full_like(x_coords,tv)])
        with torch.no_grad():
            out = model(torch.tensor(Xe,dtype=torch.float32,device=device))
        v_pred = out[:,0].cpu().numpy()
        preds[:,i] = v_pred

    fem_all   = fem_solutions.flatten()
    pinns_all = preds.flatten()
    rmse      = np.sqrt(mean_squared_error(fem_all,pinns_all))
    rel_rmse  = np.sqrt(((fem_all-pinns_all)**2).sum()/((fem_all**2).sum()+1e-16))

    model_params.update({
        "training_time":   comp_time,
        "global_rmse":     float(rmse),
        "global_rel_rmse": float(rel_rmse)
    })
    with open(params_file,"w") as fp:
        json.dump(model_params,fp,indent=2)

    # slice‐plane plots at z≈0.5
    mask = np.abs(z_coords-0.5)<1e-2
    tri  = Triangulation(x_coords[mask],y_coords[mask])
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

    print(f"→ layers={num_layers}, neurons={num_neurons}, RMSE={rmse:.3e}")
    return rmse, rel_rmse

# Main loop over architectures
test_rmse, test_rrmse = {}, {}


# Loop over all desired architectures
for nl in (3,4,5):  # Number of layers
    for nn in (128,256, 512):  # Number of neurons
        r, rr = run_pinns_simulation(nl, nn, scaling_par=None)
        test_rmse[(nl, nn)], test_rrmse[(nl, nn)] = r, rr


# Write summary for RMSES
with open(os.path.join(base_results_dir, "test_rmse.txt"), "w") as f:
    for nn in (128, 256, 512):
        row = [f"{test_rmse[(nl, nn)]:.6e}" for nl in (3, 4, 5)]
        f.write(" ".join(row) + "\n")

# Write summary for relative RMSE
with open(os.path.join(base_results_dir, "test_rel_rmse.txt"), "w") as f:
    for nn in (128, 256, 512):
        row = [f"{test_rrmse[(nl, nn)]:.6e}" for nl in (3, 4, 5)]
        f.write(" ".join(row) + "\n")

print("All simulations complete.")

