import torch
import torch.nn as nn
import numpy as np
import time 
import torch
import torch.nn as nn
from typing import Callable, Dict, Optional, Union, List, Any
import torch.nn.functional as F


class PINN(nn.Module):
    """
    Base class for Physics-Informed Neural Networks (PINNs).
    """
    def __init__(
        self,
        num_inputs: int,
        num_layers: int,
        num_neurons: int,
        num_outputs: int,
        device: str,
        activation_fn: Optional[nn.Module] = None,
        init_method: str = 'xavier'
    ):
        super(PINN, self).__init__()
        self.device = device

        if activation_fn is None:
            activation_fn = nn.Tanh()

        # Build sequential model: [Linear -> activation] * num_layers + [Linear]
        layers = []
        # Input layer
        layers.append(nn.Linear(num_inputs, num_neurons))
        layers.append(activation_fn)
        # Hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(activation_fn)
        # Output layer
        layers.append(nn.Linear(num_neurons, num_outputs))

        self.model = nn.Sequential(*layers).to(device)

        # Initialize weights
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                if init_method == 'xavier':
                    torch.nn.init.xavier_normal_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
                elif init_method == 'normal':
                    torch.nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                    torch.nn.init.zeros_(layer.bias)
                else:
                    raise ValueError(f"Unknown init_method: {init_method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the full network.
        """
        return self.model(x)
    
    
class MonodomainSolverPINNs(PINN):
    def __init__(
        self,
        num_inputs: int,
        num_layers: int,
        num_neurons: int,
        device: str,
        source_term_func: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        M: Union[float, int, list, np.ndarray, torch.Tensor] = None,
        use_ode: bool = False,
        ode_func: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        n_state_vars: int = 0,
        loss_function: str = 'L2',
        loss_weights: Optional[Dict[str, float]] = None,
        weight_strategy: str = 'manual',
        alpha: float = 0.9,
        scaling_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ):
        """
        If use_ode=True and n_state_vars>0, we interpret the PDE for v to have
        a reaction term R(v,w) = source_term_func(v, w), while the ODE handles w.
        If use_ode=False, we interpret source_term_func(x,t) as a purely spatiotemporal source for v.
        """
        if use_ode:
            output_dimension = 1 + n_state_vars  # e.g., v + w
        else:
            output_dimension = 1  # just v

        # Remove x_min, x_max, etc. as they are not used
        super(MonodomainSolverPINNs, self).__init__(
            num_inputs, num_layers, num_neurons, output_dimension, device
        )
        # Store the total number of inputs (e.g., in 3D, num_inputs=4: x,y,z,t)
        self.num_inputs = num_inputs
        self.source_term_func = source_term_func

        # Convert M to torch tensor if needed
        if isinstance(M, (float, int, list, np.ndarray)):
            self.M = torch.tensor(M, dtype=torch.float32, device=device)
        elif isinstance(M, torch.Tensor):
            self.M = M.to(device)
        else:
            raise TypeError("M must be a float, int, list, np.ndarray, or torch.Tensor.")

        self.is_scalar_M = (self.M.ndim == 0)
        self.use_ode = use_ode
        self.ode_func = ode_func
        self.n_state_vars = n_state_vars
        self.loss_function = loss_function
        self.weight_strategy = weight_strategy
        self.alpha = alpha

        # Handle manual or dynamic weighting
        if weight_strategy == 'manual':
            if loss_weights is None:
                self.loss_weights = {
                    'pde_loss': 1.0,
                    'IC_loss': 1.0,
                    'BC_loss': 1.0,
                    'data_loss': 1.0,
                    'ode_loss': 1.0
                }
            else:
                self.loss_weights = loss_weights
        else:
            self.lambda_ic = torch.tensor(1.0, device=device)
            self.lambda_bc = torch.tensor(1.0, device=device)
            self.lambda_r = torch.tensor(1.0, device=device)
            self.lambda_ode = torch.tensor(1.0, device=device)

        self.X_data = None
        self.expected_data = None

        # Setup the loss function
        if self.loss_function == 'L2':
            self.loss_function_obj = nn.MSELoss()
        elif self.loss_function == 'L1':
            self.loss_function_obj = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")

        self.scaling_func = scaling_func

    def apply_scaling(self, X: torch.Tensor) -> torch.Tensor:
        if self.scaling_func is not None:
            return self.scaling_func(X)
        return X

    def pde(self, X_collocation: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        PDE residual: 
            ∂v/∂t - ∇·(M ∇v) - R(v,w) = 0,
        plus ODE loss if use_ode=True.
        """
        X_collocation.requires_grad_(True)
        X_scaled = self.apply_scaling(X_collocation)
        outputs = self.forward(X_scaled)

        if self.use_ode:
            v = outputs[:, 0:1]
            w = outputs[:, 1:]
        else:
            v = outputs
            w = None

        # Compute partial derivatives
        du_dx = torch.autograd.grad(v.sum(), X_collocation, create_graph=True)[0]
        # Assume the last coordinate is time; spatial coordinates are the first (num_inputs - 1)
        num_spatial = self.num_inputs - 1
        # Compute spatial gradients only
        dv_dx = du_dx[:, :num_spatial]
        dv_dt = du_dx[:, -1:]  # time derivative

        # For diffusion, we compute the Laplacian in space.
        # Here, we compute second derivatives for each spatial dimension and sum them.
        laplacian = 0.0
        for i in range(num_spatial):
            grad_i = dv_dx[:, i:i+1]
            second_deriv = torch.autograd.grad(grad_i.sum(), X_collocation, create_graph=True)[0][:, i:i+1]
            laplacian = laplacian + second_deriv

        if self.is_scalar_M:
            diffusion = self.M * laplacian
        else:
            # Assume self.M is a vector/tensor of length num_spatial
            weighted_grad = dv_dx * self.M  # broadcasting over the spatial dimensions
            diffusion = torch.zeros_like(v)
            # Sum the second derivatives multiplied by corresponding M components
            for i in range(num_spatial):
                grad_i = dv_dx[:, i:i+1]
                second_deriv = torch.autograd.grad(grad_i.sum(), X_collocation, create_graph=True)[0][:, i:i+1]
                diffusion = diffusion + self.M[i] * second_deriv

        # Reaction / source term
        if self.use_ode and self.source_term_func is not None:
            reaction_term = self.source_term_func(v, w)
        elif self.use_ode:
            reaction_term = torch.zeros_like(v)
        else:
            if self.source_term_func is not None:
                reaction_term = self.source_term_func(
                    X_collocation[:, :num_spatial],
                    X_collocation[:, -1:]
                )
            else:
                reaction_term = torch.zeros_like(v)

        pde_residual = dv_dt - diffusion - reaction_term
        pde_loss = torch.mean(pde_residual**2)

        if self.use_ode and (self.ode_func is not None):
            ode_residuals = self.ode_func(v, w, X_collocation)
            ode_loss = torch.mean((ode_residuals[:, 1:])**2)
        else:
            ode_loss = torch.tensor(0.0, device=self.device)

        return pde_loss, ode_loss

    def IC(self, X_ic: torch.Tensor, expected_u0: torch.Tensor) -> torch.Tensor:
        X_scaled = self.apply_scaling(X_ic)
        outputs = self.forward(X_scaled)
        if self.use_ode:
            v0 = outputs[:, 0:1]
            loss_ic = self.loss_function_obj(v0, expected_u0[:, 0:1].to(self.device))
            if expected_u0.shape[1] > 1:
                w0 = outputs[:, 1:2]
                loss_ic += self.loss_function_obj(w0, expected_u0[:, 1:2].to(self.device))
            return loss_ic
        else:
            return self.loss_function_obj(outputs, expected_u0.to(self.device))

    def BC_neumann(
        self,
        X_boundary: torch.Tensor,
        normal_vectors: torch.Tensor,
        expected_value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        X_boundary.requires_grad_(True)
        X_scaled = self.apply_scaling(X_boundary)
        outputs = self.forward(X_scaled)
        u = outputs[:, 0:1] if self.use_ode else outputs

        # Compute the gradient with respect to spatial coordinates only.
        num_spatial = self.num_inputs - 1  # spatial dimensions = total inputs minus time
        du_dx = torch.autograd.grad(u.sum(), X_boundary, create_graph=True)[0][:, :num_spatial]

        if self.is_scalar_M:
            normal_deriv = self.M * torch.sum(du_dx * normal_vectors, dim=1, keepdim=True)
        else:
            # Assume self.M is a vector of length num_spatial
            weighted_du = du_dx * self.M  # broadcasting multiplication
            normal_deriv = torch.sum(weighted_du * normal_vectors, dim=1, keepdim=True)

        if expected_value is None:
            expected_value = torch.zeros_like(normal_deriv)
        bc_loss = torch.mean((normal_deriv - expected_value)**2)
        return bc_loss

    def data_loss(self, X_data: torch.Tensor, expected_data: torch.Tensor) -> torch.Tensor:
        X_scaled = self.apply_scaling(X_data)
        outputs = self.forward(X_scaled)
        u = outputs[:, 0:1] if self.use_ode else outputs
        return self.loss_function_obj(u, expected_data.to(self.device))

    def train_step(self, optimizer: torch.optim.Optimizer, batch_size: Optional[int] = None) -> tuple:
        self.train()
        optimizer.zero_grad()

        if batch_size is not None:
            collocation_indices = torch.randperm(self.X_collocation.size(0))[:batch_size]
            ic_indices = torch.randperm(self.X_ic.size(0))[:batch_size]
            boundary_indices = torch.randperm(self.X_boundary.size(0))[:batch_size]

            X_collocation_batch = self.X_collocation[collocation_indices]
            X_ic_batch = self.X_ic[ic_indices]
            expected_u0_batch = self.expected_u0[ic_indices]
            X_boundary_batch = self.X_boundary[boundary_indices]
            normal_vectors_batch = self.normal_vectors[boundary_indices]

            if self.X_data is not None:
                data_indices = torch.randperm(self.X_data.size(0))[:batch_size]
                X_data_batch = self.X_data[data_indices]
                expected_data_batch = self.expected_data[data_indices]
            else:
                X_data_batch = None
                expected_data_batch = None
        else:
            X_collocation_batch = self.X_collocation
            X_ic_batch = self.X_ic
            expected_u0_batch = self.expected_u0
            X_boundary_batch = self.X_boundary
            normal_vectors_batch = self.normal_vectors
            X_data_batch = self.X_data
            expected_data_batch = self.expected_data

        if self.weight_strategy == 'dynamic':
            self.update_weights(
                X_collocation_batch,
                X_ic_batch,
                expected_u0_batch,
                X_boundary_batch,
                normal_vectors_batch
            )

        IC_loss = self.IC(X_ic_batch, expected_u0_batch)
        BC_loss = self.BC_neumann(X_boundary_batch, normal_vectors_batch)
        pde_loss, ode_loss = self.pde(X_collocation_batch)

        if self.weight_strategy == 'manual':
            total_loss = (
                self.loss_weights['IC_loss'] * IC_loss +
                self.loss_weights['BC_loss'] * BC_loss +
                self.loss_weights['pde_loss'] * pde_loss +
                self.loss_weights['ode_loss'] * ode_loss
            )
        else: 
            total_loss = (
                self.lambda_ic  * IC_loss +
                self.lambda_bc  * BC_loss +
                self.lambda_r   * pde_loss +
                self.lambda_ode * ode_loss
            )


        data_loss_value = torch.tensor(0.0, device=self.device)
        if X_data_batch is not None and expected_data_batch is not None:
            data_loss_value = self.data_loss(X_data_batch, expected_data_batch)
            if self.weight_strategy == 'manual':
                total_loss += self.loss_weights['data_loss'] * data_loss_value
            else:
                total_loss += data_loss_value

        total_loss.backward()
        optimizer.step()

        return (
            pde_loss.item(),
            IC_loss.item(),
            BC_loss.item(),
            data_loss_value.item(),
            ode_loss.item(),
            total_loss.item()
        )

    def evaluate(self, X_eval: torch.Tensor, y_true: Optional[torch.Tensor] = None):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(X_eval.to(self.device))

        y_pred = outputs[:, 0:1] if self.use_ode else outputs

        if y_true is not None:
            loss = self.loss_function_obj(y_pred, y_true.to(self.device)).item()
            return y_pred.cpu(), loss
        else:
            return y_pred.cpu()

    def validate(self, X_collocation_val, X_ic_val, expected_u0_val, X_boundary_val, normal_vectors_val):
        self.eval()
        with torch.enable_grad():
            pde_loss_val, ode_loss_val = self.pde(X_collocation_val)
            ic_loss_val = self.IC(X_ic_val, expected_u0_val)
            bc_loss_val = self.BC_neumann(X_boundary_val, normal_vectors_val)

            if self.weight_strategy == 'manual':
                total_val_loss = (
                    self.loss_weights['pde_loss'] * pde_loss_val +
                    self.loss_weights['IC_loss'] * ic_loss_val +
                    self.loss_weights['BC_loss'] * bc_loss_val +
                    self.loss_weights['ode_loss'] * ode_loss_val
                )
            else:
                total_val_loss = (
                    self.lambda_ic * ic_loss_val +
                    self.lambda_bc * bc_loss_val +
                    self.lambda_r * pde_loss_val +
                    self.lambda_ode  * ode_loss_val
                )

        return total_val_loss.item()

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        self.model.to(self.device)
        self.eval()
        print(f"Model loaded from {file_path}")

    def compute_gradient_norms(self, X_collocation, X_ic, expected_u0, X_boundary, normal_vectors):
        self.zero_grad()
        pde_loss, ode_loss = self.pde(X_collocation)
        ic_loss = self.IC(X_ic, expected_u0)
        bc_loss = self.BC_neumann(X_boundary, normal_vectors)

        # grads for each
        grad_r   = torch.autograd.grad(pde_loss,     self.parameters(),
                                       create_graph=True, retain_graph=True, allow_unused=True)
        grad_ic  = torch.autograd.grad(ic_loss,      self.parameters(),
                                       create_graph=True, retain_graph=True, allow_unused=True)
        grad_bc  = torch.autograd.grad(bc_loss,      self.parameters(),
                                       create_graph=True, retain_graph=True, allow_unused=True)
        grad_ode = torch.autograd.grad(ode_loss,     self.parameters(),
                                       create_graph=True, retain_graph=True, allow_unused=True)

        def norm_of(grads):
            return torch.sqrt(
                sum(torch.sum(g**2) for g in grads if g is not None)
            ).detach()

        norm_r   = norm_of(grad_r)
        norm_ic  = norm_of(grad_ic)
        norm_bc  = norm_of(grad_bc)
        norm_ode = norm_of(grad_ode)

        return norm_r, norm_ic, norm_bc, norm_ode


    def update_weights(self, X_collocation, X_ic, expected_u0, X_boundary, normal_vectors):
        if self.weight_strategy != 'dynamic':
            return
        norm_r, norm_ic, norm_bc, norm_ode = self.compute_gradient_norms(
            X_collocation, X_ic, expected_u0, X_boundary, normal_vectors
        )
        total = norm_r + norm_ic + norm_bc + norm_ode + 1e-16
        λr, λic, λbc, λode = (
            (norm_r   / total).detach(),
            (norm_ic  / total).detach(),
            (norm_bc  / total).detach(),
            (norm_ode / total).detach(),
        )
        # exponential smoothing
        self.lambda_r   = self.alpha * self.lambda_r   + (1 - self.alpha) * λr
        self.lambda_ic  = self.alpha * self.lambda_ic  + (1 - self.alpha) * λic
        self.lambda_bc  = self.alpha * self.lambda_bc  + (1 - self.alpha) * λbc
        self.lambda_ode = self.alpha * self.lambda_ode + (1 - self.alpha) * λode


# ====================================================================================
# INVERSE PROBLEM STUFF!! (in the making)
# ===================================================================================

class InverseMonodomainSolverPINNs(MonodomainSolverPINNs):
    """
    Inverse PINN solver for the monodomain equation that estimates the diffusion parameter M.
    Here the network predicts the state variable(s) (e.g. u) while M is a learnable parameter.
    """
    def __init__(
        self,
        num_inputs: int,
        num_layers: int,
        num_neurons: int,
        device: str,
        source_term_func: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        initial_M: Union[float, int, List[float], np.ndarray, torch.Tensor] = 1.0,
        use_ode: bool = False,
        ode_func: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        n_state_vars: int = 0,
        loss_function: str = 'L2',
        loss_weights: Optional[Dict[str, float]] = None,
        weight_strategy: str = 'manual',
        alpha: float = 0.9,
        scaling_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ):
        # Call parent's initializer without passing num_outputs; 
        # MonodomainSolverPINNs computes the output dimension internally.
        super(InverseMonodomainSolverPINNs, self).__init__(
            num_inputs,
            num_layers,
            num_neurons,
            device,
            source_term_func,
            initial_M,
            use_ode,
            ode_func,
            n_state_vars,
            loss_function,
            loss_weights,
            weight_strategy,
            alpha,
            scaling_func
        )
        
        # Process initial_M so that it is always a floating-point tensor
        if isinstance(initial_M, int):
            initial_M = float(initial_M)
        if isinstance(initial_M, (float, int)):
            initial_M = torch.tensor([initial_M], dtype=torch.float32)
        elif isinstance(initial_M, (list, np.ndarray)):
            initial_M = torch.tensor(initial_M, dtype=torch.float32)
        elif isinstance(initial_M, torch.Tensor):
            initial_M = initial_M.float()
        else:
            raise TypeError("initial_M must be a float, int, list, np.ndarray, or torch.Tensor")
        
        # Replace the diffusion coefficient with a learnable parameter
        self.M = nn.Parameter(initial_M.to(device))
        self.is_scalar_M = (self.M.ndim == 0) or (self.M.ndim == 1 and self.M.shape[0] == 1)
    
    def pde(self, X_collocation: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Computes the PDE residual:
            ∂u/∂t - M ∆u - R(u) = 0,
        where u is predicted by the network and M is a learnable parameter.
        """
        X_collocation.requires_grad_(True)
        X_scaled = self.apply_scaling(X_collocation)
        outputs = self.forward(X_scaled)
        
        # Separate state variables if using ODE; otherwise, u is the network output
        if self.use_ode:
            u = outputs[:, 0:1]
            w = outputs[:, 1:]
        else:
            u = outputs
        
        # Assume input ordering is [x, y, t]
        du_dx = torch.autograd.grad(u.sum(), X_collocation, create_graph=True)[0]
        # Compute second derivatives for the spatial dimensions
        d2u_dx2 = torch.autograd.grad(du_dx[:, 0].sum(), X_collocation, create_graph=True)[0][:, 0:1]
        d2u_dy2 = torch.autograd.grad(du_dx[:, 1].sum(), X_collocation, create_graph=True)[0][:, 1:2]
        du_dt  = du_dx[:, 2:3]
        
        # Compute Laplacian using the learned parameter M
        if self.is_scalar_M:
            laplacian_term = self.M * (d2u_dx2 + d2u_dy2)
        else:
            laplacian_term = self.M[0] * d2u_dx2 + self.M[1] * d2u_dy2
        
        # Compute source (or reaction) term
        if self.use_ode and (self.ode_func is not None):
            ode_residuals = self.ode_func(u, w, X_collocation)
            source_term = ode_residuals[:, 0:1]
            ode_loss = torch.mean((ode_residuals[:, 1:])**2)
        else:
            if self.source_term_func is not None:
                source_term = self.source_term_func(X_collocation[:, :2], X_collocation[:, 2:3])
            else:
                source_term = torch.zeros_like(u)
            ode_loss = torch.tensor(0.0, device=self.device)
        
        pde_residual = du_dt - laplacian_term - source_term
        pde_loss = torch.mean(pde_residual**2)
        return pde_loss, ode_loss

#======================================================================================
# TESTING SOME NEW STUFF BASED ON THE PAPER "AN EXPERTS GUIDE TO TRAINING PINNS" (S. WANG, S. SANKARAN, H. WANG, P. PERIDKARIS)
#======================================================================================

# ----------------------------------------
# Fourier Feature Embedding
# ----------------------------------------
class FourierFeatureEmbedding(nn.Module):
    def __init__(self, num_inputs: int, mapping_size: int = 256, sigma: float = 5.0, device: str = 'cpu', use_bias: bool = True):
        super(FourierFeatureEmbedding, self).__init__()
        self.num_inputs = num_inputs
        self.mapping_size = mapping_size
        self.sigma = sigma
        self.device = device

        # Sample B from N(0, sigma^2)
        self.register_buffer('B', torch.randn(mapping_size, num_inputs, device=device) * sigma)

        if use_bias:
            # Random phase in [0, 2pi)
            b = 2 * torch.pi * torch.rand(mapping_size, device=device)
            self.register_buffer('b', b)
        else:
            self.b = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Without the extra 2pi factor, this gives: Bx + b
        x_proj = F.linear(x, self.B) + self.b
        # Now apply sin and cos without an extra 2pi multiplier:
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# ----------------------------------------
# Random Weight Factorized Linear (RWF)
# ----------------------------------------
class RandomWeightFactorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, mu: float = 1.0, sigma: float = 0.1):
        super(RandomWeightFactorizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Initialize the scaling parameters s such that alpha = exp(s) > 0.
        self.s = nn.Parameter(torch.randn(out_features) * sigma + mu)
        self.V = nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.xavier_normal_(self.V)
        # Initialize bias from N(0,1) as recommended
        self.bias = nn.Parameter(torch.randn(out_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute positive scale factors
        scale_factors = torch.exp(self.s).unsqueeze(1)
        scaled_weights = scale_factors * self.V
        return x @ scaled_weights.t() + self.bias

# ----------------------------------------
# EnhancedMonodomainSolverPINNs (inherits from MonodomainSolverPINNs)
# ----------------------------------------
class EnhancedPINN(nn.Module):
    """
    Enhanced MLP with optional Fourier features and RWF.
    """
    def __init__(self, num_inputs: int, num_layers: int, num_neurons: int, num_outputs: int,
                 device: str, use_fourier: bool = False, fourier_dim: int = 256,
                 sigma: float = 5.0, use_rwf: bool = False, mu: float = 1.0, sigma_rwf: float = 0.1):
        super(EnhancedPINN, self).__init__()
        self.device = device
        self.use_fourier = use_fourier
        self.use_rwf = use_rwf
        if self.use_fourier:
            self.fourier = FourierFeatureEmbedding(num_inputs, mapping_size=fourier_dim, sigma=sigma, device=device)
            input_dim = fourier_dim * 2
        else:
            input_dim = num_inputs
        activation = nn.Tanh()
        layers = []
        for i in range(num_layers + 1):
            in_dim = input_dim if i == 0 else num_neurons
            out_dim = num_outputs if i == num_layers else num_neurons
            if self.use_rwf:
                layers.append(RandomWeightFactorizedLinear(in_dim, out_dim, mu=mu, sigma=sigma_rwf))
            else:
                layer = nn.Linear(in_dim, out_dim)
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
                layers.append(layer)
            if i != num_layers:
                layers.append(activation)
        self.model = nn.Sequential(*layers).to(device)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fourier:
            x = self.fourier(x)
        return self.model(x)

class EnhancedMonodomainSolverPINNs(MonodomainSolverPINNs):
    """
    EnhancedMonodomainSolverPINNs incorporates Fourier features, RWF, and temporal segmentation
    with dynamic curriculum weighting.
    
    New parameters:
      - M_segments: number of temporal segments.
      - t_min, t_max: temporal domain bounds.
      - epsilon: slope for updating temporal weights.
      - f: frequency (in epochs) for global weight update.
      - curriculum_enabled: whether to update temporal weights.
    """
    def __init__(self,
                 num_inputs: int,
                 num_layers: int,
                 num_neurons: int,
                 device: str,
                 source_term_func: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 M: Union[float, int, list, np.ndarray, torch.Tensor] = None,
                 use_ode: bool = False,
                 ode_func: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 n_state_vars: int = 0,
                 loss_function: str = 'L2',
                 loss_weights: Optional[Dict[str, float]] = None,
                 weight_strategy: str = 'dynamic',
                 alpha: float = 0.9,
                 # Enhanced options:
                 use_fourier: bool = False,
                 fourier_dim: int = 256,
                 sigma: float = 5.0,
                 use_rwf: bool = False,
                 mu: float = 1.0,
                 sigma_rwf: float = 0.1,
                 # Temporal segmentation parameters:
                 M_segments: int = 5,
                 t_min: float = 0.0,
                 t_max: float = 1.0,
                 epsilon: float = 1.0,
                 f: int = 1000,
                 curriculum_enabled: bool = True):
        if use_ode:
            output_dimension = 1 + n_state_vars
        else:
            output_dimension = 1
        super(EnhancedMonodomainSolverPINNs, self).__init__(
            num_inputs=num_inputs,
            num_layers=num_layers,
            num_neurons=num_neurons,
            device=device,
            source_term_func=source_term_func,
            M=M,
            use_ode=use_ode,
            ode_func=ode_func,
            n_state_vars=n_state_vars,
            loss_function=loss_function,
            loss_weights=loss_weights,
            weight_strategy=weight_strategy,
            alpha=alpha
        )
        self.num_inputs = num_inputs
        self.source_term_func = source_term_func
        if isinstance(M, (float, int, list, np.ndarray)):
            self.M = torch.tensor(M, dtype=torch.float32, device=device)
        elif isinstance(M, torch.Tensor):
            self.M = M.to(device)
        else:
            raise TypeError("M must be a float, int, list, np.ndarray, or torch.Tensor.")
        self.is_scalar_M = (self.M.ndim == 0)
        self.use_ode = use_ode
        self.ode_func = ode_func
        self.n_state_vars = n_state_vars
        self.loss_function = loss_function
        self.weight_strategy = weight_strategy
        self.alpha = alpha
        if self.weight_strategy == 'manual':
            if loss_weights is None:
                self.loss_weights = {'pde_loss': 1.0, 'IC_loss': 1.0,
                                       'BC_loss': 1.0, 'data_loss': 1.0, 'ode_loss': 1.0}
            else:
                self.loss_weights = loss_weights
        else:
            self.lambda_ic = torch.tensor(1.0, device=device)
            self.lambda_bc = torch.tensor(1.0, device=device)
            self.lambda_r = torch.tensor(1.0, device=device)
            self.lambda_ode = torch.tensor(1.0, device=device)
        self.X_data = None
        self.expected_data = None
        if self.loss_function == 'L2':
            self.loss_function_obj = nn.MSELoss()
        elif self.loss_function == 'L1':
            self.loss_function_obj = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")
        # Enhanced network architecture:
        self.use_fourier = use_fourier
        self.use_rwf = use_rwf
        self.fourier_dim = fourier_dim
        self.sigma = sigma
        self.mu = mu
        self.sigma_rwf = sigma_rwf
        if self.use_fourier:
            self.fourier = FourierFeatureEmbedding(num_inputs, mapping_size=fourier_dim, sigma=sigma, device=device)
            input_dim = fourier_dim * 2
        else:
            input_dim = num_inputs
        layers = []
        activation = nn.Tanh()
        for i in range(num_layers + 1):
            in_dim = input_dim if i == 0 else num_neurons
            out_dim = output_dimension if i == num_layers else num_neurons
            if self.use_rwf:
                layers.append(RandomWeightFactorizedLinear(in_dim, out_dim, mu=mu, sigma=sigma_rwf))
            else:
                layer = nn.Linear(in_dim, out_dim)
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
                layers.append(layer)
            if i != num_layers:
                layers.append(activation)
        self.model = nn.Sequential(*layers).to(device)
        # Temporal segmentation & curriculum:
        self.M_segments = M_segments
        self.t_min = t_min
        self.t_max = t_max
        self.epsilon = epsilon
        self.f = f
        self.curriculum_enabled = curriculum_enabled
        self.temporal_weights = torch.ones(M_segments, device=device)
        self.segment_loss_history = [0.0] * M_segments

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fourier:
            x = self.fourier(x)
        return self.model(x)

    def train_step(self, optimizer: torch.optim.Optimizer,
                   Xc: torch.Tensor,
                   Xi: torch.Tensor,
                   u0: torch.Tensor,
                   Xb: torch.Tensor,
                   Nb: torch.Tensor,
                   epoch: Optional[int]=None) -> tuple:
        # 0) prep
        self.train()
        optimizer.zero_grad()

        # 1) compute raw PDE losses per segment
        dt = (self.t_max - self.t_min)/self.M_segments
        seg_losses = []
        for i in range(self.M_segments):
            mask = (Xc[:,2] >= self.t_min + i*dt) & (Xc[:,2] < self.t_min + (i+1)*dt)
            batch = Xc[mask]
            seg_losses.append(self.pde(batch)[0] if batch.numel()>0 else torch.tensor(0.0, device=self.device))

        # 2) update temporal weights **immediately** (no lag)
        new_w = torch.ones(self.M_segments, device=self.device)
        cum = torch.tensor(0.0, device=self.device)
        for i in range(1, self.M_segments):
            cum = cum + seg_losses[i-1]
            new_w[i] = torch.exp(- self.epsilon * cum)
        self.temporal_weights = new_w

        # 3) weighted PDE loss
        weighted_pde = sum(self.temporal_weights[i].detach()*seg_losses[i]
                           for i in range(self.M_segments)) / self.M_segments

        # 4) IC & BC losses
        IC_loss = self.IC(Xi, u0)
        BC_loss = self.BC_neumann(Xb, Nb)

        # 5) periodic global λ update
        if self.weight_strategy!='manual' and epoch is not None and (epoch % self.f ==0):
            grad_ic = torch.autograd.grad(IC_loss, self.parameters(), retain_graph=True, allow_unused=True)
            norm_ic = torch.sqrt(sum((g**2).sum() for g in grad_ic if g is not None))
            grad_bc = torch.autograd.grad(BC_loss, self.parameters(), retain_graph=True, allow_unused=True)
            norm_bc = torch.sqrt(sum((g**2).sum() for g in grad_bc if g is not None))
            grad_r  = torch.autograd.grad(weighted_pde, self.parameters(), retain_graph=True, allow_unused=True)
            norm_r  = torch.sqrt(sum((g**2).sum() for g in grad_r if g is not None))
            total_norm = norm_ic + norm_bc + norm_r

            hat_ic = total_norm/(norm_ic + 1e-8)
            hat_bc = total_norm/(norm_bc + 1e-8)
            hat_r  = total_norm/(norm_r  + 1e-8)

            self.lambda_ic = self.alpha*self.lambda_ic + (1-self.alpha)*hat_ic.detach()
            self.lambda_bc = self.alpha*self.lambda_bc + (1-self.alpha)*hat_bc.detach()
            self.lambda_r  = self.alpha*self.lambda_r  + (1-self.alpha)*hat_r .detach()

            # clamp to avoid explosion
            self.lambda_ic = self.lambda_ic.clamp(1e-4, 1e4)
            self.lambda_bc = self.lambda_bc.clamp(1e-4, 1e4)
            self.lambda_r  = self.lambda_r.clamp(1e-4, 1e4)

        # 6) form total loss
        total_loss = self.lambda_ic*IC_loss + self.lambda_bc*BC_loss + self.lambda_r*weighted_pde

        # 7) backprop + optional gradient clipping
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        # 8) return metrics
        return (weighted_pde.item(),
                IC_loss.item(),
                BC_loss.item(),
                0.0,               # ode_loss when use_ode=False
                total_loss.item())

    def validate(self, X_collocation_val, X_ic_val, expected_u0_val, X_boundary_val, normal_vectors_val):
        self.eval()
        with torch.enable_grad():
            pde_loss_val, ode_loss_val = self.pde(X_collocation_val)
            ic_loss_val = self.IC(X_ic_val, expected_u0_val)
            bc_loss_val = self.BC_neumann(X_boundary_val, normal_vectors_val)
            if self.weight_strategy == 'manual':
                total_val_loss = (self.loss_weights['pde_loss'] * pde_loss_val +
                                  self.loss_weights['IC_loss'] * ic_loss_val +
                                  self.loss_weights['BC_loss'] * bc_loss_val +
                                  self.loss_weights['ode_loss'] * ode_loss_val)
            else:
                total_val_loss = (self.lambda_ic * ic_loss_val +
                                  self.lambda_bc * bc_loss_val +
                                  self.lambda_r * pde_loss_val +
                                  ode_loss_val)
        return total_val_loss.item()

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        self.model.to(self.device)
        self.eval()
        print(f"Model loaded from {file_path}")

    def compute_gradient_norms(self, X_collocation, X_ic, expected_u0, X_boundary, normal_vectors):
        self.zero_grad()
        pde_loss, _ = self.pde(X_collocation)
        ic_loss = self.IC(X_ic, expected_u0)
        bc_loss = self.BC_neumann(X_boundary, normal_vectors)
        grad_r = torch.autograd.grad(pde_loss, self.parameters(), create_graph=True, allow_unused=True, retain_graph=True)
        grad_ic = torch.autograd.grad(ic_loss, self.parameters(), create_graph=True, allow_unused=True, retain_graph=True)
        grad_bc = torch.autograd.grad(bc_loss, self.parameters(), create_graph=True, allow_unused=True, retain_graph=True)
        norm_r = torch.sqrt(sum(torch.sum(g**2) for g in grad_r if g is not None)).detach()
        norm_ic = torch.sqrt(sum(torch.sum(g**2) for g in grad_ic if g is not None)).detach()
        norm_bc = torch.sqrt(sum(torch.sum(g**2) for g in grad_bc if g is not None)).detach()
        return norm_r, norm_ic, norm_bc

    def update_weights(self, X_collocation, X_ic, expected_u0, X_boundary, normal_vectors):
        if self.weight_strategy != 'dynamic':
            return
        norm_r, norm_ic, norm_bc = self.compute_gradient_norms(X_collocation, X_ic, expected_u0, X_boundary, normal_vectors)
        sum_norms = norm_r + norm_ic + norm_bc
        lambda_ic_new = (norm_ic / (sum_norms )).detach()
        lambda_bc_new = (norm_bc / (sum_norms )).detach()
        lambda_r_new  = (norm_r / (sum_norms )).detach()
        self.lambda_ic = self.alpha * self.lambda_ic + (1 - self.alpha) * lambda_ic_new
        self.lambda_bc = self.alpha * self.lambda_bc + (1 - self.alpha) * lambda_bc_new
        self.lambda_r  = self.alpha * self.lambda_r  + (1 - self.alpha) * lambda_r_new

