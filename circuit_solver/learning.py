import numpy as np
import torch as tc
import random
import warnings
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class CLTrainingConfig:
    """Configuration for coupled learning training"""
    x_inds: int = 'x'
    y_inds: int = 'y'
    batch_size: int = 100
    N_epochs: Optional[int] = None
    N_batches: Optional[int] = None
    eta: float = 1.0
    alpha: float = 1.0

    def __post_init__(self):
        """Validate that either N_epochs or N_batches is provided, but warn if both are given"""
        if self.N_epochs is None and self.N_batches is None:
            # Default to N_epochs = 1 if neither is provided
            self.N_epochs = 1
        elif self.N_epochs is not None and self.N_batches is not None:
            # Both provided - warn and use N_batches
            warnings.warn(
                f"Both N_epochs ({self.N_epochs}) and N_batches ({self.N_batches}) were provided. "
                f"Using N_batches = {self.N_batches}.",
                UserWarning
            )
            self.N_epochs = None  # Clear N_epochs to avoid confusion

@dataclass
class CLTestingConfig:
    """Configuration for testing circuit performance"""
    x_inds: str = 'x'
    y_inds: str = 'y'
    batch_size: int = 100
    N_epochs: Optional[int] = None
    N_batches: Optional[int] = None

    def __post_init__(self):
        """Validate that either N_epochs or N_batches is provided, but warn if both are given"""
        if self.N_epochs is None and self.N_batches is None:
            # Default to N_epochs = 1 if neither is provided
            self.N_epochs = 1
        elif self.N_epochs is not None and self.N_batches is not None:
            # Both provided - warn and use N_batches
            warnings.warn(
                f"Both N_epochs ({self.N_epochs}) and N_batches ({self.N_batches}) were provided. "
                f"Using N_batches = {self.N_batches}.",
                UserWarning
            )
            self.N_epochs = None  # Clear N_epochs to avoid confusion

@dataclass
class ILTrainingConfig:
    """Configuration for invariant learning training"""
    x_inds: int = 'x'
    y_inds: int = 'y'
    batch_size: int = 100
    N_epochs: Optional[int] = None
    N_batches: Optional[int] = None
    eta: float = 1.0
    alpha: float = 1.0

    def __post_init__(self):
        """Validate that either N_epochs or N_batches is provided, but warn if both are given"""
        if self.N_epochs is None and self.N_batches is None:
            # Default to N_epochs = 1 if neither is provided
            self.N_epochs = 1
        elif self.N_epochs is not None and self.N_batches is not None:
            # Both provided - warn and use N_batches
            warnings.warn(
                f"Both N_epochs ({self.N_epochs}) and N_batches ({self.N_batches}) were provided. "
                f"Using N_batches = {self.N_batches}.",
                UserWarning
            )
            self.N_epochs = None  # Clear N_epochs to avoid confusion

@dataclass
class AdjointTrainingConfig:
    """Configuration for adjoint learning training"""
    x_inds: int = 'x'
    y_inds: int = 'y'
    batch_size: int = 100
    N_epochs: Optional[int] = None
    N_batches: Optional[int] = None
    eta: float = 1.0
    alpha: float = 1.0

    def __post_init__(self):
        """Validate that either N_epochs or N_batches is provided, but warn if both are given"""
        if self.N_epochs is None and self.N_batches is None:
            # Default to N_epochs = 1 if neither is provided
            self.N_epochs = 1
        elif self.N_epochs is not None and self.N_batches is not None:
            # Both provided - warn and use N_batches
            warnings.warn(
                f"Both N_epochs ({self.N_epochs}) and N_batches ({self.N_batches}) were provided. "
                f"Using N_batches = {self.N_batches}.",
                UserWarning
            )
            self.N_epochs = None  # Clear N_epochs to avoid confusion

@dataclass
class GCLTrainingConfig:
    """Configuration for geometric coupled learning training"""
    x_inds: int = 'x'
    y_inds: int = 'y'
    batch_size: int = 100
    N_epochs: Optional[int] = None
    N_batches: Optional[int] = None
    eta: float = 1.0
    alpha: float = 1.0

    def __post_init__(self):
        """Validate that either N_epochs or N_batches is provided, but warn if both are given"""
        if self.N_epochs is None and self.N_batches is None:
            # Default to N_epochs = 1 if neither is provided
            self.N_epochs = 1
        elif self.N_epochs is not None and self.N_batches is not None:
            # Both provided - warn and use N_batches
            warnings.warn(
                f"Both N_epochs ({self.N_epochs}) and N_batches ({self.N_batches}) were provided. "
                f"Using N_batches = {self.N_batches}.",
                UserWarning
            )
            self.N_epochs = None  # Clear N_epochs to avoid confusion

@dataclass
class NCCLTrainingConfig:
    x_inds: int = 'x'
    y_inds: int = 'y'
    batch_size: int = 100
    N_epochs: Optional[int] = None
    N_batches: Optional[int] = None
    eta: float = 1.0
    alpha: float = 1.0
    nu: float = 1.0

    def __post_init__(self):
        """Validate that either N_epochs or N_batches is provided, but warn if both are given"""
        if self.N_epochs is None and self.N_batches is None:
            # Default to N_epochs = 1 if neither is provided
            self.N_epochs = 1
        elif self.N_epochs is not None and self.N_batches is not None:
            # Both provided - warn and use N_batches
            warnings.warn(
                f"Both N_epochs ({self.N_epochs}) and N_batches ({self.N_batches}) were provided. "
                f"Using N_batches = {self.N_batches}.",
                UserWarning
            )
            self.N_epochs = None  # Clear N_epochs to avoid confusion


def GeoCoupledLearning(X, Y, model, config=None, **params):
    # Handle configuration - use config object if provided, otherwise extract from params
    if config is not None:
        if isinstance(config, GCLTrainingConfig):
            cfg = config
        else:
            raise TypeError("config must be a GCLTrainingConfig object")
    else:
        # Use parameter extraction function
        param_dict = extract_training_params(**params)
        cfg = GCLTrainingConfig(**param_dict)

    # Convert numpy arrays to tensors if needed
    if isinstance(X, np.ndarray):
        X = tc.tensor(X, dtype=tc.float32)
    if isinstance(Y, np.ndarray):
        Y = tc.tensor(Y, dtype=tc.float32)

    # Setup
    P, N = X.shape
    PY, M = Y.shape

    if P != PY:
        raise ValueError(f"X and Y must have the same shape in the first dimension but have shapes {P} and {PY}")

    N_edges = model.circuit.number_of_edges()
    batches_per_epoch = P // cfg.batch_size

    # Calculate N_batches based on which parameter was provided
    if cfg.N_batches is not None:
        N_batches = cfg.N_batches
    else:
        # cfg.N_epochs is guaranteed to be set by __post_init__
        N_batches = int(batches_per_epoch * cfg.N_epochs)

    # change inputs to indices if default strings are passed in
    x_inds, y_inds = cfg.x_inds, cfg.y_inds
    if type(x_inds) is str:
        x_inds = model.node_type_dict[x_inds]
    if type(y_inds) is str:
        y_inds = model.node_type_dict[y_inds]

    # Initialize conductances and tracking arrays
    # k = np.ones((N_batches+1, N_edges)) * cfg.k_init

    param_history = {}
    for element in model.circuit.elements:
        name = element.name
        layer = model.element_layers[name]
        init_values = layer.theta.data.clone().detach().numpy()
        param_history[name] = [init_values]

    loss = np.zeros(N_batches)
    accuracy = np.zeros(N_batches)
    
    def loss_function(y, y_free):
        return tc.sum((y - y_free)**2)

    def argmax_classification_accuracy(y, y_free):
        return tc.sum(tc.argmax(y_free, axis=1) == np.argmax(y, axis=1))

    def compute_d_theta(V_F, V_C, layer):
        # D_V = (V_F - V_C)[:,None]       # Note: V_edge has a negative sign in it, i.e. model.V_edge() 

        return layer.trainable * cfg.alpha / cfg.eta * layer.theta * tc.mean(((V_F**2 - V_C**2) / V_F / V_C)[:,None], axis=0)
        
    # Training loop
    for t in tqdm(range(N_batches)):
        # Sample batch
        batch_inds = random.sample(range(P), cfg.batch_size)
        x = X[batch_inds]
        y = Y[batch_inds]

        # Ensure tensors are on correct device and dtype
        x = x.to(dtype=model.dtype, device=model.device)
        y = y.to(dtype=model.dtype, device=model.device)

        # Compute free and clamped states
        model.set_inputs(x_inds)
        V_node_F, _ = model(x)
        y_F = V_node_F[:, y_inds]
        V_edge_F = model.V_edge() 

        # Compute clamped state
        y_C = cfg.eta * y + (1 - cfg.eta) * y_F.detach()

        model.set_inputs(x_inds, y_inds)
        _ = model(x, y_C)
        V_edge_C = model.V_edge()
        
        # update parameters and history
        for element_name, layer in model.element_layers.items():
            edge_inds = model.element_to_edge_inds[element_name]
            V_F = V_edge_F[:, edge_inds]      
            V_C = V_edge_C[:, edge_inds]

            # apply parameter gradients
            # theta = layer.theta.data
            # d_theta = compute_d_theta(V_F, V_C, layer)
            # layer.update_parameters(theta + d_theta)
            layer.theta.data += compute_d_theta(V_F, V_C, layer)
            layer.clip_parameters()         # enforce limits on model parameter ranges
            
            # param_history[element + '_step'].append(d_theta.clone().detach().numpy())
            param_history[element_name].append(layer.theta.data.clone().detach().numpy())

        # record performance on batch
        loss_batch = loss_function(y, y_F) / cfg.batch_size
        accuracy_batch = argmax_classification_accuracy(y, y_F) / cfg.batch_size
        loss[t] = loss_batch.detach()
        accuracy[t] = accuracy_batch.detach()

    for k, v in param_history.items():
        param_history[k] = np.stack(v, axis=1)      # so that history[theta][i] is the history of theta_i, with shape (N_batch, N_edges)

    history = {
        'accuracy': accuracy,
        'loss': loss
    }
    history.update(param_history)

    return history
    

def AdjointLearning(X, Y, model, config=None, **params):
    # Handle configuration - use config object if provided, otherwise extract from params
    if config is not None:
        if isinstance(config, AdjointTrainingConfig):
            cfg = config
        else:
            raise TypeError("config must be a AdjointTrainingConfig object")
    else:
        # Use parameter extraction function
        param_dict = extract_training_params(**params)
        cfg = AdjointTrainingConfig(**param_dict)

    # Convert numpy arrays to tensors if needed
    if isinstance(X, np.ndarray):
        X = tc.tensor(X, dtype=tc.float32)
    if isinstance(Y, np.ndarray):
        Y = tc.tensor(Y, dtype=tc.float32)

    # Setup
    P, N = X.shape
    PY, M = Y.shape

    if P != PY:
        raise ValueError(f"X and Y must have the same shape in the first dimension but have shapes {P} and {PY}")

    N_edges = model.circuit.number_of_edges()
    batches_per_epoch = P // cfg.batch_size

    # Calculate N_batches based on which parameter was provided
    if cfg.N_batches is not None:
        N_batches = cfg.N_batches
    else:
        # cfg.N_epochs is guaranteed to be set by __post_init__
        N_batches = int(batches_per_epoch * cfg.N_epochs)

    # change inputs to indices if default strings are passed in
    x_inds, y_inds = cfg.x_inds, cfg.y_inds
    if type(x_inds) is str:
        x_inds = model.node_type_dict[x_inds]
    if type(y_inds) is str:
        y_inds = model.node_type_dict[y_inds]

    # Initialize conductances and tracking arrays
    # k = np.ones((N_batches+1, N_edges)) * cfg.k_init

    param_history = {}
    for element in model.circuit.elements:
        name = element.name
        layer = model.element_layers[name]
        init_values = layer.theta.data.clone().detach().numpy()
        param_history[name] = [init_values]

    loss = np.zeros(N_batches)
    accuracy = np.zeros(N_batches)
    
    def loss_function(y, y_free):
        return tc.sum((y - y_free)**2)

    def argmax_classification_accuracy(y, y_free):
        return tc.sum(tc.argmax(y_free, axis=1) == np.argmax(y, axis=1))

    def compute_d_theta(V_F, V_C, layer):
        D_V = (V_F - V_C)[:,None]       # Note: V_edge has a negative sign in it, i.e. model.V_edge() 
        return layer.trainable * cfg.alpha / cfg.eta * tc.mean(D_V * V_F[:,None], axis=0)
        
    # Training loop
    for t in tqdm(range(N_batches)):
        # Sample batch
        batch_inds = random.sample(range(P), cfg.batch_size)
        x = X[batch_inds]
        y = Y[batch_inds]

        # Ensure tensors are on correct device and dtype
        x = x.to(dtype=model.dtype, device=model.device)
        y = y.to(dtype=model.dtype, device=model.device)

        # Compute free and clamped states
        model.set_inputs(x_inds)
        V_node_F, _ = model(x)
        y_F = V_node_F[:, y_inds]
        V_edge_F = model.V_edge() 

        # Compute clamped state
        y_C = cfg.eta * y + (1 - cfg.eta) * y_F.detach()

        model.set_inputs(x_inds, y_inds)
        _ = model(x, y_C)
        V_edge_C = model.V_edge()
        
        # update parameters and history
        for element_name, layer in model.element_layers.items():
            edge_inds = model.element_to_edge_inds[element_name]
            V_F = V_edge_F[:, edge_inds]      
            V_C = V_edge_C[:, edge_inds]

            # apply parameter gradients
            # theta = layer.theta.data
            # d_theta = compute_d_theta(V_F, V_C, layer)
            # layer.update_parameters(theta + d_theta)
            layer.theta.data += compute_d_theta(V_F, V_C, layer)
            layer.clip_parameters()         # enforce limits on model parameter ranges
            
            # param_history[element + '_step'].append(d_theta.clone().detach().numpy())
            param_history[element_name].append(layer.theta.data.clone().detach().numpy())

        # record performance on batch
        loss_batch = loss_function(y, y_F) / cfg.batch_size
        accuracy_batch = argmax_classification_accuracy(y, y_F) / cfg.batch_size
        loss[t] = loss_batch.detach()
        accuracy[t] = accuracy_batch.detach()

    for k, v in param_history.items():
        param_history[k] = np.stack(v, axis=1)      # so that history[theta][i] is the history of theta_i, with shape (N_batch, N_edges)

    history = {
        'accuracy': accuracy,
        'loss': loss
    }
    history.update(param_history)

    return history
    

def InvariantLearning(X, Y, model, config=None, **params):
    # Handle configuration - use config object if provided, otherwise extract from params
    if config is not None:
        if isinstance(config, ILTrainingConfig):
            cfg = config
        else:
            raise TypeError("config must be a ILTrainingConfig object")
    else:
        # Use parameter extraction function
        param_dict = extract_training_params(**params)
        cfg = ILTrainingConfig(**param_dict)

    # Convert numpy arrays to tensors if needed
    if isinstance(X, np.ndarray):
        X = tc.tensor(X, dtype=tc.float32)
    if isinstance(Y, np.ndarray):
        Y = tc.tensor(Y, dtype=tc.float32)

    # Setup
    P, N = X.shape
    PY, M = Y.shape

    if P != PY:
        raise ValueError(f"X and Y must have the same shape in the first dimension but have shapes {P} and {PY}")

    N_edges = model.circuit.number_of_edges()
    batches_per_epoch = P // cfg.batch_size

    # Calculate N_batches based on which parameter was provided
    if cfg.N_batches is not None:
        N_batches = cfg.N_batches
    else:
        # cfg.N_epochs is guaranteed to be set by __post_init__
        N_batches = int(batches_per_epoch * cfg.N_epochs)

    # change inputs to indices if default strings are passed in
    x_inds, y_inds = cfg.x_inds, cfg.y_inds
    if type(x_inds) is str:
        x_inds = model.node_type_dict[x_inds]
    if type(y_inds) is str:
        y_inds = model.node_type_dict[y_inds]

    # Initialize conductances and tracking arrays
    # k = np.ones((N_batches+1, N_edges)) * cfg.k_init

    param_history = {}
    for element in model.circuit.elements:
        name = element.name
        layer = model.element_layers[name]
        init_values = layer.theta.data.clone().detach().numpy()
        param_history[name] = [init_values]

    loss = np.zeros(N_batches)
    accuracy = np.zeros(N_batches)
    
    def loss_function(y, y_free):
        return tc.sum((y - y_free)**2)

    def argmax_classification_accuracy(y, y_free):
        return tc.sum(tc.argmax(y_free, axis=1) == np.argmax(y, axis=1))

    def compute_d_theta(V_F, V_C, layer):
        D_V = (V_F - V_C)[:,None]       # Note: V_edge has a negative sign in it, i.e. model.V_edge() is -Del @ V_node
        d_gamma_d_x = layer.d_gamma_d_x(V_F)[:,None]
        d_gamma_d_theta = layer.d_gamma_d_theta(V_F)
        return layer.trainable * cfg.alpha / cfg.eta * tc.mean(d_gamma_d_x / d_gamma_d_theta * D_V, axis=0)
        # return layer.trainable * cfg.alpha / cfg.eta * layer.theta * tc.mean(D_V / V_F[:,None], axis=0)
        # return layer.trainable * cfg.alpha / cfg.eta * layer.theta * tc.mean(D_V * V_F[:,None], axis=0)
        
    # Training loop
    for t in tqdm(range(N_batches)):
        # Sample batch
        batch_inds = random.sample(range(P), cfg.batch_size)
        x = X[batch_inds]
        y = Y[batch_inds]

        # Ensure tensors are on correct device and dtype
        x = x.to(dtype=model.dtype, device=model.device)
        y = y.to(dtype=model.dtype, device=model.device)

        # Compute free and clamped states
        model.set_inputs(x_inds)
        V_node_F, _ = model(x)
        y_F = V_node_F[:, y_inds]
        V_edge_F = model.V_edge() 

        # Compute clamped state
        y_C = cfg.eta * y + (1 - cfg.eta) * y_F.detach()

        model.set_inputs(x_inds, y_inds)
        _ = model(x, y_C)
        V_edge_C = model.V_edge()
        
        # update parameters and history
        for element_name, layer in model.element_layers.items():
            edge_inds = model.element_to_edge_inds[element_name]
            V_F = V_edge_F[:, edge_inds]      
            V_C = V_edge_C[:, edge_inds]

            # apply parameter gradients
            # theta = layer.theta.data
            # d_theta = compute_d_theta(V_F, V_C, layer)
            # layer.update_parameters(theta + d_theta)
            layer.theta.data += compute_d_theta(V_F, V_C, layer)
            layer.clip_parameters()         # enforce limits on model parameter ranges
            
            # param_history[element + '_step'].append(d_theta.clone().detach().numpy())
            param_history[element_name].append(layer.theta.data.clone().detach().numpy())

        # record performance on batch
        loss_batch = loss_function(y, y_F) / cfg.batch_size
        accuracy_batch = argmax_classification_accuracy(y, y_F) / cfg.batch_size
        loss[t] = loss_batch.detach()
        accuracy[t] = accuracy_batch.detach()

    for k, v in param_history.items():
        param_history[k] = np.stack(v, axis=1)      # so that history[theta][i] is the history of theta_i, with shape (N_batch, N_edges)

    history = {
        'accuracy': accuracy,
        'loss': loss
    }
    history.update(param_history)

    return history
    

def NCCoupledLearning(X, Y, model, config=None, **params):
    """
    Noise contrastive coupled learning
    """

    # Handle configuration - use config object if provided, otherwise extract from params
    if config is not None:
        if isinstance(config, NCCLTrainingConfig):
            cfg = config
        else:
            raise TypeError("config must be a NCCLTrainingConfig object")
    else:
        # Use parameter extraction function
        param_dict = extract_training_params(**params)
        cfg = NCCLTrainingConfig(**param_dict)
    
    # Convert numpy arrays to tensors if needed
    if isinstance(X, np.ndarray):
        X = tc.tensor(X, dtype=tc.float32)
    if isinstance(Y, np.ndarray):
        Y = tc.tensor(Y, dtype=tc.float32)

    # Setup
    P, N = X.shape
    PY, M = Y.shape

    if P != PY:
        raise ValueError(f"X and Y must have the same shape in the first dimension but have shapes {P} and {PY}")

    N_edges = model.circuit.number_of_edges()
    batches_per_epoch = P // cfg.batch_size

    # Calculate N_batches based on which parameter was provided
    if cfg.N_batches is not None:
        N_batches = cfg.N_batches
    else:
        # cfg.N_epochs is guaranteed to be set by __post_init__
        N_batches = int(batches_per_epoch * cfg.N_epochs)

    # change inputs to indices if default strings are passed in
    x_inds, y_inds = cfg.x_inds, cfg.y_inds
    if type(x_inds) is str:
        x_inds = model.node_type_dict[x_inds]
    if type(y_inds) is str:
        y_inds = model.node_type_dict[y_inds]

    # Initialize conductances and tracking arrays
    param_history = {}
    for element in model.circuit.elements:
        name = element.name
        layer = model.element_layers[name]
        init_values = layer.theta.data.clone().detach().numpy()
        param_history[name] = [init_values]

    loss = np.zeros(N_batches)
    accuracy = np.zeros(N_batches)    
    
    def loss_function(y, y_free):
        return tc.sum((y - y_free)**2)

    def argmax_classification_accuracy(y, y_free):
        return tc.sum(tc.argmax(y_free, axis=1) == np.argmax(y, axis=1))

    # Training loop
    for t in tqdm(range(N_batches)):
        # Sample batch
        batch_inds = random.sample(range(P), cfg.batch_size)
        x = X[batch_inds]
        y = Y[batch_inds]
        P_batch = x.shape[0]

        # Ensure tensors are on correct device and dtype
        x = x.to(dtype=model.dtype, device=model.device)
        y = y.to(dtype=model.dtype, device=model.device)

        # Compute free and clamped states
        model.set_inputs(x_inds)
        V_node_f, _ = model(x)      # get free node voltages (to build clamped state)
        y_f = V_node_f[:, y_inds].detach()
        V_edge_f = model.V_edge()

        # Compute clamped state
        y_c = cfg.eta * y + (1 - cfg.eta) * y_f    #  build clamped state
        model.set_inputs(x_inds, y_inds)
        _ = model(x, y_c)
        V_edge_c = model.V_edge()

        # Compute negative clamped states
        classes = tc.unique(y, dim=0)   # get all unique output states in batch
        N_classes = len(classes)
        V_edge_n = tc.zeros((N_classes, P_batch, N_edges))
        for k, cur_y in enumerate(classes):
            cur_y = cur_y[None, :]
            y_n = cfg.eta * cur_y + (1 - cfg.eta) * y_f
            _ = model(x, y_n)
            V_edge_n[k] = model.V_edge()
        
        # update parameters and history
        for element_name, layer in model.element_layers.items():
            edge_inds = model.element_to_edge_inds[element_name]
            N_edges_cur = layer.N_edges
            V_f = V_edge_f[:, edge_inds]      
            V_c = V_edge_c[:, edge_inds]
            V_n = V_edge_n[:, :, edge_inds].reshape(N_classes * P_batch, N_edges_cur)       # This combines the negative class and batch indices. Pytorch reshapes according to the rightmost index changing the fastest. Hence if A has shape (2,3,4), then A[1,0,:] is equivalent to A.reshape(6,4)[3,:]. Therefore the ordering after reshaping here is like (c1 p1, c1 p2, c1 p3,... c1 pP, c2 p1, ...), where c1 means class 1 and p1 means point 1.

            # compute gradients
            grad_rho_f = tc.mean(layer.d_rho_d_theta(V_f), axis=0)
            grad_rho_c = tc.mean(layer.d_rho_d_theta(V_c), axis=0)
            grad_rho_n = tc.mean(layer.d_rho_d_theta(V_n), axis=0)      # includes a factor of 1 / N_classes due to mean, since d_rho_d_theta has shape (P_batch * N_classes, )

            # apply parameter gradients
            K = N_batches
            d_theta = - layer.trainable * cfg.alpha / cfg.eta * (
                (1 + cfg.nu / (K - 1)) * grad_rho_c \
                - (1 - cfg.nu) * grad_rho_f \
                - K / (K - 1) * cfg.nu * grad_rho_n
            )
            # K = N_batches
            # d_theta = - layer.trainable * cfg.alpha / cfg.eta * (
            #     (1 + cfg.nu / K) * grad_rho_c \
            #     - grad_rho_f \
            #     - cfg.nu * grad_rho_n
            # )

            layer.theta.data += d_theta
            # layer.theta.data += compute_d_theta(V_F, V_C, V_N, layer)
            layer.clip_parameters()         # enforce limits on model parameter ranges
            
            # param_history[element + '_step'].append(d_theta.clone().detach().numpy())
            param_history[element_name].append(layer.theta.data.clone().detach().numpy())

        # record performance on batch
        loss_batch = loss_function(y, y_f) / cfg.batch_size
        accuracy_batch = argmax_classification_accuracy(y, y_f) / cfg.batch_size
        loss[t] = loss_batch.detach()
        accuracy[t] = accuracy_batch.detach()

    for k, v in param_history.items():
        param_history[k] = np.stack(v, axis=1)      # so that history[theta][i] is the history of theta_i, with shape (N_batch, N_edges)

    history = {
        'accuracy': accuracy,
        'loss': loss
    }
    history.update(param_history)

    return history

def CoupledLearning(X, Y, model, config=None, **params):
    """
    X is a (P, N) ndarray, N-dim input features for P samples
    Y is a (P, M) ndarray, M-dim output targets for P samples  
    model is a CircuitModel object
    config is a CLTrainingConfig object, or parameters can be passed as kwargs
    """
    # Handle configuration - use config object if provided, otherwise extract from params
    if config is not None:
        if isinstance(config, CLTrainingConfig):
            cfg = config
        else:
            raise TypeError("config must be a CLTrainingConfig object")
    else:
        # Use parameter extraction function
        param_dict = extract_training_params(**params)
        cfg = CLTrainingConfig(**param_dict)
    
    # Convert numpy arrays to tensors if needed
    if isinstance(X, np.ndarray):
        X = tc.tensor(X, dtype=tc.float32)
    if isinstance(Y, np.ndarray):
        Y = tc.tensor(Y, dtype=tc.float32)

    # Setup
    P, N = X.shape
    PY, M = Y.shape

    if P != PY:
        raise ValueError(f"X and Y must have the same shape in the first dimension but have shapes {P} and {PY}")

    N_edges = model.circuit.number_of_edges()
    batches_per_epoch = P // cfg.batch_size

    # Calculate N_batches based on which parameter was provided
    if cfg.N_batches is not None:
        N_batches = cfg.N_batches
    else:
        # cfg.N_epochs is guaranteed to be set by __post_init__
        N_batches = int(batches_per_epoch * cfg.N_epochs)

    # change inputs to indices if default strings are passed in
    x_inds, y_inds = cfg.x_inds, cfg.y_inds
    if type(x_inds) is str:
        x_inds = model.node_type_dict[x_inds]
    if type(y_inds) is str:
        y_inds = model.node_type_dict[y_inds]

    # Initialize conductances and tracking arrays
    # k = np.ones((N_batches+1, N_edges)) * cfg.k_init

    param_history = {}
    for element in model.circuit.elements:
        name = element.name
        layer = model.element_layers[name]
        init_values = layer.theta.data.clone().detach().numpy()
        param_history[name] = [init_values]


    loss = np.zeros(N_batches)
    accuracy = np.zeros(N_batches)
    
    
    def loss_function(y, y_free):
        return tc.sum((y - y_free)**2)

    def argmax_classification_accuracy(y, y_free):
        return tc.sum(tc.argmax(y_free, axis=1) == np.argmax(y, axis=1))

    def compute_d_theta(V_F, V_C, layer):
        grad_rho_F = layer.d_rho_d_theta(V_F)
        grad_rho_C = layer.d_rho_d_theta(V_C)
        return layer.trainable * cfg.alpha / cfg.eta * tc.mean((grad_rho_F - grad_rho_C), axis=0)
        # return layer.trainable * cfg.alpha / cfg.eta / 2 * tc.mean((V_F**2 - V_C**2), axis=0)
        
    # Training loop
    for t in tqdm(range(N_batches)):
        # Sample batch
        batch_inds = random.sample(range(P), cfg.batch_size)
        x = X[batch_inds]
        y = Y[batch_inds]

        # Ensure tensors are on correct device and dtype
        x = x.to(dtype=model.dtype, device=model.device)
        y = y.to(dtype=model.dtype, device=model.device)

        # Compute free and clamped states
        model.set_inputs(x_inds)
        V_node_F, _ = model(x)
        y_F = V_node_F[:, y_inds]
        V_edge_F = model.V_edge() 

        # Compute clamped state
        y_C = cfg.eta * y + (1 - cfg.eta) * y_F.detach()

        model.set_inputs(x_inds, y_inds)
        _ = model(x, y_C)
        V_edge_C = model.V_edge()
        
        # update parameters and history
        for element_name, layer in model.element_layers.items():
            edge_inds = model.element_to_edge_inds[element_name]
            V_F = V_edge_F[:, edge_inds]      
            V_C = V_edge_C[:, edge_inds]

            # apply parameter gradients
            # theta = layer.theta.data
            # d_theta = compute_d_theta(V_F, V_C, layer)
            # layer.update_parameters(theta + d_theta)
            layer.theta.data += compute_d_theta(V_F, V_C, layer)
            layer.clip_parameters()         # enforce limits on model parameter ranges
            
            # param_history[element + '_step'].append(d_theta.clone().detach().numpy())
            param_history[element_name].append(layer.theta.data.clone().detach().numpy())

        # record performance on batch
        loss_batch = loss_function(y, y_F) / cfg.batch_size
        accuracy_batch = argmax_classification_accuracy(y, y_F) / cfg.batch_size
        loss[t] = loss_batch.detach()
        accuracy[t] = accuracy_batch.detach()

    for k, v in param_history.items():
        param_history[k] = np.stack(v, axis=1)      # so that history[theta][i] is the history of theta_i, with shape (N_batch, N_edges)

    history = {
        'accuracy': accuracy,
        'loss': loss
    }
    history.update(param_history)

    return history


### A method to test a circuit on test data
def TestClassificationAccuracy(X, Y, model, config=None, **params):
    """
    X is a (P, N) ndarray, N-dim input features for P samples
    Y is a (P, M) ndarray, M-dim output targets for P samples
    model is a CircuitModel object
    config is a CLTestConfig object, or parameters can be passed as kwargs

    Returned quantities are packaged together in the performance dictionary.
    """
    
    # Handle configuration - use config object if provided, otherwise extract from params
    if config is not None:
        if isinstance(config, CLTestingConfig):
            cfg = config
        else:
            raise TypeError("config must be a TestConfig object")
    else:
        # Create config from params - for now just use defaults and update with provided params
        cfg = CLTestingConfig(**params)

    # Convert numpy arrays to tensors if needed
    if isinstance(X, np.ndarray):
        X = tc.tensor(X, dtype=tc.float32)
    if isinstance(Y, np.ndarray):
        Y = tc.tensor(Y, dtype=tc.float32)

    P, N = X.shape
    PY, M = Y.shape

    if P != PY:
        raise ValueError(f"X and Y must have the same shape in the first dimension but have shapes {P} and {PY}")

    # Get node indices
    x_inds, y_inds = cfg.x_inds, cfg.y_inds
    if type(x_inds) is str:
        x_inds = model.node_type_dict[x_inds]
    if type(y_inds) is str:
        y_inds = model.node_type_dict[y_inds]

    # For testing, we don't have time-varying parameters like in the old version
    T = 1

    # Calculate number of batches based on which parameter was provided
    batches_per_epoch = P // cfg.batch_size
    if cfg.N_batches is not None:
        N_batches = cfg.N_batches
    else:
        # cfg.N_epochs is guaranteed to be set by __post_init__
        N_batches = int(batches_per_epoch * cfg.N_epochs)

    # Initialize arrays
    accuracy = np.zeros(N_batches)
    loss = np.zeros(N_batches)

    def loss_function(y, y_free):
        return tc.sum((y - y_free)**2)

    def argmax_classification_accuracy(y, y_free):
        return tc.sum(tc.argmax(y_free, axis=1) == tc.argmax(y, axis=1))

    # Set model to evaluation mode
    model.set_inputs(x_inds)

    # Process all batches
    for batch_idx in tqdm(range(N_batches), desc='Testing'):
        # Sample batch indices
        actual_batch_size = min(cfg.batch_size, P)
        if actual_batch_size == P:
            # Use all samples if batch size >= dataset size
            batch_inds = list(range(P))
        else:
            batch_inds = random.sample(range(P), actual_batch_size)

        # Get batch data
        x = X[batch_inds, :]
        y = Y[batch_inds, :]

        # Ensure tensors are on correct device and dtype
        x_tensor = x.to(dtype=model.dtype, device=model.device)
        y_tensor = y.to(dtype=model.dtype, device=model.device)

        # Forward pass: solve circuit in free phase
        V_node, _ = model(x_tensor)
        y_free = V_node[:, y_inds]

        # Compute metrics
        batch_loss = loss_function(y_tensor, y_free) / actual_batch_size
        batch_accuracy = argmax_classification_accuracy(y_tensor, y_free) / actual_batch_size

        accuracy[batch_idx] = batch_accuracy.detach().cpu().numpy()
        loss[batch_idx] = batch_loss.detach().cpu().numpy()
    
    # Compute statistics
    accuracy_mean = np.mean(accuracy)
    accuracy_stdev = np.std(accuracy)
    loss_mean = np.mean(loss)
    loss_stdev = np.std(loss)

    performance = {
        'accuracy': accuracy,                   # (N_batches,) ndarray
        'accuracy_mean': accuracy_mean,         # scalar
        'accuracy_stdev': accuracy_stdev,       # scalar
        'loss': loss,                           # (N_batches,) ndarray
        'loss_mean': loss_mean,                 # scalar
        'loss_stdev': loss_stdev,               # scalar
    }
    return performance

def argmax_classification_accuracy(y, y_free):
    return np.argmax(y_free) == np.argmax(y)

