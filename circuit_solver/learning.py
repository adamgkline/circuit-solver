import numpy as np
import torch as tc
import random
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class CLTrainingConfig:
    """Configuration for coupled learning training"""
    x_inds: int = 'x'
    y_inds: int = 'y'
    batch_size: int = 100
    N_epochs: int = 1
    eta: float = 0.5
    alpha: float = 1.0

@dataclass
class CLTestingConfig:
    """Configuration for testing circuit performance"""
    x_inds: str = 'x'
    y_inds: str = 'y'
    batch_size: int = 100
    # N_batches: Optional[int] = None
    N_epochs: int = 1





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
    N_epochs = cfg.N_epochs
    N_batches = int(batches_per_epoch * N_epochs)

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
        # return layer.trainable * cfg.alpha / cfg.eta * tc.mean((V_F**2 - V_C**2), axis=0)
        
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
        y_C = cfg.eta * y + (1 - cfg.eta) * y_F

        model.set_inputs(x_inds, y_inds)
        _ = model(x, y)
        V_edge_C = model.V_edge()
        
        # update parameters and history
        for element_name, layer in model.element_layers.items():
            edge_inds = model.element_to_edge_inds[element_name]
            V_F = V_edge_F[:, edge_inds]      
            V_C = V_edge_C[:, edge_inds]

            # apply parameter gradients
            layer.theta.data += compute_d_theta(V_F, V_C, layer)
            layer.clip_parameters()         # enforce limits on model parameter ranges
            
            # param_history[element + '_step'].append(d_theta.clone().detach().numpy())
            param_history[element_name].append(layer.theta.data.clone().detach().numpy())

        # record performance on batch
        loss_batch = loss_function(y, y_F) / cfg.batch_size
        accuracy_batch = argmax_classification_accuracy(y, y_F) / cfg.batch_size
        loss[t] = loss_batch
        accuracy[t] = accuracy_batch

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
    
    # Calculate number of batches
    batches_per_epoch = P // cfg.batch_size
    N_epochs = cfg.N_epochs
    N_batches = int(batches_per_epoch * N_epochs)
    # N_batches = cfg.N_batches if cfg.N_batches is not None else P // cfg.batch_size
    # if N_batches == 0:
        # N_batches = 1

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

