import numpy as np
import torch as tc
import random
import warnings
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Dict, Any
from circuit_solver import utils


@dataclass
class BaseTrainingConfig:
    """Base configuration class with common validation logic"""
    x_nodes: str = 'x'  # TODO: was annotated as `int` but should be `str` (or `Union[str, list]`)
    y_nodes: str = 'y'  # TODO: same as above
    batch_size: int = 100
    N_epochs: Optional[int] = None
    N_batches: Optional[int] = None
    clamp_method: str = 'MSE'
    parameter_update_method: Optional[str] = None
    softmax_temp: float = 1.
    overclamp_gain: float = 200

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
class CLTrainingConfig(BaseTrainingConfig):
    """Configuration for coupled learning training"""
    eta: float = 1.0
    alpha: float = 1.0

@dataclass
class CLTestingConfig(BaseTrainingConfig):
    """Configuration for testing circuit performance"""
    pass

@dataclass
class ILTrainingConfig(BaseTrainingConfig):
    """Configuration for invariant learning training"""
    eta: float = 1.0
    alpha: float = 1.0

@dataclass
class AdjointTrainingConfig(BaseTrainingConfig):
    """Configuration for adjoint learning training"""
    eta: float = 1.0
    alpha: float = 1.0

@dataclass
class GCLTrainingConfig(BaseTrainingConfig):
    """Configuration for geometric coupled learning training"""
    eta: float = 1.0
    alpha: float = 1.0

@dataclass
class NCCLTrainingConfig(BaseTrainingConfig):
    """Configuration for noise contrastive coupled learning training"""
    eta: float = 1.0
    alpha: float = 1.0
    nu: float = 1.0


# ============================================================================
# Utility Functions
# ============================================================================

def loss_function(y, y_free):
    """Compute MSE loss between target and prediction"""
    return tc.sum((y - y_free)**2)

def distance_classification_accuracy(y, y_free):
    ...

def argmax_classification_accuracy(y, y_free):
    """Compute classification accuracy based on argmax"""
    return tc.sum(tc.argmax(y_free, axis=1) == tc.argmax(y, axis=1))

def extract_training_params(**params):
    """Extract training parameters from kwargs for config creation"""
    valid_keys = {'x_nodes', 'y_nodes', 'batch_size', 'N_epochs', 'N_batches', 'eta', 'alpha', 'nu'}
    return {k: v for k, v in params.items() if k in valid_keys}

def convert_to_tensors(X, Y):
    """Convert numpy arrays to tensors if needed"""
    if isinstance(X, np.ndarray):
        X = tc.tensor(X, dtype=tc.float32)
    if isinstance(Y, np.ndarray):
        Y = tc.tensor(Y, dtype=tc.float32)
    return X, Y

def validate_data_shapes(X, Y):
    """Validate that X and Y have compatible shapes"""
    P, N = X.shape
    PY, M = Y.shape
    if P != PY:
        raise ValueError(f"X and Y must have the same shape in the first dimension but have shapes {P} and {PY}")
    return P, N, M

def resolve_node_indices(x_nodes, y_nodes, model):
    """Convert string node indices to actual indices if needed"""
    if type(x_nodes) is str:
        x_nodes = model.node_type_dict[x_nodes]
    if type(y_nodes) is str:
        y_nodes = model.node_type_dict[y_nodes]
    return x_nodes, y_nodes

def initialize_param_history(model):
    """Initialize parameter history dictionary"""
    param_history = {}
    for element in model.circuit.elements:
        name = element.name
        layer = model.element_layers[name]
        init_values = layer.theta.data.clone().detach().numpy()
        param_history[name] = [init_values]
    return param_history

def finalize_history(param_history, loss, accuracy):
    """Finalize and return history dictionary"""
    for k, v in param_history.items():
        param_history[k] = np.stack(v, axis=1)

    history = {
        'accuracy': accuracy,
        'loss': loss
    }
    history.update(param_history)
    return history

###### CLAMPING METHODS

def mse_clamping(y, y_F, cfg):
    return cfg.eta * y + (1 - cfg.eta) * y_F

def overclamping(y, y_F, cfg):
    return (1 - cfg.eta) * y_F + cfg.eta * cfg.overclamp_gain * tc.sign(y - y_F)


def mse_symmetric_clamping(y, y_F, cfg):
    return cfg.eta * y + (1 - cfg.eta) * y_F


def constant_clamping(y, y_F, eta):
    pass

softmax = tc.nn.Softmax(dim=1)   # normalize over feature dimension (batch, feature)
def ce_legacy_clamping(y, y_F, cfg):
    T = cfg.softmax_temp
    q = softmax(y_F / T)
    p = softmax(y * 10E8)
    return cfg.eta * T * (p - q) + y_F

# def ce_clamping(y, y_F, cfg):
#     T = cfg.softmax_temp
#     q = softmax(y_F / T)
#     p = softmax(y * 10E8) # this is hacky and bad, I know... Ideally I'd just pass in 1-hot vectors p but this would require some complicated stuff elsewhere
#     return y_F + cfg.eta * T / q * (p - q)

def ce_natural_clamping(y, y_F, cfg):
    T = cfg.softmax_temp
    q = softmax(y_F / T)
    p = softmax(y * 10E8) # this is hacky and bad, I know... Ideally I'd just pass in 1-hot vectors p but this would require some complicated stuff elsewhere
    return y_F + cfg.eta * T / q * (p - q)

def hinge_clamping(y, y_F, eta):
    # TODO: stub — returns None, which will crash downstream when used as a clamped state
    pass

clamping_functions = {
    'overclamping': overclamping,
    'MSE' : mse_clamping,
    'MSE_symmetric' : mse_symmetric_clamping,
    'cross_entropy' : ce_natural_clamping,
    'cross_entropy_natural' : ce_natural_clamping,
    'cross_entropy_legacy' : ce_legacy_clamping,
    'hinge': hinge_clamping,
}

###### PARAMETER UPDATE METHODS
@dataclass
class AdamConfig:
    # TODO: trailing commas make these tuples instead of floats — remove commas
    beta1 : float = 0.9,   # BUG: default is (0.9,) not 0.9
    beta2 : float = 0.999, # BUG: default is (0.999,) not 0.999
    eps : float = 1E-6,    # BUG: default is (1e-6,) not 1e-6



def adam_update(t, g, m, v, ucfg):
    """
    t: timestep
    g: original update step
    m:
    """
    # TODO: `beta1`, `beta2`, and `gamma` are undefined — should be `ucfg.beta1`,
    # `ucfg.beta2`, and a learning rate passed in as an argument
    m[t] = beta1 * m[t-1] + (1 - beta1) * g[t]        # BUG: beta1 not defined
    v[t] = beta2 * v[t-1] + (1 - beta2) * g[t] ** 2   # BUG: beta2 not defined
    m_hat = m[t] / (1 - ucfg.beta1 ** t)
    v_hat = v[t] / (1 - ucfg.beta2 ** t)
    d_theta = - gamma * m_hat / (tc.sqrt(v_hat) + ucfg.eps)  # BUG: gamma not defined
    return d_theta

parameter_update_methods = {
    'adam': adam_update
}

# ============================================================================
# Training Functions
# ============================================================================

def CoupledLearningEtaZero(X, Y, model, config=None, **params):
    """
    X is a (P, N) ndarray, N-dim input features for P samples
    Y is a (P, M) ndarray, M-dim output targets for P samples
    model is a CircuitModel object
    config is a CLTrainingConfig object, or parameters can be passed as kwargs
    """
    # Handle configuration
    if config is not None:
        if isinstance(config, CLTrainingConfig):
            cfg = config
        else:
            raise TypeError("config must be a CLTrainingConfig object")
    else:
        param_dict = extract_training_params(**params)
        cfg = CLTrainingConfig(**param_dict)

    # Convert and validate data
    X, Y = convert_to_tensors(X, Y)
    P, N, M = validate_data_shapes(X, Y)

    # Setup
    N_edges = model.circuit.number_of_edges()
    batches_per_epoch = P // cfg.batch_size
    N_batches = cfg.N_batches if cfg.N_batches is not None else int(batches_per_epoch * cfg.N_epochs)

    # Resolve node indices
    x_nodes, y_nodes = resolve_node_indices(cfg.x_nodes, cfg.y_nodes, model)
    x_inds = utils.nodes_to_inds(x_nodes, model.circuit)
    y_inds = utils.nodes_to_inds(y_nodes, model.circuit)

    # Initialize tracking arrays
    param_history = initialize_param_history(model)
    loss = np.zeros(N_batches)
    accuracy = np.zeros(N_batches)

    def compute_d_theta(V_F, epsilon, layer):
        return - layer.learning_rates * cfg.alpha * tc.mean(V_F * epsilon, axis=0)
        
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
        model.set_inputs(x_nodes)
        V_node_F, _ = model(x)
        y_F = V_node_F[:, y_inds]
        V_edge_F = model.V_edge() 

        # Compute error
        model.set_inputs(x_nodes, y_nodes)
        _ = model(x, y)
        V_edge_T = model.V_edge()
        epsilon = V_edge_T.detach() - V_edge_F.detach()
        
        # update parameters and history
        for element_name, layer in model.element_layers.items():
            edge_inds = model.element_to_inds[element_name]
            V_F = V_edge_F[:, edge_inds]      
            cur_epsilon = epsilon[:, edge_inds]

            # apply parameter gradients
            # theta = layer.theta.data
            # d_theta = compute_d_theta(V_F, V_C, layer)
            # layer.update_parameters(theta + d_theta)
            layer.theta.data += compute_d_theta(V_F, cur_epsilon, layer)
            layer.clip_parameters()         # enforce limits on model parameter ranges
            
            # param_history[element + '_step'].append(d_theta.clone().detach().numpy())
            param_history[element_name].append(layer.theta.data.clone().detach().numpy())

        # record performance on batch
        loss_batch = loss_function(y, y_F) / cfg.batch_size
        accuracy_batch = argmax_classification_accuracy(y, y_F) / cfg.batch_size
        loss[t] = loss_batch.detach()
        accuracy[t] = accuracy_batch.detach()

    return finalize_history(param_history, loss, accuracy)


def CoupledLearning(X, Y, model, config=None, **params):
    """
    X is a (P, N) ndarray, N-dim input features for P samples
    Y is a (P, M) ndarray, M-dim output targets for P samples
    model is a CircuitModel object
    config is a CLTrainingConfig object, or parameters can be passed as kwargs
    """
    # Handle configuration
    if config is not None:
        if isinstance(config, CLTrainingConfig):
            cfg = config
        else:
            raise TypeError("config must be a CLTrainingConfig object")
    else:
        param_dict = extract_training_params(**params)
        cfg = CLTrainingConfig(**param_dict)

    # Convert and validate data
    X, Y = convert_to_tensors(X, Y)
    P, N, M = validate_data_shapes(X, Y)

    # Setup
    N_edges = model.circuit.number_of_edges()
    batches_per_epoch = P // cfg.batch_size
    N_batches = cfg.N_batches if cfg.N_batches is not None else int(batches_per_epoch * cfg.N_epochs)

    # Resolve node indices
    x_nodes, y_nodes = resolve_node_indices(cfg.x_nodes, cfg.y_nodes, model)
    x_inds = utils.nodes_to_inds(x_nodes, model.circuit)
    y_inds = utils.nodes_to_inds(y_nodes, model.circuit)

    # Initialize tracking arrays
    param_history = initialize_param_history(model)
    loss = np.zeros(N_batches)
    accuracy = np.zeros(N_batches)

    def compute_d_theta(V_F, V_C, layer):
        grad_rho_F = layer.d_rho_d_theta(V_F)
        grad_rho_C = layer.d_rho_d_theta(V_C)
        return layer.learning_rates * cfg.alpha / cfg.eta * tc.mean((grad_rho_F - grad_rho_C), axis=0)
        # return layer.learning_rates * cfg.alpha / cfg.eta / 2 * tc.mean((V_F**2 - V_C**2), axis=0)
    
    compute_clamped_state = clamping_functions[cfg.clamp_method]
    
    # TODO: implement Adam update, need to initialize m and v tensors which match the shape of theta 
    # parameter_update_method = parameter_update_methods[cfg.parameter_update_method]

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
        model.set_inputs(x_nodes)
        V_node_F, _ = model(x)
        y_F = V_node_F[:, y_inds]
        V_edge_F = model.V_edge() 

        # Compute clamped state
        y_C = compute_clamped_state(y, y_F.detach(), cfg)
        # y_C = cfg.eta * y + (1 - cfg.eta) * y_F.detach()

        model.set_inputs(x_nodes, y_nodes)
        _ = model(x, y_C)
        V_edge_C = model.V_edge()
        
        # update parameters and history
        for element_name, layer in model.element_layers.items():
            edge_inds = model.element_to_inds[element_name]
            V_F = V_edge_F[:, edge_inds]      
            V_C = V_edge_C[:, edge_inds]

            # apply parameter gradients     # TODO: implement Adam update
            layer.theta.data += compute_d_theta(V_F, V_C, layer)
            layer.clip_parameters()         # enforce limits on model parameter ranges
            
            # param_history[element + '_step'].append(d_theta.clone().detach().numpy())
            param_history[element_name].append(layer.theta.data.clone().detach().numpy())

        # record performance on batch
        loss_batch = loss_function(y, y_F) / cfg.batch_size
        accuracy_batch = argmax_classification_accuracy(y, y_F) / cfg.batch_size
        loss[t] = loss_batch.detach()
        accuracy[t] = accuracy_batch.detach()

    return finalize_history(param_history, loss, accuracy)


def SymmetricCoupledLearning(X, Y, model, config=None, **params):
    """
    X is a (P, N) ndarray, N-dim input features for P samples
    Y is a (P, M) ndarray, M-dim output targets for P samples
    model is a CircuitModel object
    config is a CLTrainingConfig object, or parameters can be passed as kwargs
    """
    # Handle configuration
    if config is not None:
        if isinstance(config, CLTrainingConfig):
            cfg = config
        else:
            raise TypeError("config must be a CLTrainingConfig object")
    else:
        param_dict = extract_training_params(**params)
        cfg = CLTrainingConfig(**param_dict)

    # Convert and validate data
    X, Y = convert_to_tensors(X, Y)
    P, N, M = validate_data_shapes(X, Y)

    # Setup
    N_edges = model.circuit.number_of_edges()
    batches_per_epoch = P // cfg.batch_size
    N_batches = cfg.N_batches if cfg.N_batches is not None else int(batches_per_epoch * cfg.N_epochs)

    # Resolve node indices
    x_nodes, y_nodes = resolve_node_indices(cfg.x_nodes, cfg.y_nodes, model)
    x_inds = utils.nodes_to_inds(x_nodes, model.circuit)
    y_inds = utils.nodes_to_inds(y_nodes, model.circuit)

    # Initialize tracking arrays
    param_history = initialize_param_history(model)
    loss = np.zeros(N_batches)
    accuracy = np.zeros(N_batches)

    def compute_d_theta(V_C_neg, V_C_pos, layer):
        grad_rho_C_neg = layer.d_rho_d_theta(V_C_neg)
        grad_rho_C_pos = layer.d_rho_d_theta(V_C_pos)
        return layer.learning_rates * cfg.alpha / (2 * cfg.eta) * tc.mean((grad_rho_C_neg - grad_rho_C_pos), axis=0)
        # return layer.learning_rates * cfg.alpha / cfg.eta / 2 * tc.mean((V_F**2 - V_C**2), axis=0)
    
    compute_clamped_state = clamping_functions[cfg.clamp_method]
    
    # TODO: implement Adam update, need to initialize m and v tensors which match the shape of theta 
    # parameter_update_method = parameter_update_methods[cfg.parameter_update_method]

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
        model.set_inputs(x_nodes)
        V_node_F, _ = model(x)
        y_F = V_node_F[:, y_inds]
        V_edge_F = model.V_edge() 

        # Compute clamp states for outputs
        y_Fd = y_F.detach()
        y_C_pos = compute_clamped_state(y, y_Fd, cfg)        
        y_C_neg = y_Fd - (y_C_pos - y_Fd) # negative clamp

        # y_C_pos = compute_clamped_state(y, y_F.detach(), cfg)        
        # y_C_neg = y_F.detach() - (y_C_pos - y_F.detach()) # negative clamp

        # Compute clamped states for network
        model.set_inputs(x_nodes, y_nodes)
        _ = model(x, y_C_pos)
        V_edge_C_pos = model.V_edge()
        _ = model(x, y_C_neg)
        V_edge_C_neg = model.V_edge()
        
        # update parameters and history
        for element_name, layer in model.element_layers.items():
            edge_inds = model.element_to_inds[element_name]
            V_C_neg = V_edge_C_neg[:, edge_inds]      
            V_C_pos = V_edge_C_pos[:, edge_inds]

            # apply parameter gradients     # TODO: implement Adam update
            layer.theta.data += compute_d_theta(V_C_neg, V_C_pos, layer)
            layer.clip_parameters()         # enforce limits on model parameter ranges
            
            # param_history[element + '_step'].append(d_theta.clone().detach().numpy())
            param_history[element_name].append(layer.theta.data.clone().detach().numpy())

        # record performance on batch
        loss_batch = loss_function(y, y_F) / cfg.batch_size
        accuracy_batch = argmax_classification_accuracy(y, y_F) / cfg.batch_size
        loss[t] = loss_batch.detach()
        accuracy[t] = accuracy_batch.detach()

    return finalize_history(param_history, loss, accuracy)


# ============================================================================
# Testing Functions
# ============================================================================

def TestClassificationAccuracy(X, Y, model, config=None, **params):
    """
    X is a (P, N) ndarray, N-dim input features for P samples
    Y is a (P, M) ndarray, M-dim output targets for P samples
    model is a CircuitModel object
    config is a CLTestConfig object, or parameters can be passed as kwargs

    Returned quantities are packaged together in the performance dictionary.
    """
    # Handle configuration
    if config is not None:
        if isinstance(config, CLTestingConfig):
            cfg = config
        else:
            raise TypeError("config must be a TestConfig object")
    else:
        cfg = CLTestingConfig(**params)

    # Convert and validate data
    X, Y = convert_to_tensors(X, Y)
    P, N, M = validate_data_shapes(X, Y)

    # Resolve node indices
    x_nodes, y_nodes = resolve_node_indices(cfg.x_nodes, cfg.y_nodes, model)
    x_inds = utils.nodes_to_inds(x_nodes, model.circuit)
    y_inds = utils.nodes_to_inds(y_nodes, model.circuit)

    # Calculate number of batches
    batches_per_epoch = P // cfg.batch_size
    N_batches = cfg.N_batches if cfg.N_batches is not None else int(batches_per_epoch * cfg.N_epochs)

    # Initialize arrays
    accuracy = np.zeros(N_batches)
    loss = np.zeros(N_batches)

    # Set model to evaluation mode
    model.set_inputs(x_nodes)

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


# ============================================================================
# Experimental learning rule training functions
# ============================================================================


def GeoCoupledLearning(X, Y, model, config=None, **params):
    # Handle configuration
    if config is not None:
        if isinstance(config, GCLTrainingConfig):
            cfg = config
        else:
            raise TypeError("config must be a GCLTrainingConfig object")
    else:
        param_dict = extract_training_params(**params)
        cfg = GCLTrainingConfig(**param_dict)

    # Convert and validate data
    X, Y = convert_to_tensors(X, Y)
    P, N, M = validate_data_shapes(X, Y)

    # Setup
    N_edges = model.circuit.number_of_edges()
    batches_per_epoch = P // cfg.batch_size
    N_batches = cfg.N_batches if cfg.N_batches is not None else int(batches_per_epoch * cfg.N_epochs)

    # Resolve node indices
    x_nodes, y_nodes = resolve_node_indices(cfg.x_nodes, cfg.y_nodes, model)
    x_inds = utils.nodes_to_inds(x_nodes, model.circuit)
    y_inds = utils.nodes_to_inds(y_nodes, model.circuit)

    # Initialize tracking arrays
    param_history = initialize_param_history(model)
    loss = np.zeros(N_batches)
    accuracy = np.zeros(N_batches)

    def compute_d_theta(V_F, V_C, layer):
        # D_V = (V_F - V_C)[:,None]       # Note: V_edge has a negative sign in it, i.e. model.V_edge() 

        return layer.learning_rates * cfg.alpha / cfg.eta * layer.theta * tc.mean(((V_F**2 - V_C**2) / V_F / V_C)[:,None], axis=0)
        
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
        model.set_inputs(x_nodes)
        V_node_F, _ = model(x)
        y_F = V_node_F[:, y_inds]
        V_edge_F = model.V_edge() 

        # Compute clamped state
        y_C = cfg.eta * y + (1 - cfg.eta) * y_F.detach()

        model.set_inputs(x_nodes, y_nodes)
        _ = model(x, y_C)
        V_edge_C = model.V_edge()
        
        # update parameters and history
        for element_name, layer in model.element_layers.items():
            edge_inds = model.element_to_inds[element_name]
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

    return finalize_history(param_history, loss, accuracy)

def AdjointLearning(X, Y, model, config=None, **params):
    # Handle configuration
    if config is not None:
        if isinstance(config, AdjointTrainingConfig):
            cfg = config
        else:
            raise TypeError("config must be a AdjointTrainingConfig object")
    else:
        param_dict = extract_training_params(**params)
        cfg = AdjointTrainingConfig(**param_dict)

    # Convert and validate data
    X, Y = convert_to_tensors(X, Y)
    P, N, M = validate_data_shapes(X, Y)

    # Setup
    N_edges = model.circuit.number_of_edges()
    batches_per_epoch = P // cfg.batch_size
    N_batches = cfg.N_batches if cfg.N_batches is not None else int(batches_per_epoch * cfg.N_epochs)

    # Resolve node indices
    x_nodes, y_nodes = resolve_node_indices(cfg.x_nodes, cfg.y_nodes, model)
    x_inds = utils.nodes_to_inds(x_nodes, model.circuit)
    y_inds = utils.nodes_to_inds(y_nodes, model.circuit)

    # Initialize tracking arrays
    param_history = initialize_param_history(model)
    loss = np.zeros(N_batches)
    accuracy = np.zeros(N_batches)

    def compute_d_theta(V_F, V_C, layer):
        D_V = (V_F - V_C)[:,None]       # Note: V_edge has a negative sign in it, i.e. model.V_edge() 
        return layer.learning_rates * cfg.alpha / cfg.eta * tc.mean(D_V * V_F[:,None], axis=0)
        
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
        model.set_inputs(x_nodes)
        V_node_F, _ = model(x)
        y_F = V_node_F[:, y_inds]
        V_edge_F = model.V_edge() 

        # Compute clamped state
        y_C = cfg.eta * y + (1 - cfg.eta) * y_F.detach()

        model.set_inputs(x_nodes, y_nodes)
        _ = model(x, y_C)
        V_edge_C = model.V_edge()
        
        # update parameters and history
        for element_name, layer in model.element_layers.items():
            edge_inds = model.element_to_inds[element_name]
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

    return finalize_history(param_history, loss, accuracy)

def InvariantLearning(X, Y, model, config=None, **params):
    # Handle configuration
    if config is not None:
        if isinstance(config, ILTrainingConfig):
            cfg = config
        else:
            raise TypeError("config must be a ILTrainingConfig object")
    else:
        param_dict = extract_training_params(**params)
        cfg = ILTrainingConfig(**param_dict)

    # Convert and validate data
    X, Y = convert_to_tensors(X, Y)
    P, N, M = validate_data_shapes(X, Y)

    # Setup
    N_edges = model.circuit.number_of_edges()
    batches_per_epoch = P // cfg.batch_size
    N_batches = cfg.N_batches if cfg.N_batches is not None else int(batches_per_epoch * cfg.N_epochs)

    # Resolve node indices
    x_nodes, y_nodes = resolve_node_indices(cfg.x_nodes, cfg.y_nodes, model)
    x_inds = utils.nodes_to_inds(x_nodes, model.circuit)
    y_inds = utils.nodes_to_inds(y_nodes, model.circuit)

    # Initialize tracking arrays
    param_history = initialize_param_history(model)
    loss = np.zeros(N_batches)
    accuracy = np.zeros(N_batches)

    def compute_d_theta(V_F, V_C, layer):
        D_V = (V_F - V_C)[:,None]       # Note: V_edge has a negative sign in it, i.e. model.V_edge() is -Del @ V_node
        d_gamma_d_x = layer.d_gamma_d_x(V_F)[:,None]
        d_gamma_d_theta = layer.d_gamma_d_theta(V_F)
        return layer.learning_rates * cfg.alpha / cfg.eta * tc.mean(d_gamma_d_x / d_gamma_d_theta * D_V, axis=0)
        # return layer.learning_rates * cfg.alpha / cfg.eta * layer.theta * tc.mean(D_V / V_F[:,None], axis=0)
        # return layer.learning_rates * cfg.alpha / cfg.eta * layer.theta * tc.mean(D_V * V_F[:,None], axis=0)
        
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
        model.set_inputs(x_nodes)
        V_node_F, _ = model(x)
        y_F = V_node_F[:, y_inds]
        V_edge_F = model.V_edge() 

        # Compute clamped state
        y_C = cfg.eta * y + (1 - cfg.eta) * y_F.detach()

        model.set_inputs(x_nodes, y_nodes)
        _ = model(x, y_C)
        V_edge_C = model.V_edge()
        
        # update parameters and history
        for element_name, layer in model.element_layers.items():
            edge_inds = model.element_to_inds[element_name]
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

    return finalize_history(param_history, loss, accuracy)

def NCCoupledLearning(X, Y, model, config=None, **params):
    """
    Noise contrastive coupled learning
    """
    # Handle configuration
    if config is not None:
        if isinstance(config, NCCLTrainingConfig):
            cfg = config
        else:
            raise TypeError("config must be a NCCLTrainingConfig object")
    else:
        param_dict = extract_training_params(**params)
        cfg = NCCLTrainingConfig(**param_dict)

    # Convert and validate data
    X, Y = convert_to_tensors(X, Y)
    P, N, M = validate_data_shapes(X, Y)

    # Setup
    N_edges = model.circuit.number_of_edges()
    batches_per_epoch = P // cfg.batch_size
    N_batches = cfg.N_batches if cfg.N_batches is not None else int(batches_per_epoch * cfg.N_epochs)

    # Resolve node indices
    x_nodes, y_nodes = resolve_node_indices(cfg.x_nodes, cfg.y_nodes, model)
    x_inds = utils.nodes_to_inds(x_nodes, model.circuit)
    y_inds = utils.nodes_to_inds(y_nodes, model.circuit)

    # Initialize tracking arrays
    param_history = initialize_param_history(model)
    loss = np.zeros(N_batches)
    accuracy = np.zeros(N_batches)

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
        model.set_inputs(x_nodes)
        V_node_f, _ = model(x)      # get free node voltages (to build clamped state)
        y_f = V_node_f[:, y_inds].detach()
        V_edge_f = model.V_edge()

        # Compute clamped state
        y_c = cfg.eta * y + (1 - cfg.eta) * y_f    #  build clamped state
        model.set_inputs(x_nodes, y_nodes)
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
            edge_inds = model.element_to_inds[element_name]
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
            d_theta = - layer.learning_rates * cfg.alpha / cfg.eta * (
                (1 + cfg.nu / (K - 1)) * grad_rho_c \
                - (1 - cfg.nu) * grad_rho_f \
                - K / (K - 1) * cfg.nu * grad_rho_n
            )
            # K = N_batches
            # d_theta = - layer.learning_rates * cfg.alpha / cfg.eta * (
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

    return finalize_history(param_history, loss, accuracy)