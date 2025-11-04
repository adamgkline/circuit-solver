import numpy as np
import torch as tc
import networkx as nx
from circuit_solver import utils
from dataclasses import dataclass
from torch import nn
from torch.nn.parameter import Parameter


@dataclass
class CircuitModelConfig:
    """
    Configuration parameters for CircuitModel.

    Attributes:
        N_batch (int): Default batch size
        N_optim_steps (int): Number of optimization steps in forward pass
        optim_lr (float): Learning rate for internal optimizer
        optimizer (str): Optimizer type ('adam' or 'lbfgs'). Note: lbfgs suffers at larger batch sizes as of 25.10.08
        HIGH_voltage (float): Voltage for HIGH reference nodes
        LOW_voltage (float): Voltage for LOW reference nodes
        mode (str): Physical objective function ('cocontent' or 'current')
        dtype (torch.dtype): Default tensor data type
        use_sparse: (bool): Whether to use sparse tensor representation for incidence matrices
    """
    N_batch: int = 1
    N_optim_steps: int = None  # Will be set based on optimizer in __post_init__
    optim_lr: float = None     # Will be set based on optimizer in __post_init__
    optimizer: str = 'lbfgs'    
    HIGH_voltage: float = 1.
    LOW_voltage: float = -1.
    mode: str = 'cocontent'
    dtype: tc.dtype = tc.float32
    use_gpu: bool = False           # having trouble with SparseMPS backend right now
    use_sparse: bool = False         # use sparse tensor representation for incidence matrices

    def __post_init__(self):
        """Set optimizer-specific defaults if not explicitly provided."""
        # Define optimizer-specific defaults
        optimizer_defaults = {
            'lbfgs': {'N_optim_steps': 3, 'optim_lr': 1.0},
            'adam': {'N_optim_steps': 200, 'optim_lr': 0.1}
        }

        # Validate optimizer type
        if self.optimizer.lower() not in optimizer_defaults:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}. Supported: {list(optimizer_defaults.keys())}")

        # Set defaults for parameters that weren't explicitly provided
        defaults = optimizer_defaults[self.optimizer.lower()]

        if self.N_optim_steps is None:
            self.N_optim_steps = defaults['N_optim_steps']

        if self.optim_lr is None:
            self.optim_lr = defaults['optim_lr']
    

class Circuit(nx.Graph):
    """
    A circuit representation extending NetworkX Graph with circuit-specific functionality.
    
    This class represents an electrical circuit as a graph where:
    - Nodes represent circuit connection points
    - Edges represent electrical components (resistors, diodes, etc.)
    - Circuit elements are mapped to specific edges via element_dict
    
    The class automatically computes incidence matrices (Del and DelT) which are essential
    for circuit analysis using Kirchhoff's laws.
    
    Args:
        graph (nx.Graph): The underlying graph structure of the circuit
        element_dict (dict): Maps circuit elements to lists of edges where they occur
                           Format: {element: [edge_list]}
    
    Attributes:
        element_dict (dict): Dictionary mapping elements to their edge locations
        elements (list): List of all circuit elements
        DelT (scipy.sparse matrix): Transpose of incidence matrix (nodes x edges)
        Del (scipy.sparse matrix): Incidence matrix (edges x nodes)
    
    Example:
        >>> import networkx as nx
        >>> from elements import Resistor
        >>> graph = nx.Graph([(0,1), (1,2)])
        >>> element_dict = {Resistor: [(0,1), (1,2)]}
        >>> circuit = Circuit(graph, element_dict)
    """

    def __init__(self, graph, element_dict):
        """
        Initialize a Circuit from a graph and element dictionary.

        Args:
            graph (nx.Graph): NetworkX graph defining circuit topology
            element_dict (dict): Maps circuit elements to edge lists
        """
        # Initialize as a NetworkX graph by copying the input graph
        super(Circuit, self).__init__(graph)

        # Sort edge lists in element_dict to match the adjacency matrix ordering
        # This ensures consistency between element_dict and the incidence matrix
        sorted_element_dict = {}
        for element, edges in element_dict.items():
            # Sort edges to match the order that will be used by nx.incidence_matrix
            # NetworkX orders edges lexicographically (by node indices)
            sorted_edges = sorted(edges, key=lambda edge: (min(edge), max(edge)))
            sorted_element_dict[element] = sorted_edges

        self.element_dict = sorted_element_dict
        self.elements = list(sorted_element_dict.keys())

        # Check for duplicate element names and make them unique
        element_names = [element.name for element in self.elements]
        if len(element_names) != len(set(element_names)):
            import warnings

            # Create unique names
            name_counts = {}
            renamed_elements = []

            for element in self.elements:
                base_name = element.name
                if base_name not in name_counts:
                    name_counts[base_name] = 0
                else:
                    name_counts[base_name] += 1
                    # Modify the element's name to make it unique
                    element.name = f"{base_name}_{name_counts[base_name]}"
                    renamed_elements.append((base_name, element.name))

            # Print warning with renamed elements
            warning_msg = "\nDuplicate element names found. Renamed elements:\n"
            for old_name, new_name in renamed_elements:
                warning_msg += f"  {old_name} -> {new_name}\n"
            warnings.warn(warning_msg, UserWarning)

        self.DelT = nx.incidence_matrix(self, oriented=True)
        self.Del = self.DelT.T

        # add each element to all of the edges it is on
        for element in self.elements:
            edges = self.element_dict[element]
            # Create a dictionary mapping each edge to its element
            edge_attrs = {edge: {'element': element} for edge in edges}
            nx.set_edge_attributes(self, edge_attrs)


class CircuitModel(nn.Module):
    """
    PyTorch neural network model for simulating nonlinear resistor circuits.
    
    This model solves circuit equilibria by optimizing a physical objective function
    (either cocontent minimization or current minimization) subject to voltage constraints
    on input nodes. The model supports batched processing and automatic differentiation.
    
    Key features:
    - Hybrid sparse/dense matrix operations for performance
    - Shared parameter architecture across rho, gamma, and d_gamma modules
    - Parameter clipping to enforce physical constraints
    - Support for multiple optimization objectives
    
    Args:
        circuit (Circuit): Circuit topology and element information
        node_type_dict (dict): Maps node types to node indices
                              Special types: 'GROUND', 'HIGH', 'LOW' for reference voltages
        config (CircuitModelConfig, optional): Configuration parameters
    
    Attributes:
        circuit (Circuit): The underlying circuit
        node_type_dict (dict): Node type mappings
        element_layers (dict): Shared ElementLayer instances by element name
        rho (nn.Sequential): Cocontent function module
        gamma (nn.Sequential): Conductance function module  
        d_gamma (nn.Sequential): Conductance derivative module
        clamped_inds (list): Indices of voltage-clamped nodes
        free_inds (list): Indices of nodes with free voltages
        V_clamped (torch.Tensor): Voltages of clamped nodes (batch_size, n_clamped)
        V_free (torch.Tensor): Voltages of free nodes (batch_size, n_free)
    
    Example:
        >>> config = CircuitModelConfig(N_optim_steps=50, optim_lr=0.01)
        >>> node_types = {'GROUND': [0], 'x_inds': [1,2,3]}
        >>> model = CircuitModel(circuit, node_types, config)
        >>> x_input = torch.randn(10, 3)  # batch_size=10, 3 input nodes
        >>> voltages, obj_history = model(x_input)
    """

    def __init__(self, circuit, node_type_dict, config=None):
        """
        Initialize the CircuitModel.
        
        Args:
            circuit (Circuit): Circuit topology and element information
            node_type_dict (dict): Maps node type names to lists of node indices
            config (CircuitModelConfig, optional): Model configuration parameters
        """
        
        super(CircuitModel, self).__init__()
        self.circuit = circuit
        self.node_type_dict = node_type_dict
        self.node_types = list(node_type_dict.keys())

        ## Handle configuration settings
        # check if config is empty
        if config is not None:
            if isinstance(config, CircuitModelConfig):
                cfg = config
            else:
                raise TypeError("config must be a CircuitModelConfig object")
        else:
            cfg = CircuitModelConfig()

        # Keep track of the current batch size
        self.N_batch = cfg.N_batch

        # set high and low voltages
        self.HIGH_voltage = cfg.HIGH_voltage
        self.LOW_voltage = cfg.LOW_voltage

        # This needs to happen after allocations are made
        # set optimization mode
        self.physical_objective_dict = {
            'cocontent': self.cocontent_objective,
            'current': self.node_current_objective
        }
        self.mode = cfg.mode
        self.set_physical_objective(self.mode)
        self.N_optim_steps = cfg.N_optim_steps
        self.optim_lr = cfg.optim_lr
        self.optimizer_type = cfg.optimizer

        # set datatype
        self.dtype = cfg.dtype

        # set sparse tensor usage
        self.use_sparse = cfg.use_sparse

        # Set matrix multiplication function based on sparse usage
        self.mm_func = tc.sparse.mm if self.use_sparse else tc.mm

        # Add circuit nonlinearities
        # Create shared ElementLayer instances first
        self._create_element_layers()
        self.clip_all_parameters()

        # Then create modules that use these shared layers
        self._create_rho_module()
        self._create_gamma_module()
        self._create_d_gamma_module()

        # set device
        self.device = tc.accelerator.current_accelerator().type if tc.accelerator.is_available() and cfg.use_gpu else "cpu"   # "mps" for mac, typically "cuda" elsewhere

        ## Initialize everything needed for input structure
        # initialize clamped and free node lists
        self.clamped_inds = []      
        self.free_inds = []         
        
        # initialize clamped and free node voltage tensors
        self.V_clamped = tc.tensor([], requires_grad=False, device=self.device)
        self.V_free = tc.tensor([], requires_grad=True, device=self.device)

        # initialize clamped and free incidence matrices
        # Keep dense version for indexing operations
        self.Del_dense = tc.tensor(self.circuit.Del.toarray(), dtype=self.dtype, device=self.device)

        # Create sparse versions using helper method if enabled
        self.DelT = None
        if self.use_sparse:
            self._update_sparse_matrices()
        else:
            self.Del = self.Del_dense
            self.DelT = self.Del_dense.t()
        
        # These will be initialized in set_inputs()
        self.Del_clamped = None
        self.Del_free = None

        # set inputs appropriately
        self.set_inputs()           # at the start, clamped_inds only has reference_inds
        self.clamp()

        # Move model to device after all initialization
        self.to(self.device)
        
    def _update_sparse_matrices(self):
        """
        Update sparse Del and DelT from the current Del_dense matrix.
        Call this whenever Del_dense is modified.
        """
        # Convert dense to sparse COO format
        # Find non-zero elements
        nonzero_mask = self.Del_dense != 0
        rows, cols = tc.nonzero(nonzero_mask, as_tuple=True)
        values = self.Del_dense[nonzero_mask]
        
        # Create sparse tensor
        self.Del = tc.sparse_coo_tensor(
            indices=tc.stack([rows, cols]),
            values=values,
            size=self.Del_dense.shape,
            dtype=self.dtype,
            device=self.device
        )
        self.DelT = self.Del.t()

    def _create_element_layers(self):
        """
        Create shared ElementLayer instances that will be reused across rho, gamma, d_gamma, and d_rho_d_gamma modules
        """
        self.element_layers = {}
        self.element_to_edge_inds = {}
        
        for element in self.circuit.elements:
            edges = self.circuit.element_dict[element]
            edge_inds = utils.edges_to_inds(edges, self.circuit)
            
            # Create one ElementLayer instance per element type
            element_layer = ElementLayer(element, edges=edges)
            
            # Store the shared layer and its edge indices
            self.element_layers[element.name] = element_layer
            self.element_to_edge_inds[element.name] = edge_inds
            
            # Register as a named module for parameter tracking
            self.add_module(f"{element.name}_layer", element_layer)

    def _dense_to_sparse(self, dense_tensor):
        """
        Convert a dense tensor to sparse COO format.
        
        Args:
            dense_tensor (torch.Tensor): Dense tensor to convert
        
        Returns:
            torch.sparse.FloatTensor: Sparse tensor in COO format
        """
        nonzero_mask = dense_tensor != 0
        rows, cols = tc.nonzero(nonzero_mask, as_tuple=True)
        values = dense_tensor[nonzero_mask]
        
        return tc.sparse_coo_tensor(
            indices=tc.stack([rows, cols]),
            values=values,
            size=dense_tensor.shape,
            dtype=self.dtype,
            device=self.device
        )
 
    def _create_rho_module(self):
        """
        Create the rho (cocontent) module using shared ElementLayer instances.
        
        This module computes cocontent functions for all circuit elements,
        applying the appropriate function to each edge based on its element type.
        """
        # Use shared ElementLayer instances
        N_edges = self.circuit.number_of_edges()
        
        masked_nonlinearities = []
        for element_name, element_layer in self.element_layers.items():
            edge_indices = self.element_to_edge_inds[element_name]
            
            # Wrap shared layer's rho method in MaskedNonlinearity
            masked_nl = MaskedNonlinearity(
                nonlinearity=element_layer.rho,
                indices=edge_indices, 
                total_dim=N_edges
            )
            masked_nonlinearities.append(masked_nl)
        
        # Combine all masked nonlinearities in sequence
        self.rho = nn.Sequential(*masked_nonlinearities)

    def _create_gamma_module(self):
        """
        Create the gamma (conductance) module using shared ElementLayer instances.
        
        This module computes current-voltage relations for all circuit elements,
        applying the appropriate function to each edge based on its element type.
        """
        # Use shared ElementLayer instances
        N_edges = self.circuit.number_of_edges()
        
        masked_gamma_nonlinearities = []
        for element_name, element_layer in self.element_layers.items():
            edge_indices = self.element_to_edge_inds[element_name]
            
            # Wrap shared layer's gamma method in MaskedNonlinearity
            masked_gamma = MaskedNonlinearity(
                nonlinearity=element_layer.gamma,
                indices=edge_indices,
                total_dim=N_edges
            )
            masked_gamma_nonlinearities.append(masked_gamma)
            
        self.gamma = nn.Sequential(*masked_gamma_nonlinearities)

    def _create_d_gamma_module(self):
        """
        Create the d_gamma (differential conductance) module using shared ElementLayer instances.
        
        This module computes derivatives of current-voltage relations for all circuit elements,
        applying the appropriate function to each edge based on its element type.
        """
        # Use shared ElementLayer instances
        N_edges = self.circuit.number_of_edges()
        
        masked_d_gamma_nonlinearities = []
        for element_name, element_layer in self.element_layers.items():
            edge_indices = self.element_to_edge_inds[element_name]
            
            # Wrap shared layer's d_gamma method in MaskedNonlinearity
            masked_d_gamma = MaskedNonlinearity(
                nonlinearity=element_layer.d_gamma,
                indices=edge_indices,
                total_dim=N_edges
            )
            masked_d_gamma_nonlinearities.append(masked_d_gamma)
            
        self.d_gamma = nn.Sequential(*masked_d_gamma_nonlinearities)

    def set_physical_objective(self, mode):
        """
        Set the physical objective function for circuit optimization.
        
        Args:
            mode (str): Either 'cocontent' for cocontent minimization or 
                       'current' for current minimization
        """
        # check to see if mode is in self.physical_objective_dict. If not, throw warning and pick default
        self.physical_objective = self.physical_objective_dict[mode]
    
    def clip_all_parameters(self):
        """
        Clip all element layer parameters to their allowed ranges
        """
        # Find all ElementLayer modules in the masked nonlinearities
        for module in self.modules():
            if isinstance(module, ElementLayer):
                module.clip_parameters()

    def set_inputs(self, *node_inds):
        """
        Set which nodes will have clamped (fixed) voltages during simulation.

        This method partitions circuit nodes into clamped and free nodes:
        - Clamped nodes: Reference nodes (GROUND, HIGH, LOW) + input nodes
        - Free nodes: All other nodes whose voltages will be optimized

        Args:
            *node_inds: Variable number of iterables containing node indices to clamp,
                       OR string keys from node_type_dict

        Example:
            >>> x_inds = [1, 2, 3, 4]
            >>> y_inds = [5, 6, 7]
            >>> model.set_inputs(x_inds)  # Only clamp x_inds
            >>> model.set_inputs(x_inds, y_inds)  # Clamp both x_inds and y_inds
            >>> model.set_inputs('x_inds', 'y_inds')  # Use node_type_dict keys
        """

        # Convert string arguments to node lists using node_type_dict
        resolved_node_groups = []
        for i, node_group in enumerate(node_inds):
            if isinstance(node_group, str):
                if node_group not in self.node_type_dict:
                    raise ValueError(f"String argument '{node_group}' not found in node_type_dict. "
                                   f"Available keys: {list(self.node_type_dict.keys())}")
                resolved_node_groups.append(self.node_type_dict[node_group])
            else:
                resolved_node_groups.append(node_group)

        # verify inputs are in the correct format
        all_circuit_nodes = set(self.circuit.nodes())

        for i, node_group in enumerate(resolved_node_groups):
            # Check if node_group is iterable (list, tuple, array, etc.)
            try:
                node_list = list(node_group)
            except TypeError:
                raise TypeError(f"Argument {i} is not iterable. Expected list, tuple, or array of node indices.")

            # Check if all nodes in the group are valid circuit nodes
            for node in node_list:
                if node not in all_circuit_nodes:
                    raise ValueError(f"Node {node} in argument {i} is not a valid circuit node. "
                                   f"Valid nodes are: {sorted(all_circuit_nodes)}")

                # Check if node is an integer (assuming node indices are integers)
                if not isinstance(node, (int, np.integer)):
                    raise TypeError(f"Node {node} in argument {i} must be an integer, got {type(node)}")

        # Check for duplicate nodes across all input groups
        all_input_nodes = []
        for node_group in resolved_node_groups:
            all_input_nodes.extend(list(node_group))

        if len(all_input_nodes) != len(set(all_input_nodes)):
            duplicates = [node for node in set(all_input_nodes) if all_input_nodes.count(node) > 1]
            raise ValueError(f"Duplicate nodes found across input groups: {duplicates}")

        ## Always add reference node indices to clamped_inds 
        # 'GROUND' always set to 0 
        # 'HIGH' always set to self.HIGH_voltage
        # 'LOW' always set to self.LOW_voltage
        self.clamped_inds = []
        for n in ['GROUND', 'HIGH', 'LOW']:
            if n in self.node_types:
                self.clamped_inds.extend(self.node_type_dict[n])
        
        # Add any other inds in resolved_node_groups to clamp_inds
        for node_group in resolved_node_groups:
            self.clamped_inds.extend(list(node_group))

        # Place the remaining the indices in free_inds
        all_nodes = list(self.circuit.nodes())
        self.free_inds = [node for node in all_nodes if node not in self.clamped_inds]

        # update incidence matrices Del_clamped and Del_free
        # Create dense versions first using indexing
        Del_clamped_dense = self.Del_dense[:,self.clamped_inds]
        Del_free_dense = self.Del_dense[:,self.free_inds]

        # Convert to sparse matrices if sparse mode is enabled
        if self.use_sparse:
            self.Del_clamped = self._dense_to_sparse(Del_clamped_dense)
            self.Del_free = self._dense_to_sparse(Del_free_dense)
            # Update main sparse matrices if Del_dense was modified
            self._update_sparse_matrices()
        else:
            self.Del_clamped = Del_clamped_dense
            self.Del_free = Del_free_dense
            self.Del = self.Del_dense
            self.DelT = self.Del_dense.t()

    def clamp(self, *tensors):
        """
        Set voltages for clamped nodes from input tensors.
        
        Combines reference node voltages (GROUND=0, HIGH=HIGH_voltage, LOW=LOW_voltage)
        with user-provided input tensors to create the full clamped voltage tensor.
        
        Args:
            *tensors: Input voltage tensors, each with shape (batch_size, n_nodes_i)
                     Order should match the order used in set_inputs()
        
        Updates:
            self.V_clamped: Combined tensor of all clamped node voltages
            self.V_free: Initialized tensor of free node voltages (zeros)
            self.N_batch: Current batch size
        """
        # Update batch size from input tensors
        if tensors:
            batch_sizes = [t.shape[0] for t in tensors]
            if not all(b == batch_sizes[0] for b in batch_sizes):
                raise ValueError(f"All tensors must have same batch size, got {batch_sizes}")
            self.N_batch = batch_sizes[0]
        
        # Create reference voltage tensors for special nodes
        voltage_map = {'GROUND': 0.0, 'HIGH': self.HIGH_voltage, 'LOW': self.LOW_voltage}
        clamped_tensors = []
        
        # Add reference node voltages
        for ref_type, voltage in voltage_map.items():
            if ref_type in self.node_type_dict:
                n_nodes = len(self.node_type_dict[ref_type])
                if n_nodes > 0:
                    ref_tensor = tc.full((self.N_batch, n_nodes), voltage, dtype=self.dtype, device=self.device)
                    clamped_tensors.append(ref_tensor)
        
        # Add input tensors
        clamped_tensors.extend([t.to(dtype=self.dtype, device=self.device) for t in tensors])

        # Combine all clamped voltages
        self.V_clamped = tc.cat(clamped_tensors, dim=1) if clamped_tensors else tc.empty((self.N_batch, 0), dtype=self.dtype, device=self.device)
        # self.V_clamped.to(self.device)
        self.V_clamped.requires_grad = False

        # Initialize free node voltages
        n_free = len(self.free_inds)
        self.V_free = tc.zeros((self.N_batch, n_free), dtype=self.dtype, device=self.device, requires_grad=True)

    def forward(self, *tensors):
        """
        Forward pass: solve circuit equilibrium for given input voltages.
        
        This method:
        1. Clamps input voltages on designated nodes
        2. Optimizes free node voltages to minimize the physical objective
        3. Returns final node voltages and optimization history
        
        Args:
            *tensors: Input voltage tensors, each with shape (batch_size, n_nodes_i)
        
        Returns:
            tuple: (final_node_voltages, objective_history)
                - final_node_voltages (torch.Tensor): All node voltages (batch_size, n_nodes)
                - objective_history (list): Objective function values during optimization
        
        Example:
            >>> x_input = torch.randn(32, 10)  # batch_size=32, 10 input nodes
            >>> voltages, history = model(x_input)
            >>> print(voltages.shape)  # torch.Size([32, total_nodes])
        """
        
        # set input values for batch
        self.clamp(*tensors)

        # Check if all nodes are constrained (no free nodes)
        if len(self.free_inds) == 0:
            # All nodes are clamped - no optimization needed
            # Return current node voltages and empty history
            return self.V_node(), np.array([])

        # get objective function
        objective = self.physical_objective

        # get optimizer based on configuration
        if self.optimizer_type.lower() == 'lbfgs':
            # self.optim_lr = 1.          # best configuration based on tests on 25.0
            optimizer = tc.optim.LBFGS([self.V_free], lr=self.optim_lr)
        elif self.optimizer_type.lower() == 'adam':
            optimizer = tc.optim.Adam([self.V_free], lr=self.optim_lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}. Supported types: 'adam', 'lbfgs'")

        # carry out minimization
        obj_history = []

        if self.optimizer_type.lower() == 'lbfgs':
            # L-BFGS requires a closure function
            def closure():
                optimizer.zero_grad()
                loss = objective()
                loss.backward()
                return loss

            for n in range(self.N_optim_steps):
                loss = optimizer.step(closure)
                obj_history.append(loss)
        else:
            for n in range(self.N_optim_steps):
                optimizer.zero_grad()
                cur_obj = objective()
                cur_obj.backward()
                obj_history.append(cur_obj)
                optimizer.step()
        
        obj_history = np.array([a.cpu().detach().numpy() for a in obj_history])
        return self.V_node(), obj_history
    
    def V_node(self):
        """
        Combine clamped and free node voltages into a single tensor.
        
        Returns:
            torch.Tensor: All node voltages with shape (batch_size, n_nodes)
                         Ordered according to self.circuit.nodes()
        
        Note:
            This method performs tensor indexing and may be slow.
            Avoid using in performance-critical sections.
        """
        out = tc.zeros(self.N_batch, self.circuit.number_of_nodes(), dtype=self.dtype, device=self.device)
        out[:, self.clamped_inds] = self.V_clamped
        out[:, self.free_inds] = self.V_free
        return out

    def V_edge(self):
        """
        Compute edge voltages from node voltages using Kirchhoff's voltage law.

        Edge voltages are computed as V_edge = Del^T @ V_node, where Del is the
        incidence matrix. Uses sparse or dense matrix operations based on configuration.

        Returns:
            torch.Tensor: Edge voltages with shape (batch_size, n_edges)
        """
        # Convert einsum 'ab, cb -> ca' to matrix multiplication: Del @ V
        V_clamped_edge = - self.mm_func(self.V_clamped, self.Del_clamped.t())
        V_free_edge = - self.mm_func(self.V_free, self.Del_free.t())
        return V_clamped_edge + V_free_edge

    def I_edge(self):
        """
        Compute edge currents from edge voltages using element constitutive relations.
        
        Returns:
            torch.Tensor: Edge currents with shape (batch_size, n_edges)
        """
        return self.gamma(self.V_edge())
        
    def I_node(self):
        """
        Compute node currents from edge currents using Kirchhoff's current law.

        Node currents are computed as I_node = Del @ I_edge, where Del is the
        incidence matrix. At equilibrium, all node currents should be zero.

        Returns:
            torch.Tensor: Node currents with shape (batch_size, n_nodes)
        """
        # Convert einsum 'ab, cb -> ca' to matrix multiplication: DelT @ I
        return - self.mm_func(self.I_edge(), self.Del)

    def cocontent_objective(self):
        """
        Compute cocontent minimization objective function.
        
        The cocontent is the integral of the current-voltage characteristic.
        Minimizing cocontent finds the circuit's natural equilibrium state.
        
        Returns:
            torch.Tensor: Mean total cocontent across batch (scalar)
        """
        # return tc.mean(self.rho(self.V_edge()))
        # return tc.sum(self.rho(self.V_edge()))
        # return tc.mean(tc.sum(self.rho(self.V_edge()), axis=1))
        
        # Above options seem to give issues for LBFGS at large batch size
        return tc.sum(tc.mean(self.rho(self.V_edge()), axis=0))

    def node_current_objective(self):
        """
        Compute current minimization objective function.
        
        Minimizes the sum of squared node currents, which enforces
        Kirchhoff's current law (all node currents should be zero at equilibrium).
        
        Returns:
            torch.Tensor: Mean squared node current across batch and nodes (scalar)
        """
        # return sum of squares of currents on free nodes
        # return tc.sum(self.I_node()**2, axis=1)
        return tc.mean(self.I_node()**2)


class MaskedNonlinearity(nn.Module):
    """
    Applies a nonlinearity function only to specified indices of the input tensor.
    
    This module enables selective application of different nonlinearities to
    different edges in the circuit, allowing each element type to have its own
    constitutive relation while processing all edges in a single tensor.
    
    Args:
        nonlinearity (callable): Function to apply to selected indices
        indices (list or slice): Indices where nonlinearity should be applied
        total_dim (int): Total dimension of the input tensor
    
    Example:
        >>> nonlin = lambda x: torch.tanh(x)
        >>> masked = MaskedNonlinearity(nonlin, indices=[0, 2, 4], total_dim=10)
        >>> x = torch.randn(batch_size, 10)
        >>> y = masked(x)  # Only indices [0, 2, 4] are passed through tanh
    """
    def __init__(self, nonlinearity, indices, total_dim):
        """
        Initialize the MaskedNonlinearity.
        
        Args:
            nonlinearity (callable): Function to apply to selected indices
            indices (list, array, or slice): Indices to apply nonlinearity to
            total_dim (int): Total dimension of input tensors
        """
        super().__init__()
        self.nonlinearity = nonlinearity
        
        # Create both mask and inverse mask for efficiency
        mask = tc.zeros(total_dim, dtype=tc.bool)
        if isinstance(indices, slice):
            mask[indices] = True
        else:
            mask[tc.tensor(indices)] = True
            
        self.register_buffer('mask', mask)
        
    def forward(self, x):
        """
        Apply nonlinearity to specified indices only.
        
        Args:
            x (torch.Tensor): Input tensor with shape (..., total_dim)
        
        Returns:
            torch.Tensor: Modified tensor with nonlinearity applied to masked indices
        """
        x[..., self.mask] = self.nonlinearity(x[..., self.mask])
        return x


class ElementLayer(nn.Module):
    """
    Neural network layer representing a group of circuit elements of the same type.
    
    This layer manages learnable parameters for all edges containing the same
    element type (e.g., all diodes, all resistors). It provides vectorized
    computation of cocontent functions (rho), conductance functions (gamma),
    and conductance derivatives (d_gamma).
    
    The layer automatically enforces parameter constraints defined in the
    element's param_ranges attribute.
    
    Args:
        element (ResistiveElement): Circuit element defining the constitutive relations
        edges (list): List of edges where this element type is located
    
    Attributes:
        element (ResistiveElement): The circuit element this layer represents
        edges (list): Edges containing this element type
        theta (torch.nn.Parameter): Learnable parameters with shape (N_params, N_edges)
        N_params (int): Number of parameters per element instance
        N_edges (int): Number of edges with this element type
    
    Example:
        >>> from elements import Diode
        >>> edges = [(0,1), (1,2), (2,3)]
        >>> layer = ElementLayer(Diode, edges)
        >>> x = torch.randn(batch_size, len(edges))
        >>> cocontent = layer.rho(x)
        >>> conductance = layer.gamma(x)
        >>> d_conductance = layer.d_gamma(x)
    """
    def __init__(self, element, edges, trainable=True):
        """
        Initialize an ElementLayer for a specific circuit element type.

        Args:
            element (ResistiveElement): Circuit element defining constitutive relations
            edges (list): List of edges where this element type is present
            trainable (bool): Whether parameters should be trainable
        """

        super(ElementLayer, self).__init__()
        self.element = element
        self.edges = edges
        self.N_params = element.N_params
        self.N_edges = len(edges)
        self.trainable = trainable

        # Add parameters (e.g. conductance)
        param_shape = (self.N_params, self.N_edges)
        self.shape = param_shape

        # Initialize parameters based on element's init_mode
        initial_values = self._initialize_parameters(param_shape)
        self.theta = Parameter(initial_values, requires_grad=trainable)

    def _initialize_parameters(self, shape):
        """
        Initialize parameter values based on element's init_mode.

        Args:
            shape (tuple): Shape of the parameter tensor (N_params, N_edges)

        Returns:
            torch.Tensor: Initialized parameter values with the given shape
        """
        N_params, N_edges = shape
        element = self.element

        if element.init_mode == 'constant':
            # Initialize with constant values
            if element.init_params is None:
                raise ValueError(f"init_params must be provided for element '{element.name}' with 'constant' initialization mode")

            # Allow single value or list of values
            if isinstance(element.init_params, (int, float)):
                values = tc.full(shape, element.init_params)
            else:
                # Assume list of values, one per parameter
                if len(element.init_params) != N_params:
                    raise ValueError(f"init_params length ({len(element.init_params)}) must match N_params ({N_params}) for element '{element.name}'")
                values = tc.tensor(element.init_params, dtype=tc.float32)[:, None].expand(shape).clone()

        elif element.init_mode == 'uniform':
            # Initialize uniformly within param_ranges
            param_ranges = tc.tensor(element.param_ranges, dtype=tc.float32)
            param_lower = param_ranges[:, 0, None]
            param_upper = param_ranges[:, 1, None]
            values = tc.rand(shape) * (param_upper - param_lower) + param_lower

        elif element.init_mode == 'normal':
            # Initialize with normal distribution
            if element.init_params is None:
                raise ValueError(f"init_params must be provided for element '{element.name}' with 'normal' initialization mode")

            # Allow single (mean, std) tuple or list of tuples
            if isinstance(element.init_params[0], (int, float)):
                # Single (mean, std) for all parameters
                mean, std = element.init_params
                values = tc.randn(shape) * std + mean
            else:
                # List of (mean, std) tuples, one per parameter
                if len(element.init_params) != N_params:
                    raise ValueError(f"init_params length ({len(element.init_params)}) must match N_params ({N_params}) for element '{element.name}'")
                values = tc.zeros(shape)
                for i, (mean, std) in enumerate(element.init_params):
                    values[i] = tc.randn(N_edges) * std + mean

        elif element.init_mode == 'geometric_mean':
            # Initialize with geometric mean of param_ranges (default behavior)
            param_ranges = tc.tensor(element.param_ranges, dtype=tc.float32)
            geom_mean = tc.sqrt(param_ranges[:, 0] * param_ranges[:, 1])
            values = tc.ones(shape) * geom_mean[:, None]

        else:
            raise ValueError(f"Unknown init_mode '{element.init_mode}' for element '{element.name}'. "
                           f"Supported modes: 'constant', 'uniform', 'normal', 'geometric_mean'")

        # Clip to param_ranges if specified
        with tc.no_grad():
            for i, (min_val, max_val) in enumerate(element.param_ranges):
                if min_val is not None:
                    values[i].clamp_(min=min_val)
                if max_val is not None:
                    values[i].clamp_(max=max_val)

        return values
    
    def set_parameters(self, theta):
        self.theta.data = tc.tensor(theta)
        self.clip_parameters()

    def clip_parameters(self):
        """
        Clip parameters to their allowed ranges based on element.param_ranges.
        
        This method enforces physical constraints on circuit element parameters
        (e.g., conductance must be positive, voltage ranges must be realistic).
        Called automatically during model initialization and can be called
        manually during training.
        """
        """
        Clip parameters to their allowed ranges based on element.param_ranges
        """
        with tc.no_grad():
            for i, (min_val, max_val) in enumerate(self.element.param_ranges):
                if min_val is not None:
                    self.theta[i].clamp_(min=min_val)
                if max_val is not None:
                    self.theta[i].clamp_(max=max_val)

    def forward(self, x):
        """
        Compute cocontent function for this element type.
        
        Args:
            x (torch.Tensor): Edge voltages with shape (batch_size, N_edges)
        
        Returns:
            torch.Tensor: Cocontent values with shape (batch_size, N_edges)
        
        Note:
            This method is equivalent to self.rho(x)
        """
        return self.element.rho(x, self.theta)

    def vectorize(self, func):
        """
        Create a vectorized version of an element function for batch processing.
        
        This method wraps element functions (rho, gamma, d_gamma) to handle
        the expected tensor shapes: (batch_size, N_edges) for inputs and
        (N_params, N_edges) for parameters.
        
        Args:
            func (callable): Element function to vectorize
        
        Returns:
            callable: Vectorized function that processes batched inputs
        
        Note:
            Uses nested torch.vmap to vectorize over both batch and edge dimensions
        """
        # First vectorize over batch index, then vectorize over edge index
        return tc.vmap(tc.vmap(func, in_dims=(0, None), out_dims=0), in_dims=(1, 1), out_dims=-1)
        # return tc.vmap(tc.vmap(func, in_dims=(1,1)), in_dims=(0,None))

    def rho(self, x):
        """
        Compute cocontent function (integral of current-voltage relation).
        
        Args:
            x (torch.Tensor): Edge voltages with shape (batch_size, N_edges)
        
        Returns:
            torch.Tensor: Cocontent values with shape (batch_size, N_edges)
        """
        return self.vectorize(self.element.rho)(x, self.theta)

    def gamma(self, x):
        """
        Compute conductance function (current-voltage relation).
        
        Args:
            x (torch.Tensor): Edge voltages with shape (batch_size, N_edges)
        
        Returns:
            torch.Tensor: Edge currents with shape (batch_size, N_edges)
        """
        return self.vectorize(self.element.gamma)(x, self.theta)

    def d_gamma(self, x):
        """
        Compute derivative of conductance function (differential conductance).
        
        Args:
            x (torch.Tensor): Edge voltages with shape (batch_size, N_edges)
        
        Returns:
            torch.Tensor: Differential conductance with shape (batch_size, N_edges)
        """
        return self.vectorize(self.element.d_gamma)(x, self.theta)

    def d_rho_d_theta(self, x):
        """
        Compute derivative of cocontent function with respect to parameters

        Args:
            x (torch.Tensor):
        
        Returns:
            torch.Tensor: Cocontent gradient with shape (batch_size, N_params, N_edges)
        """
        return self.vectorize(self.element.d_rho_d_theta)(x, self.theta)

