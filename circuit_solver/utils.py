import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

import networkx as nx
import numpy as np
import json
import pandas as pd
import pickle
import os


######################################## GRAPHICS UTILITIES ########################################


def alpha_n(n, min_alpha=0.001, max_alpha=1.0):
    """
    Calculate optimal alpha transparency for scatter plots based on number of points.

    Uses square root scaling to account for quadratic density effects from overlapping
    points, ensuring good visual density without oversaturation.

    Args:
        n: Number of points in the scatter plot
        min_alpha: Minimum alpha value (default: 0.02) - prevents points from being invisible
        max_alpha: Maximum alpha value (default: 1.0) - cap for small datasets

    Returns:
        float: Alpha value between min_alpha and max_alpha

    Example:
        plt.scatter(x, y, alpha=scatter_alpha(len(x)))

    """
    alpha = 2 / np.sqrt(n)
    return np.clip(alpha, min_alpha, max_alpha)


######################################## EXPERIMENT INDEX MANAGEMENT ########################################


def update_index(namespace, dataframe_keys, save_dir, index_filename='index.csv'):
    """
    Update the experiment index with metadata from a saved experiment.

    Args:
        namespace: The argparse.Namespace or object containing experiment variables
        dataframe_keys: Set of keys from namespace to include in the index
        save_dir: Root directory where experiments are saved
        index_filename: Name of the index file (default: 'index.csv')

    Returns:
        pandas.DataFrame: The updated index

    Example:
        v = run_params()
        dataframe_keys = set(['k_min', 'k_max', 'input_gain', 'timestamp'])
        update_index(v, dataframe_keys, save_dir)
    """
    index_path = os.path.join(save_dir, index_filename)

    # Load existing index or create new one
    if os.path.exists(index_path):
        index_df = pd.read_csv(index_path)
    else:
        index_df = pd.DataFrame()

    # Extract metadata from namespace
    metadata = {}
    for key in dataframe_keys:
        if hasattr(namespace, key):
            value = getattr(namespace, key)
            # Convert non-serializable objects to string representation
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[key] = value
            else:
                metadata[key] = str(value)

    # Add file path if available
    if hasattr(namespace, 'full_file_path'):
        metadata['full_file_path'] = namespace.full_file_path

    # Check for duplicate timestamp
    if 'timestamp' in metadata and not index_df.empty and 'timestamp' in index_df.columns:
        duplicate_mask = index_df['timestamp'] == metadata['timestamp']
        if duplicate_mask.any():
            duplicate_idx = index_df[duplicate_mask].index[0]
            print(f"Warning: Timestamp '{metadata['timestamp']}' already exists in index.")
            print(f"Overwriting entry at index {duplicate_idx}")
            # Update the existing row
            for key, value in metadata.items():
                index_df.loc[duplicate_idx, key] = value
        else:
            # Append new row
            new_row = pd.DataFrame([metadata])
            index_df = pd.concat([index_df, new_row], ignore_index=True)
    else:
        # Append to index (no timestamp or empty index)
        new_row = pd.DataFrame([metadata])
        index_df = pd.concat([index_df, new_row], ignore_index=True)

    # Save updated index
    index_df.to_csv(index_path, index=False)

    return index_df


def search_index(save_dir, index_filename='index.csv', **query):
    """
    Search the experiment index based on metadata queries.

    Args:
        save_dir: Root directory where experiments are saved
        index_filename: Name of the index file (default: 'index.csv')
        **query: Key-value pairs to filter by. Supports:
            - Exact match: k_min=0.1
            - Range query: k_min_min=0.1, k_min_max=10.0
            - Contains (for strings): description_contains='crossbar'

    Returns:
        pandas.DataFrame: Filtered index with matching experiments

    Example:
        # Find all experiments with k_min between 0.1 and 1.0
        results = search_index(save_dir, k_min_min=0.1, k_min_max=1.0)

        # Find experiments with specific description
        results = search_index(save_dir, description_contains='collapse')

        # Exact match
        results = search_index(save_dir, input_gain=10.0)
    """
    index_path = os.path.join(save_dir, index_filename)

    if not os.path.exists(index_path):
        print(f"No index found at {index_path}")
        return pd.DataFrame()

    index_df = pd.read_csv(index_path)

    # Apply filters
    mask = pd.Series([True] * len(index_df))

    for key, value in query.items():
        # Handle range queries
        if key.endswith('_min'):
            col = key[:-4]  # Remove '_min' suffix
            if col in index_df.columns:
                mask &= index_df[col] >= value
        elif key.endswith('_max'):
            col = key[:-4]  # Remove '_max' suffix
            if col in index_df.columns:
                mask &= index_df[col] <= value
        # Handle contains queries for strings
        elif key.endswith('_contains'):
            col = key[:-9]  # Remove '_contains' suffix
            if col in index_df.columns:
                mask &= index_df[col].astype(str).str.contains(value, na=False, case=False)
        # Exact match
        else:
            if key in index_df.columns:
                mask &= index_df[key] == value

    return index_df[mask]


def load_from_index(search_results, indices=None):
    """
    Load experiment data from search results.

    Args:
        search_results: DataFrame from search_index()
        indices: List of row indices to load (default: load all)

    Returns:
        List of loaded namespace objects

    Example:
        results = search_index(save_dir, k_min_min=0.1)
        experiments = load_from_index(results, indices=[0, 2])  # Load first and third
        experiments = load_from_index(results)  # Load all
    """
    if indices is None:
        indices = range(len(search_results))

    loaded_data = []
    for idx in indices:
        if idx >= len(search_results):
            print(f"Warning: Index {idx} out of range (max: {len(search_results)-1})")
            continue

        file_path = search_results.iloc[idx]['full_file_path']
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue

        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            loaded_data.append(data)

    return loaded_data


def view_index(save_dir, columns=None, index_filename='index.csv',):
    """
    View the experiment index with selected columns.

    Args:
        save_dir: Root directory where experiments are saved
        index_filename: Name of the index file (default: 'index.csv')
        columns: List of columns to display (default: all)

    Returns:
        pandas.DataFrame: The index (or selected columns)

    Example:
        view_index(save_dir, columns=['timestamp', 'k_min', 'k_max', 'description'])
    """
    index_path = os.path.join(save_dir, index_filename)

    if not os.path.exists(index_path):
        print(f"No index found at {index_path}")
        return pd.DataFrame()

    index_df = pd.read_csv(index_path)

    if columns is not None:
        # Filter to only existing columns
        valid_columns = [col for col in columns if col in index_df.columns]
        return index_df[valid_columns]

    return index_df


def refresh_index(save_dir, dataframe_keys=None, index_filename='index.csv', backup=True):
    """
    Rebuild the experiment index from all .dat files in the save directory.

    This scans the save directory for all .dat files, loads them, and rebuilds
    the index from scratch based on the actual files present.

    Args:
        save_dir: Root directory where experiments are saved
        dataframe_keys: Set of keys to include in index. If None, uses all keys
                       that are common across all loaded experiments.
        index_filename: Name of the index file (default: 'index.csv')
        backup: If True, backs up existing index before overwriting (default: True)

    Returns:
        pandas.DataFrame: The rebuilt index

    Example:
        # Rebuild with specific keys
        keys = {'timestamp', 'k_min', 'k_max', 'description', 'input_gain'}
        refresh_index(save_dir, dataframe_keys=keys)

        # Rebuild with auto-detected keys
        refresh_index(save_dir)
    """
    import glob

    index_path = os.path.join(save_dir, index_filename)

    # Backup existing index if requested
    if backup and os.path.exists(index_path):
        backup_path = index_path.replace('.csv', '_backup.csv')
        import shutil
        shutil.copy(index_path, backup_path)
        print(f"Backed up existing index to {backup_path}")

    # Find all .dat files recursively in save_dir
    dat_files = glob.glob(os.path.join(save_dir, '**', '*.dat'), recursive=True)
    print(f"Found {len(dat_files)} .dat files in {save_dir}")

    if len(dat_files) == 0:
        print("No data files found. Index not modified.")
        return pd.DataFrame()

    # Load all data files and collect metadata
    all_metadata = []
    failed_loads = []

    for file_path in dat_files:
        try:
            with open(file_path, 'rb') as f:
                namespace = pickle.load(f)

            # If dataframe_keys not specified, collect all attributes
            if dataframe_keys is None:
                keys_to_extract = set(vars(namespace).keys())
            else:
                keys_to_extract = dataframe_keys

            # Extract metadata
            metadata = {}
            for key in keys_to_extract:
                if hasattr(namespace, key):
                    value = getattr(namespace, key)
                    # Convert non-serializable objects to string representation
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        metadata[key] = value
                    else:
                        metadata[key] = str(value)

            # Always add file path
            metadata['full_file_path'] = file_path

            all_metadata.append(metadata)

        except Exception as e:
            failed_loads.append((file_path, str(e)))
            print(f"Warning: Failed to load {file_path}: {e}")

    # Report failures
    if failed_loads:
        print(f"\nFailed to load {len(failed_loads)} files:")
        for path, error in failed_loads:
            print(f"  - {path}: {error}")

    # Create new index
    if all_metadata:
        index_df = pd.DataFrame(all_metadata)

        # Sort by timestamp if available
        if 'timestamp' in index_df.columns:
            index_df = index_df.sort_values('timestamp').reset_index(drop=True)

        # Save new index
        index_df.to_csv(index_path, index=False)
        print(f"\nSuccessfully rebuilt index with {len(index_df)} entries")
        print(f"Index saved to {index_path}")

        return index_df
    else:
        print("No metadata could be extracted. Index not created.")
        return pd.DataFrame()


######################################## MNIST STUFF ########################################


def downsample_images(images, size=(5, 5)):
    return np.array([resize(img, size, anti_aliasing=True, preserve_range=True).flatten() for img in images])


def preprocess_MNIST_data_legacy(X_in, size):
    # downsample
    X = X_in.T.reshape((-1, 28, 28))
    X = downsample_images(X, size).T

    # Add negative entries
    X = np.concatenate((X, -X), axis=0)

    # # Add ground node
    # P = X_in.shape[1]
    # X = np.concatenate((X, np.zeros((1,P))))

    # Normalize
    x_max = np.max(X)
    X = X / x_max
    
    return X

def preprocess_MNIST_data(X_in, Y_in, size, X_gain=1., Y_gain=1., add_negatives=True):
    # downsample inputs
    X = downsample_images(X_in, size)

    # Encode labels as one-hot
    Y = index_to_one_hot(Y_in)

    # Add negative entries
    if add_negatives:
        X = np.concatenate((X, -X), axis=1)
    # Y = np.concatenate((Y, -Y), axis=1)
    
    # Normalize
    x_max = np.max(X)
    X = X / x_max

    # Apply gains
    X = X_gain * X
    Y = Y_gain * Y

    # # Add ground node
    # P = X_in.shape[1]
    # X = np.concatenate((X, np.zeros((1,P))))
    return X, Y

def index_to_one_hot(Y_in):
    Y = np.zeros((Y_in.size, Y_in.max() + 1))
    Y[np.arange(Y_in.size), Y_in] = 1
    return Y


def look(img):
    fig, ax = plt.subplots()
    sp = img.shape
    cmap = 'Greys'
    if len(sp) == 1:
        K = int(np.sqrt(len(img)))
        ax.imshow(img.reshape(K, K), cmap=cmap)
    elif len(sp) == 2:
        ax.imshow(img, cmap=cmap)
    return fig, ax


######################################## GRAPH TOOLS ########################################


def list_difference(list_1, list_2):
    return list(set(list_1) - set(list_2))

def all_pairs_between(a_nodes, b_nodes):
    """
    Return a list of tuples (u, v) connecting all u in a_nodes to all v in b_nodes
    """
    pairs_between = []
    for a in a_nodes:
        for b in b_nodes:
            pairs_between.append((a, b))
    
    return pairs_between

def get_edges_between(a_nodes, b_nodes, graph):
    """
    Given two lists of nodes for a graph, return a list of edge tuples for each edge connecting one set to the other.

    Args:
        a_nodes: List of nodes
        b_nodes: List of nodes
        graph: NetworkX graph object
        
    Returns:
        List of edge tuples connecting nodes in a_nodes to nodes in b_nodes
        
    Example:
        graph = nx.Graph([(0,1), (1,2), (2,3), (0,3)])
        a_nodes = [0, 1]
        b_nodes = [2, 3]
        edges = get_edges_between(a_nodes, b_nodes, graph)  # Returns [(1,2), (0,3)]
    """
    a_set = set(a_nodes)
    b_set = set(b_nodes)
    
    edges_between = []
    for edge in graph.edges():
        u, v = edge
        if (u in a_set and v in b_set) or (u in b_set and v in a_set):
            edges_between.append(edge)
    
    return edges_between


def nodes_to_inds(nodes, graph):
    """
    For a list of nodes, return their corresponding indices in list(graph.nodes())
    """

    node_list = list(graph.nodes())
    node_to_index = {node: i for i, node in enumerate(node_list)}
    indices = []
    for node in nodes:
        if node in node_to_index:
            indices.append(node_to_index[node])
        else:
            raise ValueError(f"Node {node} not found in graph")
    
    return indices

def edges_to_inds(edges, graph):
    """
    Given a graph and a list of edges, return the indices of the edges in graph.edges().
    
    Args:
        edges: List of edge tuples, e.g., [(0,1), (2,3), ...]
        graph: NetworkX graph object
        
    Returns:
        List of integers representing the indices of the edges in graph.edges()
        
    Raises:
        ValueError: If any edge is not found in the graph
        
    Example:
        graph = nx.Graph([(0,1), (1,2), (2,0)])
        edges = [(1,2), (0,2)]
        indices = edges_to_inds(graph, edges)  # Returns [1, 2]
    """
    edge_list = list(graph.edges())
    edge_to_index = {edge: i for i, edge in enumerate(edge_list)}
    
    indices = []
    for edge in edges:
        if edge in edge_to_index:
            indices.append(edge_to_index[edge])
        else:
            raise ValueError(f"Edge {edge} not found in graph")
    
    return indices


def segment_array(array, segment_sizes):
    """
    Segment an array into chunks based on given sizes.
    
    Args:
        array: Array-like object to segment (e.g., list, numpy array, range)
        segment_sizes: Tuple of sizes for each segment, with None for auto-sized segments
        
    Returns:
        List of segments (each segment is a list of elements from array)
        
    Example:
        segment_array(range(10), (2, 3, None, None)) 
        # Returns segments of sizes [2, 3, 2, 3] (remainder evenly split)
    """
    
    array = list(array)  # Convert to list for easy indexing
    total_length = len(array)
    
    # Identify which segments have definite sizes
    is_definite = [size is not None for size in segment_sizes]
    
    # Calculate allocated space and remainder
    allocated = sum(size for size, definite in zip(segment_sizes, is_definite) if definite)
    remainder = total_length - allocated
    
    if remainder < 0:
        raise ValueError(f"Segment sizes sum to {allocated} but array length is {total_length}")
    
    # Count None segments and distribute remainder
    none_count = sum(1 for size in segment_sizes if size is None)
    if none_count == 0:
        if remainder != 0:
            raise ValueError(f"Remainder {remainder} but no None segments to distribute to")
        final_sizes = list(segment_sizes)
    else:
        base_size = remainder // none_count
        extra = remainder % none_count
        
        final_sizes = []
        none_idx = 0
        for size in segment_sizes:
            if size is None:
                # First 'extra' None segments get one additional element
                final_size = base_size + (1 if none_idx < extra else 0)
                final_sizes.append(final_size)
                none_idx += 1
            else:
                final_sizes.append(size)
    
    # Create segments
    segments = []
    start_idx = 0
    for size in final_sizes:
        end_idx = start_idx + size
        segment = [int(a) for a in array[start_idx:end_idx]]
        segments.append(segment)
        start_idx = end_idx
    
    return segments

def get_mask_from_inds(inds, full_list):
    return np.array([a in inds for a in full_list])

def get_preimage(dict):
    """
    For a dictionary with a structure like 

        {2:1, 3:1, 5:2, 7:2, 8:2}

    Return

        {1:{2,3}, 2: {5,7,8}}
    
    """
    values = dict.values()
    inverted_dict = {
        val: set(key for key in dict.keys() if dict[key] == val) for val in values
    }
    return inverted_dict


def er_network(n, p, directed=True, relabel_nodes=True, max_connected_attempts=100):
    graph = None
    not_connected = True
    counter = 0
    while not_connected:
        if counter > max_connected_attempts:
            print('no connected graph found')
            return None
        graph = nx.erdos_renyi_graph(n, p, directed=directed)
        if directed: 
            check_graph = graph.to_undirected()
        else:
            check_graph = graph
        not_connected = not nx.is_connected(check_graph)
        counter += 1


    for i, node in enumerate(graph.nodes):
        theta = i * 2 * np.pi / n
        graph.nodes[node]['pos'] = np.array([np.cos(theta), np.sin(theta)])
    if relabel_nodes:
        graph = nx.convert_node_labels_to_integers(graph)

    return graph

def bipartite_network(n, m, relabel_nodes=True):
    """
    Bipartite graph with partitions of size m and n
    """
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(n+m))
    for i in range(n):
        for j in range(n, n+m):
            graph.add_edge(i,j)

    for i in range(n):
        graph.nodes[i]['pos'] = np.array([0, i / n])

    for i in range(n, n+m):
        graph.nodes[i]['pos'] = np.array([1, (i-n) / m])

    if relabel_nodes:
        graph = nx.convert_node_labels_to_integers(graph)

    return graph


def complete_network(n, relabel_nodes=True):
    graph = nx.complete_graph(n)
    for i, node in enumerate(graph.nodes):
        theta = i * 2 * np.pi / n
        graph.nodes[node]['pos'] = np.array([np.cos(theta), np.sin(theta)])
    if relabel_nodes:
        graph = nx.convert_node_labels_to_integers(graph)
    
    return graph

def grid_network(n, m, periodic=False, size_uc = (1,1), relabel_nodes=True):
    graph = nx.grid_2d_graph(n, m, periodic=periodic)
    for node in graph.nodes:
        graph.nodes[node]['pos'] = np.array(node)*size_uc
    if relabel_nodes:
        graph = nx.convert_node_labels_to_integers(graph)

    return graph


def generate_layer_graph(circuit):
    """
    Generate a layer graph for a Circuit.
    """
    graph = circuit.graph
    inputs = set(circuit.inputs)
    outputs = set(circuit.outputs)
    hiddens = set(graph.nodes) - inputs - outputs

    # generate layer structure for outputs
    # isolate output subgraph O and solve the coloring problem
    O = nx.induced_subgraph(graph, outputs)
    O_node_to_color = coloring.greedy_color(O)

    # construct map from colors back to nodes
    O_color_to_node = get_preimage(O_node_to_color) # dict[color] = {nodes with that color}

    # now solve coloring problem to get layer structure for hidden node graph H
    H = nx.induced_subgraph(graph, hiddens)
    H_node_to_color = coloring.greedy_color(H)
    H_color_to_node = get_preimage(H_node_to_color)

    # finally, generate the full layer graph. Each node is a layer (color), and two colors are conneced if they contain nodes which were originally connected

    # Now add the color nodes to the graph
    layer_graph = nx.Graph()
    layer_graph.add_node('I0')   # the input layer (always only one)
    layer_graph.nodes['I0']['nodes'] = inputs
    all_colors = ['I0']

    # Hidden node colors
    H_colors = H_color_to_node.keys()
    for color in H_colors:
        new_color = 'H'+str(color)
        all_colors.append(new_color)
        layer_graph.add_node(new_color)
        layer_graph.nodes[new_color]['nodes'] = H_color_to_node[color]

    # Output node colors
    O_colors = O_color_to_node.keys()
    for color in O_colors:
        new_color = 'O'+str(color)
        all_colors.append(new_color)
        layer_graph.add_node(new_color)
        layer_graph.nodes[new_color]['nodes'] = O_color_to_node[color]

    # Find the edges of the layer graph
    for i, c1 in enumerate(all_colors):
        for j in range(i):
            c2 = all_colors[j]
            nodes_1 = layer_graph.nodes[c1]['nodes']
            nodes_2 = layer_graph.nodes[c2]['nodes']
            nodes_12 = nodes_1 | nodes_2
            subgraph_12 = nx.induced_subgraph(graph, nodes_12)
            cur_edges = subgraph_12.edges
            if len(cur_edges) > 0:      # because each layer has no internal connections, any edges are between layers.
                layer_graph.add_edge(c2,c1)
            else:   # else add no edge and do nothing
                continue

            layer_graph.edges[c2, c1]['edges'] = set(subgraph_12.edges)


    return layer_graph