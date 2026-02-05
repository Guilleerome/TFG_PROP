import numpy as np
import networkx as nx
from typing import Dict, Any



def build_graph_from_instance(plant) -> nx.Graph:
    """
    Build an undirected weighted graph from a Plant instance.
    Nodes: 0..n-1 with attribute "size".
    Edges: weight = matrix[i][j] + matrix[j][i] for i<j, included only if weight>0.
    """
    n = int(plant.number)
    G = nx.Graph(name=plant.name)
    # Nodes with "size"
    for i, size in enumerate(plant.facilities):
        G.add_node(i, size=float(size))
    # Edges with symmetric weights
    M = plant.matrix
    for i in range(n):
        for j in range(i+1, n):
            wij = 0.0
            if i < len(M) and j < len(M[i]):
                wij += float(M[i][j])
            if j < len(M) and i < len(M[j]):
                wij += float(M[j][i])
            if wij > 0.0:
                G.add_edge(i, j, weight=wij)
    return G

def _safe_stats(vals):
    arr = np.array(list(vals), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(np.max(arr)), float(np.min(arr)), float(np.mean(arr)), float(np.std(arr))

def extract_graph_features(G: "nx.Graph") -> Dict[str, Any]:
    """
    Mirror of the notebook features: degrees, sizes, weights, density, connectedness, etc.
    """
    n_vertices = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # Degrees (unweighted degree like in the notebook)
    degrees = [d for _, d in G.degree()]
    v_max_deg, v_min_deg, v_avg_deg, v_std_deg = _safe_stats(degrees)

    # Node sizes (from attribute)
    node_sizes = list(nx.get_node_attributes(G, "size").values())
    v_max_size, v_min_size, v_avg_size, v_std_size = _safe_stats(node_sizes)

    # Edge weights
    weights = list(nx.get_edge_attributes(G, "weight").values())
    e_max_w, e_min_w, e_avg_w, e_std_w = _safe_stats(weights)

    density = nx.density(G)
    is_connected = nx.is_connected(G) if n_vertices > 0 else False
    components = nx.number_connected_components(G) if n_vertices > 0 else 0
    is_regular = nx.is_regular(G)

    return {
        "id": G.name,
        "num_vertices": n_vertices,
        "num_edges": n_edges,
        "vertex_max_degree": v_max_deg,
        "vertex_min_degree": v_min_deg,
        "vertex_avg_degree": v_avg_deg,
        "vertex_std_degree": v_std_deg,
        "vertex_max_size": v_max_size,
        "vertex_min_size": v_min_size,
        "vertex_avg_size": v_avg_size,
        "vertex_std_size": v_std_size,
        "edge_max_weight": e_max_w,
        "edge_min_weight": e_min_w,
        "edge_avg_weight": e_avg_w,
        "edge_std_weight": e_std_w,
        "density": density,
        "is_connected": bool(is_connected),
        "components": int(components),
        "is_regular": bool(is_regular),
    }
