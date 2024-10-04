import networkx as nx
from itertools import combinations
import time
from pybktree import BKTree
import Levenshtein as lev  # Efficient Levenshtein distance computation
import numpy as np
import scipy.linalg as spla

# load "PDB Seqres.fasta"

with open("PDB Seqres.fasta", "r") as file:
    data = file.read()


proteins = data.split(">")[1:]
aminos = []
labels = []


for idx, protein in enumerate(proteins):
    try:
        aminos.append(protein.split("\n")[1])
        labels.append(protein.split("\n")[0])
    except:
        pass

print(len(aminos))
print(len(proteins) - len(aminos))
print(aminos[0])

# dedupe
aminos = list(set(aminos))
aminos = [amino for amino in aminos if "X" not in amino]

n = 1000000
aminos = aminos[:n]


def create_graph_bktree(aminos, k=1):
    """
    Create a graph where nodes are amino sequences and edges connect sequences with
    Levenshtein distance up to k, including insertions and deletions at the ends.

    Parameters:
    - aminos (list): List of amino acid sequences.
    - k (int): Maximum Levenshtein distance for edge creation.

    Returns:
    - G (networkx.Graph): The constructed graph.
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(aminos)))  # Using indices as node identifiers

    # Build BK-tree using Levenshtein distance as the metric
    tree = BKTree(lev.distance, aminos)

    start_time = time.time()
    length = len(aminos)

    for idx, seq in enumerate(aminos):
        # Query for all sequences within distance k
        matches = tree.find(seq, k)
        for distance, match_seq in matches:
            neighbor_idx = aminos.index(match_seq)
            if neighbor_idx != idx:
                G.add_edge(idx, neighbor_idx)

        if idx % 50 == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(f"Processed {idx} sequences in {elapsed:.2f} seconds.")
            estimated_time = (elapsed / idx) * length
            print(f"Estimated time to completion: {estimated_time:.2f} seconds.")

    total_time = time.time() - start_time
    print(f"Total time for BK-tree graph creation: {total_time:.2f} seconds")
    return G


# Create the optimized graph with Hamming distance threshold k=2
graph = create_graph_bktree(aminos, k=1)
print(f"Number of nodes in the graph: {graph.number_of_nodes()}")
print(f"Number of edges in the graph: {graph.number_of_edges()}")

# fine the length of the longest path in the graph, given that the graph is not connected nor acyclic
# The graph is not a DAG (Directed Acyclic Graph), so we can't use dag_longest_path_length
# Instead, we can find the diameter of the graph (longest shortest path between any two nodes)
# Note: This can be computationally expensive for large graphs
try:
    diameter = nx.diameter(graph)
    print(f"The diameter of the graph (longest shortest path) is: {diameter}")
except nx.NetworkXError:
    print(
        "The graph is not connected. Finding the longest path in the largest component:"
    )
    largest_cc = max(nx.connected_components(graph), key=len)
    subgraph = graph.subgraph(largest_cc)
    diameter = nx.diameter(subgraph)
    print(f"The diameter of the largest connected component is: {diameter}")
    # print the text of the aminos in the largest connected component
    print(
        f"The aminos in the largest connected component are: {[aminos[node] for node in largest_cc]}"
    )


# Optional: Visualize the graph (requires matplotlib)
# Optional: Visualize the graph (requires matplotlib)
import matplotlib.pyplot as plt

# Create a subgraph with nodes that have at least one edge
non_isolated_nodes = [node for node, degree in graph.degree() if degree > 0]
isolated_nodes = [node for node, degree in graph.degree() if degree == 0]
print(len(isolated_nodes))
print("max degree", max([graph.degree(node) for node in graph.nodes]))

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(graph)  # Calculate layout for the entire graph

# Draw non-isolated nodes
nx.draw_networkx_nodes(
    graph,
    pos,
    nodelist=non_isolated_nodes,
    node_size=10,
    node_color="skyblue",
    alpha=0.6,
)

# Draw edges
nx.draw_networkx_edges(
    graph,
    pos,
    edgelist=graph.edges(),
    edge_color="gray",
    alpha=0.6,
)

plt.title(
    f"Amino Acid Sequences Graph with Hamming Distance of 1 ({len(isolated_nodes)} Isolated Nodes in Red) "
)
plt.tight_layout()
plt.savefig("amino_acid_graph_with_isolated.png", dpi=300, bbox_inches="tight")
plt.show()

# Print some statistics
print(f"Total nodes in the graph: {graph.number_of_nodes()}")
print(f"Nodes with at least one edge: {len(non_isolated_nodes)}")
print(f"Isolated nodes: {len(isolated_nodes)}")
