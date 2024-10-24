from pennylane.tape import QuantumTape
from pennylane.operation import Operation
import networkx as nx
from networkx.algorithms.isomorphism.ismags import ISMAGS
from typing import Tuple
from copy import deepcopy
from itertools import combinations
from pennylane_snowflurry.monarq_data import connectivity, build_benchmark
from pennylane_snowflurry.api_utility import ApiUtility

def is_directly_connected(op : Operation, machine_topology : nx.Graph) -> bool:
    return op.wires[1] in machine_topology.neighbors(op.wires[0])

def circuit_graph(tape : QuantumTape) -> nx.Graph:
    """
    input : 
    tape : QuantumTape, a tape representing the quantum circuit

    output : nx.Graph, a graph representing the connections between the wires in the circuit
    """
    links : list[Tuple[int, int]] = []

    for op in tape.operations:
        if len(op.wires) != 2:
            continue
        toAdd = (op.wires[0], op.wires[1])
        links.append(toAdd)
    g = nx.Graph(set(links))
    g.add_nodes_from([w for w in tape.wires if w not in g.nodes])
    return g

def machine_graph(use_benchmark = True):
    benchmark = build_benchmark() if use_benchmark else None
    broken_nodes = benchmark[ApiUtility.keys.qubits] if use_benchmark else []
    broken_couplers = benchmark[ApiUtility.keys.couplers] if use_benchmark else []

    links = [(v[0], v[1]) for (_, v) in connectivity[ApiUtility.keys.couplers].items()]
    
    return nx.Graph([i for i in links if i[0] not in broken_nodes and i[1] not in broken_nodes \
            and i not in broken_couplers and list(reversed(i)) not in broken_couplers])

def _find_isomorphisms(circuit : nx.Graph, machine : nx.Graph) -> dict[int, int]:
    vf2 = nx.isomorphism.GraphMatcher(machine, circuit)
    for mono in vf2.subgraph_monomorphisms_iter():
       return {v : k for k, v in mono.items()}
    return None

def find_largest_subgraph_isomorphism_vf2(circuit : nx.Graph, machine : nx.Graph):
    """
    Uses vf2 and combinations to find the largest common graph between two graphs
    """
    edges = [e for e in circuit.edges]
    for i in reversed(range(len(edges) + 1)):
        for comb in combinations(edges, i):
            result = _find_isomorphisms(nx.Graph(comb), machine)
            if result: return result

def find_largest_subgraph_isomorphism_imags(circuit : nx.Graph, machine : nx.Graph):
    """
    Uses IMAGS to find the largest common graph between two graphs
    """

    ismags = ISMAGS(machine, circuit)
    for mapping in ismags.largest_common_subgraph():
        return {v:k for (k, v) in mapping.items()} if mapping is not None and len(mapping) > 0 else mapping

def most_connected_node(source : int, graph : nx.Graph):
    """
    find node in graph with most connections with the given source node
    """
    g_copy = deepcopy(graph)
    return max(g_copy.nodes, \
        key = lambda n : sum(1 for g in graph.edges if g[0] == n and g[1] == source or g[1] == n and g[0] == source))

def shortest_path(a : int, b : int, graph : nx.Graph, excluding : list[int] = [], prioritized_nodes : list[int] = []):
    """
    find the shortest path between node a and b

    Args :
        a : start node
        b : end node
        graph : the graph to find a path in
        excluding : nodes we dont want to use
        prioritized_nodes : nodes we want to use if possible
    """
    g_copy = deepcopy(graph)
    g_copy.remove_nodes_from(excluding)
    return nx.astar_path(g_copy, a, b, 
                         weight = lambda u, v, _: 1 if any(node in (u, v) 
                                                           for node in prioritized_nodes) else 2)

def find_best_wire(graph : nx.Graph):
    """
    find node with highest degree in graph
    """
    return max(graph.nodes, key=lambda n: graph.degree(n))

def find_closest_wire(a : int, machine_graph : nx.Graph, excluding : list[int]):
    """
    find node in graph that is closest to given node, not considering arbitrary excluding list
    """
    min_node = None
    min_value = 100000
    for b in machine_graph.nodes:
        if b in excluding:
            continue
        value = len(shortest_path(a, b, machine_graph))
        
        if value < min_value:
            min_value = value
            min_node = b
    return min_node

def node_with_shortest_path_from_selection(source : int, selection : list[int], graph : nx.Graph):
    """
    find the unmapped node node in graph minus mapped nodes that has shortest path to given source node
    """
    # all_unmapped_nodes = [n for n in graph.nodes if n not in mapping and n != source]
    # mapping_minus_source = [n for n in mapping if n != source]

    nodes_minus_source = [node for node in selection if node != source]
    return min(nodes_minus_source, key=lambda n: len(shortest_path(source, n, graph)))
    # return min(all_unmapped_nodes, key = lambda n : len(_shortest_path(source, n, graph, mapping_minus_source)))
