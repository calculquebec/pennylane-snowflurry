from copy import deepcopy
from functools import reduce
from pennylane.tape import QuantumTape
import networkx as nx


class VF2:
    _pattern : nx.Graph
    _target : nx.Graph
    _pattern_nodes : list[int]
    _target_nodes : list[int]
    _matches : dict[int, int]

    def __init__(self, pattern : nx.Graph, target : nx.Graph):
        self._pattern = pattern
        self._target = target
        self._pattern_nodes = list(pattern.nodes)
        self._target_nodes = list(target.nodes)
        self._matches = {}  # pattern node -> target node mapping

    def _is_valid_mapping(self, pattern_node : int, target_node : int) -> bool:
        # Check if the current mapping is valid
        for p in self._pattern.neighbors(pattern_node):
            if p in self._matches:
                t = self._matches[p]
                if not target_node in self._target.neighbors(t):
                    return False
        return True

    def backtrack(self, pattern_index : int) -> bool:
        if pattern_index == len(self._pattern_nodes):
            return True

        pattern_node = self._pattern_nodes[pattern_index]

        for target_node in self._target_nodes:
            if target_node not in self._matches.values() and self._is_valid_mapping(pattern_node, target_node):
                self._matches[pattern_node] = target_node

                if self.backtrack(pattern_index + 1):
                    return True

                del self._matches[pattern_node]

        return False

    def find_subgraph_isomorphisms(self) -> dict[int, int]:
        if self.backtrack(0):
            return self._matches
        else:
            return None

# TODO : change to an actual call to an API
def get_broken_infra():
    return [], []

def circuit_graph(tape : QuantumTape) -> nx.Graph:
    """
    input : 
    tape : QuantumTape, a tape representing the quantum circuit

    output : nx.Graph, a graph representing the connections between the wires in the circuit
    """
    links : list[list[int]] = []

    for op in tape.operations:
        if len(op.wires) != 2:
            continue
        toAdd = list(int(i) for i in op.wires)
        links.append(toAdd)
    return nx.Graph(links)

def machine_graph(broken_nodes : list[int] = [], broken_couplers : list[list[int]] = []):
#       00
#       |
#    08-04-01
#    |  |  | 
# 16-12-09-05-02
# |  |  |  |  |
# 20-17-13-10-06-03
#    |  |  |  |  |
#    21-18-14-11-07
#       |  |  |
#       22-19-15
#          |
#          23
    
    links = [[0, 4], [1, 4], [1, 5], [2, 5], [2, 6], 
             [3, 6], [3, 7], [4, 8], [4, 9], 
             [5, 9], [5, 10], [6, 10], [6, 11], 
             [7, 11], [8, 12], [9, 12], [9, 13], 
             [10, 13], [10, 14], [11, 14], [11, 15], 
             [12, 16], [12, 17], [13, 17], [13, 18],
             [14, 18], [14, 19], [15, 19], [16, 20],
             [17, 20], [17, 21], [18, 21], [18, 22],
             [19, 22], [19, 23]]
    return nx.Graph([i for i in links if i[0] not in broken_nodes and i[1] not in broken_nodes \
            and i not in broken_couplers and list(reversed(i)) not in broken_couplers])

# TODO : try to make it work with networkx instead of handwritten algo
def _find_isomorphisms(circuit : nx.Graph, machine : nx.Graph) -> dict[int, int]:
    vf2 = VF2(circuit, machine)
    isomorphisms = vf2.find_subgraph_isomorphisms()
    return isomorphisms

def _find_largest_subgraph_isomorphism(circuit : nx.Graph, machine : nx.Graph):
    from itertools import combinations

    edges = [e for e in circuit.edges]
    for i in reversed(range(len(edges) + 1)):
        for comb in combinations(edges, i):
            result = _find_isomorphisms(nx.Graph(comb), machine)
            if result: return result

def _most_connected_node(source : int, graph : nx.Graph, excluding : list[int]):
    """
    find node in graph minus excluded nodes with most connections with the given source node
    """
    g_copy = deepcopy(graph)
    g_copy.remove_nodes_from(excluding)
    return max(g_copy.nodes, \
        key = lambda n : sum(1 for g in graph.edges if g[0] == n and g[1] == source or g[1] == n and g[0] == source))

def _shortest_path(a : int, b : int, graph : nx.Graph, excluding : list[int] = []):
    """
    find the shortest path between node a and b in graph minus excluded nodes
    """
    g_copy = deepcopy(graph)
    g_copy.remove_nodes_from(excluding)
    return nx.astar_path(g_copy, a, b)

def _node_with_shortest_path_from_selection(source : int, selection : list[int], graph : nx.Graph):
    """
    find the unmapped node node in graph minus mapped nodes that has shortest path to given source node
    """
    # all_unmapped_nodes = [n for n in graph.nodes if n not in mapping and n != source]
    # mapping_minus_source = [n for n in mapping if n != source]

    nodes_minus_source = [node for node in selection if node != source]
    return min(nodes_minus_source, key=lambda n: _shortest_path(source, n, graph))
    # return min(all_unmapped_nodes, key = lambda n : len(_shortest_path(source, n, graph, mapping_minus_source)))

def physical_placement(tape : QuantumTape) -> QuantumTape:
    # on veut que les fils soient mappés à des qubits de la machine en fonction d'un isomorphisme ou d'un placement efficace
    
    broken_nodes, broken_couplers = get_broken_infra()
    circuit_topology = circuit_graph(tape)
    machine_topology = machine_graph(broken_nodes, broken_couplers)

    
    # 1. trouver un isomorphisme entre le circuit et la machine
    mapping = _find_isomorphisms(circuit_topology, machine_topology)

    if mapping is None: 
        # 2. si aucun isomorphisme n'existe, trouver un isomorphisme entre le sous-graphe le plus gros du circuit et la machine
        mapping = _find_largest_subgraph_isomorphism(circuit_topology, machine_topology)
        # 3. identifier les noeuds du circuit manquant dans le mapping (a)
        missing = [node for node in circuit_topology.nodes if node not in mapping.keys()]
        
        for node in missing:
            # 4. trouver un noeud du circuit (b) qui minimise le chemin entre b et le noeud a non-mappé
            shortest_path_node = _node_with_shortest_path_from_selection(node, mapping.keys(), circuit_topology)
            # 5. trouver un noeud dans la machine (a') qui minimise le chemin entre a' et b'
            
            possibles = [node for node in machine_topology.nodes if node not in mapping.values()]
            shortest_path_mapping = _node_with_shortest_path_from_selection(mapping[shortest_path_node], possibles, machine_topology)
            
            mapping[node] = shortest_path_mapping
    
    # 6. corriger les connexions dans le circuit en fonction du mappage choisi
    new_tape = type(tape)([op.map_wires(mapping) for op in tape.operations], [m.map_wires(mapping) for m in tape.measurements], shots=tape.shots)

    return new_tape

if __name__ == "__main__":
    import pennylane as qml
    dev = qml.device("default.qubit")

    @qml.qnode(dev)
    def circuit():
        qml.CNOT([0, 1])
        qml.CNOT([0, 2])
        qml.CNOT([0, 3])
        qml.CNOT([0, 4])
        qml.CNOT([0, 5])
        return qml.probs()
    
    circuit()

    tape = physical_placement(circuit.tape)
    print(tape)