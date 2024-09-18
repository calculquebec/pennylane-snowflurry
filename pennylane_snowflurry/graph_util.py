from audioop import reverse
from functools import reduce
from platform import machine
import networkx as nx
from networkx.algorithms import isomorphism
from networkx.algorithms.isomorphism import vf2userfunc
from pennylane import qml
import pennylane.tape as tape

def get_isomorphism(graph : list[list[int]], subgraph : list[list[int]]) -> dict[int, int]:
    nodes_in_graph = list(set(reduce(lambda a, b: a + b, graph)))
    nodes_in_subgraph = list(set(reduce(lambda a, b: a + b, subgraph)))

    for node in nodes_in_graph:
        
        print(nodes_in_graph)

def circuit_graph(tape : tape.QuantumScript):
    links : list[list[int]] = []

    for op in tape.operations:
        if len(op.wires) != 2:
            continue
        toAdd = list(int(i) for i in op.wires)
        links.append(toAdd)
    return links
    
# TODO : permettre d'enlever des liens non-fonctionnels 
# 
# comment faire ça?
# cnot(0, 1)
# cnot(1, 2)
# cnot(2, 0)
#
# comment faire un graph qui a plus que 5 liens sur un noeud?


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
    return [i for i in links if i[0] not in broken_nodes and i[1] not in broken_nodes \
            and i not in broken_couplers and list(reversed(i)) not in broken_couplers]


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

def find_isomorphisms(circuit : tape.QuantumScript | list[list[int]], broken_nodes : list[int] = [], broken_couplers : list[list[int]] = []) -> dict[int, int]:
    import networkx as nx

    circuit = circuit if circuit is list else circuit_graph(circuit)
    circ_graph = nx.Graph(circuit)
    machine = nx.Graph(machine_graph(broken_nodes, broken_couplers))
    vf2 = VF2(circ_graph, machine)
    isomorphisms = vf2.find_subgraph_isomorphisms()
    return isomorphisms
    
    
def subgraph_is_connected(graph : list[list[int]], subgraph_nodes : list[int]):
    if any(link in graph for link in [[node1, node2] for node1 in subgraph_nodes for node2 in subgraph_nodes if node1 != node2]):
        return True
    return False

        
def find_path(graph : list[list[int]], start : int, length : int) -> list[list[int]]:
    import random as rand
    path : list[list[int]]= []
    current : int = start
    for _ in range(length):
        link : list[int] = rand.choice([l for l in graph if l[0] == current and l[1] not in [p[0] for p in path]])
        current = link[1]
        path.append(link)
    return path

def nx_isomoprhism():
    import networkx as nx
    import time
    circuit = [[0, 1], [1, 2], [1, 3]]
    machine = machine_graph()
    
    g_circuit = nx.DiGraph()
    g_circuit.add_edges_from(circuit)
    g_circuit = g_circuit.to_undirected()
    g_machine = nx.DiGraph()
    g_machine.add_edges_from(machine)
    g_machine = g_machine.to_undirected()

    print("circuit : " + g_circuit.edges)
    print("circuit : " + g_machine.edges)
    iso = nx.isomorphism.GraphMatcher(g_machine, g_circuit)
    iso.initialize()
    
    for match in iso.isomorphisms_iter():
        print(match)

def find_largest_subgraph_isomorphism(tape, broken_nodes : list[int] = [], broken_couplers : list[list[int]] = []):
    graph = circuit_graph(tape)
    from itertools import combinations
    for i in reversed(range(len(graph) + 1)):
        for comb in combinations(graph, i):
            result = find_isomorphisms(comb, broken_nodes, broken_couplers)
            if result: return result

if __name__ == "__main__":
    qtape = tape.QuantumTape([qml.CNOT([0, 1]), qml.CNOT([1, 2]), qml.CNOT([2, 0])])
    result = find_isomorphisms(qtape)
    if result is None:
        largest = find_largest_subgraph_isomorphism(qtape)
        if len(largest) < 3: raise Exception("impossible to simulate connectivity with swaps on a graph that has less than 3 nodes")

    print(result)
