import networkx as nx
import pennylane.tape as tape


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


def machine_graph(*unavailable : int):
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
    return [i for i in links if i[0] not in unavailable and i[1] not in unavailable] + [list(reversed(i)) for i in links if i[0] not in unavailable and i[1] not in unavailable]


class VF2:
    def __init__(self, pattern : nx.Graph, target : nx.Graph):
        self.pattern = pattern
        self.target = target
        self.pattern_nodes = list(pattern.nodes)
        self.target_nodes = list(target.nodes)
        self.matches = {}  # pattern node -> target node mapping

    def is_valid_mapping(self, pattern_node, target_node):
        # Check if the current mapping is valid
        for p in self.pattern.neighbors(pattern_node):
            if p in self.matches:
                t = self.matches[p]
                if not target_node in self.target.neighbors(t):
                    return False
        return True

    def backtrack(self, pattern_index):
        if pattern_index == len(self.pattern_nodes):
            return True

        pattern_node = self.pattern_nodes[pattern_index]

        for target_node in self.target_nodes:
            if target_node not in self.matches.values() and self.is_valid_mapping(pattern_node, target_node):
                self.matches[pattern_node] = target_node

                if self.backtrack(pattern_index + 1):
                    return True

                del self.matches[pattern_node]

        return False

    def find_subgraph_isomorphisms(self):
        if self.backtrack(0):
            return self.matches
        else:
            return None


def find_isomorphisms(circuit : tape.QuantumScript) -> list[int]:
    import networkx as nx
    circ_graph = nx.Graph(circuit_graph(circuit))
    machine = nx.Graph(machine_graph())
    vf2 = VF2(circ_graph, machine)
    isomorphisms = vf2.find_subgraph_isomorphisms()
    return [isomorphisms[i] for i in range(len(isomorphisms))]
    
    
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
