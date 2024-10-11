import networkx as nx

class VF2:
    """
    vf2 algorithm for finding subgraph isomorphisms. 
    deprecated : it's better to use the networkx methods for finding subgraph isomorphisms
    """
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
