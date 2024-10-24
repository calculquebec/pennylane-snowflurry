from pennylane.tape import QuantumTape
import pennylane_snowflurry.utility.graph_utility as graph_util


def placement_imags(tape : QuantumTape, use_benchmark = True) -> QuantumTape:
    """
    places the circuit on the machine's connectivity using IMAGS algorithm
    """
    circuit_topology = graph_util.circuit_graph(tape)
    machine_topology = graph_util.machine_graph(use_benchmark)

    
    # 1. trouver un isomorphisme de sous-graph entre le circuit et la machine, maximisant le nombre de noeuds pris en compte
    mapping = graph_util.find_largest_subgraph_isomorphism_imags(circuit_topology, machine_topology)

    # 2. identifier les noeuds du circuit manquant dans le mapping (a)
    missing = [node for node in circuit_topology.nodes if node not in mapping.keys()]
        
    for node in missing:
        # 3. Trouver le noeud le plus connecté à A dans le circuit
        most_connected_node = graph_util.most_connected_node(node, circuit_topology)

        # 4. trouver un noeud dans la machine (a') qui minimise le chemin entre a' et b'  
        possibles = [n for n in machine_topology.nodes if n not in mapping.values()]
        shortest_path_mapping = graph_util.node_with_shortest_path_from_selection(mapping[most_connected_node], possibles, machine_topology)
            
        mapping[node] = shortest_path_mapping
    
    # 5. corriger les connexions dans le circuit en fonction du mappage choisi
    new_tape = type(tape)([op.map_wires(mapping) for op in tape.operations], [m.map_wires(mapping) for m in tape.measurements], shots=tape.shots)

    return new_tape

def placement_vf2(tape : QuantumTape, use_benchmark = True) -> QuantumTape:
    """
    places the circuit on the machine's connectivity using VF2 algorithm
    """
    circuit_topology = graph_util.circuit_graph(tape)
    machine_topology = graph_util.machine_graph(use_benchmark)

    
    # 1. trouver un isomorphisme de sous-graph entre le circuit et la machine, maximisant le nombre de noeuds pris en compte
    mapping = graph_util.find_largest_subgraph_isomorphism_vf2(circuit_topology, machine_topology)

    # 2. identifier les noeuds du circuit manquant dans le mapping (a)
    missing = [node for node in circuit_topology.nodes if node not in mapping.keys()]
        
    for node in missing:
        # 3. Trouver le noeud le plus connecté à A dans le circuit
        most_connected_node = graph_util.most_connected_node(node, circuit_topology)

        # 4. trouver un noeud dans la machine (a') qui minimise le chemin entre a' et b'  
        possibles = [n for n in machine_topology.nodes if n not in mapping.values()]
        shortest_path_mapping = graph_util.node_with_shortest_path_from_selection(mapping[most_connected_node], possibles, machine_topology)
            
        mapping[node] = shortest_path_mapping
    
    # 5. corriger les connexions dans le circuit en fonction du mappage choisi
    new_tape = type(tape)([op.map_wires(mapping) for op in tape.operations], [m.map_wires(mapping) for m in tape.measurements], shots=tape.shots)

    return new_tape

def _recurse(a_key, b_key, mapping, to_explore, machine_topology, circuit_topology):
        if b_key in mapping:
            return

        to_explore.pop(to_explore.index(b_key))
        mapping[b_key] = graph_util.find_closest_wire(mapping[a_key], machine_topology, [v for (k, v) in mapping.items()])

        a_key2 = b_key
        for b_key2 in to_explore:
            if (a_key2, b_key2) not in circuit_topology.edges:
                continue

            _recurse(a_key2, b_key2, mapping, to_explore, machine_topology, circuit_topology)

def placement_astar(tape : QuantumTape, use_benchmark = True) -> QuantumTape:
    """
    places the circuit on the machine's connectivity using astar algorithm and comparing path lengths
    """
    circuit_topology = graph_util.circuit_graph(tape)
    machine_topology = graph_util.machine_graph(use_benchmark)

    mapping = {}
    # sort nodes by degree descending, so that we map the most connected node first
    to_explore = list(reversed(sorted([w for w in tape.wires], key=lambda w: circuit_topology.degree(w))))

    while len(to_explore) > 0:
        a_key = to_explore.pop(0)
        mapping[a_key] = graph_util.find_best_wire(machine_topology)

        for b_key in to_explore:
            if (a_key, b_key) not in circuit_topology.edges: 
                continue
            
            _recurse(a_key, b_key, mapping, to_explore, machine_topology, circuit_topology)

    new_tape = type(tape)([op.map_wires(mapping) for op in tape.operations], [m.map_wires(mapping) for m in tape.measurements], shots=tape.shots)
    return new_tape