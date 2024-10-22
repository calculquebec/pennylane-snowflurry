from pennylane.tape import QuantumTape
from pennylane_snowflurry.utility.graph_utility import find_largest_subgraph_isomorphism, circuit_graph, machine_graph, node_with_shortest_path_from_selection, most_connected_node, find_closest_wire, find_best_wire

def placement_vf2(tape : QuantumTape, benchmark : dict[str, any] = None) -> QuantumTape:

    if benchmark is None: benchmark = { "qubits": [], "couplers": [] }
    # on veut que les fils soient mapp�s � des qubits de la machine en fonction d'un isomorphisme ou d'un placement efficace
    broken_nodes = benchmark["qubits"]
    broken_couplers = benchmark["couplers"]

    circuit_topology = circuit_graph(tape)
    machine_topology = machine_graph(broken_nodes, broken_couplers)

    
    # 1. trouver un isomorphisme de sous-graph entre le circuit et la machine, maximisant le nombre de noeuds pris en compte
    mapping = find_largest_subgraph_isomorphism(circuit_topology, machine_topology)

    # 2. identifier les noeuds du circuit manquant dans le mapping (a)
    missing = [node for node in circuit_topology.nodes if node not in mapping.keys()]
        
    for node in missing:
        # 3. Trouver le noeud le plus connecté à A dans le circuit
        most_connected_node = most_connected_node(node, circuit_topology)

        # 4. trouver un noeud dans la machine (a') qui minimise le chemin entre a' et b'  
        possibles = [n for n in machine_topology.nodes if n not in mapping.values()]
        shortest_path_mapping = node_with_shortest_path_from_selection(mapping[most_connected_node], possibles, machine_topology)
            
        mapping[node] = shortest_path_mapping
    
    # 5. corriger les connexions dans le circuit en fonction du mappage choisi
    new_tape = type(tape)([op.map_wires(mapping) for op in tape.operations], [m.map_wires(mapping) for m in tape.measurements], shots=tape.shots)

    return new_tape

def placement_astar(tape : QuantumTape, benchmark : dict[str, any] = None) -> QuantumTape:
    """
    use astar to place qubits using their connectedness
    TODO : mapping doesn't fit the results I'm expecting meaning there must be an error somewhere in this function
    """
    if benchmark is None: benchmark = { "qubits": [], "couplers": [] }
    broken_nodes = benchmark["qubits"]
    broken_couplers = benchmark["couplers"]
    circuit_topology = circuit_graph(tape)
    machine_topology = machine_graph(broken_nodes, broken_couplers)

    mapping = {}

    to_explore = [w for w in tape.wires]
    while len(to_explore) > 0:
        a = to_explore.pop(0)
        mapping[a] = find_best_wire(machine_topology)

        for link in circuit_topology.edges:
            if a not in link: 
                continue
            
            b_key = link[0] if a == link[1] else link[1]
            if b_key in mapping:
                continue

            to_explore.pop(to_explore.index(b_key))
            mapping[b_key] = find_closest_wire(a, machine_topology, mapping.values())

    new_tape = type(tape)([op.map_wires(mapping) for op in tape.operations], [m.map_wires(mapping) for m in tape.measurements], shots=tape.shots)
    return new_tape