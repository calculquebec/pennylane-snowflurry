from pennylane.tape import QuantumTape
from pennylane.operation import Operation
from copy import deepcopy
import networkx as nx

from transpiler.physical_placement import _shortest_path, circuit_graph, machine_graph, get_broken_infra
from transpiler.native_gate_decomposition import _custom_swap
def _is_directly_connected(op : Operation, machine_topology : nx.Graph) -> bool:
    return op.wires[1] in machine_topology.neighbors(op.wires[0])

# TODO : there are better alternatives than using swap gates
def swap_routing(tape : QuantumTape):
    # en fonction du mappage choisi, connecter les qubits non-couplés à la position des portes touchées en utilisant des swaps
    broken_nodes, broken_couplers = get_broken_infra()
    machine_topology = machine_graph(broken_nodes, broken_couplers)
    new_operations : list[Operation] = []
    list_copy = tape.operations.copy()

    for oper in list_copy:
        # s'il s'agit d'une porte à 2 qubit n'étant pas placée sur un coupleur physique, 
        # on la route avec des cnots en utilisant astar
        if oper.num_wires == 2 and not _is_directly_connected(oper, machine_topology):
            path = _shortest_path(oper.wires[0], oper.wires[1], machine_topology)
            for i in range(1, len(path) - 1): 
                new_operations += _custom_swap(path[i:i+2])

            new_operations += [oper.map_wires({k:v for (k,v) in zip(oper.wires, path[0:2])})]

            for i in reversed(range(1, len(path) - 1)):
                new_operations += _custom_swap(path[i:i+2])
        else:
            new_operations += [oper]

    new_tape = type(tape)(new_operations, tape.measurements, shots=tape.shots)

    return new_tape
