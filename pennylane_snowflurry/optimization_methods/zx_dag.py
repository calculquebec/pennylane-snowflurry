import pennylane as qml
import numpy as np
from pennylane.operation import Operation
from pennylane.tape import QuantumTape
from pennylane.wires import Wires
import pennylane.transforms as transforms
from copy import deepcopy
import networkx as nx

DEBUG = False

class ZX_DAG:
    """
    this class is a Directed Acyclic Graph containing only Z or X rotations and CZ gates
    It is possible to pass any type of gate at initialization, but the circuit will be decomposed.
    """
    
    _index_generator = -1
    nodes : dict[int, Operation]
    tape : QuantumTape

    def __init__(self, tape : QuantumTape):

        self.nodes = {}
        self.graph = nx.MultiDiGraph()
       
        tape = ZX_DAG._decompose(tape)
        
        for i, op in enumerate(tape.operations):
            op.index_for_placement = f"{i}"

        self.tape = deepcopy(tape)
        for op in tape.operations:
            idx = self.add_node(op)

            for wire in op.wires:
                # create links between predecessors and current operation
                for pred_idx in reversed(self.nodes):
                    if pred_idx == idx:
                        continue
                    if wire in Wires.shared_wires([self.nodes[idx].wires, self.nodes[pred_idx].wires]):
                        self.graph.add_edge(pred_idx, idx)
                        # we skip all other nodes, and go to next wire
                        break
        return
    
    @staticmethod
    def _decompose(tape : QuantumTape) -> QuantumTape:
        """
        expands the circuit to only RX, RZ and CZ gates
        """
    
        def custom_ry(wires : Wires, **kwarg) -> list[Operation]:
            return [
                qml.RZ(np.pi/2, wires),
                qml.RX(kwarg["params"][0], wires),
                qml.RZ(-np.pi/2, wires)
            ]
        
        def custom_cnot(wires : Wires, **kwarg) -> list[Operation]:
            return custom_h([wires[1]]) + [qml.CZ(wires)] + custom_h([wires[1]])
    
        def custom_h(wires : Wires, **kwarg) -> list[Operation]:
            return [
                qml.RZ(np.pi/2, wires[0]),
                qml.RX(np.pi/2, wires[0]),
                qml.RZ(np.pi/2, wires[0])
            ]
        
        equivalences = {
            "CNOT" : custom_cnot,
            "Hadamard" : custom_h,
            "RY" : custom_ry,
        }
    
        tape = transforms.create_expand_fn(depth=9, stop_at=lambda op : op.name in ["RX", "RZ", "CZ"])(tape)
    
        new_operations = []

        for op in tape.operations:
            new_operations += equivalences[op.name](wires=op.wires, params=op.parameters) if op.name in equivalences else [op]
    
        return type(tape)(new_operations, tape.measurements, tape.shots)
    
    @staticmethod             
    def _is_z_gate(op : Operation, epsilon : float = 1E-8) -> bool:
        return ZX_DAG._is_cz_gate(op) or ZX_DAG._is_single_z_gate(op, epsilon)

    @staticmethod
    def _is_single_z_gate(op : Operation, epsilon : float = 1E-8) -> bool:
        mat = op.matrix()
        return op.num_wires == 1 and np.abs(mat[0][1]) < epsilon and np.abs(mat[1][0]) < epsilon
    
    def _floor(value : float, upTo : float = 2):
        return int(value * (10 ** upTo))/(10.0 ** upTo)

    @staticmethod
    def _is_cz_gate(op : Operation) -> bool:
        return op.name == "CZ"

    @staticmethod
    def _can_merge(a : Operation, b : Operation) -> bool:
        """
        assumes that all operations in tape have been decomposed to rz, rx and cz
        """
        merge = ZX_DAG._is_on_same_wire(a, b) and \
            ZX_DAG._is_single_z_gate(a) and ZX_DAG._is_single_z_gate(b) \
            or not ZX_DAG._is_z_gate(a) and not ZX_DAG._is_z_gate(b)

        return merge

    @staticmethod
    def _is_on_same_wire(a : Operation, b : Operation) -> bool:
        shared = Wires.shared_wires([a.wires, b.wires])
        return len(shared) > 0

    @staticmethod
    def _merge(a : Operation, b : Operation, epsilon = 1E-8) -> qml.RX | qml.RZ:
        """
        assumes that checks for mergeability have been done prior to this method
        """

        angle = a.parameters[0] + b.parameters[0]
        angle %= 2 * np.pi

        if ZX_DAG._is_single_z_gate(a):
            return qml.RZ(angle, a.wires)

        if not ZX_DAG._is_z_gate(a):
            return qml.RX(angle, a.wires)

        raise Exception("cannot merge other than RZ -> RZ and RX -> RX")

    @staticmethod
    def _can_commute(a : Operation, b : Operation) -> bool:
        """
        assumes that all operations in tape have been decomposed to rz, rx and cz
        we only commute if a is z and b is cz, since we dont want to have cases where we commute forever
        """
        commute = ZX_DAG._is_single_z_gate(a) and ZX_DAG._is_cz_gate(b) 
        return commute

    def roots(self) -> list[int]:
        """
        returns all root operations indices (operations with no predecessor)
        """
        roots = [n for n,d in self.graph.in_degree() if d==0] 
        return roots

    def leaves(self) -> list[int]:
        """
        return all leaves operation indices (operations with no successor)
        """
        leaves = [n for n,d in self.graph.out_degree() if d==0]
        return leaves

    def predecessors(self, op : int) -> list[int]:
        """
        returns the direct predecessors of an operation
        """
        return list(self.graph.predecessors(op))

    def successors(self, op : int) -> list[int]:
        """
        returns the direct successors of an operation
        """

        return list(self.graph.successors(op))

    def remove_node(self, op : int):
        """
        removes an operation, and all links to it
        """
        # remove all link from or to idx
        self.nodes.pop(op)
        self.graph.remove_node(op)

    def add_node(self, op : Operation) -> int:
        """
        adds an operation and gives it an identifier
        """
        ZX_DAG._index_generator += 1
        self.nodes[ZX_DAG._index_generator] = op
        self.graph.add_node(ZX_DAG._index_generator, color="green" if self._is_z_gate(op) else "red", gate = "Z" if self._is_z_gate(op) else "X")
        return ZX_DAG._index_generator

    def is_root_z(self, op: int) -> bool:
        """
        returns true if operation is a z gate with no predecessor
        """
        predecessor = self.predecessors(op)
        if len(predecessor) >= 1: return False
        return self._is_z_gate(self.nodes[op])

    def is_leaf_z(self, op: int) -> bool:
        """
        returns true if operation is a z gate with no successor
        """
        successors = self.successors(op)
        if len(successors) >= 1: return False
        return self._is_z_gate(self.nodes[op])

    def can_merge_with_successor(self, op : int) -> bool:
        """
        return true if there is only one successor and successor is on same axis (RX | RZ)
        """
        successors = self.successors(op)
        if len(successors) != 1:
            return False
        return ZX_DAG._can_merge(self.nodes[op], self.nodes[successors[0]])
    
    def merge_with_successor(self, a : int):
        """
        merges two rotation operations together
        assumes mergeability has been verified beforehand
        """

        successors = self.successors(a)
        b = successors[0]

        wires = self.nodes[a].wires

        b_succ = [succ for succ in self.graph.successors(b) if any([w in self.nodes[succ].wires for w in wires])]
        a_pred = [pred for pred in self.graph.predecessors(a) if any([w in self.nodes[pred].wires for w in wires])]
        
        a_b_node = ZX_DAG._merge(self.nodes[a], self.nodes[b])
        a_b_node.index_for_placement = f"{self.nodes[a].index_for_placement}+{self.nodes[b].index_for_placement}"

        self.remove_node(a)
        self.remove_node(b)

        a_b = self.add_node(a_b_node)
        
        self.graph.add_edges_from([pred, a_b] for pred in a_pred)
        self.graph.add_edges_from([a_b, succ] for succ in b_succ)
    
    def can_commute_with_successor(self, op : int):
        """
        returns true if operation is a z rotation gate and successor is a cz gate
        other commutation scenarios are discarded to limit redundant commutations
        """
        successors = self.successors(op)
        if len(successors) != 1:
            return False
        return ZX_DAG._can_commute(self.nodes[op], self.nodes[successors[0]])

    def commute_with_successor(self, a : int):
        """
        swaps operation and it's successor.
        assumes commutativity has been verified beforehand
        """

        succ = self.successors(a)
        b = succ[0]

        wire = self.nodes[a].wires[0] # there is only one wire
        
        b_succ = [succ for succ in self.graph.successors(b) if wire in self.nodes[succ].wires]
        a_pred = [pred for pred in self.graph.predecessors(a) if wire in self.nodes[pred].wires]
        # remove all links to successors of b if on same wire
        self.graph.remove_edges_from([b, succ] for succ in b_succ)

        # remove all links to predecessors of a if on same wire
        self.graph.remove_edges_from([pred, a] for pred in a_pred)

        # remove link between a and b
        self.graph.remove_edge(a, b)

        self.graph.add_edge(b, a)
        self.graph.add_edges_from([pred, b] for pred in a_pred)
        self.graph.add_edges_from([a, succ] for succ in b_succ)

    def is_trivial(self, a : int, epsilon = 1E-8):
        op = self.nodes[a]
        return len(op.parameters) > 0 and abs(op.parameters[0]) % (np.pi * 2) < epsilon

    def remove_trivial(self, a : int):
        preds = self.predecessors(a)
        succs = self.successors(a)

        self.remove_node(a)

        self.graph.add_edges_from([pred, succ] for pred in preds for succ in succs if ZX_DAG._is_on_same_wire(self.nodes[pred], self.nodes[succ]))
        # for pred in preds:
        #     for succ in succs:
        #         if ZX_DAG._is_on_same_wire(self.nodes[pred], self.nodes[succ]):
        #             self.graph.add_edge(pred, succ)

    def apply_step(self, condition, action):
        repeat = True
        while repeat:
                repeat = False
                i = 0
                while i < self.graph.number_of_nodes():
                    n = list(self.nodes)[i]
                    if condition(n):
                        action(n)
                        repeat = True
                        continue
                    i += 1

    def simplify(self, iterations = 10) -> "ZX_DAG":
        """
        eliminates z root and leaf operations, right-commutes and merges rotations until max iteration is reached, or no change is added
        """     

        for _ in range(iterations):
            comparison = deepcopy(self.graph)
            # remove all z axis roots and leaves
            self.apply_step(self.is_root_z, self.remove_node)
            self.apply_step(self.is_leaf_z, self.remove_node)
            
            # commute and merge
            
            # TODO : make this work
            self.apply_step(self.can_commute_with_successor, self.commute_with_successor)
            self.apply_step(self.can_merge_with_successor, self.merge_with_successor)
            # self.apply_step(self.is_trivial, self.remove_trivial)
            

            if comparison.nodes == self.graph.nodes and comparison.edges == self.graph.edges:
                break;
            
        
        return self

    def to_tape(self) -> QuantumTape:
        generations = nx.topological_generations(self.graph)
        new_operations = []
        for gen in generations:
            for i in gen:
                new_operations.append(self.nodes[i])
        
        return type(self.tape)(new_operations, self.tape.measurements, self.tape.shots)


def zx_dag_optimisation(tape : QuantumTape):
    """
    uses a dag containing only RZ, RX and CZ gates to optimize by commuting and merging rotations. 
    deprecated : this should not be used
    """
    
    dag = ZX_DAG(tape)
    
    dag = dag.simplify(20)
    
    if DEBUG:
        from optimization_utility import to_qasm
        print(to_qasm(dag.to_tape(), False))

    return tape

def zx_calculus(tape : QuantumTape) -> QuantumTape:
    import pyzx
    from pennylane.transforms import to_zx, from_zx
    
    g = to_zx(tape)
    pyzx.simplify.teleport_reduce(g)
    test = from_zx(g)
    return test
