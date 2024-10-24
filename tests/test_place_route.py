from pennylane_snowflurry.transpiler.placement import placement_astar
from pennylane_snowflurry.transpiler.routing import swap_routing
import unittest
import pennylane as qml
from pennylane.tape import QuantumTape

class test_place_route(unittest.TestCase):
    def test_place_trivial(self):
        answer = [4, 0, 1, 8, 9]
        ops = [qml.CNOT([0, 1]), 
               qml.CNOT([0, 2]), 
               qml.CNOT([0, 3]), 
               qml.CNOT([0, 4])]

        tape = QuantumTape(ops=ops)
        new_tape = placement_astar(tape, False)
        self.assertListEqual(sorted(answer), sorted(int(w) for w in new_tape.wires))

    def test_place_too_connected(self):
        answer = [4, 0, 9, 8, 1, 5]
        ops = [qml.CNOT([0, 1]), 
               qml.CNOT([0, 2]), 
               qml.CNOT([0, 3]), 
               qml.CNOT([0, 4]), 
               qml.CNOT([0, 5])]

        tape = QuantumTape(ops=ops)
        new_tape = placement_astar(tape, False)
        self.assertListEqual(sorted(answer), sorted(int(w) for w in new_tape.wires))

    def test_route_trivial(self):
        ops = [qml.CNOT([0, 4])]
        tape = QuantumTape(ops=ops)
        new_tape = swap_routing(tape, False)
        self.assertListEqual(ops, new_tape.operations)

    def test_route_distance1(self):
        results = [qml.SWAP([4, 1]), qml.CNOT([0, 4]), qml.SWAP([4, 1])]
        ops = [qml.CNOT([0, 1])]
        tape = QuantumTape(ops=ops)
        new_tape = swap_routing(tape, False)
        self.assertListEqual(results, new_tape.operations)

    def test_route_distance2(self):
        results = [qml.SWAP([1, 5]), qml.SWAP([4, 1]), qml.CNOT([0, 4]), qml.SWAP([4, 1]), qml.SWAP([1, 5])]
        ops = [qml.CNOT([0, 5])]
        tape = QuantumTape(ops=ops)
        new_tape = swap_routing(tape, False)
        self.assertListEqual(results, new_tape.operations)

    def test_route_impossible_loop(self):
        results = [qml.CNOT([4, 1]), qml.CNOT([1, 5]), qml.SWAP([1, 4]), qml.CNOT([5, 1]), qml.SWAP([1, 4])]
        ops = [qml.CNOT([4, 1]), 
               qml.CNOT([1, 5]), 
               qml.CNOT([5, 4])]
        tape = QuantumTape(ops=ops)
        new_tape = swap_routing(tape, False)
        self.assertListEqual(results, new_tape.operations)

if __name__ == "__main__":
    unittest.main()