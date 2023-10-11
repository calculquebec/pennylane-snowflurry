import unittest
from pennylane_snowflurry.snowflurry_device import SnowflurryQubitDevice
from julia import Snowflurry
from julia import Base
from julia import Main
class TestSnowflurryQubitDeviceInitialization(unittest.TestCase):
    def test_initialization(self):
        # Example parameters
        num_wires = 4
        num_shots = 1000
        seed = 42

        # Create an instance of the SnowflurryQubitDevice
        device = SnowflurryQubitDevice(wires=num_wires, shots=num_shots, seed=seed)

        self.assertEqual(device.num_wires, num_wires)

        self.assertEqual(device.shots, num_shots)

    def test_hadamard_pyjulia(self):
        c = Snowflurry.QuantumCircuit(qubit_count=3)
        j = julia.julia()
        j.eval("jl.eval('push!(c,hadamard(1))')")



if __name__ == '__main__':
    unittest.main()
