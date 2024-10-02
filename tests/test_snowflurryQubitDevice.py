import unittest
from pennylane_snowflurry.snowflurry_device import SnowflurryQubitDevice
from juliacall import newmodule

# TODO : this namespace should be imported from the plugin
Snowflurry = newmodule("Snowflurry")
Snowflurry.seval("using Snowflurry")


class TestSnowflurryQubitDeviceInitialization(unittest.TestCase):
    def test_initialization(self):
        # Example parameters
        num_wires = 4
        num_shots = 1000
        seed = 42

        # Create an instance of the SnowflurryQubitDevice
        device = SnowflurryQubitDevice(wires=num_wires, shots=num_shots)

        self.assertEqual(device.num_wires, num_wires)

        self.assertEqual(device.shots.total_shots, num_shots)


    def test_hadamard_juliacall(self):
        Snowflurry.c = Snowflurry.QuantumCircuit(qubit_count=3)
        Snowflurry.seval("push!(c,hadamard(1))")


if __name__ == "__main__":
    unittest.main()
