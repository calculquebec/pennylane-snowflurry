import unittest
from pennylane_snowflurry.snowflurry_device import SnowflurryQubitDevice

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


if __name__ == '__main__':
    unittest.main()
