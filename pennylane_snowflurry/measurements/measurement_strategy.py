from abc import ABC, abstractmethod


class MeasurementStrategy:

    def __init__(self):
        from pennylane_snowflurry.pennylane_converter import Snowflurry
        self.Snowflurry = Snowflurry

    def measure(self, converter, mp, shots):
        pass
