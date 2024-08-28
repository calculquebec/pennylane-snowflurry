from abc import ABC, abstractmethod

class MeasurementProcess(ABC): # TODO find a different name for this class to avoid confusion with Pennylanes's MP

    @abstractmethod
    def measure(self):
        pass