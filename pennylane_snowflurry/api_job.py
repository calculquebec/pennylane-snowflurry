from pennylane.tape import QuantumTape
import json
import time
from pennylane_snowflurry.api_adapter import ApiAdapter
from pennylane_snowflurry.api_utility import ApiUtility

class JobException(Exception):
    def __init__(self, message : str):
        self.message = message
    
    def __str__(self): self.message
    
class Job:
    host : str
    user : str
    access_token : str
    realm : str
    
    def __init__(self, circuit : QuantumTape, circuit_name = None):
        self.adapter = ApiAdapter()
        self.circuit_dict = ApiUtility.convert_circuit(circuit)
        self.circuit_name = circuit_name if circuit_name is not None else "default"
        self.shots = circuit.shots.total_shots

    def run(self, max_tries : int = -1):
        """
        converts a quantum tape into a dictionary, readable by thunderhead
        creates a job on thunderhead
        fetches the result until the job is successfull, and returns the result
        """

        if max_tries == -1: max_tries = 2 ** 15
        if circuit_name is None: circuit_name = "default"

        response = self.adapter.create_job(self.circuit_dict, self.circuit_name, self.shots)
        
        if(response.status_code == 200):
            current_status = ""
            job_id = json.loads(response.text)["job"]["id"]
            for i in range(max_tries):
                time.sleep(0.2)
                response = self.adapter.job_by_id(job_id)

                if response.status_code != 200: 
                    continue

                content = json.loads(response.text)
                status = content["job"]["status"]["type"]
                if(current_status != status):
                    current_status = status

                if(status != "SUCCEEDED"): 
                    continue

                return content["result"]["histogram"]
            raise JobException("Couldn't finish job. Stuck on status : " + str(current_status))
        else:
            raise JobException(response.text)
    
    

