from pennylane.tape import QuantumTape
import json
import time
from pennylane_snowflurry.api_adapter import ApiAdapter, internal

class JobException(Exception):
    def __init__(self, message : str):
        self.message = message
    
    def __str__(self): self.message
    
class Job:
    host : str
    user : str
    access_token : str
    realm : str
    
    def __init__(self, 
                 host = "", 
                 user = "", 
                 access_token = "", 
                 realm = "", ):
        self.adapter = ApiAdapter(host, user, access_token, realm)
        
    def verbosePrint(content, verbose : bool = False):
        if verbose:
            print(content)

    def run(self, circuit : QuantumTape, circuit_name : str, project_id : str, machine_name : str, verbose = False, max_tries = 100):
        """
        converts a quantum tape into a dictionary, readable by thunderhead
        creates a job on thunderhead
        fetches the result until the job is successfull, and returns the result
        """

        circuit_dict = internal.convert_circuit(circuit)

        response = None

        try:
            response = self.adapter.create_job(circuit_dict, 
                                               circuit_name, 
                                               project_id, 
                                               machine_name, 
                                               circuit.shots.total_shots)
        except Exception as e:
            print(e)
            raise e;
        
        if(response.status_code == 200):
            current_status = ""
            job_id = json.loads(response.text)["job"]["id"]
            Job.verbosePrint("sent job with id : " + job_id, verbose)
            for _ in range(max_tries):
                time.sleep(0.1)
                response = self.adapter.job_by_id(job_id)

                if response.status_code != 200: 
                    continue

                content = json.loads(response.text)
                status = content["job"]["status"]["type"]
                if(current_status != status):
                    current_status = status
                    Job.verbosePrint(current_status, verbose)

                if(status != "SUCCEEDED"): 
                    continue

                return content["result"]["histogram"]
            raise JobException("Couldn't finish job. Stuck on status : " + str(current_status))
        else:
            raise JobException(response.text)
    
    

