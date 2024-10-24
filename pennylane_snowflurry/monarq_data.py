from dotenv import dotenv_values
from pennylane_snowflurry.api_adapter import ApiAdapter
from pennylane_snowflurry.api_utility import ApiUtility

"""
#       00
#       |
#    08-04-01
#    |  |  | 
# 16-12-09-05-02
# |  |  |  |  |
# 20-17-13-10-06-03
#    |  |  |  |  |
#    21-18-14-11-07
#       |  |  |
#       22-19-15
#          |
#          23
"""
connectivity = {
  ApiUtility.keys.qubits : [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 ],
  ApiUtility.keys.couplers : {
      "0": [0, 4],
      "1": [4, 1],
      "2": [1, 5],
      "3": [5, 2],
      "4": [2, 6],
      "5": [6, 3],
      "6": [3, 7],
      "7": [8, 4],
      "8": [4, 9],
      "9": [9, 5],
      "10": [5, 10],
      "11": [10, 6],
      "12": [6, 11],
      "13": [11, 7],
      "14": [8, 12],
      "15": [12, 9],
      "16": [9, 13],
      "17": [13, 10],
      "18": [10, 14],
      "19": [14, 11],
      "20": [11, 15],
      "21": [16, 12],
      "22": [12, 17],
      "23": [17, 13],
      "24": [13, 18],
      "25": [18, 14],
      "26": [14, 19],
      "27": [19, 15],
      "28": [16, 20],
      "29": [20, 17],
      "30": [17, 21],
      "31": [21, 18],
      "32": [18, 22],
      "33": [22, 19],
      "34": [19, 23]
  }
}

def build_benchmark():
    """
    creates a dictionary that contains unreliable qubits and couplers
    """
    config = dotenv_values(".env")
    q1acceptance = float(config["Q1_ACCEPTANCE"])
    q2acceptance = float(config["Q2_ACCEPTANCE"])

    # call to api to get qubit and couplers benchmark
    api = ApiAdapter()
    qubits_and_couplers = api.get_qubits_and_couplers()

    the_benchmark = { ApiUtility.keys.qubits : [], ApiUtility.keys.couplers : [] }

    for coupler_id in qubits_and_couplers[ApiUtility.keys.couplers]:
        benchmark_coupler = qubits_and_couplers[ApiUtility.keys.couplers][coupler_id]
        conn_coupler = connectivity[ApiUtility.keys.couplers][coupler_id]

        if benchmark_coupler[ApiUtility.keys.czGateFidelity] >= q2acceptance:
            continue

        the_benchmark[ApiUtility.keys.couplers].append(conn_coupler)

    for qubit_id in qubits_and_couplers[ApiUtility.keys.qubits]:
        benchmark_qubit = qubits_and_couplers[ApiUtility.keys.qubits][qubit_id]

        if benchmark_qubit[ApiUtility.keys.readoutState1Fidelity] >= q1acceptance:
            continue

        the_benchmark[ApiUtility.keys.qubits].append(int(qubit_id))
    return the_benchmark
