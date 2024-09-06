from itertools import combinations
from squlearn.observables import CustomObservable, SinglePauli

def SumOneTwoQubitPauli(num_qubits, meas_str):
    measurements = []
    
    # one-qubit measurements
    for mstr in meas_str:
        for i in range(num_qubits):
            measurements.append(SinglePauli(num_qubits, i, op_str=mstr))
    
    # two-qubit measurements
    initial_string = 'I' * num_qubits
    positions = range(num_qubits)
    two_positions_combinations = list(combinations(positions, 2))
    for op_str in meas_str:
        for pos in two_positions_combinations:
            if pos[1] - pos[0] <= num_qubits - 1:
                measurements.append(CustomObservable(num_qubits=num_qubits,operator_string=''.join(initial_string[:pos[0]] + op_str + initial_string[pos[0]+1:pos[1]] + op_str + initial_string[pos[1]+1:])))
    return measurements

def TwoQubitPauli(num_qubits, meas_str):
    initial_string = 'I' * num_qubits
    # generate all possible positions to place op_str(= 'X', 'Y', 'Z')
    positions = range(num_qubits)
    two_positions_combinations = list(combinations(positions, 2))
    
    measurements = []
    for op_str in meas_str:
        for pos in two_positions_combinations:
            if pos[1] - pos[0] <= num_qubits - 1:
                measurements.append(CustomObservable(num_qubits=num_qubits,operator_string=''.join(initial_string[:pos[0]] + op_str + initial_string[pos[0]+1:pos[1]] + op_str + initial_string[pos[1]+1:])))
    return measurements

def DimensionlessTransverseFieldIsingHamiltonian(num_qubits: int):
    
    def gen_double_ising_str(i,j):
        H = 'I' * num_qubits
        H = H[i+1 :] + 'Z' + H[:i]
        if i != j:
            H = H[: num_qubits-j-1] + 'Z' + H[num_qubits - j :]
        return H
    
    def gen_single_ising_str(i):
        H = 'I' * num_qubits
        H = H[i+1 :] + 'Z' + H[:i]
        return H
    
    op_list = []
    for i in range(num_qubits):
        op_list.append(gen_single_ising_str(i))
    
    for i in range(num_qubits):
        for j in range(i):
            op_list.append(gen_double_ising_str(i,j))
    
    #return op_list
    return CustomObservable(num_qubits=num_qubits, operator_string=op_list)