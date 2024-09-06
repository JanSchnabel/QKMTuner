import warnings

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, PauliTwoDesign

from squlearn.encoding_circuit import *

def separable_encoding_circuit(
    num_qubits: int,
    num_features: int,
    num_layers: int,
    gate: str
):
    
    circuit = QuantumCircuit(num_qubits)

    def h_rz_gate(theta, qubit):
        circuit.h(qubit)
        circuit.rz(theta, qubit)
    
    def rx_ry_rz_sequence(theta, qubit):
        circuit.rx(theta, qubit)
        circuit.ry(theta, qubit)
        circuit.rz(theta, qubit)
    
    def h_rx_ry_rz_sequence(theta, qubit):
        circuit.h(qubit)
        circuit.rx(theta, qubit)
        circuit.ry(theta, qubit)
        circuit.rz(theta, qubit)

    gate_mapping = {
        'rx': circuit.rx,
        'ry': circuit.ry,
        'rz': circuit.rz,
        'h_rz': h_rz_gate,
        'rx_ry_rz': rx_ry_rz_sequence,
        'h_rx_ry_rz': h_rx_ry_rz_sequence
    }

    rotation_func = gate_mapping.get(gate)
    if rotation_func is None:
        raise ValueError("Invalid rotation_gate value. Choose 'rx', 'ry', 'rz', or 'h_rz'")

    features = ParameterVector('x', num_features)
    for _ in range(num_layers):
        for i in range(num_qubits):
            rotation_func(features[i % num_features], i)
    
    return QiskitEncodingCircuit(circuit)

def separable_encoding_circuit_variant_2(
    num_qubits: int,
    num_features: int,
    num_layers: int,
    gate: str
):
    
    circuit = QuantumCircuit(num_qubits)

    def h_rz_gate(theta, qubit):
        circuit.h(qubit)
        circuit.rz(theta, qubit)
    
    def rx_ry_rz_sequence(theta, qubit):
        circuit.rx(theta, qubit)
        circuit.ry(theta, qubit)
        circuit.rz(theta, qubit)
    
    def h_rx_ry_rz_sequence(theta, qubit):
        circuit.h(qubit)
        circuit.rx(theta, qubit)
        circuit.ry(theta, qubit)
        circuit.rz(theta, qubit)

    gate_mapping = {
        'rx': circuit.rx,
        'ry': circuit.ry,
        'rz': circuit.rz,
        'h_rz': h_rz_gate,
        'rx_ry_rz': rx_ry_rz_sequence,
        'h_rx_ry_rz': h_rx_ry_rz_sequence
    }

    rotation_func = gate_mapping.get(gate)
    if rotation_func is None:
        raise ValueError("Invalid rotation_gate value. Choose 'rx', 'ry', 'rz', or 'h_rz'")

    features = ParameterVector('x', num_features)
    feature_offset = 0
    for _ in range(num_layers):
        for i in range(num_qubits):
            rotation_func(features[feature_offset % num_features], i)
            feature_offset += 1
    
    return QiskitEncodingCircuit(circuit)

def z_encoding_circuit(
    num_qubits: int,
    num_features: int,
    num_layers: int
):
    if num_qubits != num_features:
        raise ValueError("num_qubits has to be the same as num_features")
    
    return QiskitEncodingCircuit(ZFeatureMap, feature_dimension=num_features, reps=num_layers)

def hardware_efficient_encoding(
    num_qubits: int,
    num_features: int,
    num_layers: int,
    gate: str
):
    circuit = QuantumCircuit(num_qubits)

    def h_rz_gate(theta, qubit):
        circuit.h(qubit)
        circuit.rz(theta, qubit)
    
    def rx_ry_rz_sequence(theta, qubit):
        circuit.rx(theta, qubit)
        circuit.ry(theta, qubit)
        circuit.rz(theta, qubit)
    
    def h_rx_ry_rz_sequence(theta, qubit):
        circuit.h(qubit)
        circuit.rx(theta, qubit)
        circuit.ry(theta, qubit)
        circuit.rz(theta, qubit)

    gate_mapping = {
        'rx': circuit.rx,
        'ry': circuit.ry,
        'rz': circuit.rz,
        'h_rz': h_rz_gate,
        'rx_ry_rz': rx_ry_rz_sequence,
        'h_rx_ry_rz': h_rx_ry_rz_sequence
    }

    rotation_func = gate_mapping.get(gate)
    if rotation_func is None:
        raise ValueError("Invalid rotation_gate value. Choose 'rx', 'ry', 'rz', or 'h_rz'")

    features = ParameterVector('x', num_features)
    for _ in range(num_layers):
        for i in range(num_qubits):
            rotation_func(features[i % num_features], i)
        for i in range(num_qubits - 1):
            circuit.cx(i, i+1)

    return QiskitEncodingCircuit(circuit)

def hardware_efficient_encoding_variant_2(
    num_qubits: int,
    num_features: int,
    num_layers: int,
    gate: str
):
    circuit = QuantumCircuit(num_qubits)

    def h_rz_gate(theta, qubit):
        circuit.h(qubit)
        circuit.rz(theta, qubit)
    
    def rx_ry_rz_sequence(theta, qubit):
        circuit.rx(theta, qubit)
        circuit.ry(theta, qubit)
        circuit.rz(theta, qubit)
    
    def h_rx_ry_rz_sequence(theta, qubit):
        circuit.h(qubit)
        circuit.rx(theta, qubit)
        circuit.ry(theta, qubit)
        circuit.rz(theta, qubit)

    gate_mapping = {
        'rx': circuit.rx,
        'ry': circuit.ry,
        'rz': circuit.rz,
        'h_rz': h_rz_gate,
        'rx_ry_rz': rx_ry_rz_sequence,
        'h_rx_ry_rz': h_rx_ry_rz_sequence
    }

    rotation_func = gate_mapping.get(gate)
    if rotation_func is None:
        raise ValueError("Invalid rotation_gate value. Choose 'rx', 'ry', 'rz', or 'h_rz'")

    features = ParameterVector('x', num_features)
    feature_offset = 0
    for _ in range(num_layers):
        for i in range(num_qubits):
            rotation_func(features[feature_offset % num_features], i)
            feature_offset += 1
        for i in range(num_qubits - 1):
            circuit.cx(i, i+1)

    return QiskitEncodingCircuit(circuit)

def zz_encoding_circuit(
    num_qubits: int,
    num_features: int,
    num_layers: int
):
    if num_qubits != num_features:
        raise ValueError("num_qubits has to be the same as num_features")
    
    return QiskitEncodingCircuit(ZZFeatureMap, feature_dimension=num_features, reps=num_layers)