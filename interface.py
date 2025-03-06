import numpy as np
import time
import sys

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import random_unitary
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
from numpy import ctypeslib

from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit

from ctypes import cdll
import ctypes

import opt_einsum as oe
import cupy as cp  # For GPU support

import cuquantum as cq

from tqdm import tqdm

# Define the complex number structure to match C++

class cgate(ctypes.Structure):
    _fields_ = [("qubits", ctypes.POINTER(ctypes.c_ubyte)),
                ("unitary", ctypes.POINTER(ctypes.c_double)),
                ("rank", ctypes.c_size_t)]

class Gate:
    def __init__(self, qubits, unitary, name=None):
        self.qubits = qubits  # span of qubits on which the gate acts
        self.unitary = unitary  # unitary matrix representing the gate
        self.name = name

    def __repr__(self):
        return f'Gate(qubits={self.qubits}, unitary=\n{self.unitary})'

def get_gate_list(qc):
    gates_list = []
    n_qubits = qc.num_qubits

    for instruction in qc.data:
        gate_operation = instruction.operation
        qubits = [qubit._index for qubit in instruction.qubits]
        gates_list.append(Gate(qubits, instruction.matrix, gate_operation.name))

    return gates_list

def replace_swap_with_unitary(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Replace SWAP, controlled-SWAP, X, CX, and CCX gates in a quantum circuit with custom unitary gates,
    preserving the original position of each gate.
    
    Args:
        circuit (QuantumCircuit): Input quantum circuit
        
    Returns:
        QuantumCircuit: Modified circuit with SWAP, CSWAP, X, CX, and CCX gates replaced by custom unitaries
    """
    # Define the SWAP unitary matrix
    swap_unitary = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]])
    
    # Define the X (NOT) gate unitary matrix
    x_unitary = np.array([[0, 1],
                         [1, 0]])
    
    # Create controlled version of the SWAP unitary (8x8 matrix)
    # Initialize with identity matrix
    cswap_unitary = np.eye(8)
    # When control qubit is |1⟩, apply SWAP
    cswap_unitary[4:8, 4:8] = swap_unitary
    
    # Create controlled-X (CNOT) unitary (4x4 matrix)
    # Initialize with identity matrix
    cx_unitary = np.eye(4)
    # When control qubit is |1⟩, apply X
    cx_unitary[2:4, 2:4] = x_unitary
    
    # Create controlled-controlled-X (Toffoli) unitary (8x8 matrix)
    # Initialize with identity matrix
    ccx_unitary = np.eye(8)
    # When both control qubits are |1⟩, apply X
    ccx_unitary[6:8, 6:8] = x_unitary
    
    # Create the custom gates
    custom_swap  = UnitaryGate(swap_unitary, label='CustomSWAP')
    custom_cswap = UnitaryGate(cswap_unitary, label='CustomCSWAP')
    custom_x     = UnitaryGate(x_unitary, label='CustomX')
    custom_cx    = UnitaryGate(cx_unitary, label='CustomCX')
    custom_ccx   = UnitaryGate(ccx_unitary, label='CustomCCX')
    
    # Create a new circuit with the same registers as the original circuit
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    
    # Iterate through the circuit instructions using the new attribute access method
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = instruction.qubits
        clbits = instruction.clbits
        
        if operation.name == 'swap':
            # Replace SWAP with custom unitary
            new_circuit.append(custom_swap, qubits)
        elif operation.name == 'cswap':
            # Replace controlled-SWAP with custom controlled unitary
            new_circuit.append(custom_cswap, qubits)
        elif operation.name == 'x':
            # Replace X (NOT) with custom unitary
            new_circuit.append(custom_x, qubits)
        elif operation.name == 'cx':
            # Replace CX (CNOT) with custom controlled unitary
            new_circuit.append(custom_cx, qubits)
        elif operation.name == 'ccx':
            # Replace CCX (Toffoli) with custom controlled-controlled unitary
            new_circuit.append(custom_ccx, qubits)
        else:
            # Keep other gates as they are
            new_circuit.append(operation, qubits, clbits)
    
    return new_circuit

def get_unitary_with_qiskit(qc):
    simulator = AerSimulator(method='unitary', device='GPU')

    transpiled_circuit = transpile(qc, simulator)
    transpiled_circuit.save_unitary()

    start_time = time.time()

    job = simulator.run(transpiled_circuit)
    result = job.result()
    unitary_matrix = result.get_unitary(transpiled_circuit)

    end_time = time.time()

    execution_time_ms = (end_time - start_time) * 1000

    return unitary_matrix, execution_time_ms

def contract_cppsim(qc):
    gate_list = get_gate_list(qc)
    num_qubits = qc.num_qubits

    unitary_matrix_cpp = np.zeros((2**num_qubits, 2**num_qubits), dtype=np.complex128)

    # Convert the gate list to a list of cpx objects
    gates = []
    for gate in gate_list:
        qubits = (ctypes.c_ubyte * len(gate.qubits))(*gate.qubits)
        rank = len(gate.qubits)
        
        unitary = np.ascontiguousarray(gate.unitary, dtype=np.complex128)
        
        gates.append(cgate(qubits, unitary.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), rank))

    lib = cdll.LoadLibrary('./lib/libTTC.so')

    # Set up function argument types
    lib.contract_circuit.argtypes = [
        ctypes.POINTER(cgate),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t
    ]

    start_time = time.time()

    # Call the C++ function
    lib.contract_circuit(
        (cgate * len(gates))(*gates),
        len(gates),
        unitary_matrix_cpp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        num_qubits
    )

    end_time = time.time()

    # Convert result back to numpy array
    unitary_matrix_cpp = unitary_matrix_cpp.reshape((2**num_qubits, 2**num_qubits))

    execution_time_ms = (end_time - start_time) * 1000

    return unitary_matrix_cpp, execution_time_ms

def unify_qubit_registers(original_qc):
     # Count how many total qubits are in the original circuit
    total_qubits = len(original_qc.qubits)

    # Create a single quantum register for all qubits
    new_qreg = QuantumRegister(total_qubits, 'q')

    # Recreate classical registers (same names, same sizes)
    new_cregs = []
    for creg in original_qc.cregs:
        cr = ClassicalRegister(creg.size, name=creg.name)
        new_cregs.append(cr)

    # Create the new circuit with one big Qreg + the original classical regs
    new_qc = QuantumCircuit(new_qreg, *new_cregs, name=original_qc.name)

    # Build a map from each old Qubit object -> new (qreg, index)
    qubit_map = {}
    for idx, qubit_obj in enumerate(original_qc.qubits):
        qubit_map[qubit_obj] = idx  # qubit_obj -> integer index

    # Build a map from each old Clbit object -> (the new ClassicalRegister, local_idx)
    # We do this by iterating over the old circuit's classical registers in order,
    # and hooking each clbit to the corresponding position in the new register.
    cbit_map = {}
    for creg in original_qc.cregs:
        # Find the matching new creg we just created
        new_creg = next(cr for cr in new_cregs if cr.name == creg.name and cr.size == creg.size)
        for idx in range(creg.size):
            # old cbit object: creg[idx]
            old_cbit_obj = creg[idx]
            cbit_map[old_cbit_obj] = (new_creg, idx)

    # Reconstruct all instructions
    for instr in original_qc.data:
        # 'instr' is a circuit instruction object
        operation = instr.operation
        old_qubits = instr.qubits
        old_cbits = instr.clbits

        # Map the qubits to the new register
        new_qargs = []
        for q in old_qubits:
            new_index = qubit_map[q]
            new_qargs.append(new_qreg[new_index])

        # Map the classical bits
        new_cargs = []
        for c in old_cbits:
            (mapped_creg, mapped_idx) = cbit_map[c]
            new_cargs.append(mapped_creg[mapped_idx])

        # Add the same operation to the new circuit
        new_qc.append(operation, new_qargs, new_cargs)

    return new_qc