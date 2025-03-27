import argparse
import os
import numpy as np
from qiskit.quantum_info import random_unitary
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.compiler import transpile
from tqdm import tqdm
import pandas as pd

from interface import get_unitary_with_qiskit, contract_cppsim, replace_swap_with_unitary, unify_qubit_registers

def pick_random_qubits(num_qubits, seed):
    # choose a random number of qubits from 1 to num_qubits
    num_qubits_chosen = np.random.randint(1, num_qubits + 1)
    # choose a random set of qubits
    qubits = np.random.choice(num_qubits, num_qubits_chosen, replace=False)
    qubits = sorted([int(q) for q in qubits])
    return qubits, num_qubits_chosen

def create_random_unitary(num_qubits_chosen, seed):
    return random_unitary(2**num_qubits_chosen, seed=seed)

def get_error(matrix1, matrix2):
    diff = matrix1 - matrix2
    error = np.sqrt(np.sum(np.abs(diff) ** 2))
    return error

def generate_random_qiskit_circuit(num_qubits, depth, seed):
    # create a circuit with a gate spanning all 10 qubits and one spanning the first 9
    circuit = QuantumCircuit(num_qubits)

    np.random.seed(seed)

    circuit = random_circuit(num_qubits, depth, max_operands=4)

    circuit.data = circuit.data[:depth]
    
    return circuit

def generate_random_unitary_circuit(num_qubits, depth, seed):
    # create a circuit with a gate spanning all 10 qubits and one spanning the first 9
    circuit = QuantumCircuit(num_qubits)

    np.random.seed(seed)

    ## create 100 random unitaries
    for _ in range(depth):
        qubits_chosen, num_qubits_chosen = pick_random_qubits(num_qubits, seed)
        # qubits_chosen = list(range(num_qubits))
        # num_qubits_chosen = num_qubits
        circuit.unitary(create_random_unitary(num_qubits_chosen, seed), qubits_chosen)
    return circuit


def circuit_contraction(qcs, sanity_check, output_file, filenames=['rcs']):
    # create df with additional columns for filename and standard deviations
    df = pd.DataFrame(columns=['filename', 'num_qubits', 'num_gates', 'qiskit_time', 'qiskit_std', 'cpp_time', 'cpp_std', 'error'], dtype=float)

    for i, qc in enumerate(qcs):
        qc = replace_swap_with_unitary(qc)

        qc = unify_qubit_registers(qc)

        # strip the circuit of any measurements
        qc = qc.copy()
        qc.remove_final_measurements()

        num_ops = sum(qc.count_ops().values())

        if qc.num_qubits > 10:
            print(f"Skipping circuit with {qc.num_qubits} qubits and {num_ops} gates from {filenames[i]}")
            continue

        # Run each method 10 times and calculate averages
        qiskit_times = []
        cpp_times = []

        print(f"Running circuit with {qc.num_qubits} qubits and {num_ops} gates from {filenames[i]}")

        # print(f"Running circuit with {qc.num_qubits} qubits and {qc.count_ops().values()} depth from {filenames[i]}")

        n_samples = 1 if sanity_check else 10
        
        for _ in tqdm(range(n_samples)):
            if not sanity_check:
                if qc.num_qubits <= 10:
                    try:
                        unitary_matrix, time_ms = get_unitary_with_qiskit(qc)
                        qiskit_times.append(time_ms)
                    except Exception as e:
                        print(f"Error processing circuit:\n{qc}")
                        print(f"Error message: {str(e)}")
                        raise

            unitary_matrix_cpp, time_ms = contract_cppsim(qc)
            cpp_times.append(time_ms)

        if n_samples > 2 and not sanity_check:
            # remove best and worst times
            if qc.num_qubits <= 10:
                qiskit_times.remove(max(qiskit_times))
                qiskit_times.remove(min(qiskit_times))
            cpp_times.remove(max(cpp_times))
            cpp_times.remove(min(cpp_times))

        execution_time_ms_cpp = np.mean(cpp_times)
        cpp_std = np.std(cpp_times) if len(cpp_times) > 1 else None

        # append a new row to the df
        if qc.num_qubits <= 10 and not sanity_check:
            error = get_error(unitary_matrix, unitary_matrix_cpp)
            execution_time_ms_qiskit = np.mean(qiskit_times)
            qiskit_std = np.std(qiskit_times) if len(qiskit_times) > 1 else None
        else:
            error = None
            execution_time_ms_qiskit = None
            qiskit_std = None

        new_row = {
            'filename': filenames[i],
            'num_qubits': qc.num_qubits,
            'num_gates': num_ops,
            'qiskit_time': execution_time_ms_qiskit,
            'qiskit_std': qiskit_std,
            'cpp_time': execution_time_ms_cpp,
            'cpp_std': cpp_std,
            'error': error
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(output_file, index=False)
    return df

def parse_args():
    parser = argparse.ArgumentParser(description='Circuit contraction test')
    parser.add_argument('--test', type=str, choices=['rcs', 'rcs_unitary', 'literature'], 
                        required=True, help='Test type to run')
    parser.add_argument('--n_qubits', type=int, help='Number of qubits (required for rcs and rcs_unitary)')
    parser.add_argument('--depth', type=int, help='Circuit depth (required for rcs and rcs_unitary)')
    parser.add_argument('--folder', type=str, help='Folder path for literature test circuits')
    parser.add_argument('--sanity', action='store_true', help='Run sanity check without time comparison')
    args = parser.parse_args()
    
    if args.test == 'literature' and not args.folder:
        parser.error("--folder is required when test type is 'literature'")
    
    if args.test in ['rcs', 'rcs_unitary']:
        if not args.n_qubits:
            parser.error("--n_qubits is required for rcs and rcs_unitary tests")
        if not args.depth:
            parser.error("--depth is required for rcs and rcs_unitary tests")
    
    return args

if __name__ == '__main__':
    args = parse_args()
    qcs = []
    filenames = []
    if args.test == 'literature': 
        for filename in sorted(os.listdir(args.folder)):
            if filename.endswith('.qasm'):
                qcs.append(QuantumCircuit.from_qasm_file(os.path.join(args.folder, filename)))
                filenames.append(filename)
    else:
        # Add tqdm progress bar for circuit generation
        for n in tqdm(
            range(1, args.n_qubits + 1, 1),
            desc="Generating circuits for testing",
            unit="qubit"
        ):
            for d in range(100, args.depth + 50, 50):
                if args.test == 'rcs':
                    qcs.append(generate_random_qiskit_circuit(n, d, 0))
                    filenames.append(args.test)
                if args.test == 'rcs_unitary':
                    qcs.append(generate_random_unitary_circuit(n, d, 0))
                    filenames.append(args.test)

    output_file = f'nvidia_results/{args.test}_{"sanity" if args.sanity else "full"}.csv'
    circuit_contraction(qcs, args.sanity, output_file, filenames)