import os
import sys
from typing import Dict

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.append(os.path.dirname(SCRIPT_DIR))

import networkx as nx
import numpy as np
from orquestra.integrations.qulacs.simulator import QulacsSimulator
from orquestra.opt.problems.maxcut import MaxCut
from orquestra.quantum.operators import PauliSum, PauliTerm

from src.data_generation import generate_data, prepare_cost_function, save_hamiltonians
from src.hamiltonian_generation import assign_random_weights, assign_weight_for_term
from src.utils import calculate_plot_extents, generate_timestamp_str
from src.metrics import roughness_fourier_sparsity_using_norms, roughness_tv_from_scan, igsd_from_scan

backend = QulacsSimulator()
scan_resolution = 101


def main():
    cost_period = [np.pi, np.pi]
    fourier_period = [2 * np.pi, 2 * np.pi]

    hamiltonians_dict: Dict[str, PauliSum] = {}
    for num_qubits in [3, 6, 12]:
        G_complete_graph = nx.complete_graph(num_qubits)
        weighted_MaxCut_hamil = MaxCut().get_hamiltonian(G_complete_graph)
        weighted_MaxCut_hamil = assign_weight_for_term(
            weighted_MaxCut_hamil, PauliTerm("I0"), 0
        )

        hamiltonians_dict[
            f"ham_MAXCUT_weighted_-10_to_10_qubits_{num_qubits}"
        ] = assign_random_weights(weighted_MaxCut_hamil)

    timestamp = generate_timestamp_str()

    save_hamiltonians(hamiltonians_dict)

    for file_label, hamiltonian in hamiltonians_dict.items():
        cost_function = prepare_cost_function(hamiltonian, backend)
        origin = np.array([0, 0])
        dir_x = np.array([1, 0])
        dir_y = np.array([0, 1])

        calculate_plot_extents(hamiltonian)

        scan2D_result, fourier_result, metrics_dict = generate_data(
            cost_function,
            origin=origin,
            dir_x=dir_x,
            dir_y=dir_y,
            file_label=file_label,
            scan_resolution=scan_resolution,
            cost_period=cost_period,
            fourier_period=fourier_period,
            timestamp_str=timestamp,
        )

        tv = roughness_tv_from_scan(scan2D_result)
        print("Total Variation")
        print(tv)

        fd = roughness_fourier_sparsity_using_norms(fourier_result)
        print("Fourier Density")
        print(fd)

        print("IGSD for Parameter gamma")
        igsd_param_0, igsd_param_1 = igsd_from_scan(scan2D_result)
        print("IGSD for Parameter gamma")
        print(igsd_param_0)
        print("IGSD for Parameter beta")
        print(igsd_param_1)


if __name__ == "__main__":
    main()
