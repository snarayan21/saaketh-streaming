# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""shuffle quality metric based on batch composition of streams for input run yamls."""

import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import matplotlib.pyplot as plt
import numpy as np
from core.yaml_processing import create_simulation_dataset, ingest_yaml
from core.batch_composition_metrics import probability_l1_metric, probability_l2_metric, \
    get_stream_batch_probabilities

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate your training yaml from the command \
                                     line.')
    parser.add_argument('-f', '--file', type=str, help='path to yaml file', required=True)
    parser.add_argument('-n', '--nodes', type=int, help='number of physical nodes', required=False)
    parser.add_argument('-d',
                        '--devices',
                        type=int,
                        help='number of devices per node',
                        required=False)
    args = parser.parse_args()

    # Read in yaml file
    filepath = args.file
    total_devices, workers, max_duration, global_batch_size, train_dataset = \
        ingest_yaml(filepath=filepath)

    # Check if we have to ask for any parameters
    args = parser.parse_args()
    nodes = args.nodes
    if nodes is None:
        nodes = int(input('Number of physical nodes: '))
    # devices may be specified in the yaml file.
    if total_devices is None:
        devices = args.devices
    else:
        if total_devices % nodes != 0:
            raise ValueError('The number of devices must be divisible by the number of nodes.')
        devices = total_devices // nodes

    # Create SimulationDataset
    print('Constructing SimulationDataset...')
    dataset = create_simulation_dataset(nodes, devices, workers, global_batch_size, train_dataset)

    stream_probabilities, batch_probabilities = get_stream_batch_probabilities(dataset, max_duration)

    # Calculate shuffle quality metrics for each batch -- deviation from true stream probability distribution
    l1_metric = probability_l1_metric(stream_probabilities, batch_probabilities)
    l2_metric = probability_l2_metric(stream_probabilities, batch_probabilities)

    # plot the results
    num_batches = batch_probabilities.shape[0]
    plt.plot(np.arange(num_batches), l1_metric, label='L1 metric')
    plt.plot(np.arange(num_batches), l2_metric, label='L2 metric')
    plt.legend()
    plt.show()