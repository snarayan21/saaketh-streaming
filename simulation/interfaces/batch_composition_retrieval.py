# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Retrieve and save stream and batch probabilities from shuffle experiment yamls."""

import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.yaml_processing import create_simulation_dataset, ingest_yaml
from core.batch_composition_metrics import get_stream_batch_probabilities
import pickle

# All shuffle experiment runs were on 2 nodes, 8 devices per node.
nodes = 2
devices = 8

experiment_names_filepaths = [
    ("mpt-1b-py1e-NCN16-SBS1000000-stratified-1","/Users/saaketh.narayan/Desktop/yamls/shuffling_experiments/stratified_py1e/NCN16-SBS1000000/mpt-1b-py1e-NCN16-SBS1000000-stratified-1.yaml"),
    ]

for experiment_name, filepath in experiment_names_filepaths:

    # read in yaml file
    total_devices, workers, max_duration, global_batch_size, train_dataset = \
        ingest_yaml(filepath=filepath, nodes=nodes, devices=devices)
    # Create SimulationDataset
    print(f'Constructing SimulationDataset for {experiment_name}...')
    dataset = create_simulation_dataset(nodes, devices, workers, global_batch_size, train_dataset)
    stream_probabilities, batch_probabilities = get_stream_batch_probabilities(dataset, max_duration)

    probabilities = {'stream_probabilities': stream_probabilities,
                     'batch_probabilities': batch_probabilities}
    prob_file = open(f'./stream_batch_probabilities/probs_{experiment_name}', 'wb')
    pickle.dump(probabilities, prob_file)
    prob_file.close()