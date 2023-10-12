# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""shuffle quality metric based on batch composition of streams for input run yamls."""

import numpy as np
from numpy.typing import NDArray
from core.simulation_dataset import SimulationDataset
from core.utils import get_batches_epochs
from core.sim_time import Time

def probability_l1_metric(true_probabilities: NDArray, batch_probabilities: NDArray) -> NDArray:
    """Calculate the L1 metric between the true stream probability distribution and approximations.

    Args:
        true_probabilities (NDArray): true stream probability distribution
        batch_probabilities (NDArray): batch probabilities from streams

    Returns:
        NDArray: L1 shuffle quailty metric
    """
    return np.sum(np.abs(batch_probabilities - true_probabilities), axis=1)

def probability_l2_metric(true_probabilities: NDArray, batch_probabilities: NDArray) -> NDArray:
    """Calculate the L2 metric between the true stream probability distribution and approximations.

    Args:
        true_probabilities (NDArray): true stream probability distribution
        batch_probabilities (NDArray): batch probabilities from streams

    Returns:
        NDArray: L2 shuffle quality metric
    """
    return np.linalg.norm(batch_probabilities - true_probabilities, axis=1)

def probability_l1_metric_diffs(probability_diffs: NDArray) -> NDArray:
    """Calculate the L1 metric between the true stream probability distribution and approximations.

    Args:
        probability_diffs (NDArray): differences between two discrete probability distributions

    Returns:
        NDArray: L1 shuffle quailty metric
    """
    return np.sum(np.abs(probability_diffs), axis=1)

def probability_l2_metric_diffs(probability_diffs: NDArray) -> NDArray:
    """Calculate the L2 metric between the true stream probability distribution and approximations.

    Args:
        probability_diffs (NDArray): differences between two discrete probability distributions

    Returns:
        NDArray: L2 shuffle quality metric
    """
    return np.linalg.norm(probability_diffs, axis=1)

def get_stream_batch_probabilities(dataset: SimulationDataset, max_duration: Time):

    # Get the true stream probability distribution
    stream_probabilities = dataset.get_stream_probabilities()
    num_streams = len(stream_probabilities)

    # Iterate through dataset and get the stream composition of every batch.
    # Sample partition structured as (node, rank, worker, batches per worker, batch)
    sample_partition = dataset.get_sample_partition(0, 0)
    # Transpose the partition and reshape to (number of global batches, global batch size)
    global_batch_size = dataset.get_global_batch_size()
    global_batches = sample_partition.transpose(3, 2, 0, 1, 4).reshape(-1, global_batch_size)
    # Get sample to shard map and shard to stream map
    sample_to_shard = dataset.get_spanner()
    shard_to_stream = dataset.get_stream_per_shard()
    # Get batch compositions by stream for every batch
    _, _, total_batches = get_batches_epochs(dataset, max_duration)
    batch_compositions = []
    for i in range(total_batches):
        # Only get batch composition if sample does not have padding
        if i % 1000 == 0:
            print(f'Getting batch composition for batch {i}...')
        batch = global_batches[i]
        if np.any(batch == -1):
            continue
        # Get the streams in the batch
        batch_streams = [shard_to_stream[sample_to_shard[s]] for s in batch]
        # Get the number of samples in the batch from each stream
        batch_compositions.append(np.bincount(np.array(batch_streams), minlength=num_streams))
    batch_compositions = np.array(batch_compositions)

    # Turn batch compositions into stream probabilities per batch
    batch_probabilities = batch_compositions / global_batch_size
    
    return stream_probabilities, batch_probabilities
