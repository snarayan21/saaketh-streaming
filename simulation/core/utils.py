# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Peripheral functions for simulation functionality."""

import numpy as np
from core.sim_time import Time, TimeUnit
from core.simulation_dataset import SimulationDataset
from numpy.typing import NDArray
import math


def get_batches_epochs(dataset: SimulationDataset, max_duration: Time) -> tuple[int, int, int]:
    """Get batches per epoch, epochs, and total epochs from a Time object.

    Args:
        dataset (SimulationDataset): The dataset being simulated.
        max_duration (Time): The maximum duration, can be specified in yaml.

    Returns:
        tuple[int, int, int]: batches per epoch, epochs, and the total batches.
    """
    # get epochs, batches_per_epoch, and total_batches from a Time obect
    dataset_batches = dataset.get_num_batches()
    batches_per_epoch = dataset_batches
    epochs = 1
    total_batches = dataset_batches
    if max_duration.unit == TimeUnit.EPOCH:
        epochs = max_duration.value
        batches_per_epoch = dataset_batches
        total_batches = epochs * batches_per_epoch
    elif max_duration.unit == TimeUnit.BATCH:
        full_epochs = max_duration.value // dataset_batches
        # check if there is a partial epoch we should fulfill
        if max_duration.value % dataset_batches != 0:
            full_epochs += 1
        # make sure we don't simulate past the duration set.
        if max_duration.value < dataset_batches:
            batches_per_epoch = max_duration.value
        else:
            batches_per_epoch = dataset_batches
        total_batches = max_duration.value
    else:
        raise ValueError('Simulator currently only supports max_duration in epochs or batches.')

    return batches_per_epoch, epochs, total_batches


def get_total_batches(dataset: SimulationDataset, max_duration: Time) -> int:
    """Get total batches from a Time object.

    Args:
        dataset (SimulationDataset): The dataset being simulated.
        max_duration (Time): The maximum duration, can be specified in yaml.

    Returns:
        int: The total batches.
    """
    dataset_batches = dataset.get_num_batches()
    total_batches = dataset_batches
    if max_duration.unit == TimeUnit.EPOCH:
        epochs = max_duration.value
        batches_per_epoch = dataset_batches
        total_batches = epochs * batches_per_epoch
    elif max_duration.unit == TimeUnit.BATCH:
        total_batches = max_duration.value
    else:
        raise ValueError('Simulator currently only supports max_duration in epochs or batches.')

    return total_batches


def remove_padded_samples(samples: NDArray) -> NDArray:
    """Remove padded samples from a batch.

    Args:
        samples (NDArray): The samples to remove padded samples from.

    Returns:
        NDArray: The samples with padded samples removed.
    """
    return np.delete(samples, np.where(samples == -1))


def bytes_to_time(bytes: int, bandwidth: int) -> float:
    """Convert bytes to time.

    Args:
        bytes (int): The bytes to convert.
        bandwidth (int): The bandwidth available.

    Returns:
        float: The time it takes to transfer the bytes.
    """
    return bytes / bandwidth


def time_to_bytes(time: float, bandwidth: int) -> int:
    """Convert time to bytes.

    Args:
        time (float): The time to convert.
        bandwidth (int): The bandwidth available.

    Returns:
        int: The bytes transferred in the time.
    """
    return int(time * bandwidth)


def get_rolling_avg_throughput(step_times: NDArray, window: int = 10) -> NDArray:
    """Get rolling average throughput from step times.

    Args:
        step_times (NDArray): time per step, as calculated by simulation
        window (int): window size for rolling average

    Returns:
        NDArray: rolling average throughput
    """
    step_times_rolling_avg = np.convolve(step_times, np.ones(window) / window, mode='valid')
    batch_throughput_rolling_avg = 1 / step_times_rolling_avg
    batch_throughput_rolling_avg = np.concatenate(
        (np.array([0] * (window - 1)), batch_throughput_rolling_avg))

    return batch_throughput_rolling_avg


def get_simulation_stats(step_times: NDArray, time_per_sample: float,
                         device_batch_size: int) -> tuple[int, float, int, int]:
    """Gets simulation stats for web UI.

    Args:
        step_times (NDArray): time per step, as calculated by simulation
        time_per_sample (float): time to process one sample on one device (seconds)
        device_batch_size (int): batch size per device

    Returns:
        tuple[int, float, int, int]: number of steps with throughput drops, time till warmup,
            step number of warmup, number of steps with throughput drops after warmup
    """
    # calculate percent of download-limited steps
    min_step_time = time_per_sample * device_batch_size
    all_throughput_drops = int(np.count_nonzero(step_times > (min_step_time)))

    epsilon = 1e-6

    # calculate warmup time (time to first max possible rolling average throughput) within epsilon
    max_throughput = 1 / min_step_time
    rolling_avg_throughput = get_rolling_avg_throughput(step_times)
    if np.max(rolling_avg_throughput) >= max_throughput - epsilon:
        warmup_step = int(np.argmax(rolling_avg_throughput >= (max_throughput)) + 1)
        warmup_time = float(np.sum(step_times[:warmup_step]))
    else:
        # we never hit the max possible throughput
        warmup_step = int(rolling_avg_throughput.shape[0])
        warmup_time = float(np.sum(step_times))

    # see if there are throughput drops after warmup so we can notify users
    if warmup_step != rolling_avg_throughput.shape[0]:
        # if we did hit the max throughput then we check for later drops
        post_warmup_tp_drops = int(np.count_nonzero(step_times[warmup_step:] > min_step_time))
    else:
        # since warmup was the whole time, there are no post-warmup throughput drops
        post_warmup_tp_drops = 0

    return all_throughput_drops, warmup_time, warmup_step, post_warmup_tp_drops

def exponential_weighted_moving_average(data, alpha, row_size=None, dtype=None, order='C', out=None):
    """
    Reshapes data before calculating EWMA, then iterates once over the rows
    to calculate the offset without precision issues
    :param data: Input data, will be flattened.
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param row_size: int, optional
        The row size to use in the computation. High row sizes need higher precision,
        low values will impact performance. The optimal value depends on the
        platform and the alpha being used. Higher alpha values require lower
        row size. Default depends on dtype.
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the desired output. If not provided or `None`,
        a freshly-allocated array is returned.
    :return: The flattened result.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = float
    else:
        dtype = np.dtype(dtype)

    row_size = int(row_size) if row_size is not None else get_max_row_size(alpha, dtype)

    if data.size <= row_size:
        # The normal function can handle this input, use that
        return ewma_vectorized(data, alpha, dtype=dtype, order=order, out=out)
    
    if data.ndim > 1:
        # flatten input
        data = np.reshape(data, -1, order=order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    row_n = int(data.size // row_size)  # the number of rows to use
    trailing_n = int(data.size % row_size)  # the amount of data leftover
    first_offset = data[0]

    if trailing_n > 0:
        # set temporary results to slice view of out parameter
        out_main_view = np.reshape(out[:-trailing_n], (row_n, row_size))
        data_main_view = np.reshape(data[:-trailing_n], (row_n, row_size))
    else:
        out_main_view = out
        data_main_view = data

    # get all the scaled cumulative sums with 0 offset
    ewma_vectorized_2d(data_main_view, alpha, axis=1, offset=0, dtype=dtype,
                       order='C', out=out_main_view)

    scaling_factors = (1 - alpha) ** np.arange(1, row_size + 1)
    last_scaling_factor = scaling_factors[-1]

    # create offset array
    offsets = np.empty(out_main_view.shape[0], dtype=dtype)
    offsets[0] = first_offset
    # iteratively calculate offset for each row
    for i in range(1, out_main_view.shape[0]):
        offsets[i] = offsets[i - 1] * last_scaling_factor + out_main_view[i - 1, -1]

    # add the offsets to the result
    out_main_view += offsets[:, np.newaxis] * scaling_factors[np.newaxis, :]

    if trailing_n > 0:
        # process trailing data in the 2nd slice of the out parameter
        ewma_vectorized(data[-trailing_n:], alpha, offset=out_main_view[-1, -1],
                        dtype=dtype, order='C', out=out[-trailing_n:])
    return out
    
def ewma_vectorized(data, alpha, offset=None, dtype=None, order='C', out=None):
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out

def ewma_vectorized_2d(data, alpha, axis=None, offset=None, dtype=None, order='C', out=None):
    """
    Calculates the exponential moving average over a given axis.
    :param data: Input data, must be 1D or 2D array.
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param axis: The axis to apply the moving average on.
        If axis==None, the data is flattened.
    :param offset: optional
        The offset for the moving average. Must be scalar or a
        vector with one element for each row of data. If set to None,
        defaults to the first value of each row.
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Ignored if axis is not None.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the desired output. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    assert data.ndim <= 2

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if axis is None or data.ndim < 2:
        # use 1D version
        if isinstance(offset, np.ndarray):
            offset = offset[0]
        return ewma_vectorized(data, alpha, offset, dtype=dtype, order=order,
                               out=out)

    assert -data.ndim <= axis < data.ndim

    # create reshaped data views
    out_view = out
    if axis < 0:
        axis = data.ndim - int(axis)

    if axis == 0:
        # transpose data views so columns are treated as rows
        data = data.T
        out_view = out_view.T

    if offset is None:
        # use the first element of each row as the offset
        offset = np.copy(data[:, 0])
    elif np.size(offset) == 1:
        offset = np.reshape(offset, (1,))

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # calculate the moving average
    row_size = data.shape[1]
    row_n = data.shape[0]
    scaling_factors = np.power(1. - alpha, np.arange(row_size + 1, dtype=dtype),
                               dtype=dtype)
    # create a scaled cumulative sum array
    np.multiply(
        data,
        np.multiply(alpha * scaling_factors[-2], np.ones((row_n, 1), dtype=dtype),
                    dtype=dtype)
        / scaling_factors[np.newaxis, :-1],
        dtype=dtype, out=out_view
    )
    np.cumsum(out_view, axis=1, dtype=dtype, out=out_view)
    out_view /= scaling_factors[np.newaxis, -2::-1]

    if not (np.size(offset) == 1 and offset == 0):
        offset = offset.astype(dtype, copy=False)
        # add the offsets to the scaled cumulative sums
        out_view += offset[:, np.newaxis] * scaling_factors[np.newaxis, 1:]

    return out

def get_max_row_size(alpha, dtype=float):
    assert 0. <= alpha < 1.
    # This will return the maximum row size possible on 
    # your platform for the given dtype. I can find no impact on accuracy
    # at this value on my machine.
    # Might not be the optimal value for speed, which is hard to predict
    # due to numpy's optimizations
    # Use np.finfo(dtype).eps if you  are worried about accuracy
    # and want to be extra safe.
    epsilon = np.finfo(dtype).tiny
    # If this produces an OverflowError, make epsilon larger
    return int(np.log(epsilon)/np.log(1-alpha)) + 1


def lr_schedule_cosine_warmup(alpha_f: float, warmup_steps: int, max_steps: int, step: int):
    """Cosine learning rate multiplier with warmup, at a particular step.

    Args:
        alpha_f (float): Final learning rate.
        warmup_steps (int): Number of warmup steps.
        max_steps (int): Max number of steps.
        step (int): Current step.

    Returns:
        float: learning rate at that step.
    """
    if step < warmup_steps:
        return (step+1) / (warmup_steps+1)
    else:
        tau_w = (step - warmup_steps) / max_steps
        return alpha_f + ((1-alpha_f) * 0.5*(1 + math.cos(math.pi * tau_w)))