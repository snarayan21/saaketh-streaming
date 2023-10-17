# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze batch tokens from actual training runs."""

import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle
import numpy as np
from core.utils import exponential_weighted_moving_average
from core.batch_composition_metrics import probability_l1_metric, probability_l2_metric, \
    probability_l1_metric_diffs, probability_l2_metric_diffs
import matplotlib.pyplot as plt

token_counts_file = open(f'/Users/saaketh.narayan/Desktop/saaketh-streaming/simulation/interfaces/token_counts/2stream', 'rb')
token_counts = pickle.load(token_counts_file)
token_counts_file.close()

token_probabilities = token_counts / np.sum(token_counts, axis=1, keepdims=True)

steps, vocab_size = token_counts.shape
print("max number of steps:", steps)

window = 100

# take exponential moving average of token counts over window
token_counts_ema = []
token_idx = 0
for token_composition in token_counts.T:
    if token_idx % 1000 == 0:
        print("processing token", token_idx)
    token_idx += 1
    token_counts_ema.append(exponential_weighted_moving_average(token_composition, 2/(window+1)))
token_counts_ema = np.array(token_counts_ema).T
token_probabilities_ema = token_counts_ema / np.sum(token_counts_ema, axis=1, keepdims=True)

token_probability_diffs_ema = token_probabilities[1:] - token_probabilities_ema[:-1]
token_probability_diffs_ema = np.insert(token_probability_diffs_ema, 0, token_probabilities[0], axis=0)
token_l1_metric_ema = probability_l1_metric_diffs(token_probability_diffs_ema)
token_l2_metric_ema = probability_l2_metric_diffs(token_probability_diffs_ema)

token_start = 0
token_end = 1000
start = 0
end = 5600

# plot some token probabilities over time
# plt.plot(np.arange(steps)[start:end], token_probabilities_ema[start:end, token_start:token_end])
# plt.show()

plt.plot(np.arange(len(token_l1_metric_ema))[start:end], token_l1_metric_ema[start:end])
# plt.plot(np.arange(len(token_l2_metric_ema))[start:end], token_l2_metric_ema[start:end])
plt.show()
