# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Get token counts from dataset batches of actual training runs."""

import pickle
import numpy as np


batch_token_counts = None

vocab_size = 50368
num_batch_files = 14

for filenum in range(num_batch_files):
    print("processing file", filenum)
    batch_file = open(f'/Users/saaketh.narayan/Desktop/yamls/shuffling_experiments/dataset_shards/2stream_pickled/batch_file_{filenum}', 'rb')
    batch_tokens = pickle.load(batch_file)
    batch_file.close()

    # Count token occurrences in each batch
    steps, samples, seqlen = batch_tokens.shape
    batch_tokens = batch_tokens.reshape(steps, samples*seqlen)
    batch_tokens = np.apply_along_axis(lambda x: np.bincount(x, minlength=vocab_size), 1, batch_tokens)

    if batch_token_counts is None:
        batch_token_counts = batch_tokens
    else:
        batch_token_counts = np.vstack((batch_token_counts, batch_tokens))

token_counts_file = open(f'/Users/saaketh.narayan/Desktop/saaketh-streaming/simulation/interfaces/token_counts/2stream', 'wb')
pickle.dump(batch_token_counts, token_counts_file)
token_counts_file.close()