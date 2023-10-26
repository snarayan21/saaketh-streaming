# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Retrieve actual training data from shuffle experiment yamls."""

import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.yaml_processing import create_simulation_dataset, ingest_yaml
import pickle
from streaming.base import Stream
from llmfoundry.data.text_data import StreamingTextDataset
from llmfoundry.utils.builders import build_tokenizer
from torch.utils.data import DataLoader
import transformers
import numpy as np

# All shuffle experiment runs were on 2 nodes, 8 devices per node.
nodes = 2
devices = 8

# dataset parameters go here
device_batch_size = 512 # we only have one device (local CPU lol) so we get the whole global batch at once.
max_seq_len = 2048
num_canonical_nodes = 1
predownload = 1024
shuffle = True
shuffle_algo = 'py1br'
shuffle_block_size = 100000
shuffle_seed = 1
# all_streams = [
#     {'local': '/Users/saaketh.narayan/Desktop/yamls/shuffling_experiments/dataset_shards/wiki',
#     'remote': 'oci://mosaicml-internal-dataset-red-pajama/wikipedia/gptneox-tok/en/'},
#     {'local': '/Users/saaketh.narayan/Desktop/yamls/shuffling_experiments/dataset_shards/github',
#     'remote': 'oci://mosaicml-internal-dataset-red-pajama/github/gptneox-tok/'}
# ]
all_streams = [
    {'local': '/Users/saaketh.narayan/Desktop/yamls/shuffling_experiments/dataset_shards/onesample',
     'remote': 'oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/0pt8/train'}
]
streams = []
for streamdict in all_streams:
    streams.append(Stream(local=streamdict['local'], remote=streamdict['remote']))

# tokenizer parameters go here
tokenizer_name = "EleutherAI/gpt-neox-20b"
tokenizer_config = {
    'model_max_length': 2048
}
# build tokenizer
tokenizer = build_tokenizer(tokenizer_name, tokenizer_config)

# dataloader parameters go here
drop_last = True
mlm_probability = None

# Create StreamingTextDataset
dataset = StreamingTextDataset(
    tokenizer=tokenizer,
    streams=streams,
    batch_size=device_batch_size,
    max_seq_len=max_seq_len,
    num_canonical_nodes=num_canonical_nodes,
    predownload=predownload,
    shuffle=shuffle,
    shuffle_algo=shuffle_algo,
    shuffle_block_size=shuffle_block_size,
    shuffle_seed=shuffle_seed,
)

collate_fn = transformers.DataCollatorForLanguageModeling(
    tokenizer=dataset.tokenizer,
    mlm=mlm_probability is not None,
    mlm_probability=mlm_probability)

dataloader = DataLoader(
    dataset,
    batch_size=device_batch_size,
    drop_last=drop_last,
    pin_memory=True,
    timeout=0,
)

batches_buffer = []
file_num = 0
for step, batch in enumerate(dataloader):
    if step == 1:
        break
    print(f'Step {step}')
    print(batch)
    print(batch.dtype)

    

