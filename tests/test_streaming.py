# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import math
import shutil
from typing import Any, Tuple

import pytest
from torch.utils.data import DataLoader

from streaming.base import StreamingDataLoader, StreamingDataset
from tests.common.utils import convert_to_mds


@pytest.mark.parametrize('batch_size', [4])
@pytest.mark.parametrize('seed', [2222])
@pytest.mark.parametrize('shuffle', [False])
@pytest.mark.parametrize('drop_last', [False, True])
@pytest.mark.parametrize('num_workers', [3, 6])
@pytest.mark.parametrize('num_canonical_nodes', [4, 8])
@pytest.mark.parametrize('epoch_size', [10, 200])
@pytest.mark.usefixtures('local_remote_dir')
def test_dataloader_epoch_size_no_streams(local_remote_dir: Tuple[str,
                                                                  str], batch_size: int, seed: int,
                                          shuffle: bool, drop_last: bool, num_workers: int,
                                          num_canonical_nodes: int, epoch_size: int):
    local, remote = local_remote_dir
    convert_to_mds(out_root=remote,
                   dataset_name='sequencedataset',
                   num_samples=117,
                   size_limit=1 << 8)

    # Build StreamingDataset
    dataset = StreamingDataset(local=local,
                               remote=remote,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed,
                               num_canonical_nodes=num_canonical_nodes,
                               epoch_size=epoch_size)

    # Build DataLoader
    dataloader = StreamingDataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     drop_last=drop_last)

    samples_seen = 0
    for batch in dataloader:
        print(batch['sample'])
        samples_seen += batch['sample'].size(dim=0)

    if epoch_size % num_canonical_nodes != 0:
        assert samples_seen == math.ceil(epoch_size / num_canonical_nodes) * num_canonical_nodes
    else:
        if drop_last:
            assert samples_seen == epoch_size - (epoch_size % batch_size)
        else:
            assert samples_seen == epoch_size



@pytest.mark.parametrize('batch_size', [4])
@pytest.mark.parametrize('seed', [2222])
@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('drop_last', [False])
@pytest.mark.parametrize('num_workers', [3, 6])
@pytest.mark.parametrize('num_canonical_nodes', [4, 8])
@pytest.mark.parametrize('epoch_size', [10, 200])
@pytest.mark.usefixtures('local_remote_dir')
def test_dataloader_fixed_sampling(local_remote_dir: Any, batch_size: int, seed: int,
                                   shuffle: bool, drop_last: bool, num_workers: int, num_canonical_nodes: int,
                                   epoch_size: int):
    remote_dir, local_dir = local_remote_dir
    convert_to_mds(out_root=remote_dir,
                   dataset_name='sequencedataset',
                   num_samples=117,
                   size_limit=1 << 8)

    # Build StreamingDataset
    dataset = StreamingDataset(local=local_dir,
                               remote=remote_dir,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed,
                               num_canonical_nodes=num_canonical_nodes,
                               epoch_size=epoch_size,
                               sampling_method='fixed')

    # Build DataLoader
    dataloader = StreamingDataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     drop_last=drop_last)

    # iterate once over the dataloader (first epoch)
    first_samples_seen = {}
    for batch in dataloader:
        for element in batch['sample']:
            int_element = int(element)
            if int_element in first_samples_seen:
                first_samples_seen[int_element] += 1
            else:
                first_samples_seen[int_element] = 1

    # check 2 more epochs to see if samples are the same
    first_samples_seen = {}
    for epoch in range(3):
        samples_seen = first_samples_seen if epoch == 0 else {}
        for batch in dataloader:
            for element in batch['sample']:
                int_element = int(element)
                if element in samples_seen:
                    samples_seen[int_element] += 1
                else:
                    samples_seen[int_element] = 1

        if epoch > 0:
            assert samples_seen == first_samples_seen


@pytest.mark.parametrize('batch_size', [128])
@pytest.mark.parametrize('drop_last', [False, True])
@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('num_workers', [0, 4])
@pytest.mark.parametrize('num_samples', [9867, 30_000])
@pytest.mark.parametrize('size_limit', [8_192])
@pytest.mark.parametrize('seed', [1234])
@pytest.mark.usefixtures('local_remote_dir')
def test_dataloader_single_device(local_remote_dir: Tuple[str, str], batch_size: int,
                                  drop_last: bool, shuffle: bool, num_workers: int,
                                  num_samples: int, size_limit: int, seed: int):
    local, remote = local_remote_dir
    convert_to_mds(out_root=remote,
                   dataset_name='sequencedataset',
                   num_samples=num_samples,
                   size_limit=size_limit)

    # Build a StreamingDataset
    dataset = StreamingDataset(local=local,
                               remote=remote,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed)

    # Build DataLoader
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            drop_last=drop_last)

    # Expected number of batches based on batch_size and drop_last
    expected_num_batches = (num_samples // batch_size) if drop_last else math.ceil(num_samples /
                                                                                   batch_size)
    expected_num_samples = expected_num_batches * batch_size if drop_last else num_samples

    # Iterate over DataLoader
    rcvd_batches = 0
    sample_order = []

    for batch_ix, batch in enumerate(dataloader):
        rcvd_batches += 1

        # Every batch should be complete except (maybe) final one
        if batch_ix + 1 < expected_num_batches:
            assert len(batch['id']) == batch_size
        else:
            if drop_last:
                assert len(batch['id']) == batch_size
            else:
                assert len(batch['id']) <= batch_size

        sample_order.extend(batch['id'][:])

    # Test dataloader length
    assert len(dataloader) == expected_num_batches
    assert rcvd_batches == expected_num_batches

    # Test that all samples arrived with no duplicates
    assert len(set(sample_order)) == expected_num_samples
    if not drop_last:
        assert len(set(sample_order)) == num_samples


@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('seed', [1111])
@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('num_workers', [0, 8])
@pytest.mark.usefixtures('local_remote_dir')
def test_dataloader_determinism(local_remote_dir: Any, batch_size: int, seed: int, shuffle: bool,
                                num_workers: int):
    remote_dir, local_dir = local_remote_dir
    convert_to_mds(out_root=remote_dir,
                   dataset_name='sequencedataset',
                   num_samples=117,
                   size_limit=1 << 8)

    # Build StreamingDataset
    dataset = StreamingDataset(local=local_dir,
                               remote=remote_dir,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed)

    # Build DataLoader
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)

    # Append sample ID
    sample_order = []
    for batch in dataloader:
        sample_order.extend(batch['id'][:])

    del dataloader
    del dataset

    # Build StreamingDataset again to test deterministic sample ID
    dataset = StreamingDataset(local=local_dir,
                               remote=remote_dir,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed)

    # Build DataLoader
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)

    # Append sample ID
    second_sample_order = []
    for batch in dataloader:
        second_sample_order.extend(batch['id'][:])

    assert len(sample_order) == len(second_sample_order)
    assert sample_order == second_sample_order


@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('seed', [2222])
@pytest.mark.parametrize('shuffle', [False])
@pytest.mark.parametrize('drop_last', [False, True])
@pytest.mark.parametrize('num_workers', [0, 8])
@pytest.mark.parametrize('num_canonical_nodes', [1])
@pytest.mark.usefixtures('local_remote_dir')
def test_dataloader_sample_order(local_remote_dir: Any, batch_size: int, seed: int, shuffle: bool,
                                 drop_last: bool, num_workers: int, num_canonical_nodes: int):
    remote_dir, local_dir = local_remote_dir
    convert_to_mds(out_root=remote_dir,
                   dataset_name='sequencedataset',
                   num_samples=117,
                   size_limit=1 << 8)

    # Build StreamingDataset
    dataset = StreamingDataset(local=local_dir,
                               remote=remote_dir,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed,
                               num_canonical_nodes=num_canonical_nodes)

    # Build DataLoader
    dataloader = StreamingDataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     drop_last=drop_last)

    if drop_last:
        num_samples = (len(dataset) // batch_size) * batch_size
        expected_sample_order = [f'{value:06}' for value in range(num_samples)]
    else:
        expected_sample_order = [f'{value:06}' for value in range(len(dataset))]

    # Append sample ID
    sample_order = []
    for batch in dataloader:
        sample_order.extend(batch['id'][:])

    assert expected_sample_order == sample_order


@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('seed', [3456])
@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('num_workers', [0, 4])
@pytest.mark.parametrize('num_canonical_nodes', [1])
@pytest.mark.usefixtures('local_remote_dir')
def test_streamingdataloader_mid_epoch_resumption(local_remote_dir: Any, batch_size: int,
                                                  seed: int, shuffle: bool, num_workers: int,
                                                  num_canonical_nodes: int):
    remote_dir, local_dir = local_remote_dir
    convert_to_mds(out_root=remote_dir,
                   dataset_name='sequencedataset',
                   num_samples=117,
                   size_limit=1 << 8)

    # Build StreamingDataset
    dataset = StreamingDataset(local=local_dir,
                               remote=remote_dir,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed,
                               num_canonical_nodes=num_canonical_nodes)

    # Build DataLoader
    dataloader = StreamingDataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     drop_last=False)

    expected_sample_order = [f'{ix:06}' for ix in range(len(dataset))]

    sample_order = []
    for idx, batch in enumerate(dataloader):
        if idx == len(dataset) // (batch_size * 2):
            sample_order.extend(batch['id'][:])
            state_dict = dataloader.state_dict()
            assert state_dict is not None
            break
        sample_order.extend(batch['id'][:])

    del dataloader
    del dataset

    dataset = StreamingDataset(local=local_dir,
                               remote=remote_dir,
                               shuffle=shuffle,
                               batch_size=batch_size,
                               shuffle_seed=seed)

    dataloader = StreamingDataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     drop_last=False)

    dataloader.load_state_dict(state_dict)  # pyright: ignore
    for idx, batch in enumerate(dataloader):
        sample_order.extend(batch['id'][:])

    # sort the sample to check for missing and duplicate samples
    sample_order.sort()
    assert len(sample_order) == len(expected_sample_order), 'Missing samples'
    assert len(set(sample_order)) == len(set(expected_sample_order)), 'Duplicate samples'
    assert sample_order == expected_sample_order, 'Incorrect sample order'


@pytest.mark.parametrize('shuffle_seed', [(9876, 9876), (12345, 1567)])
@pytest.mark.usefixtures('local_remote_dir')
def test_multiple_dataset_instantiation(local_remote_dir: Any, shuffle_seed: tuple):
    remote_dir, local_dir = local_remote_dir
    convert_to_mds(out_root=remote_dir,
                   dataset_name='sequencedataset',
                   num_samples=117,
                   size_limit=1 << 8)
    batch_size = 8
    num_workers = 2

    shuffle_seed_train, shuffle_seed_val = shuffle_seed

    # Build train StreamingDataset
    train_dataset = StreamingDataset(local=local_dir,
                                     remote=remote_dir,
                                     batch_size=batch_size,
                                     shuffle_seed=shuffle_seed_train)

    # Build train DataLoader
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers)

    # Append train sample ID
    train_sample_order = []
    for batch in train_dataloader:
        train_sample_order.extend(batch['id'][:])

    shutil.rmtree(local_dir)
    del train_dataloader
    del train_dataset

    # Build val StreamingDataset
    val_dataset = StreamingDataset(local=local_dir,
                                   remote=remote_dir,
                                   batch_size=batch_size,
                                   shuffle_seed=shuffle_seed_val)

    # Build val DataLoader
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers)

    # Append val sample ID
    val_sample_order = []
    for batch in val_dataloader:
        val_sample_order.extend(batch['id'][:])

    assert len(train_sample_order) == len(val_sample_order), 'Missing samples'
    assert len(set(train_sample_order)) == len(set(val_sample_order)), 'Duplicate samples'
