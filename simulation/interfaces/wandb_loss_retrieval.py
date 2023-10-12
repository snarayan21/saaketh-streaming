# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""retrieve training loss curves from wandb runs with potential resumptions."""

import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import wandb
import matplotlib.pyplot as plt
import pickle

api = wandb.Api()

project_id = 'mosaic-ml/shuffling_experiments-1b'
project_runs = api.runs(path=project_id, per_page=300)
project_run_names = [run.name for run in project_runs]
project_run_ids = [run.id for run in project_runs]

run_groups = ["mpt-1b-py1br-NCN1-SBS100000-2stream-1"]

for run_group in run_groups:

    # get all the runs in the project corresponding to the run group
    run_names_ids = [(project_run_names[i], project_run_ids[i]) \
                     for i in range(len(project_run_names)) if run_group in project_run_names[i]]

    # stitch together all resumptions for this run group
    print("Processing:", run_group)
    run_train_step_loss = {}
    for run_name, run_id in run_names_ids:
        run = api.run(f'{project_id}/{run_id}')

        history = run.scan_history()

        step = [row["_step"] for row in history]
        step = [s for s in step if s is not None]
        
        # if number of steps is less than 1000 and if run was not finished here, skip this run.
        run_finished = (max(step) == 24800)
        if len(step) < 1000 and not run_finished:
            continue

        train_loss = [row["loss/train/total"] for row in history]
        train_loss = [t for t in train_loss if t is not None]
        
        # clip the train loss and steps to the nearest multiple of 1000
        # if max step is not 24800 (run was finished)
        if not run_finished:
            train_loss = train_loss[:-(len(train_loss) % 1000)]
            step = step[:-(len(step) % 1000)]
        
        # add train losses from this run to the overall run train loss dict
        for i in range(len(train_loss)):
            run_train_step_loss[step[i]] = train_loss[i]
    
    sorted_run_train_loss_steps = sorted(list(run_train_step_loss.items()), key=lambda x: x[0])
    run_train_loss = np.array([x[1] for x in sorted_run_train_loss_steps])

    # make sure this run group has a loss value for every single step (24800 total)
    assert len(run_train_loss) == 24800

    # pickle the train loss for this run group
    file = open(f'./run_losses/train_loss_{run_group}', 'wb')
    pickle.dump(run_train_loss, file)
    file.close()

    plt.plot(np.arange(len(run_train_loss)), run_train_loss, label=run_group)
    plt.show()
