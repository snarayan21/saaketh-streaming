# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""compare loss from wandb runs with shuffle batch composition difference metric."""

import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core.batch_composition_metrics import probability_l1_metric, probability_l2_metric, \
    probability_l1_metric_diffs, probability_l2_metric_diffs
import matplotlib.pyplot as plt
import pickle
from core.utils import exponential_weighted_moving_average, lr_schedule_cosine_warmup
from scipy.signal import savgol_filter

global_batch_size = 512
lr = 0.0002
alpha_f = 0.1
warmup_steps = 100
max_steps = 24800
window = 20
# get learning rate schedule (cosine with warmup)
lr_schedule = lr*np.array([lr_schedule_cosine_warmup(alpha_f, warmup_steps, max_steps, step) for step in range(max_steps)])
stream2_lr_schedule = lr*np.array([lr_schedule_cosine_warmup(alpha_f, warmup_steps, 10000, step) for step in range(10000)])
baseline_name = 'mpt-1b-py1e-NCN16-SBS1000000-random-1'

# Get the baseline train loss curve (strongest shuffle, smooth curve)
baseline_loss_file = open(f'./run_losses/train_loss_{baseline_name}', 'rb')
baseline_train_loss = pickle.load(baseline_loss_file)
smooth_baseline_train_loss = savgol_filter(baseline_train_loss, 50, 3)
baseline_loss_file.close()

# Get the baseline stream and batch probabilities
baseline_prob_file = open(f'./stream_batch_probabilities/probs_{baseline_name}', 'rb')
baseline_probs = pickle.load(baseline_prob_file)
baseline_prob_file.close()
baseline_stream_probabilities = baseline_probs['stream_probabilities']
baseline_batch_probabilities = baseline_probs['batch_probabilities']
baseline_batch_composition = global_batch_size * baseline_batch_probabilities
# make batch composition lr-aware
# baseline_batch_raw_composition = global_batch_size * baseline_batch_probabilities
# baseline_batch_composition = (lr_schedule[:baseline_batch_probabilities.shape[0]]*baseline_batch_raw_composition.T).T
baseline_l1_metric = probability_l1_metric(baseline_stream_probabilities, baseline_batch_probabilities)
baseline_l2_metric = probability_l2_metric(baseline_stream_probabilities, baseline_batch_probabilities)

# Get the exponential moving average of the batch composition (for each stream separately)
baseline_batch_composition_ema = []
for stream_composition in baseline_batch_composition.T:
    baseline_batch_composition_ema.append(
        exponential_weighted_moving_average(stream_composition, 2/(window+1)))
baseline_batch_composition_ema = np.array(baseline_batch_composition_ema).T
# now calculate the probability of each stream from the moving average compositions
baseline_batch_probabilities_ema = baseline_batch_composition_ema / np.sum(
    baseline_batch_composition_ema, axis=1, keepdims=True)
#baseline_batch_prob_diffs_ema = np.diff(baseline_batch_probabilities_ema, axis=0, prepend=baseline_batch_probabilities_ema[:1])
# subtract the ema batch probabilities from the previous step from the current step probabilities
baseline_batch_prob_diffs_ema = baseline_batch_probabilities[1:] - baseline_batch_probabilities_ema[:-1]
baseline_batch_prob_diffs_ema = np.insert(baseline_batch_prob_diffs_ema, 0, baseline_batch_probabilities[0], axis=0)
baseline_l1_metric_ema = probability_l1_metric_diffs(baseline_batch_prob_diffs_ema)
baseline_l2_metric_ema = probability_l2_metric_diffs(baseline_batch_prob_diffs_ema)

# plot probabilities of streams over time
# plt.plot(np.arange(len(baseline_batch_probabilities)), baseline_batch_probabilities, label='raw')
# plt.plot(np.arange(len(baseline_batch_probabilities_ema)), baseline_batch_probabilities_ema, label='ema')
# plt.show()

# plt.title(f'Deviation from dataset composition for baseline run')
# plt.plot(np.arange(len(baseline_l1_metric)), baseline_l1_metric, label='L1 metric')
# plt.plot(np.arange(len(baseline_l2_metric)), baseline_l2_metric, label='L2 metric')
# plt.legend()
# plt.show()

#experiment_names = ["mpt-1b-py1e-NCN16-SBS1000000-random-1", "mpt-1b-py1b-NCN16-SBS1000000-random-1", "mpt-1b-py1br-NCN16-SBS1000000-random-1", "mpt-1b-py1br-NCN1-SBS100000-random-1"]
experiment_names = ["mpt-1b-py1b-NCN16-SBS1000000-random-1"]
# experiment_names = ["mpt-1b-py1br-NCN1-SBS100000-2stream-1"]
# experiment_names = ["mpt-1b-py1br-NCN1-SBS100000-random-1"]

for experiment_name in experiment_names:

    # get the train loss curve for this experiment
    experiment_file = open(f'./run_losses/train_loss_{experiment_name}', 'rb')
    experiment_train_loss = pickle.load(experiment_file)
    smooth_experiment_train_loss = savgol_filter(experiment_train_loss, 50, 3)
    experiment_file.close()

    # get the stream and batch probabilities for this experiment
    experiment_prob_file = open(f'./stream_batch_probabilities/probs_{experiment_name}', 'rb')
    experiment_probs = pickle.load(experiment_prob_file)
    experiment_prob_file.close()
    experiment_stream_probabilities = experiment_probs['stream_probabilities']
    experiment_batch_probabilities = experiment_probs['batch_probabilities']
    experiment_batch_composition = global_batch_size * experiment_batch_probabilities
    # make batch composition lr-aware
    # experiment_batch_raw_composition = global_batch_size * experiment_batch_probabilities
    # experiment_batch_composition = (stream2_lr_schedule[:experiment_batch_probabilities.shape[0]]*experiment_batch_raw_composition.T).T
    experiment_l1_metric = probability_l1_metric(experiment_stream_probabilities, experiment_batch_probabilities)
    experiment_l2_metric = probability_l2_metric(experiment_stream_probabilities, experiment_batch_probabilities)

    # Get the exponential moving average of the batch composition (for each stream separately)
    experiment_batch_composition_ema = []
    for stream_composition in experiment_batch_composition.T:
        experiment_batch_composition_ema.append(
            exponential_weighted_moving_average(stream_composition, 2/(window+1)))
    experiment_batch_composition_ema = np.array(experiment_batch_composition_ema).T
    # now calculate the probability of each stream from the moving average compositions
    experiment_batch_probabilities_ema = experiment_batch_composition_ema / np.sum(
        experiment_batch_composition_ema, axis=1, keepdims=True)
    #experiment_batch_prob_diffs_ema = np.diff(experiment_batch_probabilities_ema, axis=0, prepend=experiment_batch_probabilities_ema[:1])
    experiment_batch_prob_diffs_ema = experiment_batch_probabilities[1:] - experiment_batch_probabilities_ema[:-1]
    experiment_batch_prob_diffs_ema = np.insert(experiment_batch_prob_diffs_ema, 0, experiment_batch_probabilities[0], axis=0)
    experiment_l1_metric_ema = probability_l1_metric_diffs(experiment_batch_prob_diffs_ema)
    experiment_l2_metric_ema = probability_l2_metric_diffs(experiment_batch_prob_diffs_ema)

    # make loss deviation positive as well.
    loss_deviation = smooth_experiment_train_loss - smooth_baseline_train_loss
    # TODO: replace this troll loss deviation with something actual lol
    #poly_coeffs = np.polyfit(np.arange(len(experiment_train_loss)), experiment_train_loss, 10)
    #troll_baseline_loss = np.polyval(poly_coeffs, np.arange(len(experiment_train_loss)))
    #loss_deviation = smooth_experiment_train_loss - troll_baseline_loss
    if np.min(loss_deviation) < 0:
        loss_deviation -= np.min(loss_deviation)
    loss_difference = np.diff(loss_deviation, prepend=loss_deviation[0])
    start = 0
    end = 19500

    # get the block stops for this experiment
    shuffle_block_size = 1000000
    global_batch_size = 512
    num_canonical_nodes = 16
    total_samples = len(experiment_train_loss)*global_batch_size
    block_stops = np.arange(total_samples, step=num_canonical_nodes*shuffle_block_size)
    def plot_block_ranges(block_stops, global_batch_size, start, end, ax):
        step_stops = [block / global_batch_size for block in block_stops if block/global_batch_size > start and block/global_batch_size < end]
        ax.vlines(step_stops, 0, 1, color='lightgray')

    
    # plot the l1 and l2 shuffle quality metrics for the experiment
    # plot the l1 shuffle quality metric on the x axis and the excess loss on the y axis as a scatter plot
    # plot the l2 shuffle quality metric on the x axis and the excess loss on the y axis as a scatter plot
    # plot the loss curve for the experiment and the baseline
    # plot the "loss deviation" from the experiment vs the baseline
    # 2 x 2 grid of subplots
    fig, ((ax1, ax3), (ax4, ax6)) = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(f'Shuffle quality and loss results for {experiment_name}')
    first_color = 'tab:red'
    second_color = 'tab:blue'
    third_color = 'tab:green'

    # ax1.set_title('metrics, batch composition on underlying dataset')
    # ax1.set_xlabel('step')
    # ax1.set_ylabel('metric value')
    # #plot_block_ranges(block_stops, global_batch_size, start, end, ax1)
    # ax1.plot(np.arange(len(experiment_l1_metric))[start:end], experiment_l1_metric[start:end], label='exp L1')
    # ax1.plot(np.arange(len(experiment_l2_metric))[start:end], experiment_l2_metric[start:end], label='exp L2')
    # #ax1.plot(np.arange(len(baseline_l1_metric))[start:end], baseline_l1_metric[start:end], label='base L1')
    # #ax1.plot(np.arange(len(baseline_l2_metric))[start:end], baseline_l2_metric[start:end], label='base L2')
    # ax1.legend()

    # ax2.set_title('metrics (on underlying dataset) vs. loss deviation')
    # ax2.set_xlabel('loss deviation')
    # ax2.set_ylabel('l1 metric', color=first_color)
    # ax2.scatter(loss_deviation[start:end], experiment_l1_metric[start:end], s=10, color=first_color)
    # ax2a = ax2.twinx()
    # ax2a.set_ylabel('l2 metric', color=second_color)
    # ax2a.scatter(loss_deviation[start:end], experiment_l2_metric[start:end], s=10, color=second_color)

    # ax2.set_title('metrics (on previously seen) vs. abs loss deviation')
    # ax2.set_xlabel('loss deviation')
    # ax2.set_ylabel('l1 metric', color=first_color)
    # ax2.scatter(loss_deviation[start:end], experiment_l1_metric_ema[start:end], s=10, color=first_color)
    # ax2a = ax2.twinx()
    # ax2a.set_ylabel('l2 metric', color=second_color)
    # ax2a.scatter(loss_deviation[start:end], experiment_l2_metric_ema[start:end], s=10, color=second_color)

    ax1.set_title('metrics, batch composition on previously seen')
    ax1.set_xlabel('step')
    ax1.set_ylabel('metric value')
    plot_block_ranges(block_stops, global_batch_size, start, end, ax1)
    ax1.plot(np.arange(len(experiment_l1_metric_ema))[start:end], experiment_l1_metric_ema[start:end], label='exp L1 ema')
    ax1.plot(np.arange(len(experiment_l2_metric_ema))[start:end], experiment_l2_metric_ema[start:end], label='exp L2 ema')

    #ax1.plot(np.arange(len(baseline_l1_metric_ema))[start:end], baseline_l1_metric_ema[start:end], label='base L1 ema')
    #ax1.plot(np.arange(len(baseline_l2_metric_ema))[start:end], baseline_l2_metric_ema[start:end], label='base L2 ema')

    # plot how batch probabilities changed over time
    ax3.set_title('batch probabilities')
    ax3.set_xlabel('step')
    ax3.set_ylabel('probability')
    plot_block_ranges(block_stops, global_batch_size, start, end, ax3)
    ax3.plot(np.arange(len(experiment_batch_probabilities))[start:end], experiment_batch_probabilities[start:end], label='raw')
    ax3.plot(np.arange(len(experiment_batch_probabilities_ema))[start:end], experiment_batch_probabilities_ema[start:end], label='ema')

    ax4.set_title('train losses')
    ax4.set_xlabel('step')
    ax4.set_ylabel('train loss')
    plot_block_ranges(block_stops, global_batch_size, start, end, ax4)
    ax4.plot(np.arange(len(experiment_train_loss))[start:end], experiment_train_loss[start:end], label='exp')
    ax4.plot(np.arange(len(baseline_train_loss))[start:end], baseline_train_loss[start:end], label='base')
    ax4.plot(np.arange(len(smooth_baseline_train_loss))[start:end], smooth_baseline_train_loss[start:end], label='base smooth')
    ax4.plot(np.arange(len(loss_difference))[start:end], loss_difference[start:end], label='difference')
    ax4.plot(np.arange(len(loss_deviation))[start:end], loss_deviation[start:end], label='deviation')
    #ax4.plot(np.arange(len(troll_baseline_loss))[start:end], troll_baseline_loss[start:end], label='troll baseline')
    ax4.legend()

    # ax5.set_title('metrics (on underlying dataset) and loss deviation')
    # ax5.set_xlabel('step')
    # ax5.set_ylabel('loss deviation', color=first_color)
    # plot_block_ranges(block_stops, global_batch_size, start, end, ax5)
    # ax5.plot(np.arange(len(loss_deviation))[start:end], loss_deviation[start:end], label='deviation', color=first_color)
    # ax5a = ax5.twinx()
    # ax5a.set_ylabel('metric value')
    # ax5a.plot(np.arange(len(experiment_l1_metric))[start:end], experiment_l1_metric[start:end], label='exp L1')
    # ax5a.plot(np.arange(len(experiment_l2_metric))[start:end], experiment_l2_metric[start:end], label='exp L2')
    # ax5a.legend()

    #ax6.set_title('metrics (on underlying dataset) and loss deviation')
    ax6.set_title('loss deviation')
    ax6.set_xlabel('step')
    ax6.set_ylabel('loss deviation', color=first_color)
    plot_block_ranges(block_stops, global_batch_size, start, end, ax6)
    ax6.plot(np.arange(len(loss_deviation))[start:end], loss_deviation[start:end], label='deviation', color=first_color)
    ax6a = ax6.twinx()
    ax6a.set_ylabel('metric value')
    #ax6a.plot(np.arange(len(experiment_l1_metric_ema))[start:end], experiment_l1_metric_ema[start:end], label='exp L1 ema')
    #ax6a.plot(np.arange(len(experiment_l2_metric_ema))[start:end], experiment_l2_metric_ema[start:end], label='exp L2 ema')
    ax6a.legend()
    
    fig.tight_layout()
    plt.show()