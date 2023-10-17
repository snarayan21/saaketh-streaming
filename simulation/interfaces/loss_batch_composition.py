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
window = 100
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
# experiment_names = ["mpt-1b-py1b-NCN16-SBS1000000-random-1"]
experiment_names = ["mpt-1b-py1br-NCN1-SBS100000-2stream-1"]
# experiment_names = ["mpt-1b-py1br-NCN1-SBS100000-random-1"]

for experiment_name in experiment_names:

    # get and process token counts into probability metrics for this run
    token_counts_file = open(f'/Users/saaketh.narayan/Desktop/saaketh-streaming/simulation/interfaces/token_counts/2stream', 'rb')
    token_counts = pickle.load(token_counts_file)
    steps, vocab_size = token_counts.shape
    token_counts_file.close()

    token_probabilities = token_counts / np.sum(token_counts, axis=1, keepdims=True)
    # take exponential moving average of token counts over window
    token_counts_ema = []
    token_idx = 0
    for token_composition in token_counts.T:
        if token_idx % 10000 == 0:
            print("processing token", token_idx)
        token_idx += 1
        token_counts_ema.append(exponential_weighted_moving_average(token_composition, 2/(window+1)))
    token_counts_ema = np.array(token_counts_ema).T
    token_probabilities_ema = token_counts_ema / np.sum(token_counts_ema, axis=1, keepdims=True)
    token_probability_diffs_ema = token_probabilities[1:] - token_probabilities_ema[:-1]
    token_probability_diffs_ema = np.insert(token_probability_diffs_ema, 0, token_probabilities[0], axis=0)
    token_l1_metric_ema = probability_l1_metric_diffs(token_probability_diffs_ema)
    token_l2_metric_ema = probability_l2_metric_diffs(token_probability_diffs_ema)

    # make "new token" metric for each step over window range
    new_tokens = []
    for step in range(steps):
        if step % 1000 == 0:
            print("new tokens for step", step)
        if step == 0:
            # case when no window to compare to
            new_tokens.append(0)
        else:
            if step < window:
                # case when we have to limit window size
                window_tokens = token_counts[0:step]
            else:
                # case when we can take the full window
                window_tokens = token_counts[step-window:step]
            window_no_token_indices = np.where(np.sum(window_tokens, axis=0) == 0)[0]
            step_tokens_window_no_token_indices = token_counts[step][window_no_token_indices]
            new_tokens.append(np.sum(step_tokens_window_no_token_indices, axis=0))
    new_tokens = np.array(new_tokens)

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
    # loss_deviation = smooth_experiment_train_loss - smooth_baseline_train_loss
    # TODO: replace this troll loss deviation with something actual lol
    poly_coeffs = np.polyfit(np.arange(len(experiment_train_loss)), experiment_train_loss, 10)
    troll_baseline_loss = np.polyval(poly_coeffs, np.arange(len(experiment_train_loss)))
    loss_deviation = smooth_experiment_train_loss - troll_baseline_loss
    if np.min(loss_deviation) < 0:
        loss_deviation -= np.min(loss_deviation)
    loss_difference = np.diff(loss_deviation, prepend=loss_deviation[0])
    
    start = 0
    end = 5600

    # get the block stops for this experiment
    # shuffle_block_size = 1000000
    # global_batch_size = 512
    # num_canonical_nodes = 16
    # total_samples = len(experiment_train_loss)*global_batch_size
    # block_stops = np.arange(total_samples, step=num_canonical_nodes*shuffle_block_size)

    # These are block stops for 2stream ONLY (NCN 1, SBS 100k)
    block_stops = [105423, 205090, 320983, 421749, 540217, 633238, 728265, 821824, 897345, 1017281, 1095179, 1206504, 1304991, 1393563, 1505046, 1597612, 1700879, 1823526, 1937536, 2031635, 2126276, 2226633, 2321386, 2433076, 2542323, 2628637, 2752506, 2859967, 2962664, 3049371, 3134824, 3252835, 3357886, 3453689, 3570334, 3694171, 3809908, 3888113, 4005028, 4106732, 4214364, 4312095, 4412404, 4501970, 4590801, 4672021, 4759613, 4838630, 4937906, 5053779, 5161517, 5245266, 5333908, 5410517, 5517832, 5620779, 5745449, 5831574, 5920483, 6013971, 6137096, 6252602, 6348986, 6454056, 6549387, 6631845, 6714174, 6795608, 6885888, 6990726, 7097250, 7214371, 7324230, 7422874, 7540500, 7649215, 7735064, 7849654, 7969541, 8052599, 8153375, 8261158, 8355886, 8479724, 8570335, 8679659, 8798519, 8908837, 9001970, 9082458, 9168103, 9271912, 9362805, 9445149, 9563687, 9674219, 9780172, 9879085, 9989767, 10067631, 10153385, 10239336, 10347949, 10442456, 10524257, 10606501, 10724267, 10815896, 10906725, 11031688, 11129256, 11215245, 11321598, 11425290, 11508843, 11584064, 11689896, 11812284, 11898275, 12015434, 12114584, 12193430, 12309359, 12392562, 12485360, 12589805, 12690477, 12773388, 12858953, 12949884, 13069193, 13183535, 13286394, 13395925, 13499321, 13606291, 13701451, 13778550, 13895189, 13980811, 14086495, 14167476, 14263026, 14366203, 14469314, 14581640, 14704024, 14803103, 14897063, 15012362, 15097919, 15176699, 15290528, 15372971, 15476484, 15554639, 15657103, 15752448, 15876286, 15961203, 16054346, 16175601, 16267998, 16350760, 16457402, 16572554, 16672323, 16750736, 16854694, 16941585, 17038146, 17130219, 17247153, 17367701, 17443221, 17566239, 17689421, 17797186, 17891010, 17985304, 18107159, 18225556, 18318908, 18421287, 18505873, 18585246, 18669202, 18761911, 18873830, 18955516, 19069602, 19184875, 19260733, 19347005, 19463330, 19555574, 19679377, 19786620, 19900134, 20003144, 20092734, 20206330, 20304885, 20417528, 20531541, 20652949, 20758885, 20860436, 20972757, 21066763, 21142535, 21220250, 21297949, 21379660, 21499184, 21584404, 21685528, 21765687, 21858745, 21933856, 22011136, 22107673, 22223707, 22304502, 22417761, 22508835, 22584943, 22675281, 22779900, 22875582, 22951437, 23053361, 23141755, 23232888, 23317029, 23403529, 23509692, 23617811, 23719188, 23799694, 23917416, 24009672, 24132577, 24215309, 24316068, 24392342, 24488931, 24605077, 24689449, 24801971, 24917732, 25035611, 25147115, 25254361, 25330117, 25446653, 25555661, 25652238, 25740619, 25839901, 25934009, 26033236, 26142734, 26221293, 26336351, 26453942, 26539117, 26656767, 26743204, 26867433, 26977083, 27094877, 27187203, 27308825, 27429900, 27553776, 27668687, 27763564, 27863363, 27941937, 28025063, 28128068, 28203988, 28317802, 28436344, 28559257, 28634667, 28723055, 28816827, 28908667, 29026190, 29149180, 29228791, 29344181, 29449360, 29532956, 29635063, 29758497, 29868438, 29944973, 30032803, 30108204, 30216946, 30301341, 30406478, 30491117, 30574790, 30664395, 30771262, 30876413, 30958839, 31079842, 31155204]
    
    def plot_block_ranges(block_stops, global_batch_size, start, end, ax):
        step_stops = [block / global_batch_size for block in block_stops if block/global_batch_size > start and block/global_batch_size < end]
        ax.vlines(step_stops, 0, 1, color='lightgray')
    
    # plot the l1 and l2 shuffle quality metrics for the experiment
    # plot the l1 shuffle quality metric on the x axis and the excess loss on the y axis as a scatter plot
    # plot the l2 shuffle quality metric on the x axis and the excess loss on the y axis as a scatter plot
    # plot the loss curve for the experiment and the baseline
    # plot the "loss deviation" from the experiment vs the baseline
    # 2 x 2 grid of subplots
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 10))
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

    ax2.set_title('metrics, token composition on previously seen')
    ax2.set_xlabel('step')
    ax2.set_ylabel('metric value')
    plot_block_ranges(block_stops, global_batch_size, start, end, ax2)
    ax2.plot(np.arange(len(token_l1_metric_ema))[start:end], token_l1_metric_ema[start:end], label='token L1 ema')
    ax2.plot(np.arange(len(token_l2_metric_ema))[start:end], token_l2_metric_ema[start:end], label='token L2 ema')

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

    # ax5.set_title('new tokens compared to window per step')
    # ax5.set_xlabel('step')
    # ax5.set_ylabel('new tokens')
    # plot_block_ranges(block_stops, global_batch_size, start, end, ax5)
    # ax5.plot(np.arange(len(new_tokens))[start:end], new_tokens[start:end], label='new tokens')

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
    ax5.set_title('loss deviation')
    ax5.set_xlabel('step')
    ax5.set_ylabel('loss deviation', color=first_color)
    plot_block_ranges(block_stops, global_batch_size, start, end, ax5)
    ax5.plot(np.arange(len(loss_deviation))[start:end], loss_deviation[start:end], label='deviation', color=first_color)
    ax5a = ax5.twinx()
    ax5a.set_ylabel('metric value')
    #ax5a.plot(np.arange(len(experiment_l1_metric_ema))[start:end], experiment_l1_metric_ema[start:end], label='exp L1 ema')
    #ax5a.plot(np.arange(len(experiment_l2_metric_ema))[start:end], experiment_l2_metric_ema[start:end], label='exp L2 ema')
    ax5a.legend()
    
    fig.tight_layout()
    plt.show()