# -*- coding: utf-8 -*-
"""
Code for analyzing safe vs risky human data

@author: Thomas Elston
"""
# --------------------------
import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import pingouin as pg
from glob import glob
# --------------------------


def load_process_data(datadir, context):
    """Loads and pre-processes data for a given experiment
    Args:
        datadir (string): the folder where each experiment's folder is
        context (string): whether to look at "gain" or "loss" data
    Returns:
        all_data: the trial-wise responses of each participant
        p_choice_data: the participant mean responses for each condition
        p_rt_data: the participant mean RTs for each condition
    """

    # get names of each experiment's folder    
    data_folders = sorted(glob(datadir + '*/', recursive=True))

    # initialize outputs
    all_data = pd.DataFrame()
    p_choice_data = pd.DataFrame()
    p_rt_data = pd.DataFrame()

    ctr = 0

    for exp_num, exp_folder in enumerate(data_folders):
        print('\n')
        print(context + ' Experiment #: ' + str(exp_num + 1) +
              ' of ' + str(len(data_folders)))

        # get files names with path
        fnames = [exp_folder + _ for _ in os.listdir(exp_folder) if _.endswith('.csv')]

        # load a csv file and assess each one
        for i in range(len(fnames)):

            print('\rSubject #: ' + str(i + 1) + ' of ' + str(len(fnames)) + '   ', end='')

            df = pd.read_csv(fnames[i], header=[0])

            p_perf = np.zeros((len(fnames), 4))

            # get overall training performance so we can reject subjects
            all_train_ix = df.phase == "training"
            all_exp_ix = df.phase == "exp"
            trials2include = all_train_ix | all_exp_ix
            all_safe_ix = (df.imgLeftType == 'Safe') & (df.imgRightType == 'Safe')
            all_risky_ix = (df.imgLeftType == 'Risky') & (df.imgRightType == 'Risky')
            ue_ix = (df.imgLeftType != df.imgRightType) & (df.probLeft != df.probRight)
            gain_ix = df.blkType == 'Gain'
            loss_ix = df.blkType == 'Loss'

            # define the best options 
            all_picked_best = df.highProbSelected.astype(int)

            # get subject overall performance       
            gs_perf = all_picked_best[trials2include & gain_ix & all_safe_ix].mean()
            gr_perf = all_picked_best[trials2include & gain_ix & all_risky_ix].mean()
            ls_perf = 1 - all_picked_best[trials2include & loss_ix & all_safe_ix].mean()
            lr_perf = 1 - all_picked_best[trials2include & loss_ix & all_risky_ix].mean()
            g_ue_perf = all_picked_best[trials2include & gain_ix & ue_ix].mean()
            l_ue_perf = 1 - all_picked_best[trials2include & loss_ix & ue_ix].mean()

            p_perf[i, :] = np.array([gs_perf, gr_perf, ls_perf, lr_perf])

            # inclusion crit for overall perf
            c = 0.60

            if np.nanmean(p_perf[i, :]) > c:

                # only assess the context specified as input argument
                df = df[df.blkType == context]

                # drop trials with RTs that were too slow or too fast - cut-offs determined via manual inspection
                df = df.drop(df[(df.rt > 1500) | (df.rt < 200)].index)
                df = df.reset_index()

                # change a few datatypes
                df.highProbSelected = df.highProbSelected.astype(int)
                df.probLeft = df.probLeft.astype(float)
                df.probRight = df.probRight.astype(float)

                # get some useful indices
                # trial types and task phases
                trainix = df.phase == "training"
                testix = df.phase == "exp"

                # infer trial types from the left/right stimulus types
                safe_ix = (df.imgLeftType == 'Safe') & (df.imgRightType == 'Safe')
                risky_ix = (df.imgLeftType == 'Risky') & (df.imgRightType == 'Risky')
                EQ = df.probLeft == df.probRight
                UE = (df.imgLeftType != df.imgRightType) & ~EQ

                # define the best options for the two contexts (gain/loss)            
                if context == 'Loss':
                    picked_best = df.highProbSelected == 0
                else:
                    picked_best = df.highProbSelected == 1

                # keep track of which stimulus type was chosen on each trial
                picked_risky = np.zeros(shape=(len(EQ),))
                picked_risky[(df.responseSide == 'left') & (df.imgLeftType == 'Risky')] = 1
                picked_risky[(df.responseSide == 'right') & (df.imgRightType == 'Risky')] = 1
                picked_risky = picked_risky.astype(int)

                # define reaction times
                rt = df.rt

                # define trials that were "hits"
                tHit = df['rewardCode']

                # find out the chosen_prob of each trial
                chosen_prob = np.empty(len(df.rt))
                picked_left = df['responseSide'] == 'left'
                picked_right = df['responseSide'] == 'right'
                chosen_prob[picked_left] = df['probLeft'].loc[picked_left]
                chosen_prob[picked_right] = df['probRight'].loc[picked_right]

                # add these values to the dataframe
                df['exp_num'] = np.ones(shape=(len(df),)) * (exp_num + 1)
                df['chosen_prob'] = chosen_prob

                # what is the best option's type?
                best_type = np.empty((len(df.rt)))

                # find the trials where the safe/risky options were better
                for t in range(len(df.rt)):
                    if df.highProbSide[t] == 'left':
                        if df.imgLeftType[t] == 'Risky':
                            best_type[t] = 1
                        else:
                            best_type[t] = 0
                    else:
                        if df.imgRightType[t] == 'Risky':
                            best_type[t] = 1
                        else:
                            best_type[t] = 0
                    if picked_risky[t] == 1:
                        df.at[t, 'chosen_type'] = 'Risky'
                    else:
                        df.at[t, 'chosen_type'] = 'Safe'

                # choice conditions
                risky_best = best_type == 1
                safe_best = best_type == 0
                t20v20 = (df.probLeft == .2) & (df.probRight == .2)
                t50v50 = (df.probLeft == .5) & (df.probRight == .5)
                t80v80 = (df.probLeft == .8) & (df.probRight == .8)

                # -----------------------------
                #    summarize each subject
                # -----------------------------
                # get experiment number, subject id, version, age, and gender
                p_choice_data.at[ctr, 'exp_num'] = exp_num + 1
                p_choice_data.at[ctr, 'vpnum'] = df.vpNum[0]
                p_choice_data.at[ctr, 'version'] = df.version[0]
                p_choice_data.at[ctr, 'context'] = context

                p_rt_data.at[ctr, 'exp_num'] = exp_num + 1
                p_rt_data.at[ctr, 'vpnum'] = df.vpNum[0]
                p_rt_data.at[ctr, 'version'] = df.version[0]
                p_rt_data.at[ctr, 'context'] = context

                if 'age' in df:
                    p_choice_data.at[ctr, 'age'] = df.age[0]
                    p_rt_data.at[ctr, 'age'] = df.age[0]
                else:
                    p_choice_data.at[ctr, 'age'] = np.nan
                    p_rt_data.at[ctr, 'age'] = np.nan

                if 'gender' in df:
                    p_choice_data.at[ctr, 'sex'] = df.gender[0]
                    p_rt_data.at[ctr, 'sex'] = df.gender[0]

                else:
                    p_choice_data.at[ctr, 'sex'] = 'na'
                    p_rt_data.at[ctr, 'sex'] = 'na'

                # training choice data
                p_choice_data.at[ctr, 't_safe'] = np.nanmean(picked_best[trainix & safe_ix])
                p_choice_data.at[ctr, 't_risky'] = np.nanmean(picked_best[trainix & risky_ix])

                # main block pure trials
                p_choice_data.at[ctr, 'p_safe'] = np.nanmean(picked_best[testix & safe_ix])
                p_choice_data.at[ctr, 'p_risky'] = np.nanmean(picked_best[testix & risky_ix])

                # main block unequal trials
                p_choice_data.at[ctr, 'UE_safe'] = np.nanmean(picked_best[UE & safe_best])
                p_choice_data.at[ctr, 'UE_risky'] = np.nanmean(picked_best[UE & risky_best])

                # main block equivaluable trials
                p_choice_data.at[ctr, 'EQ20'] = np.nanmean(picked_risky[t20v20])
                p_choice_data.at[ctr, 'EQ50'] = np.nanmean(picked_risky[t50v50])
                p_choice_data.at[ctr, 'EQ80'] = np.nanmean(picked_risky[t80v80])

                # check how often the risky options paid off during training
                t_risky = trainix & (picked_risky == 1)
                t20_pHit = np.sum(t_risky & (tHit==1) & (chosen_prob == .2)) / np.sum((chosen_prob == .2) & t_risky)
                t50_pHit = np.sum(t_risky & (tHit==1) & (chosen_prob == .5)) / np.sum((chosen_prob == .5) & t_risky)
                t80_pHit = np.sum(t_risky & (tHit==1) & (chosen_prob == .8)) / np.sum((chosen_prob == .8) & t_risky)

                p_choice_data.at[ctr, 't20'] = t20_pHit
                p_choice_data.at[ctr, 't50'] = t50_pHit
                p_choice_data.at[ctr, 't80'] = t80_pHit

                # do the same but with RTs
                p_rt_data.at[ctr, 't_safe'] = np.nanmean(rt[trainix & safe_ix])
                p_rt_data.at[ctr, 't_risky'] = np.nanmean(rt[trainix & risky_ix])

                # main block pure trials
                p_rt_data.at[ctr, 'p_safe'] = np.nanmean(rt[testix & safe_ix])
                p_rt_data.at[ctr, 'p_risky'] = np.nanmean(rt[testix & risky_ix])

                # main block unequal trials
                p_rt_data.at[ctr, 'UE_safe'] = np.nanmean(rt[UE & safe_best])
                p_rt_data.at[ctr, 'UE_risky'] = np.nanmean(rt[UE & risky_best])

                # main block equivaluable trials
                p_rt_data.at[ctr, 'EQ20'] = np.nanmean(rt[t20v20])
                p_rt_data.at[ctr, 'EQ50'] = np.nanmean(rt[t50v50])
                p_rt_data.at[ctr, 'EQ80'] = np.nanmean(rt[t80v80])

                # increment the subject counter
                ctr = ctr + 1

                # add all data to the aggregate dataframe
                all_data = pd.concat([all_data, df], axis=0, ignore_index=True)

    return p_choice_data, p_rt_data, all_data


# END of load_processData


def plot_task_validation(gain_choice, loss_choice, save_fig):
    """Plots results from the training blocks and main block conditions where there
    was a best choice.
    Args:
        gain_choice (dataframe): participant means in gain context
        loss_choice (dataframe): participant means in loss context
        save_fig (boolean): flag of whether or not to save the figure as a .svg file
    """

    experiment_ids = (np.unique(gain_choice['exp_num'])).astype(int)
    # create figure and define the color map
    cmap = plt.cm.Paired(np.linspace(0, 1, 12))
    fig = plt.figure(figsize=(8, 8), dpi=300)
    gs = fig.add_gridspec(3, 12)



    for exp_ix, exp in enumerate(experiment_ids):

        # pull the data for this experiment
        gain_e_data = gain_choice.loc[gain_choice['exp_num'] == exp]
        loss_e_data = loss_choice.loc[loss_choice['exp_num'] == exp]

        # create subplots for this experiment
        sampling_ax = fig.add_subplot(gs[exp_ix, 0: 4])
        train_ax = fig.add_subplot(gs[exp_ix, 6: 8])
        pure_ax = fig.add_subplot(gs[exp_ix, 8: 10])
        UE_ax = fig.add_subplot(gs[exp_ix, 10: 12])

        # sampling behavior during training
        sampling_ax.plot([0,1], [0,1], color='gray')
        sampling_ax.errorbar([.2, .5, .8], [gain_e_data['t20'].mean(),
                                            gain_e_data['t50'].mean(),
                                            gain_e_data['t80'].mean()],
                                            [gain_e_data['t20'].sem(),
                                            gain_e_data['t50'].sem(),
                                            gain_e_data['t80'].sem()],
                             color=cmap[1, :], capsize=0, linewidth=2, marker='s')
        sampling_ax.errorbar([.2, .5, .8], [loss_e_data['t20'].mean(),
                                            loss_e_data['t50'].mean(),
                                            loss_e_data['t80'].mean()],
                                            [loss_e_data['t20'].sem(),
                                            loss_e_data['t50'].sem(),
                                            loss_e_data['t80'].sem()],
                             color=cmap[5, :], capsize=0, linewidth=2, marker='s')

        sampling_ax.set_ylim([0, 1])
        sampling_ax.set_yticks([.2, .5, .8])
        sampling_ax.set_xlim([0, 1])
        sampling_ax.set_xticks([.2, .5, .8])
        sampling_ax.spines['top'].set_visible(False)
        sampling_ax.spines['right'].set_visible(False)
        sampling_ax.set_ylabel('Experienced Prob.')


        # overall training performance
        train_ax.errorbar(np.array([1, 2]), [gain_e_data['t_safe'].mean(), gain_e_data['t_risky'].mean()],
                                            [gain_e_data['t_safe'].sem(), gain_e_data['t_risky'].sem()],
                                             color=cmap[1, :], capsize=0, linewidth=2, marker='s')
        train_ax.errorbar(np.array([1, 2]), [loss_e_data['t_safe'].mean(), loss_e_data['t_risky'].mean()],
                                            [loss_e_data['t_safe'].sem(), loss_e_data['t_risky'].sem()],
                                             color=cmap[5, :], capsize=0, linewidth=2, marker='s')
        train_ax.set_ylim([0, 1])
        train_ax.set_yticks([0, .5, 1])
        train_ax.set_xlim([.8, 2.2])
        train_ax.set_xticks([])
        train_ax.set_ylabel('p(Choose Optimal)')
        train_ax.spines['top'].set_visible(False)
        train_ax.spines['right'].set_visible(False)

        # 'Pure' safe and risky trials
        pure_ax.errorbar(np.array([1, 2]), [gain_e_data['p_safe'].mean(), gain_e_data['p_risky'].mean()],
                                            [gain_e_data['p_safe'].sem(), gain_e_data['p_risky'].sem()],
                                             color=cmap[1, :], capsize=0, linewidth=2, marker='s')
        pure_ax.errorbar(np.array([1, 2]), [loss_e_data['p_safe'].mean(), loss_e_data['p_risky'].mean()],
                                            [loss_e_data['p_safe'].sem(), loss_e_data['p_risky'].sem()],
                                             color=cmap[5, :], capsize=0, linewidth=2, marker='s')
        pure_ax.set_ylim([0, 1])
        pure_ax.set_xlim([.8, 2.2])
        pure_ax.set_yticks([])
        pure_ax.set_xticks([])
        pure_ax.spines['top'].set_visible(False)
        pure_ax.spines['right'].set_visible(False)
        pure_ax.spines['left'].set_visible(False)

        # Unequal safe vs risky trials
        UE_ax.errorbar(np.array([1, 2]), [gain_e_data['UE_safe'].mean(), gain_e_data['UE_risky'].mean()],
                                            [gain_e_data['UE_safe'].sem(), gain_e_data['UE_risky'].sem()],
                                             color=cmap[1, :], capsize=0, linewidth=2, marker='s')
        UE_ax.errorbar(np.array([1, 2]), [loss_e_data['UE_safe'].mean(), loss_e_data['UE_risky'].mean()],
                                            [loss_e_data['UE_safe'].sem(), loss_e_data['UE_risky'].sem()],
                                             color=cmap[5, :], capsize=0, linewidth=2, marker='s')
        UE_ax.set_ylim([0, 1])
        UE_ax.set_xlim([.8, 2.2])
        UE_ax.set_yticks([])
        UE_ax.set_xticks([])
        UE_ax.spines['top'].set_visible(False)
        UE_ax.spines['right'].set_visible(False)
        UE_ax.spines['left'].set_visible(False)

        if exp == 1:
            sampling_ax.legend(['Unity','Gain', 'Loss'])
            sampling_ax.set_title('Adequate Sampling')
            train_ax.set_title('Training')
            pure_ax.set_title('Pure S/U')
            UE_ax.set_title('UE SvsU')

        if exp == 3:
            sampling_ax.set_xlabel('Objective Prob.')
            train_ax.set_xticks([1, 2])
            train_ax.set_xticklabels(['S', 'U'])
            pure_ax.set_xticks([1, 2])
            pure_ax.set_xticklabels(['S', 'U'])
            pure_ax.set_xlabel('Choice Condition')
            UE_ax.set_xticks([1, 2])
            UE_ax.set_xticklabels(['S>U', 'U>S'])

    plt.show()
    # save plot
    if save_fig == True:
        fig.savefig('sampling_and_performance.svg', transparent=True)



def plot_choice_or_rt(gain_data, loss_data, data_type):
    """Plots the mean+sem results for each condition
    Args:
        gain_data (dataframe): participant means in gain context
        loss_data (dataframe): participant means in loss context
        data_type (string): whether it's "choice" or "rt" data 
    """

    # get IDs of experiments
    experiment_ids = (np.unique(gain_data['exp_num'])).astype(int)

    # initialize arrays for data 
    train_means = np.empty(shape=(len(experiment_ids), 4))
    train_sems = np.empty(shape=(len(experiment_ids), 4))
    pure_means = np.empty(shape=(len(experiment_ids), 4))
    pure_sems = np.empty(shape=(len(experiment_ids), 4))
    UE_means = np.empty(shape=(len(experiment_ids), 4))
    UE_sems = np.empty(shape=(len(experiment_ids), 4))
    EQ_means = np.empty(shape=(len(experiment_ids), 6))
    EQ_sems = np.empty(shape=(len(experiment_ids), 6))

    # collect the data for plotting
    for exp_ix, exp in enumerate(experiment_ids):

        # are we dealing with gain or loss data right now?
        for cond in range(2):
            if cond == 0:
                e_data = gain_data.loc[gain_data['exp_num'] == exp]
            else:
                e_data = loss_data.loc[loss_data['exp_num'] == exp]

            train_means[exp_ix, 0 + 2 * cond] = e_data['t_safe'].mean()
            train_means[exp_ix, 1 + 2 * cond] = e_data['t_risky'].mean()
            train_sems[exp_ix, 0 + 2 * cond] = e_data['t_safe'].sem()
            train_sems[exp_ix, 1 + 2 * cond] = e_data['t_risky'].sem()

            pure_means[exp_ix, 0 + 2 * cond] = e_data['p_safe'].mean()
            pure_means[exp_ix, 1 + 2 * cond] = e_data['p_risky'].mean()
            pure_sems[exp_ix, 0 + 2 * cond] = e_data['p_safe'].sem()
            pure_sems[exp_ix, 1 + 2 * cond] = e_data['p_risky'].sem()

            UE_means[exp_ix, 0 + 2 * cond] = e_data['UE_safe'].mean()
            UE_means[exp_ix, 1 + 2 * cond] = e_data['UE_risky'].mean()
            UE_sems[exp_ix, 0 + 2 * cond] = e_data['UE_safe'].sem()
            UE_sems[exp_ix, 1 + 2 * cond] = e_data['UE_risky'].sem()

            EQ_means[exp_ix, 0 + 3 * cond] = e_data['EQ20'].mean()
            EQ_means[exp_ix, 1 + 3 * cond] = e_data['EQ50'].mean()
            EQ_means[exp_ix, 2 + 3 * cond] = e_data['EQ80'].mean()

            EQ_sems[exp_ix, 0 + 3 * cond] = e_data['EQ20'].sem()
            EQ_sems[exp_ix, 1 + 3 * cond] = e_data['EQ50'].sem()
            EQ_sems[exp_ix, 2 + 3 * cond] = e_data['EQ80'].sem()

    # create figure and define the color map
    cmap = plt.cm.Paired(np.linspace(0, 1, 12))

    fig = plt.figure(figsize=(8, 8), dpi=300)

    gs = fig.add_gridspec(3, 12)

    xlims = np.array([.8, 2.2])

    if data_type == 'choice':
        ylims = np.array([0, 1])
        eq_ylim = np.array([.2, .8])
        ylbl = 'p(Choose Best)'
        eq_ylbl = 'p(Choose Risky)'
        ytcks = np.array([.0, .5, 1])
        eq_ytcks = np.array([.2, .5, .8])
    else:
        ylims = np.array([400, 1000])
        ylbl = 'RT (ms)'
        eq_ylim = ylims
        eq_ylbl = ylbl
        ytcks = np.array([400, 700, 1000])
        eq_ytcks = ytcks

    for exp_ix, exp in enumerate(experiment_ids):

        # create subplots for this experiment
        train_ax = fig.add_subplot(gs[exp_ix, 0: 2])
        pure_ax = fig.add_subplot(gs[exp_ix, 2: 4])
        UE_ax = fig.add_subplot(gs[exp_ix, 4: 6])
        EQ_ax = fig.add_subplot(gs[exp_ix, 8: 12])

        train_ax.errorbar(np.array([1, 2]), train_means[exp_ix, 0:2], train_sems[exp_ix, 0:2],
                          color=cmap[1, :], capsize=0, linewidth=2, marker='s')

        train_ax.errorbar(np.array([1, 2]), train_means[exp_ix, 2:4], train_sems[exp_ix, 2:4],
                          color=cmap[5, :], capsize=0, linewidth=2, marker='s')

        train_ax.set_ylim(ylims)
        train_ax.set_yticks(ytcks)
        train_ax.set_ylabel('Experiment ' + str(exp) + '\n' + ylbl)
        train_ax.set_xticks([])
        train_ax.set_xlim(xlims)
        train_ax.spines['top'].set_visible(False)
        train_ax.spines['right'].set_visible(False)

        pure_ax.errorbar(np.array([1, 2]), pure_means[exp_ix, 0:2], pure_sems[exp_ix, 0:2],
                         color=cmap[1, :], capsize=0, linewidth=2, marker='s')

        pure_ax.errorbar(np.array([1, 2]), pure_means[exp_ix, 2:4], pure_sems[exp_ix, 2:4],
                         color=cmap[5, :], capsize=0, linewidth=2, marker='s')

        pure_ax.set_ylim(ylims)
        pure_ax.set_yticks([])
        pure_ax.set_xticks([])
        pure_ax.set_xlim(xlims)
        pure_ax.spines['top'].set_visible(False)
        pure_ax.spines['right'].set_visible(False)
        pure_ax.spines['left'].set_visible(False)

        UE_ax.errorbar(np.array([1, 2]), UE_means[exp_ix, 0:2], UE_sems[exp_ix, 0:2],
                       color=cmap[1, :], capsize=0, linewidth=2, marker='s')

        UE_ax.errorbar(np.array([1, 2]), UE_means[exp_ix, 2:4], UE_sems[exp_ix, 2:4],
                       color=cmap[5, :], capsize=0, linewidth=2, marker='s')

        UE_ax.set_ylim(ylims)
        UE_ax.set_yticks([])
        UE_ax.set_xticks([])
        UE_ax.set_xlim(xlims)
        UE_ax.spines['top'].set_visible(False)
        UE_ax.spines['right'].set_visible(False)
        UE_ax.spines['left'].set_visible(False)

        EQ_ax.errorbar(np.array([1, 2, 3]), EQ_means[exp_ix, 0:3], EQ_sems[exp_ix, 0:3],
                       color=cmap[1, :], capsize=0, linewidth=2, marker='s')

        EQ_ax.errorbar(np.array([1, 2, 3]), EQ_means[exp_ix, 3:6], EQ_sems[exp_ix, 3:6],
                       color=cmap[5, :], capsize=0, linewidth=2, marker='s')

        EQ_ax.set_ylim(eq_ylim)
        EQ_ax.set_yticks(eq_ytcks)
        EQ_ax.set_xticks([])
        EQ_ax.spines['top'].set_visible(False)
        EQ_ax.spines['right'].set_visible(False)
        EQ_ax.set_ylabel(eq_ylbl)

        if exp == 1:
            train_ax.legend(['Gain', 'Loss'])
            train_ax.set_title('Training')
            pure_ax.set_title('Pure S/R')
            UE_ax.set_title('UE SvsR')
            EQ_ax.set_title('=SvsR')
            EQ_ax.legend(['Gain', 'Loss'])

        if exp == 3:
            train_ax.set_xticks([1, 2])
            train_ax.set_xticklabels(['S', 'R'])
            pure_ax.set_xticks([1, 2])
            pure_ax.set_xticklabels(['S', 'R'])
            UE_ax.set_xticks([1, 2])
            UE_ax.set_xticklabels(['S>R', 'R>S'])
            EQ_ax.set_xticks([1, 2, 3])
            EQ_ax.set_xticklabels(['20%', '50%', '80%'])

            pure_ax.set_xlabel('Choice Condition')
            EQ_ax.set_xlabel('Equivaluable Condition')

    plt.show()
    # save plot
    plot_name = data_type + "_fig.svg"
    fig.savefig(plot_name, transparent=True)


def assess_conds_with_best_choice(gain_choice, loss_choice):
    """t-tests against chance (.5) for conditions where there was best option

    Args:
        gain_choice (dataframe): participant choice means
        loss_choice (dataframe): participant choice means

    Returns:
        gain_results (dataframe): t-test results for each condition in each experiment
                                  in gain context
        loss_results (dataframe): t-test results for each condition in each experiment
                                  in loss context
    """

    gain_results = pd.DataFrame()
    loss_results = pd.DataFrame()

    conds = ['t_safe', 't_risky', 'p_safe', 'p_risky', 'UE_safe', 'UE_risky']

    ctr = 0

    # loop over experiments
    for exp in np.unique(gain_choice['exp_num']):

        for cond in conds:

            # run the t-tests for this condition in this experiment
            gain_ttest = pg.ttest(gain_choice[cond].loc[gain_choice['exp_num'] == exp], .5)
            loss_ttest = pg.ttest(loss_choice[cond].loc[loss_choice['exp_num'] == exp], .5)

            # accumulate the results
            gain_results.at[ctr, 'experiment'] = exp
            gain_results.at[ctr, 'cond'] = cond
            loss_results.at[ctr, 'experiment'] = exp
            loss_results.at[ctr, 'cond'] = cond

            for i in gain_ttest.columns.values:
                gain_results.at[ctr, i] = gain_ttest[i].values
                loss_results.at[ctr, i] = loss_ttest[i].values

            ctr = ctr + 1

    return gain_results, loss_results


# END of assess_conds_with_best_choice()

def run_rmANOVAs(gain_data, loss_data):
    """Runs repeated measures anovas on the 4 major task conditions
    Args:
        gain_data (dataframe): participant mean responses in the gain context
        loss_data (dataframe): participant mean responses in the loss context
    
    Returns:
        exp1_stats (dict): anova tables from each condition of experiment 1
        exp2_stats (dict): anova tables from each condition of experiment 2
        exp3_stats (dict): anova tables from each condition of experiment 3
    """

    # merge the data
    conds = ['t_safe', 't_risky', 'p_safe', 'p_risky', 'UE_safe', 'UE_risky',
             'EQ20', 'EQ50', 'EQ80']
    all_data = pd.melt(pd.concat([gain_data, loss_data]),
                       id_vars=['vpnum', 'exp_num', 'context'],
                       value_vars=conds,
                       var_name='condition', value_name='resp')

    # make some basic indices for selecting the major conditions
    train_ix = all_data['condition'].str.contains('t_')
    pure_ix = all_data['condition'].str.contains('p_')
    UE_ix = all_data['condition'].str.contains('UE')
    EQ_ix = all_data['condition'].str.contains('EQ')

    # loop over each experiment
    for exp in np.unique(all_data['exp_num']):

        exp_dict = {}

        exp_ix = all_data['exp_num'] == exp

        exp_dict['train'] = pg.rm_anova(data=all_data.loc[exp_ix & train_ix],
                                        dv='resp',
                                        subject='vpnum',
                                        within=['condition', 'context'])

        exp_dict['pure'] = pg.rm_anova(data=all_data.loc[exp_ix & pure_ix],
                                       dv='resp',
                                       subject='vpnum',
                                       within=['condition', 'context'])

        exp_dict['UE'] = pg.rm_anova(data=all_data.loc[exp_ix & UE_ix],
                                     dv='resp',
                                     subject='vpnum',
                                     within=['condition', 'context'])

        exp_dict['EQ'] = pg.rm_anova(data=all_data.loc[exp_ix & EQ_ix],
                                     dv='resp',
                                     subject='vpnum',
                                     within=['condition', 'context'])

        # store the dict appropriately 
        if exp == 1:
            exp1_stats = exp_dict
        if exp == 2:
            exp2_stats = exp_dict
        if exp == 3:
            exp3_stats = exp_dict

    return exp1_stats, exp2_stats, exp3_stats


def eq_bias_by_prob_posthoc(gain_choice, loss_choice):
    """This analysis plots and assess the context-related differences in risk attitudes
    as a function of probability.

    Args:
        gain_choice (dataframe): mean participant choice patterns in the gain context
        loss_choice (dataframe): mean participant choice patterns in the loss context

    Returns:
        prob_glms (dict): summary of GLMs showing effect of probability on contextual differences
                          in p(Choose Risky) in the =SvsR trials
    """

    # initialize output
    prob_glms = {}

    # create the figure and define color map
    cmap = plt.cm.Paired(np.linspace(0, 1, 12))

    fig = plt.figure(figsize=(9, 6), dpi=300)
    gs = fig.add_gridspec(7, 3)


    # loop over each experiment
    for exp_num, exp in enumerate(np.unique(gain_choice['exp_num'])):

        exp_ix = gain_choice['exp_num'] == exp

        # create axes for this experiment
        choice_ax = fig.add_subplot(gs[0: 3, exp_num])
        diff_ax = fig.add_subplot(gs[4: 7, exp_num])

        choice_ax.errorbar([0, 1, 2], [gain_choice['EQ20'].loc[exp_ix].mean(),
                                                 gain_choice['EQ50'].loc[exp_ix].mean(),
                                                 gain_choice['EQ80'].loc[exp_ix].mean()],
                                                [gain_choice['EQ20'].loc[exp_ix].sem(),
                                                 gain_choice['EQ50'].loc[exp_ix].sem(),
                                                 gain_choice['EQ80'].loc[exp_ix].sem()],
                                     color=cmap[1, :], capsize=0, linewidth=2, marker='s')

        choice_ax.errorbar([0, 1, 2], [loss_choice['EQ20'].loc[exp_ix].mean(),
                                                 loss_choice['EQ50'].loc[exp_ix].mean(),
                                                 loss_choice['EQ80'].loc[exp_ix].mean()],
                                                [loss_choice['EQ20'].loc[exp_ix].sem(),
                                                 loss_choice['EQ50'].loc[exp_ix].sem(),
                                                 loss_choice['EQ80'].loc[exp_ix].sem()],
                                     color=cmap[5, :], capsize=0, linewidth=2, marker='s')
        choice_ax.set_title('Experiment ' + str(int(exp)))
        choice_ax.set_ylim([.2, .8])
        choice_ax.set_yticks([.2, .5, .8])
        choice_ax.set_xlim([-.2, 2.2])
        choice_ax.set_xticks([0, 1, 2])
        choice_ax.set_xticklabels(['20%', '50%', '80%'])
        choice_ax.spines['right'].set_visible(False)
        choice_ax.spines['top'].set_visible(False)

        eq20 = (gain_choice['EQ20'].loc[gain_choice['exp_num'] == exp] -
                loss_choice['EQ20'].loc[loss_choice['exp_num'] == exp])
        eq50 = (gain_choice['EQ50'].loc[gain_choice['exp_num'] == exp] -
                loss_choice['EQ50'].loc[loss_choice['exp_num'] == exp])
        eq80 = (gain_choice['EQ80'].loc[gain_choice['exp_num'] == exp] -
                loss_choice['EQ80'].loc[loss_choice['exp_num'] == exp])

        diff_ax.bar([0, 1, 2], [eq20.mean(), eq50.mean(), eq80.mean()], color=cmap[8, :])
        diff_ax.plot([-2, 4], [0, 0], color='Gray')
        diff_ax.errorbar([0, 1, 2], [eq20.mean(), eq50.mean(), eq80.mean()],
                                  [eq20.sem(), eq50.sem(), eq80.sem()], color='black',
                                   capsize=0, linewidth=2, marker='s')
        diff_ax.set_xticks([0, 1, 2])
        diff_ax.set_xticklabels(['20%', '50%', '80%'])
        diff_ax.set_ylim([-.2, .4])
        diff_ax.set_xlim([-.8, 2.8])
        diff_ax.spines['right'].set_visible(False)
        diff_ax.spines['top'].set_visible(False)

        if exp > 1:
            choice_ax.spines['left'].set_visible(False)
            diff_ax.spines['left'].set_visible(False)


        # fit a regression
        n_subs = len(eq20)
        reg_df = pd.DataFrame()
        reg_df['bias'] = pd.concat([eq20, eq80, eq80], axis=0)
        reg_df['prob'] = np.concatenate([np.ones(shape=(n_subs,)) * .2,
                                         np.ones(shape=(n_subs,)) * .5,
                                         np.ones(shape=(n_subs,)) * .8, ])

        exp_glm = smf.glm('bias~prob', data=reg_df).fit()

        prob_glms['Exp_' + str(int(exp))] = exp_glm.summary()

    plt.show()
    #fig.savefig("EqualSvsU.svg", transparent=True)
    return prob_glms


def assess_prob_expval_infoformat_regression(gain_choice, loss_choice):
    """This function assesses the effects of information format and expected value across the
    three experiments.

    Args:
        gain_choice (dataframe): participant mean choices in gain context
        loss_choice (dataframe): participant mean choices in loss context
    Returns:
        bias_mdl (glm summary): summary of a GLM assessing role of probability,
            information format, and expected value in cross-context choice biases
    """

    biases = np.concatenate([gain_choice['EQ20'] - loss_choice['EQ20'],
                             gain_choice['EQ50'] - loss_choice['EQ50'],
                             gain_choice['EQ80'] - loss_choice['EQ80']], axis=0)

    subj_factor = np.concatenate([gain_choice['vpnum'],
                                  gain_choice['vpnum'],
                                  gain_choice['vpnum']], axis=0).astype((int))

    exp_factor = np.concatenate([gain_choice['exp_num'],
                                  gain_choice['exp_num'],
                                  gain_choice['exp_num']], axis=0).astype(int)

    ev_factor = np.ones_like(exp_factor)
    info_format_factor = np.ones_like(exp_factor)
    prob_factor = np.concatenate([np.ones(shape=(len(gain_choice)))*.2,
                  np.ones(shape=(len(gain_choice))) * .5,
                  np.ones(shape=(len(gain_choice))) * .8], axis=0)


    ev_factor[exp_factor == 2] = -1
    info_format_factor[exp_factor == 3] = -1

    # put everything into a dataframe
    reg_df = pd.DataFrame()
    reg_df['bias'] = biases
    reg_df['prob'] = prob_factor
    reg_df['subj'] = subj_factor
    reg_df['ev'] = ev_factor
    reg_df['exp'] = exp_factor
    reg_df['info_type'] = info_format_factor
    bias_mdl = smf.glm('bias~prob*ev*info_type', data=reg_df).fit().summary()

    return bias_mdl

def win_stay_analysis(gain_all, loss_all):
    """This analysis looks for instances where a certain option was chosen 
    and yielded a non-zero outcome (in gain context; opposite in loss context) 
    and asks how likely the person is to select that option the next time it's presented.

    Args:
        gain_all (dataframe): trial-by-trial response data from gain context
        loss_all (dataframe): trial-by-trial response data from loss context
    Returns:
        winstay_long(dataframe): a large data frame with columns: vpnum, context, winstay probability, 
                                experiment #, and EQbias
        winstay_wide (dataframe): wide-format data frame that's convenient for plotting later
    """

    # combine the data
    all_data = pd.concat([gain_all, loss_all])

    # only keep the "pure" trials
    eq_ix = all_data['probLeft'] == all_data['probRight']
    main_block = all_data['phase'] == 'exp'
    all_data = all_data.loc[~eq_ix & main_block]

    # get subject IDs
    subj_ids = all_data['vpNum'].unique()
    probabilities = np.sort(all_data['probLeft'].unique())

    # initialize the results array
    winstay_long = pd.DataFrame()
    winstay_wide = pd.DataFrame()
    ctr = 0
    ctx_ctr = 0
    first_pass = True

    # loop over each experiment
    for exp in np.unique(all_data['exp_num']):

        # loop through each subject in this experiment
        for s in all_data['vpNum'].loc[all_data['exp_num'] == exp].unique():

            sdata = all_data.loc[all_data.vpNum == s]

            # loop over the contexts
            for ctx in np.unique(sdata['blkType']):

                ctx_data = sdata.loc[sdata['blkType'] == ctx]

                # now loop over each stim_type (safe/risky)
                for t_ix, im_type in enumerate(np.sort(np.unique(ctx_data['imgLeftType']))):

                    l_t_ix = ctx_data['imgLeftType'] == im_type
                    r_t_ix = ctx_data['imgRightType'] == im_type

                    if first_pass:
                        first_pass = False
                    else:
                        ctx_ctr = ctx_ctr + 1

                    winstay_wide.at[ctx_ctr, 'exp'] = exp
                    winstay_wide.at[ctx_ctr, 'subj'] = s
                    winstay_wide.at[ctx_ctr, 'context'] = ctx
                    winstay_wide.at[ctx_ctr, 'type'] = im_type

                    # now loop over each probability
                    for p_ix, prob in enumerate(probabilities):

                        # get trials where this probability was present
                        prob_tnums = np.argwhere(((((ctx_data['probLeft'] == prob) & l_t_ix) |
                                                   ((ctx_data['probRight'] == prob) & r_t_ix)).to_numpy()))

                        # get trial numbers of when selecting this prob led to a non-zero outcome       
                        hit_tnums = np.argwhere((((ctx_data['chosen_prob'] == prob) &
                                                  (ctx_data['chosen_type'] == im_type)) &
                                                 (ctx_data['rewardCode'] == 1)).to_numpy())

                        prob_ws = np.zeros(shape=len(hit_tnums))
                        prob_ws[:] = np.nan

                        # loop through each of these trials and find the next time this
                        # stim was present           
                        for i in range(len(hit_tnums) - 1):
                            # candidate future trials where this prob was present
                            trials_in_range = np.argwhere(prob_tnums > hit_tnums[i])
                            next_t = prob_tnums[trials_in_range[0, 0]]
                            prob_ws[i] = (ctx_data['chosen_prob'].iloc[next_t] == prob).values.astype(int)

                        # get the mean WS / LS probability for this probability and aggregate results
                        winstay_long.at[ctr, 'exp'] = exp
                        winstay_long.at[ctr, 'subj'] = s
                        winstay_long.at[ctr, 'context'] = ctx
                        winstay_long.at[ctr, 'type'] = im_type
                        winstay_long.at[ctr, 'prob'] = prob

                        if np.sum(~np.isnan(prob_ws)) > 0:
                            winstay_mean = np.nanmean(prob_ws)
                        else:
                            winstay_mean = 0

                        winstay_long.at[ctr, 'winstay'] = winstay_mean
                        ctr = ctr + 1

                        # now add data to the wide-format dataframe
                        winstay_wide.at[ctx_ctr, prob] = winstay_mean

                    # END of looping over probabilities
                # END of looping over stim types (safe, risky)
            # END of looping over contexts
        # END of looping over subjects
    # END of looping over experiments

    return winstay_long, winstay_wide


# END of WinStayAnalysis


def plot_assess_hit_stay(winstay_long, winstay_wide, gain_choice, loss_choice):
    """Plots and statistically assesses how hit-stay probabilities vary as a function of
    stimulus type, probability, and context. Also examines how win-stay probabilities
    correspond to risk attitudes (as measured in the =SvsR trials)

    Args:
        winstay_long (dataframe): long-format dataframe convenient for stats
        winstay_wide (dataframe): wide-format dataframe convenient for plotting 
        gain_choice (dataframe): participant choice means in gain context
        loss_choice (dataframe): participant choice means in loss context

    Returns:
        _type_: _description_
    """
    GLM_results = {}
    all_choice = pd.concat([gain_choice, loss_choice])

    # create the figure and define color map
    cmap = plt.cm.Paired(np.linspace(0, 1, 12))

    fig = plt.figure(figsize=(10, 10), dpi=300)
    gs = fig.add_gridspec(3, 17)

    # plot raw win-stay results
    for exp_ix, exp in enumerate(np.unique(winstay_wide['exp'])):
        e_df = winstay_wide.loc[winstay_wide['exp'] == exp]
        e_ws_mean = e_df.groupby(['context', 'type'])[[0.2, 0.5, 0.8]].mean().to_numpy()
        e_ws_sem = e_df.groupby(['context', 'type'])[[0.2, 0.5, 0.8]].sem().to_numpy()

        # compute slopes for hit-stay curves
        gain_ix = e_df['context'] == 'Gain'
        loss_ix = e_df['context'] == 'Loss'
        gain_df = pd.DataFrame()
        loss_df = pd.DataFrame()
        gain_df['hit_stay'] = pd.concat([e_df[0.2].loc[gain_ix],
                                         e_df[0.5].loc[gain_ix],
                                         e_df[0.8].loc[gain_ix]])
        gain_df['prob'] = np.concatenate([np.ones(shape=np.sum(gain_ix), ) * .2,
                                          np.ones(shape=np.sum(gain_ix), ) * .5,
                                          np.ones(shape=np.sum(gain_ix), ) * .8, ])
        loss_df['hit_stay'] = pd.concat([e_df[0.2].loc[loss_ix],
                                         e_df[0.5].loc[loss_ix],
                                         e_df[0.8].loc[loss_ix]])
        loss_df['prob'] = np.concatenate([np.ones(shape=np.sum(loss_ix), ) * .2,
                                          np.ones(shape=np.sum(loss_ix), ) * .5,
                                          np.ones(shape=np.sum(loss_ix), ) * .8, ])

        gain_hs_mdl = smf.glm('hit_stay ~ prob', gain_df).fit()
        loss_hs_mdl = smf.glm('hit_stay ~ prob', loss_df).fit()

        GLM_results['Exp_' + str(int(exp)) + 'hs_gain'] = gain_hs_mdl.summary()
        GLM_results['Exp_' + str(int(exp)) + 'hs_loss'] = loss_hs_mdl.summary()

        # create subplots for this experiment
        ws_ax = fig.add_subplot(gs[exp_ix, 0: 5])
        gain_ax = fig.add_subplot(gs[exp_ix, 6: 11])
        loss_ax = fig.add_subplot(gs[exp_ix, 12: 17])
        ws_ax.errorbar([.2, .5, .8], e_ws_mean[0, :], e_ws_sem[0, :], marker='s', markersize=5,
                       linewidth=2, color=cmap[1, :])
        ws_ax.errorbar([.2, .5, .8], e_ws_mean[1, :], e_ws_sem[1, :], marker='s', markersize=5,
                       linewidth=2, color=cmap[0, :])
        ws_ax.errorbar([.2, .5, .8], e_ws_mean[2, :], e_ws_sem[2, :], marker='s', markersize=5,
                       linewidth=2, color=cmap[5, :])
        ws_ax.errorbar([.2, .5, .8], e_ws_mean[3, :], e_ws_sem[3, :], marker='s', markersize=5,
                       linewidth=2, color=cmap[4, :])
        ws_ax.set_xlim([.1, .9])
        ws_ax.set_ylim([0, 1])
        ws_ax.set_xticks([.2, .5, .8])
        ws_ax.set_yticks([0, .5, 1])
        ws_ax.set_ylabel('Experiment ' + str(int(exp)) + '\n' + 'p(Hit-Stay)')

        for ctx_ix, ctx in enumerate(np.unique(winstay_wide['context'])):

            safe_c_df = e_df.loc[(e_df['context'] == ctx) & (e_df['type'] == 'Safe')]
            risky_c_df = e_df.loc[(e_df['context'] == ctx) & (e_df['type'] == 'Risky')]

            choice_df = all_choice.loc[(all_choice['exp_num'] == exp) & (all_choice['context'] == ctx)]

            eq_bias = np.array([choice_df['EQ20'].values, choice_df['EQ50'].values,
                                choice_df['EQ80'].values]).T

            subj_ids = np.unique(e_df['subj'])
            n_subjs = len(subj_ids)

            probs = [0.2, 0.5, 0.8]
            ws_diffs = np.empty(shape=(n_subjs, 3))

            for p_ix, p in enumerate(probs):
                ws_diffs[:, p_ix] = risky_c_df[p].values - safe_c_df[p].values

                if ctx == 'Gain':
                    gain_ax.scatter(ws_diffs[:, p_ix], eq_bias[:, p_ix], s=15)
                    gain_ax.set_xlim([-.7, .7])
                    gain_ax.set_ylabel('p(Choose Risky in =SvsR)')
                else:
                    loss_ax.scatter(ws_diffs[:, p_ix], eq_bias[:, p_ix], s=15)
                    loss_ax.set_xlim([-.7, .7])

            if exp == 3:
                ws_ax.set_xlabel('Risk Level')
                gain_ax.set_xlabel('Hit-Stay Safe - Hit-Stay Risky')

            if exp == 1:
                ws_ax.set_title('Hit-Stay Analysis')
                ws_ax.legend(['Gain, Risky', 'Gain, Safe', 'Loss, Risky', 'Loss, Safe'],
                             ncol=2, fontsize=8)
                gain_ax.legend(['20%', '50%', '80%'], fontsize=8)
                gain_ax.set_title('Relating HS to Gain Risk Attitude')
                loss_ax.set_title('Relating HS to Loss Risk Attitude')

            # create dataframe for mixed effects model
            reg_df = pd.DataFrame()
            reg_df['subj'] = np.concatenate((subj_ids, subj_ids, subj_ids))
            reg_df['ws_diff'] = np.concatenate((ws_diffs[:, 0], ws_diffs[:, 1], ws_diffs[:, 2]))
            reg_df['eq_bias'] = np.concatenate((eq_bias[:, 0], eq_bias[:, 1], eq_bias[:, 2]))
            reg_df['prob'] = np.concatenate((np.ones((n_subjs, 1)) * .2, np.ones((n_subjs, 1)) * .5,
                                             np.ones((n_subjs, 1)) * .8))

            # fit the general linear model
            plt_mdl = smf.glm('eq_bias ~ ws_diff', data=reg_df).fit()

            # pull out slope and intercept for plotting the regression line
            if ctx == 'Gain':
                xlims = np.array(gain_ax.get_xlim())
                gain_ax.plot(xlims, (xlims * plt_mdl.params[1]) + plt_mdl.params[0], color='black', linewidth=2)
                gain_ax.set_yticks([0, .5, 1])
            else:
                xlims = np.array(loss_ax.get_xlim())
                loss_ax.plot(xlims, (xlims * plt_mdl.params[1]) + plt_mdl.params[0], color='black', linewidth=2)
                loss_ax.set_yticks([0, .5, 1])

            # store the results of the GLM
            save_name = 'Exp' + str(int(exp)) + '_' + ctx
            GLM_results[save_name] = plt_mdl.summary()
    plt.show()
    # save plot
    fig.savefig("hit_stay_fig.svg", transparent=True)
    return GLM_results


# END of plotting and analyzing win-stay data


def risk_sensitive_RL(gain_all, loss_all):
    """Fits a risk-sensitive RL model for each participant in each context
    Args:
        gain_all (dataframe): trial-by-trial choice data in gain context
        loss_all (dataframe): trial-by-trial choice data in loss context

    Returns:
        gain_fits (dataframe): best fitting quantile and other parameters
        loss_fits (dataframe): same as with gain_fits
    """

    # initialize output
    gain_fits = pd.DataFrame()
    loss_fits = pd.DataFrame()
    grid_step = .05
    alpha_vals = np.linspace(grid_step, 1, int(1 / grid_step))
    betas = np.linspace(1, 10, 10)
    n_params = 3

    # get the combinations of alphas and betas
    # 1st col = alpha_plus, 2nd = alpha_minus, 3rd = beta
    Q_params = np.array(np.meshgrid(alpha_vals, alpha_vals, betas)).T.reshape(-1, 3)

    # merge the data
    all_data = pd.concat([gain_all, loss_all])

    # keep track of parameter sets per participant
    ctr = 0

    # loop over experiments
    for exp in np.unique(all_data['exp_num']):
        # loop over contexts
        for ctx in np.unique(all_data['blkType']):

            print('\n')
            print(ctx + ' Experiment #: ' + str(int(exp)) +
                  ' of ' + str(len(np.unique(all_data['exp_num']))))

            exp_subjs = np.unique(all_data['vpNum'].loc[(all_data['exp_num'] == exp) &
                                                        (all_data['blkType'] == ctx)])
            n_subjs = len(exp_subjs)

            # loop over participants
            for p_ix, p in enumerate(exp_subjs):

                print('\rSubject #: ' + str(p_ix + 1) + ' of ' + str(n_subjs) + '   ', end='')

                # extract some pieces so we can run using numpy
                p_data = all_data.loc[(all_data['exp_num'] == exp) &
                                      (all_data['blkType'] == ctx) &
                                      (all_data['vpNum'] == p)]

                t_stim = np.empty((len(p_data), 2))
                participant_choice = np.empty((len(p_data), 1))
                for ix, i in enumerate(np.unique(p_data['probLeft'])):
                    t_stim[(p_data['probLeft'] == i) & (p_data['imgLeftType'] == 'Safe'), 0] = ix
                    t_stim[(p_data['probLeft'] == i) & (p_data['imgLeftType'] == 'Risky'), 0] = ix + 3
                    t_stim[(p_data['probRight'] == i) & (p_data['imgRightType'] == 'Safe'), 1] = ix
                    t_stim[(p_data['probRight'] == i) & (p_data['imgRightType'] == 'Risky'), 1] = ix + 3
                    participant_choice[(p_data['chosen_prob'] == i) & (p_data['chosen_type'] == 'Safe')] = ix
                    participant_choice[(p_data['chosen_prob'] == i) & (p_data['chosen_type'] == 'Risky')] = ix + 3

                participant_choice = participant_choice.astype(int)
                n_stim = len(np.unique(t_stim))
                t_stim = t_stim.astype(int)
                t_outcomes = p_data['rewardPoints'].values.astype(int)
                t_chosen_type = p_data['chosen_type'].to_numpy()

                eq20_ix = ((p_data['probLeft'] == .2) & (p_data['probRight'] == .2)).to_numpy()
                eq50_ix = ((p_data['probLeft'] == .5) & (p_data['probRight'] == .5)).to_numpy()
                eq80_ix = ((p_data['probLeft'] == .8) & (p_data['probRight'] == .8)).to_numpy()
                all_eq = eq20_ix | eq50_ix | eq80_ix
                picked_risky = (t_chosen_type == 'Risky').astype(int)

                # initialize intermediate output for the model runs for this participant / context
                sum_log_likelihood = np.zeros((len(Q_params), 1))
                eq_bias_LL = np.zeros((len(Q_params), 1))
                accuracy = np.zeros((len(Q_params), 1))
                eq_bias = np.zeros((len(Q_params), 3))

                # now loop over Q_params
                for q in range(len(Q_params)):

                    # initialize the output of this agent
                    t_log_likelihood = np.empty((len(p_data), 1))
                    t_agent_choices = np.zeros((len(p_data), 1)).astype(int)

                    # instantiate the agent with this param set
                    agent = RS_agent(n_stim, Q_params[q, 0], Q_params[q, 1], Q_params[q, 2])

                    # step through the trials
                    for t in range(len(p_data)):

                        # what's the probability the agent picks the left option?
                        p_choose_left, t_agent_choices[t] = agent.softmax(t_stim[t, 0], t_stim[t, 1])

                        # store the log likelihood that the agent picked the same option
                        # as the human participant
                        if (p_choose_left > .5) & (t_stim[t, 0] == participant_choice[t]):
                            t_log_likelihood[t] = np.log(p_choose_left)
                        else:
                            t_log_likelihood[t] = np.log(1 - p_choose_left)

                        # update the model
                        agent.update(participant_choice[t], t_outcomes[t], t_chosen_type[t])

                    # assess some characteristics of this model run
                    sum_log_likelihood[q] = np.sum(t_log_likelihood)
                    eq_bias_LL[q] = np.sum(t_log_likelihood[eq20_ix | eq50_ix | eq80_ix])
                    accuracy[q] = np.mean(t_agent_choices == participant_choice)
                    eq_bias[q, 0] = np.mean(t_agent_choices[eq20_ix] == 3)
                    eq_bias[q, 1] = np.mean(t_agent_choices[eq50_ix] == 4)
                    eq_bias[q, 2] = np.mean(t_agent_choices[eq80_ix] == 5)

                # save the best fitting parameters for this participant
                best_params_ix = np.argmax(sum_log_likelihood)
                quantile = Q_params[best_params_ix, 0] / np.sum(Q_params[best_params_ix, 0:2])
                beta = Q_params[best_params_ix, 2]

                if ctx == 'Gain':
                    gain_fits.at[ctr, 'exp'] = exp
                    gain_fits.at[ctr, 'context'] = ctx
                    gain_fits.at[ctr, 'subj'] = p
                    gain_fits.at[ctr, 'aic'] = (-2 * sum_log_likelihood[best_params_ix]) + (2 * n_params)
                    gain_fits.at[ctr, 'quantile'] = quantile
                    gain_fits.at[ctr, 'beta'] = beta
                    gain_fits.at[ctr, 'accuracy'] = accuracy[best_params_ix]
                    gain_fits.at[ctr, 'total_p_risk_pref'] = np.mean(picked_risky[all_eq])
                    gain_fits.at[ctr, 'agent_eq20'] = eq_bias[best_params_ix, 0]
                    gain_fits.at[ctr, 'agent_eq50'] = eq_bias[best_params_ix, 1]
                    gain_fits.at[ctr, 'agent_eq80'] = eq_bias[best_params_ix, 2]
                    gain_fits.at[ctr, 'subj_eq20'] = np.mean(picked_risky[eq20_ix])
                    gain_fits.at[ctr, 'subj_eq50'] = np.mean(picked_risky[eq50_ix])
                    gain_fits.at[ctr, 'subj_eq80'] = np.mean(picked_risky[eq80_ix])

                if ctx == 'Loss':
                    loss_fits.at[ctr, 'exp'] = exp
                    loss_fits.at[ctr, 'context'] = ctx
                    loss_fits.at[ctr, 'subj'] = p
                    loss_fits.at[ctr, 'aic'] = (-2 * sum_log_likelihood[best_params_ix]) + (2 * n_params)
                    loss_fits.at[ctr, 'quantile'] = quantile
                    loss_fits.at[ctr, 'beta'] = beta
                    loss_fits.at[ctr, 'accuracy'] = accuracy[best_params_ix]
                    loss_fits.at[ctr, 'total_p_risk_pref'] = np.mean(picked_risky[all_eq])
                    loss_fits.at[ctr, 'agent_eq20'] = eq_bias[best_params_ix, 0]
                    loss_fits.at[ctr, 'agent_eq50'] = eq_bias[best_params_ix, 1]
                    loss_fits.at[ctr, 'agent_eq80'] = eq_bias[best_params_ix, 2]
                    loss_fits.at[ctr, 'subj_eq20'] = np.mean(picked_risky[eq20_ix])
                    loss_fits.at[ctr, 'subj_eq50'] = np.mean(picked_risky[eq50_ix])
                    loss_fits.at[ctr, 'subj_eq80'] = np.mean(picked_risky[eq80_ix])

                ctr = ctr + 1
    return gain_fits, loss_fits


def differential_risk_sensitive_RL(gain_all, loss_all):
    """Fits a differential risk-sensitive RL model for each participant in each context

    Args:
        gain_all (dataframe): trial-by-trial choice data in gain context
        loss_all (dataframe): trial-by-trial choice data in loss context

    Returns:
        gain_fits (dataframe): best fitting quantile and other parameters
        loss_fits (dataframe): same as with gain_fits
    """

    # initialize output
    gain_fits = pd.DataFrame()
    loss_fits = pd.DataFrame()

    alphavals = np.linspace(.1, 1, int(1 / .1))
    # betas = np.linspace(1,10,10)
    n_params = 4
    betas = np.array([1])  # this is for debugging

    # get the combinations of alphas and betas
    # 1st col = safe_a_plus, 2nd = safe_a_minus, 3rd/4th = safe/risky_a_plus, 5th = beta
    Qparams = np.array(np.meshgrid(alphavals, alphavals,
                                   alphavals, alphavals, betas)).T.reshape(-1, 5)

    all_data = pd.concat([gain_all, loss_all])

    ctr = 0

    # loop over experiments
    for exp in np.unique(all_data['exp_num']):
        # loop over contexts
        for ctx in np.unique(all_data['blkType']):

            print('\n')
            print(ctx + ' Experiment #: ' + str(int(exp)) +
                  ' of ' + str(len(np.unique(all_data['exp_num']))))

            exp_subjs = np.unique(all_data['vpNum'].loc[(all_data['exp_num'] == exp) &
                                                        (all_data['blkType'] == ctx)])
            n_subjs = len(exp_subjs)

            # loop over participants
            for p_ix, p in enumerate(exp_subjs):

                print('\rSubject #: ' + str(p_ix + 1) + ' of ' + str(n_subjs) + '   ', end='')

                # extract some pieces so we can run using numpy
                p_data = all_data.loc[(all_data['exp_num'] == exp) &
                                      (all_data['blkType'] == ctx) &
                                      (all_data['vpNum'] == p)]

                t_stim = np.empty((len(p_data), 2))
                participant_choice = np.empty((len(p_data), 1)).astype(int)
                for ix, i in enumerate(np.unique(p_data['probLeft'])):
                    t_stim[(p_data['probLeft'] == i) & (p_data['imgLeftType'] == 'Safe'), 0] = ix
                    t_stim[(p_data['probLeft'] == i) & (p_data['imgLeftType'] == 'Risky'), 0] = ix + 3
                    t_stim[(p_data['probRight'] == i) & (p_data['imgRightType'] == 'Safe'), 1] = ix
                    t_stim[(p_data['probRight'] == i) & (p_data['imgRightType'] == 'Risky'), 1] = ix + 3
                    participant_choice[(p_data['chosen_prob'] == i) & (p_data['chosen_type'] == 'Safe')] = ix
                    participant_choice[(p_data['chosen_prob'] == i) & (p_data['chosen_type'] == 'Risky')] = ix + 3

                n_stim = len(np.unique(t_stim))
                t_stim = t_stim.astype(int)
                t_outcomes = p_data['rewardPoints'].values.astype(int)
                t_chosen_type = p_data['chosen_type'].to_numpy()

                eq20_ix = ((p_data['probLeft'] == .2) & (p_data['probRight'] == .2)).to_numpy()
                eq50_ix = ((p_data['probLeft'] == .5) & (p_data['probRight'] == .5)).to_numpy()
                eq80_ix = ((p_data['probLeft'] == .8) & (p_data['probRight'] == .8)).to_numpy()
                all_eq = eq20_ix | eq50_ix | eq80_ix
                picked_risky = (t_chosen_type == 'Risky').astype(int)

                # initialize intermediate output for the model runs for this participant / context
                sum_log_likelihood = np.zeros((len(Qparams), 1))
                eq_bias_LL = np.zeros((len(Qparams), 1))
                accuracy = np.zeros((len(Qparams), 1))
                eq_bias = np.zeros((len(Qparams), 3))

                # now loop over Qparams
                for q in range(len(Qparams)):

                    # initialize the output of this agent
                    t_log_likelihood = np.empty((len(p_data), 1))
                    t_agent_choices = np.zeros((len(p_data), 1)).astype(int)

                    # instantiate the agent with this param set
                    agent = differential_RS_agent(n_stim,
                                                  Qparams[q, 0], Qparams[q, 1],
                                                  Qparams[q, 2], Qparams[q, 3], Qparams[q, 4])

                    # step through the trials
                    for t in range(len(p_data)):

                        # what's the probability the agent picks the left option?
                        p_choose_left, t_agent_choices[t] = agent.softmax(t_stim[t, 0], t_stim[t, 1])

                        # store the log likelihood that the agent picked the same option
                        # as the human participant
                        if (p_choose_left > .5) & (t_stim[t, 0] == participant_choice[t]):
                            t_log_likelihood[t] = np.log(p_choose_left)
                        else:
                            t_log_likelihood[t] = np.log(1 - p_choose_left)

                        # update the model
                        agent.update(participant_choice[t], t_outcomes[t], t_chosen_type[t])

                    # assess some characteristics of this model run
                    sum_log_likelihood[q] = np.sum(t_log_likelihood)
                    eq_bias_LL[q] = np.sum(t_log_likelihood[eq20_ix | eq50_ix | eq80_ix])
                    accuracy[q] = np.mean(t_agent_choices == participant_choice)
                    eq_bias[q, 0] = np.mean(t_agent_choices[eq20_ix] == 3)
                    eq_bias[q, 1] = np.mean(t_agent_choices[eq50_ix] == 4)
                    eq_bias[q, 2] = np.mean(t_agent_choices[eq80_ix] == 5)

                # save the best fitting parameters for this participant
                best_params_ix = np.argmax(sum_log_likelihood)
                safe_quantile = Qparams[best_params_ix, 0] / np.sum(Qparams[best_params_ix, 0:2])
                risky_quantile = Qparams[best_params_ix, 2] / np.sum(Qparams[best_params_ix, 2:4])

                if ctx == 'Gain':
                    gain_fits.at[ctr, 'exp'] = exp
                    gain_fits.at[ctr, 'context'] = ctx
                    gain_fits.at[ctr, 'subj'] = p
                    gain_fits.at[ctr, 'aic'] = (-2 * sum_log_likelihood[best_params_ix]) + (2 * n_params)
                    gain_fits.at[ctr, 'safe_quantile'] = safe_quantile
                    gain_fits.at[ctr, 'risky_quantile'] = risky_quantile
                    gain_fits.at[ctr, 'accuracy'] = accuracy[best_params_ix]
                    gain_fits.at[ctr, 'total_p_risk_pref'] = np.mean(picked_risky[all_eq])
                    gain_fits.at[ctr, 'agent_eq20'] = eq_bias[best_params_ix, 0]
                    gain_fits.at[ctr, 'agent_eq50'] = eq_bias[best_params_ix, 1]
                    gain_fits.at[ctr, 'agent_eq80'] = eq_bias[best_params_ix, 2]
                    gain_fits.at[ctr, 'subj_eq20'] = np.mean(picked_risky[eq20_ix])
                    gain_fits.at[ctr, 'subj_eq50'] = np.mean(picked_risky[eq50_ix])
                    gain_fits.at[ctr, 'subj_eq80'] = np.mean(picked_risky[eq80_ix])

                if ctx == 'Loss':
                    loss_fits.at[ctr, 'exp'] = exp
                    loss_fits.at[ctr, 'context'] = ctx
                    loss_fits.at[ctr, 'subj'] = p
                    loss_fits.at[ctr, 'aic'] = (-2 * sum_log_likelihood[best_params_ix]) + (2 * n_params)
                    loss_fits.at[ctr, 'safe_quantile'] = safe_quantile
                    loss_fits.at[ctr, 'risky_quantile'] = risky_quantile
                    loss_fits.at[ctr, 'accuracy'] = accuracy[best_params_ix]
                    loss_fits.at[ctr, 'total_p_risk_pref'] = np.mean(picked_risky[all_eq])
                    loss_fits.at[ctr, 'agent_eq20'] = eq_bias[best_params_ix, 0]
                    loss_fits.at[ctr, 'agent_eq50'] = eq_bias[best_params_ix, 1]
                    loss_fits.at[ctr, 'agent_eq80'] = eq_bias[best_params_ix, 2]
                    loss_fits.at[ctr, 'subj_eq20'] = np.mean(picked_risky[eq20_ix])
                    loss_fits.at[ctr, 'subj_eq50'] = np.mean(picked_risky[eq50_ix])
                    loss_fits.at[ctr, 'subj_eq80'] = np.mean(picked_risky[eq80_ix])

                ctr = ctr + 1
    return gain_fits, loss_fits


def plot_RSRL_results(gain_fits, loss_fits):
    """Plots and analyzes results of the RS-RL models.
    Args:
        gain_fits (dataframe): participant mean responses in the gain context
        loss_fits (dataframe): participant mean responses in the loss context

    Returns:
        expectile_models (dict): summary of GLMs fit to model expectiles and risk preference
        match_models (dict): summary of GLM fit to a given experiment
    """

    # initialize output
    expectile_models = {}
    match_models = {}

    # create figure and define colormap
    cmap = plt.cm.Paired(np.linspace(0, 1, 12))
    cmap2 = plt.cm.tab20(np.linspace(0, 1, 20))
    fig = plt.figure(figsize=(14, 8), dpi=300)
    gs = fig.add_gridspec(3, 16)

    experiment_ids = np.unique(gain_fits['exp']).astype(int)

    for exp_ix, exp in enumerate(experiment_ids):

        # create subplots for this experiment
        bias_ax = fig.add_subplot(gs[exp_ix, 0: 3])
        accuracy_ax = fig.add_subplot(gs[exp_ix, 4: 6])
        qreg_ax = fig.add_subplot(gs[exp_ix, 7: 10])
        prob_match_ax = fig.add_subplot(gs[exp_ix, 11: 14])

        # pull this experiment's data
        e_gain = gain_fits.loc[gain_fits['exp'] == exp]
        e_loss = loss_fits.loc[loss_fits['exp'] == exp]

        # pull individual human biases for each experiment
        gain_bias = np.array([e_gain['subj_eq20'],
                              e_gain['subj_eq50'],
                              e_gain['subj_eq80']])
        loss_bias = np.array([e_loss['subj_eq20'],
                              e_loss['subj_eq50'],
                              e_loss['subj_eq80']])

        # pull the biases of the best-fitting agents for each human/experiment
        a_gain_bias = np.array([e_gain['agent_eq20'],
                                e_gain['agent_eq50'],
                                e_gain['agent_eq80']])
        a_loss_bias = np.array([e_loss['agent_eq20'],
                                e_loss['agent_eq50'],
                                e_loss['agent_eq80']])

        bias_ax.plot(np.array([.2, .5, .8]), gain_bias, color=cmap[0, :], linewidth=1)
        bias_ax.plot(np.array([.2, .5, .8]), loss_bias, color=cmap[4, :], linewidth=1)

        bias_ax.plot(np.array([.2, .5, .8]), np.mean(gain_bias, axis=1),
                     color=cmap[1, :], marker='s', linewidth=3, markersize=8, label='Gain')

        bias_ax.plot(np.array([.2, .5, .8]), np.mean(loss_bias, axis=1),
                     color=cmap[5, :], marker='s', linewidth=3, markersize=8, label='Loss')
        bias_ax.set_xticks([.2, .5, .8])
        bias_ax.set_yticks([0, .5, 1])
        bias_ax.set_xlim([.15, .85])
        bias_ax.set_ylabel('Experiment ' + str(exp) + '\n p(Choose Risky)')

        # plot mean model accuracy
        # make some jitered x vals
        gain_x = np.random.uniform(0, .4, len(e_gain))
        loss_x = np.random.uniform(.6, 1, len(e_loss))

        accuracy_ax.scatter(gain_x, e_gain['accuracy'], marker='s', color=cmap[0, :],
                            s=10)
        accuracy_ax.scatter(loss_x, e_loss['accuracy'], marker='s', color=cmap[4, :],
                            s=10)
        accuracy_ax.scatter(.2, e_gain['accuracy'].mean(), marker='s', color=cmap[1, :],
                            s=100, edgecolors='black')
        accuracy_ax.scatter(.8, e_loss['accuracy'].mean(), marker='s', color=cmap[5, :],
                            s=100, edgecolors='black')
        accuracy_ax.set_xticks([.2, .8])
        accuracy_ax.set_yticks([0, .5, 1])
        accuracy_ax.set_ylim([0, 1])
        accuracy_ax.set_xticklabels(['Gain', 'Loss'])
        accuracy_ax.set_ylabel('p(Agent = Human Choice)')

        # plot relationship between quantile and risk-attitude
        qreg_ax.scatter(e_gain['quantile'], e_gain['total_p_risk_pref'], marker='s',
                        s=10, color=cmap[0, :])
        qreg_ax.scatter(e_loss['quantile'], e_loss['total_p_risk_pref'], marker='s',
                        s=10, color=cmap[4, :])
        qreg_ax.set_xlim([0, 1])
        qreg_ax.set_ylim([0, 1])

        # do regression and plot regression line
        reg_df = pd.DataFrame()
        reg_df['quantile'] = np.concatenate((e_gain['quantile'], e_loss['quantile']))
        reg_df['risk_bias'] = np.concatenate((e_gain['total_p_risk_pref'], e_loss['total_p_risk_pref']))
        reg_df['context'] = np.concatenate((np.ones((len(e_gain), 1)),
                                            np.ones((len(e_gain), 1)) * -1))

        # fit the GLM
        q_mdl = smf.glm('risk_bias ~ quantile*context', data=reg_df).fit()
        q_xlim = np.array([0, 1])
        qreg_ax.plot(q_xlim, (q_xlim * q_mdl.params[1]) + q_mdl.params[0], color='black', linewidth=2)
        qreg_ax.set_xticks([0, .5, 1])
        qreg_ax.set_yticks([0, .5, 1])
        qreg_ax.set_ylabel('p(Choose Risky in =SvsR)')

        # now plot how the model's choices predict the human choices at each probability
        prob_match_ax.scatter(a_gain_bias[0, :], gain_bias[0, :], marker='o', s=10, color=cmap2[4, :])
        prob_match_ax.scatter(a_gain_bias[1, :], gain_bias[1, :], marker='^', s=10, color=cmap2[2, :])
        prob_match_ax.scatter(a_gain_bias[2, :], gain_bias[2, :], marker='s', s=10, color=cmap2[8, :])
        prob_match_ax.scatter(a_loss_bias[0, :], loss_bias[0, :], marker='o', s=10, color=cmap2[5, :])
        prob_match_ax.scatter(a_loss_bias[1, :], loss_bias[1, :], marker='^', s=10, color=cmap2[3, :])
        prob_match_ax.scatter(a_loss_bias[2, :], loss_bias[2, :], marker='s', s=10, color=cmap2[9, :])

        # now get/plot the regression line
        p_match_df = pd.DataFrame()
        p_match_df['human_choice'] = np.concatenate((gain_bias[0, :], gain_bias[1, :], gain_bias[2, :],
                                                     loss_bias[0, :], loss_bias[1, :], loss_bias[2, :]))
        p_match_df['agent_choice'] = np.concatenate((a_gain_bias[0, :], a_gain_bias[1, :], a_gain_bias[2, :],
                                                     a_loss_bias[0, :], a_loss_bias[1, :], a_loss_bias[2, :]))
        p_match_df['context'] = np.concatenate((np.ones((len(e_gain), 1)),
                                                np.ones((len(e_gain), 1)),
                                                np.ones((len(e_gain), 1)),
                                                np.ones((len(e_gain), 1)) * -1,
                                                np.ones((len(e_gain), 1)) * -1,
                                                np.ones((len(e_gain), 1)) * -1))

        p_match_df['prob'] = np.concatenate((np.ones((len(e_gain), 1)) * .2,
                                             np.ones((len(e_gain), 1)) * .5,
                                             np.ones((len(e_gain), 1)) * .8,
                                             np.ones((len(e_gain), 1)) * .2,
                                             np.ones((len(e_gain), 1)) * .5,
                                             np.ones((len(e_gain), 1)) * .8))

        match_mdl = smf.glm('human_choice ~ agent_choice*context*prob', data=p_match_df).fit()
        prob_match_ax.plot(q_xlim, (q_xlim * match_mdl.params[1]) + match_mdl.params[0], color='black',
                           linewidth=2)

        prob_match_ax.set_xticks([0, .5, 1])
        prob_match_ax.set_yticks([0, .5, 1])
        prob_match_ax.set_ylabel('Human Choice')

        if exp == 1:
            bias_ax.set_title('Individual Biases')
            accuracy_ax.set_title('Model Accuracy')
            qreg_ax.set_title('RS-RL Expectile Predicts \n Individual Risk Attitudes')
            prob_match_ax.set_title('RS-RL Choice Predicts \n Human Choice')

            bias_ax.legend()
            prob_match_ax.legend(['20%', '50%', '80%'], fontsize=8)

        if exp == 3:
            bias_ax.set_xlabel('=SvsR Condition')
            accuracy_ax.set_xlabel('Context')
            qreg_ax.set_xlabel('RS-RL Expectile')
            prob_match_ax.set_xlabel('RS-RL Agent Choice')

        # save the model output
        save_name = 'Exp' + str(exp)
        expectile_models[save_name] = q_mdl.summary()
        match_models[save_name] = match_mdl.summary()

    plt.show()
    # save plot
    fig.savefig("RSRL_fig.svg", transparent=True)
    return expectile_models, match_models


class RS_agent(object):
    def __init__(self, n_states, alpha_plus, alpha_minus, beta):
        self.alpha_plus = alpha_plus
        self.alpha_minus = alpha_minus
        self.beta = beta
        self.Qtable = np.zeros(shape=(1, n_states))[0]

    def softmax(self, option1, option2):

        Q1 = self.Qtable[option1]
        Q2 = self.Qtable[option2]
        b = self.beta

        if np.absolute(Q1 - Q2) > 0:
            # prob of choosing option 1
            softmax = np.exp(b * Q1) / (np.exp(b * Q1) + np.exp(b * Q2))
            # softmax = 1/(1+np.exp(b*(Q1-Q2)))
        else:
            softmax = np.random.rand(1)

        # which option did the agent pick?
        if softmax > .5:
            agent_choice = option1
        else:
            agent_choice = option2

        if softmax == 0:
            softmax = .0000000001
        if softmax == 1:
            softmax = .9999999999

        return softmax, agent_choice

    def update(self, chosen_opt, outcome, chosen_type):

        prediction_error = outcome - self.Qtable[chosen_opt]

        if prediction_error > 0:
            a = self.alpha_plus
        else:
            a = self.alpha_minus

        # update the Q table
        self.Qtable[chosen_opt] = self.Qtable[chosen_opt] + (a * prediction_error)
