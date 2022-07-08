# -*- coding: utf-8 -*-
"""
Code for analyzing safe vs risky human data

@author: Thomas Elston
"""

import os
import pandas as pd
import numpy as np
import numpy.matlib
import pdb
import matplotlib.pyplot as plt
from regex import F
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
import pingouin as pg
import itertools as it
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import math
import scipy as sp
import utils as ut


def load_processData(datadir,context): 
        
    # get files names with path
    fnames = [os.path.join(datadir, _) for _ in os.listdir(datadir) if _.endswith('.csv')]
    
    # initialize a dataframe for ALL of the data
    alldata = pd.DataFrame()
    
    # initialize a dataframe for each participant's mean choice behavior
    pChoicedata = pd.DataFrame()
   
    # dataframe for RTs
    pRTdata = pd.DataFrame()
    
    # array of subject performance
    p_perf = np.zeros((len(fnames),4))


    ctr = 0
    
    print('assessing file#:', end=' ')
    # load a csv file and assess each one
    
    for i in range(len(fnames)):
                
        print(str(i),end=' ')
        
        df = pd.read_csv(fnames[i],header=[0])
        
        # get overall training performance so we can reject subjects
        all_train_ix = df.phase == "training"
        all_exp_ix   = df.phase == "exp"
        all_safe_ix  = (df.imgLeftType == 'Safe') & (df.imgRightType == 'Safe')
        all_risky_ix = (df.imgLeftType == 'Risky') & (df.imgRightType == 'Risky')
        gain_ix = df.blkType == 'Gain'
        loss_ix = df.blkType == 'Loss'
        
        # define the best options 
        all_picked_best = df.highProbSelected.astype(int)
        
        # get subject overall performance       
        gs_perf = all_picked_best[all_exp_ix & gain_ix & all_safe_ix].mean()
        gr_perf = all_picked_best[all_exp_ix & gain_ix & all_risky_ix].mean()
        ls_perf = 1-all_picked_best[all_exp_ix & loss_ix & all_safe_ix].mean()
        lr_perf = 1-all_picked_best[all_exp_ix & loss_ix & all_risky_ix].mean()
        
        p_perf[i,:] = np.array([gs_perf,gr_perf,ls_perf,lr_perf])
        
        # inclusion crit for overall perf
        c = 0.60
    
        
        #if (gs_perf > c) & (gr_perf > c) & (ls_perf > c) & (lr_perf > c):
        if np.nanmean(p_perf[i,:]) > c:
  
            # only assess the context specified as input argument
            df = df[df.blkType == context]
    
    
            # drop trials with RTs that were too slow or too fast
            df = df.drop(df[(df.rt > 3000) | (df.rt < 200)].index)
            df = df.reset_index()
            
            # change a few datatypes
            df.highProbSelected = df.highProbSelected.astype(int)
            df.probLeft = df.probLeft.astype(float)
            df.probRight = df.probRight.astype(float)
            
            # get some useful indices
            # trial types and task phases
            trainix = df.phase == "training"
            testix  = df.phase == "exp"
            
            # infer trial types from the left/right stimulus types
            safe_ix  = (df.imgLeftType == 'Safe') & (df.imgRightType == 'Safe')
            risky_ix = (df.imgLeftType == 'Risky') & (df.imgRightType == 'Risky')
            EQ    = df.probLeft == df.probRight
            UE    = (df.imgLeftType != df.imgRightType) & ~EQ
            
            # define the best options for the two contexts (gain/loss)
            picked_best = np.zeros(len(EQ))
            
            if context == 'Loss':
                picked_best = df.highProbSelected ==0
            else:
                picked_best = df.highProbSelected ==1
                

            # keep track of which stimulus type was chosen on each trial
            picked_risky = np.zeros(shape = (len(EQ),))
            picked_risky[(df.responseSide == 'left') & (df.imgLeftType == 'Risky')] = 1
            picked_risky[(df.responseSide == 'right') & (df.imgRightType == 'Risky')] = 1
            picked_risky = picked_risky.astype(int)
            
            chose_risky_ix = np.array(picked_risky, dtype=bool)
    
            # define reaction times
            rt = df.rt
            
            # infer image types and assign identifying numbers (useful for WS,LS analysis)
            left_imgnum = np.empty(len(df.rt))
            right_imgnum = np.empty(len(df.rt))
            
            left_imgnum[df.probLeft ==.2] = 0
            left_imgnum[df.probLeft ==.5] = 1
            left_imgnum[df.probLeft ==.8] = 2
            right_imgnum[df.probRight ==.2] = 0
            right_imgnum[df.probRight ==.5] = 1
            right_imgnum[df.probRight ==.8] = 2
            
            left_imgnum[df.imgLeftType=='Risky'] = left_imgnum[df.imgLeftType=='Risky']+3
            right_imgnum[df.imgRightType=='Risky'] = right_imgnum[df.imgRightType=='Risky']+3
    
            # add these values to the dataframe
            df['imageNumberLeft'] = left_imgnum
            df['imageNumberRight'] = right_imgnum  
                    
            # what is the best option's type?
            besttype = np.empty((len(df.rt)))
                    
            # find the trials where the safe/risky options were better
            for t in range(len(df.rt)):
                if df.highProbSide[t] == 'left':               
                    if df.imgLeftType[t] == 'Risky':                   
                        besttype[t] = 1                    
                    else:
                        besttype[t] = 0               
                else:               
                    if df.imgRightType[t] == 'Risky':                  
                        besttype[t] = 1               
                    else:
                        besttype[t] = 0               
                            
            
            # choice conditions
            a20 = (df.probLeft ==.2) | (df.probRight ==.2)
            a50 = (df.probLeft ==.5) | (df.probRight ==.5)
            a80 = (df.probLeft ==.8) | (df.probRight ==.8)
            riskybest = besttype == 1
            safebest = besttype == 0
            t20v50 = a20 & a50
            t50v80 = a80 & a50
            t20v80 = a20 & a80
            t20v20 = (df.probLeft ==.2) & (df.probRight ==.2)
            t50v50 = (df.probLeft ==.5) & (df.probRight ==.5)
            t80v80 = (df.probLeft ==.8) & (df.probRight ==.8)
                  
            #-----------------------------
            #    summarize each subject
            #-----------------------------   
        
            # get subject idnum, version, sex, and age
            pChoicedata.at[ctr,'vpnum']   = df.vpNum[0]
            pChoicedata.at[ctr,'version'] = df.version[0]
            pChoicedata.at[ctr,'context'] = context
            pChoicedata.at[ctr,'age']     = df.age[0]
            pChoicedata.at[ctr,'sex']     = df.gender[0]
            
            # look at training choice data
            pChoicedata.at[ctr,'t_s_20v50'] = np.nanmean(picked_best[trainix & t20v50 & safe_ix])
            pChoicedata.at[ctr,'t_s_50v80'] = np.nanmean(picked_best[trainix & t50v80 & safe_ix])
            pChoicedata.at[ctr,'t_s_20v80'] = np.nanmean(picked_best[trainix & t20v80 & safe_ix])
            
            pChoicedata.at[ctr,'t_r_20v50'] = np.nanmean(picked_best[trainix & t20v50 & risky_ix])
            pChoicedata.at[ctr,'t_r_50v80'] = np.nanmean(picked_best[trainix & t50v80 & risky_ix])
            pChoicedata.at[ctr,'t_r_20v80'] = np.nanmean(picked_best[trainix & t20v80 & risky_ix])
    
            # main block pure trials
            pChoicedata.at[ctr,'s_20v50'] = np.nanmean(picked_best[testix & t20v50 & safe_ix])
            pChoicedata.at[ctr,'s_50v80'] = np.nanmean(picked_best[testix & t50v80 & safe_ix])
            pChoicedata.at[ctr,'s_20v80'] = np.nanmean(picked_best[testix & t20v80 & safe_ix])
            
            pChoicedata.at[ctr,'r_20v50'] = np.nanmean(picked_best[testix & t20v50 & risky_ix])
            pChoicedata.at[ctr,'r_50v80'] = np.nanmean(picked_best[testix & t50v80 & risky_ix])
            pChoicedata.at[ctr,'r_20v80'] = np.nanmean(picked_best[testix & t20v80 & risky_ix])
            
            # main block unequal trials
            # safe is better
            pChoicedata.at[ctr,'UE_s_20v50'] = np.nanmean(picked_best[UE & t20v50 & riskybest])
            pChoicedata.at[ctr,'UE_s_50v80'] = np.nanmean(picked_best[UE & t50v80 & riskybest])
            pChoicedata.at[ctr,'UE_s_20v80'] = np.nanmean(picked_best[UE & t20v80 & riskybest])
            
            # risky is better
            pChoicedata.at[ctr,'UE_r_20v50'] = np.nanmean(picked_best[UE & t20v50 & safebest])
            pChoicedata.at[ctr,'UE_r_50v80'] = np.nanmean(picked_best[UE & t50v80 & safebest])
            pChoicedata.at[ctr,'UE_r_20v80'] = np.nanmean(picked_best[UE & t20v80 & safebest])
            
            # main block equivaluable trials
            pChoicedata.at[ctr,'EQ20'] = np.nanmean(picked_risky[t20v20])
            pChoicedata.at[ctr,'EQ50'] = np.nanmean(picked_risky[t50v50])
            pChoicedata.at[ctr,'EQ80'] = np.nanmean(picked_risky[t80v80])
            
            
            # do the same but with RTs
            pRTdata.at[ctr,'vpnum']   = df.vpNum[0]
            pRTdata.at[ctr,'version'] = df.version[0]
            pRTdata.at[ctr,'context'] = context
            pRTdata.at[ctr,'age']     = df.age[0]
            pRTdata.at[ctr,'sex']     = df.gender[0]
            
            # look at training choice data
            pRTdata.at[ctr,'t_s_20v50'] = np.nanmean(rt[trainix & t20v50 & safe_ix])
            pRTdata.at[ctr,'t_s_50v80'] = np.nanmean(rt[trainix & t50v80 & safe_ix])
            pRTdata.at[ctr,'t_s_20v80'] = np.nanmean(rt[trainix & t20v80 & safe_ix])
            
            pRTdata.at[ctr,'t_r_20v50'] = np.nanmean(rt[trainix & t20v50 & risky_ix])
            pRTdata.at[ctr,'t_r_50v80'] = np.nanmean(rt[trainix & t50v80 & risky_ix])
            pRTdata.at[ctr,'t_r_20v80'] = np.nanmean(rt[trainix & t20v80 & risky_ix])
    
            # main block pure trials
            pRTdata.at[ctr,'s_20v50'] = np.nanmean(rt[testix & t20v50 & safe_ix])
            pRTdata.at[ctr,'s_50v80'] = np.nanmean(rt[testix & t50v80 & safe_ix])
            pRTdata.at[ctr,'s_20v80'] = np.nanmean(rt[testix & t20v80 & safe_ix])
            
            pRTdata.at[ctr,'r_20v50'] = np.nanmean(rt[testix & t20v50 & risky_ix])
            pRTdata.at[ctr,'r_50v80'] = np.nanmean(rt[testix & t50v80 & risky_ix])
            pRTdata.at[ctr,'r_20v80'] = np.nanmean(rt[testix & t20v80 & risky_ix])
            
            # main block unequal trials
            # narrow is better
            pRTdata.at[ctr,'UE_s_20v50'] = np.nanmean(rt[UE & t20v50 & safebest])
            pRTdata.at[ctr,'UE_s_50v80'] = np.nanmean(rt[UE & t50v80 & safebest])
            pRTdata.at[ctr,'UE_s_20v80'] = np.nanmean(rt[UE & t20v80 & safebest])
            
            pRTdata.at[ctr,'UE_r_20v50'] = np.nanmean(rt[UE & t20v50 & riskybest])
            pRTdata.at[ctr,'UE_r_50v80'] = np.nanmean(rt[UE & t50v80 & riskybest])
            pRTdata.at[ctr,'UE_r_20v80'] = np.nanmean(rt[UE & t20v80 & riskybest])
            
            # main block equivaluable trials
            pRTdata.at[ctr,'EQ20'] = np.nanmean(rt[t20v20])
            pRTdata.at[ctr,'EQ50'] = np.nanmean(rt[t50v50])
            pRTdata.at[ctr,'EQ80'] = np.nanmean(rt[t80v80])
            
            ctr = ctr+1
            
            # add all data to the aggregate dataframe
            alldata = alldata.append(df)



        
        
                
    xx=[]
    return pChoicedata , pRTdata, alldata, p_perf
# END of load_processData 


def show_excluded_subjects(p_perf):
        
    # calculate different exclusison crit
    
    # exclude based on overall performance
    overall_ix = p_perf.mean(axis=1) > .6  
    
    # exclude based on performance in each condition
    cond_ix = np.sum(p_perf > .6, axis=1) > 3
    
    
    
    # create figure
    fig, ax = plt.subplots(1,2,figsize=(6, 3), dpi=150)
    fig.tight_layout(h_pad=4)
  
    ax[0].plot(np.transpose(p_perf[overall_ix,:]), linewidth=1, color='tab:gray', label='keep')
    ax[0].plot(np.transpose(p_perf[~overall_ix,:]), linewidth=1, color='tab:red', label='exclude')
    ax[0].set_ylim([.2,1])
    ax[0].set_title('all trials > 60%')
    ax[0].set_xticks(np.array([0,1,2,3]))
    ax[0].set_yticks(np.array([.2,.4,.6,.8,1]))
    ax[0].set_xticklabels(['G_S','G_R','L_S','L_R'])
    ax[0].set_ylabel('% corr')
    ax[0].set_xlabel('Condition')
    
    
    ax[1].plot(np.transpose(p_perf[cond_ix,:]), linewidth=1, color='tab:gray', label='keep')
    ax[1].plot(np.transpose(p_perf[~cond_ix,:]), linewidth=1, color='tab:red', label='exclude')
    ax[1].set_ylim([.2,1])
    ax[1].set_title('all conds > 60%')
    ax[1].set_xticks(np.array([0,1,2,3]))
    ax[1].set_yticks(np.array([.2,.4,.6,.8,1]))
    ax[1].set_xticklabels(['G_S','G_R','L_S','L_R'])

    
    
    xx=[]
# END of show_excluded_subjects


def plot_mean_perf(gain_data, loss_data, datatype):
    
    '''
    This function plots and statistically assess choice and rt data by condition
    '''
            

    # collect the mean performance for the training trials
    mean_gain_train = pd.DataFrame()
    mean_gain_train['vpnum']   = gain_data.vpnum
    mean_gain_train['context'] = gain_data.context
    mean_gain_train['safe']    = np.nanmean(gain_data.iloc[:,5:8],axis=1)
    mean_gain_train['risky']   = np.nanmean(gain_data.iloc[:,8:10],axis=1)
    
    mean_loss_train = pd.DataFrame()
    mean_loss_train['vpnum']   = loss_data.vpnum
    mean_loss_train['context'] = loss_data.context
    mean_loss_train['safe']    = np.nanmean(loss_data.iloc[:,5:8],axis=1)
    mean_loss_train['risky']   = np.nanmean(loss_data.iloc[:,8:10],axis=1)
    
    n_subs = len(mean_loss_train)
    
    # aggregate the train data for an rm_anova later
    all_train = pd.DataFrame()
    all_train['resp'] = np.concatenate([mean_gain_train['risky'], mean_gain_train['safe'],
                                 mean_loss_train['risky'], mean_loss_train['safe'],])
    
    all_train['vpnum'] = np.concatenate([mean_gain_train['vpnum'],mean_gain_train['vpnum'],
                            mean_loss_train['vpnum'],mean_loss_train['vpnum']])
    
    all_train['context'] = np.concatenate([np.ones([n_subs,]), np.ones([n_subs,]),
                              np.ones([n_subs,])*-1, np.ones([n_subs,])*-1])
    
    all_train['cond'] = np.concatenate([np.ones([n_subs,]), np.ones([n_subs,])*-1,
                              np.ones([n_subs,]), np.ones([n_subs,])*-1])
    
    

    
    # collect the mean performance for the main block pure trials 
    mean_gain = pd.DataFrame()
    mean_gain['vpnum']   = gain_data.vpnum
    mean_gain['context'] = gain_data.context
    mean_gain['safe']    = np.nanmean(gain_data.iloc[:,11:14],axis=1)
    mean_gain['risky']   = np.nanmean(gain_data.iloc[:,14:17],axis=1)
    
    mean_loss = pd.DataFrame()
    mean_loss['vpnum']   = loss_data.vpnum
    mean_loss['context'] = loss_data.context
    mean_loss['safe']    = np.nanmean(loss_data.iloc[:,11:14],axis=1)
    mean_loss['risky']   = np.nanmean(loss_data.iloc[:,14:17],axis=1)
    
    # aggregate the main block data
    all_main = pd.DataFrame()
    all_main['resp'] = np.concatenate([mean_gain['risky'], mean_gain['safe'],
                                 mean_loss['risky'], mean_loss['safe'],])
    
    all_main['vpnum'] = np.concatenate([mean_gain['vpnum'],mean_gain['vpnum'],
                            mean_loss['vpnum'],mean_loss['vpnum']])
    
    all_main['context'] = np.concatenate([np.ones([n_subs,]), np.ones([n_subs,]),
                              np.ones([n_subs,])*-1, np.ones([n_subs,])*-1])
    
    all_main['cond'] = np.concatenate([np.ones([n_subs,]), np.ones([n_subs,])*-1,
                              np.ones([n_subs,]), np.ones([n_subs,])*-1])
    
    # collect the mean performance for the main block UE trials 
    mean_gain_UE = pd.DataFrame()
    mean_gain_UE['vpnum']   = gain_data.vpnum
    mean_gain_UE['context'] = gain_data.context
    mean_gain_UE['safe']    = np.nanmean(gain_data.iloc[:,17:20],axis=1)
    mean_gain_UE['risky']   = np.nanmean(gain_data.iloc[:,20:23],axis=1)
    
    mean_loss_UE = pd.DataFrame()
    mean_loss_UE['vpnum']   = loss_data.vpnum
    mean_loss_UE['context'] = loss_data.context
    mean_loss_UE['safe']    = np.nanmean(loss_data.iloc[:,17:20],axis=1)
    mean_loss_UE['risky']   = np.nanmean(loss_data.iloc[:,20:23],axis=1)
    
    # aggregate the main UE trials data
    all_UE = pd.DataFrame()
    all_UE['resp'] = np.concatenate([mean_gain_UE['risky'], mean_gain_UE['safe'],
                                 mean_loss_UE['risky'], mean_loss_UE['safe'],])
    
    all_UE['vpnum'] = np.concatenate([mean_gain_UE['vpnum'],mean_gain_UE['vpnum'],
                            mean_loss_UE['vpnum'],mean_loss_UE['vpnum']])
    
    all_UE['context'] = np.concatenate([np.ones([n_subs,]), np.ones([n_subs,]),
                              np.ones([n_subs,])*-1, np.ones([n_subs,])*-1])
    
    all_UE['cond'] = np.concatenate([np.ones([n_subs,]), np.ones([n_subs,])*-1,
                              np.ones([n_subs,]), np.ones([n_subs,])*-1])

    
    # plot data
    # define the color map
    cmap = plt.cm.Paired(np.linspace(0, 1, 12))

    fig = plt.figure(figsize=(8, 2), dpi=300)

    gs = fig.add_gridspec(1,12)
    ax0 = fig.add_subplot(gs[0: 2])
    ax1 = fig.add_subplot(gs[2: 4])
    ax2 = fig.add_subplot(gs[4: 6])
    ax3 = fig.add_subplot(gs[8: 12])
    
    xlims = np.array([.8,2.2])
    
    if datatype == 'choice':
        ylims = np.array([0,1])
        eq_ylim = np.array([.2,.8])
        ylbl= 'p(Choose Best)'
        eq_ylbl = 'p(Choose Risky)'
        ytcks = np.array([.0,.5,1])
        eq_ytcks = np.array([.2, .5, .8])
        
    else:
        ylims = np.array([300,900])
        ylbl = 'RT (ms)'
        eq_ylim = ylims
        eq_ylbl = ylbl
        ytcks = np.array([300, 600, 900])
        eq_ytcks = ytcks

    
    # gain_train means
    gain_train_y = np.array([mean_gain_train.safe.mean(),
                             mean_gain_train.risky.mean()])
    
    gain_train_err = np.array([mean_gain_train.safe.sem(),
                               mean_gain_train.risky.sem()])
    
    # loss_train means
    loss_train_y = np.array([mean_loss_train.safe.mean(),
                             mean_loss_train.risky.mean()])
    
    loss_train_err = np.array([mean_loss_train.safe.sem(),
                               mean_loss_train.risky.sem()])
    
    
    ax0.errorbar(np.array([1,2]), gain_train_y, gain_train_err,
                   color = cmap[1,:],capsize=0, linewidth=2, marker='.')
    
    ax0.errorbar(np.array([1,2]), loss_train_y, loss_train_err,
                   color = cmap[5,:],capsize=0,linewidth=2, marker='.')
    
    ax0.set_ylim(ylims)
    ax0.set_yticks(ytcks)
    ax0.set_xticks([1,2])
    ax0.set_xlim(xlims)
    ax0.set_xticklabels(['S', 'R'])
    ax0.legend(['Gain', 'Loss'])
    ax0.set_ylabel(ylbl)
    ax0.set_title('Training')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    
    # gain means
    gain_y = np.array([mean_gain.safe.mean(),
                             mean_gain.risky.mean()])
    
    gain_err = np.array([mean_gain.safe.sem(),
                               mean_gain.risky.sem()])
    
    # loss means
    loss_y = np.array([mean_loss.safe.mean(),
                             mean_loss.risky.mean()])
    
    loss_err = np.array([mean_loss.safe.sem(),
                         mean_loss.risky.sem()])
    
    
    ax1.errorbar(np.array([1,2]), gain_y, gain_err,
                   color = cmap[1,:],capsize=0, linewidth=2, marker='.')
    
    ax1.errorbar(np.array([1,2]), loss_y, loss_err,
                   color = cmap[5,:],capsize=0,linewidth=2, marker='.')
    
    ax1.set_ylim(ylims)
    ax1.get_yaxis().set_visible(False)
    ax1.set_xticks([1,2])
    ax1.set_xlim(xlims)
    ax1.set_xticklabels(['S', 'R'])
    ax1.set_title('Pure')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    # gain means
    UE_gain_y = np.array([mean_gain_UE.safe.mean(),
                          mean_gain_UE.risky.mean()])
    
    UE_gain_err = np.array([mean_gain_UE.safe.sem(),
                            mean_gain_UE.risky.sem()])
    
    # loss means
    UE_loss_y = np.array([mean_loss_UE.safe.mean(),
                          mean_loss_UE.risky.mean()])
    
    UE_loss_err = np.array([mean_loss_UE.safe.sem(),
                            mean_loss_UE.risky.sem()])
    
    
    ax2.errorbar(np.array([1,2]), UE_gain_y, UE_gain_err,
                   color = cmap[1,:],capsize=0, linewidth=2, marker='.')
    
    ax2.errorbar(np.array([1,2]), UE_loss_y, UE_loss_err,
                   color = cmap[5,:],capsize=0,linewidth=2, marker='.')
    
    ax2.set_ylim(ylims)
    ax2.set_xticks([1,2])
    ax2.set_xlim(xlims)
    ax2.get_yaxis().set_visible(False)
    ax2.set_xticklabels(['S>R', 'R>S'])
    ax2.set_title('Unequal S vs R')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    
     
    # now get the EQ Safe vs Risky trials
    gain_EQ_y = np.array([gain_data.EQ20.mean(),
                          gain_data.EQ50.mean(),
                          gain_data.EQ80.mean()])
    
    gain_EQ_yerr = np.array([gain_data.EQ20.sem(),
                             gain_data.EQ50.sem(),
                             gain_data.EQ80.sem()])
    
    loss_EQ_y = np.array([loss_data.EQ20.mean(),
                          loss_data.EQ50.mean(),
                          loss_data.EQ80.mean()])
    
    loss_EQ_yerr = np.array([loss_data.EQ20.sem(),
                             loss_data.EQ50.sem(),
                             loss_data.EQ80.sem()])
    
    all_EQ=pd.DataFrame()
    all_EQ['resp'] = np.concatenate([gain_data.EQ20,gain_data.EQ50,gain_data.EQ80,
                                     loss_data.EQ20,loss_data.EQ50,loss_data.EQ80])
    
    all_EQ['vpnum'] = np.concatenate([mean_gain_UE['vpnum'],mean_gain_UE['vpnum'],
                                      mean_gain_UE['vpnum'],mean_loss_UE['vpnum'],
                                      mean_loss_UE['vpnum'],mean_loss_UE['vpnum']])
    
    all_EQ['context'] = np.concatenate([np.ones([n_subs,]), np.ones([n_subs,]),
                                        np.ones([n_subs,]), np.ones([n_subs,])*-1, 
                                        np.ones([n_subs,])*-1, np.ones([n_subs,])*-1])
    
    all_EQ['cond'] = np.concatenate([np.ones([n_subs,]), np.ones([n_subs,])*2,
                                     np.ones([n_subs,])*3, np.ones([n_subs,]),
                                     np.ones([n_subs,])*2,np.ones([n_subs,])*3])
    
    
    
    ax3.errorbar(np.array([1,2,3]),gain_EQ_y,gain_EQ_yerr,
                   color=cmap[1,:], capsize=0, linewidth=2, marker='.')
    
    ax3.errorbar(np.array([1,2,3]),loss_EQ_y,loss_EQ_yerr,
                   color=cmap[5,:], capsize=0, linewidth=2, marker='.')
    
    ax3.set_ylim(eq_ylim)
    ax3.set_ylabel(eq_ylbl)
    ax3.set_xticks([1,2,3])
    ax3.set_yticks(eq_ytcks)
    ax3.set_xticklabels(['EQ20', 'EQ50', 'EQ80'])
    ax3.set_title('Equal S vs R')

    
 #-----------------------
 #         STATS
 #-----------------------   

 # training
    train_stats = pg.rm_anova(data = all_train,
                           dv = 'resp',
                           within = ['context','cond'],
                           subject = 'vpnum')
 
 # pure trials 
    pure_stats = pg.rm_anova(data = all_main,
                           dv = 'resp',
                           within = ['context','cond'],
                           subject = 'vpnum')
 
 # UE trials 
    pure_stats = pg.rm_anova(data = all_UE,
                           dv = 'resp',
                           within = ['context','cond'],
                           subject = 'vpnum')
 
 # EQ trials
    EQ_stats = pg.rm_anova(data = all_EQ,
                           dv = 'resp',
                           within = ['context','cond'],
                           subject = 'vpnum')
 

    # set break point here to check stats
    xx=[]
    
# END of plot_mean_perf

def assess_conds_with_best_choice(gain_data, loss_data):
    xx=[]

    gain_df = pd.DataFrame()
    loss_df = pd.DataFrame()

    gain_df['train_safe'] = np.nanmean(gain_data.iloc[:,5:8],axis=1)
    gain_df['train_risky'] = np.nanmean(gain_data.iloc[:,8:11],axis=1)
    gain_df['main_safe'] = np.nanmean(gain_data.iloc[:,11:14],axis=1)
    gain_df['main_risky'] = np.nanmean(gain_data.iloc[:,14:17],axis=1)
    gain_df['UE_safe_better'] = np.nanmean(gain_data.iloc[:,17:20],axis=1)
    gain_df['UE_risky_better'] = np.nanmean(gain_data.iloc[:,20:23],axis=1)

    loss_df['train_safe'] = np.nanmean(loss_data.iloc[:,5:8],axis=1)
    loss_df['train_risky'] = np.nanmean(loss_data.iloc[:,8:11],axis=1)
    loss_df['main_safe'] = np.nanmean(loss_data.iloc[:,11:14],axis=1)
    loss_df['main_risky'] = np.nanmean(loss_data.iloc[:,14:17],axis=1)
    loss_df['UE_safe_better'] = np.nanmean(loss_data.iloc[:,17:20],axis=1)
    loss_df['UE_risky_better'] = np.nanmean(loss_data.iloc[:,20:23],axis=1)

    gain_results = pd.DataFrame()
    loss_results = pd.DataFrame()

    for cond in range(6):
        gain_results = gain_results.append(pg.ttest(gain_df.iloc[:,cond], .5))
        loss_results = loss_results.append(pg.ttest(loss_df.iloc[:,cond], .5))

    gain_results.index = gain_df.columns
    loss_results.index = gain_df.columns

    return gain_results, loss_results

# END of assess_conds_with_best_choice()





def collect_data_for_stats(gain_data, loss_data):
    
    '''
    This function aggregates the data from each experiment into a dict
    which stats can later be done on
    '''

    out_data = {} # initialize dict as to store data in
            
    # collect the mean performance for the training trials
    mean_gain_train = pd.DataFrame()
    mean_gain_train['vpnum']   = gain_data.vpnum
    mean_gain_train['context'] = gain_data.context
    mean_gain_train['safe']    = np.nanmean(gain_data.iloc[:,5:8],axis=1)
    mean_gain_train['risky']   = np.nanmean(gain_data.iloc[:,8:11],axis=1)
    
    mean_loss_train = pd.DataFrame()
    mean_loss_train['vpnum']   = loss_data.vpnum
    mean_loss_train['context'] = loss_data.context
    mean_loss_train['safe']    = np.nanmean(loss_data.iloc[:,5:8],axis=1)
    mean_loss_train['risky']   = np.nanmean(loss_data.iloc[:,8:11],axis=1)
    
    n_subs = len(mean_loss_train)
    
    # aggregate the train data for an rm_anova later
    all_train = pd.DataFrame()
    all_train['resp'] = np.concatenate([mean_gain_train['risky'], mean_gain_train['safe'],
                                 mean_loss_train['risky'], mean_loss_train['safe'],])
    
    all_train['vpnum'] = np.concatenate([mean_gain_train['vpnum'],mean_gain_train['vpnum'],
                            mean_loss_train['vpnum'],mean_loss_train['vpnum']])
    
    all_train['context'] = np.concatenate([np.ones([n_subs,]), np.ones([n_subs,]),
                              np.ones([n_subs,])*-1, np.ones([n_subs,])*-1])
    
    all_train['cond'] = np.concatenate([np.ones([n_subs,]), np.ones([n_subs,])*-1,
                              np.ones([n_subs,]), np.ones([n_subs,])*-1])
    
    out_data['train'] = all_train

    
    # collect the mean performance for the main block pure trials 
    mean_gain = pd.DataFrame()
    mean_gain['vpnum']   = gain_data.vpnum
    mean_gain['context'] = gain_data.context
    mean_gain['safe']    = np.nanmean(gain_data.iloc[:,11:14],axis=1)
    mean_gain['risky']   = np.nanmean(gain_data.iloc[:,14:17],axis=1)
    
    mean_loss = pd.DataFrame()
    mean_loss['vpnum']   = loss_data.vpnum
    mean_loss['context'] = loss_data.context
    mean_loss['safe']    = np.nanmean(loss_data.iloc[:,11:14],axis=1)
    mean_loss['risky']   = np.nanmean(loss_data.iloc[:,14:17],axis=1)
    
    # aggregate the main block data
    all_main = pd.DataFrame()
    all_main['resp'] = np.concatenate([mean_gain['risky'], mean_gain['safe'],
                                 mean_loss['risky'], mean_loss['safe'],])
    
    all_main['vpnum'] = np.concatenate([mean_gain['vpnum'],mean_gain['vpnum'],
                            mean_loss['vpnum'],mean_loss['vpnum']])
    
    all_main['context'] = np.concatenate([np.ones([n_subs,]), np.ones([n_subs,]),
                              np.ones([n_subs,])*-1, np.ones([n_subs,])*-1])
    
    all_main['cond'] = np.concatenate([np.ones([n_subs,]), np.ones([n_subs,])*-1,
                              np.ones([n_subs,]), np.ones([n_subs,])*-1])

    out_data['pure'] = all_main

    
    # collect the mean performance for the main block UE trials 
    mean_gain_UE = pd.DataFrame()
    mean_gain_UE['vpnum']   = gain_data.vpnum
    mean_gain_UE['context'] = gain_data.context
    mean_gain_UE['safe']    = np.nanmean(gain_data.iloc[:,17:20],axis=1)
    mean_gain_UE['risky']   = np.nanmean(gain_data.iloc[:,20:23],axis=1)
    
    mean_loss_UE = pd.DataFrame()
    mean_loss_UE['vpnum']   = loss_data.vpnum
    mean_loss_UE['context'] = loss_data.context
    mean_loss_UE['safe']    = np.nanmean(loss_data.iloc[:,17:20],axis=1)
    mean_loss_UE['risky']   = np.nanmean(loss_data.iloc[:,20:23],axis=1)
    
    # aggregate the main UE trials data
    all_UE = pd.DataFrame()
    all_UE['resp'] = np.concatenate([mean_gain_UE['risky'], mean_gain_UE['safe'],
                                 mean_loss_UE['risky'], mean_loss_UE['safe'],])
    
    all_UE['vpnum'] = np.concatenate([mean_gain_UE['vpnum'],mean_gain_UE['vpnum'],
                            mean_loss_UE['vpnum'],mean_loss_UE['vpnum']])
    
    all_UE['context'] = np.concatenate([np.ones([n_subs,]), np.ones([n_subs,]),
                              np.ones([n_subs,])*-1, np.ones([n_subs,])*-1])
    
    all_UE['cond'] = np.concatenate([np.ones([n_subs,]), np.ones([n_subs,])*-1,
                              np.ones([n_subs,]), np.ones([n_subs,])*-1])

    out_data['UE'] = all_UE

    
    
    all_EQ=pd.DataFrame()
    all_EQ['resp'] = np.concatenate([gain_data.EQ20,gain_data.EQ50,gain_data.EQ80,
                                     loss_data.EQ20,loss_data.EQ50,loss_data.EQ80])
    
    all_EQ['vpnum'] = np.concatenate([mean_gain_UE['vpnum'],mean_gain_UE['vpnum'],
                                      mean_gain_UE['vpnum'],mean_loss_UE['vpnum'],
                                      mean_loss_UE['vpnum'],mean_loss_UE['vpnum']])
    
    all_EQ['context'] = np.concatenate([np.ones([n_subs,]), np.ones([n_subs,]),
                                        np.ones([n_subs,]), np.ones([n_subs,])*-1, 
                                        np.ones([n_subs,])*-1, np.ones([n_subs,])*-1])
    
    all_EQ['cond'] = np.concatenate([np.ones([n_subs,]), np.ones([n_subs,])*2,
                                     np.ones([n_subs,])*3, np.ones([n_subs,]),
                                     np.ones([n_subs,])*2,np.ones([n_subs,])*3])

    out_data['EQ'] = all_EQ

    return out_data

# END of collect_data_for_stats()


def do_stats(exp1, exp2):

    '''
    This function statistically assesses the training, pure, unequal safe vs risky,
    and equal safe vs risky conditions both within and across exps 1 and 2

    Results are aggregated into dicts
    '''

    exp1_stats = {}
    exp2_stats = {}
    both_exp_stats = {}

    #-----------------------
    #     Experiment 1
    #-----------------------   
    # training
    exp1_stats['train'] = pg.rm_anova(data = exp1['train'],
                           dv = 'resp',
                           within = ['context','cond'],
                           subject = 'vpnum')
 
    # pure trials 
    exp1_stats['pure'] = pg.rm_anova(data = exp1['pure'],
                           dv = 'resp',
                           within = ['context','cond'],
                           subject = 'vpnum')
 
    # UE trials 
    exp1_stats['UE'] = pg.rm_anova(data = exp1['UE'],
                           dv = 'resp',
                           within = ['context','cond'],
                           subject = 'vpnum')
 
    # EQ trials
    exp1_stats['EQ'] = pg.rm_anova(data = exp1['EQ'],
                           dv = 'resp',
                           within = ['context','cond'],
                           subject = 'vpnum')

    #-----------------------
    #     Experiment 2
    #-----------------------   
    # training
    exp2_stats['train'] = pg.rm_anova(data = exp2['train'],
                           dv = 'resp',
                           within = ['context','cond'],
                           subject = 'vpnum')
 
    # pure trials 
    exp2_stats['pure'] = pg.rm_anova(data = exp2['pure'],
                           dv = 'resp',
                           within = ['context','cond'],
                           subject = 'vpnum')
 
    # UE trials 
    exp2_stats['UE'] = pg.rm_anova(data = exp2['UE'],
                           dv = 'resp',
                           within = ['context','cond'],
                           subject = 'vpnum')
 
    # EQ trials
    exp2_stats['EQ'] = pg.rm_anova(data = exp2['EQ'],
                           dv = 'resp',
                           within = ['context','cond'],
                           subject = 'vpnum')


    #----------------
    # mixed effects models combining exp1 and exp2
    #---------------
    # make an array indicating which experiment some data came from
    exp_factor = np.concatenate([np.ones(len(exp1['train'])),
                            np.ones(len(exp2['train']))*-1]).astype(int)

    eq_exp_factor = np.concatenate([np.ones(len(exp1['EQ'])),
                            np.ones(len(exp2['EQ']))*-1])
    
    # aggregate data for mixed anova
    all_train = pd.concat([exp1['train'], exp2['train']])
    all_train['exp_factor'] = exp_factor

    all_pure = pd.concat([exp1['pure'], exp2['pure']])
    all_pure['exp_factor'] = exp_factor

    all_UE = pd.concat([exp1['UE'], exp2['UE']])
    all_UE['exp_factor'] = exp_factor

    all_EQ = pd.concat([exp1['EQ'], exp2['EQ']])
    all_EQ['exp_factor'] = eq_exp_factor

    # do the mixed effects linear models
    both_exp_stats['train'] = smf.mixedlm('resp ~ C(context)*C(cond)*C(exp_factor)',
                                all_train, groups = all_train['vpnum']).fit()

    both_exp_stats['pure'] = smf.mixedlm('resp ~ C(context)*C(cond)*C(exp_factor)',
                                all_pure, groups = all_pure['vpnum']).fit()

    both_exp_stats['UE'] = smf.mixedlm('resp ~ C(context)*C(cond)*C(exp_factor)',
                                all_UE, groups = all_UE['vpnum']).fit()

    both_exp_stats['EQ'] = smf.mixedlm('resp ~ C(context)*C(cond)*C(exp_factor)',
                                all_EQ, groups = all_EQ['vpnum']).fit()

 
    return exp1_stats, exp2_stats

# end of do_stats
    
    

def plot_both_experiments_perf(exp1_gain_data, exp1_loss_data,
                               exp2_gain_data, exp2_loss_data, datatype):
    
    '''
    This function plots and statistically assess choice and rt data by condition 
    for exp 1 (top row) and exp 2 (bottom row)
    '''
    exp1_gain_train = pd.DataFrame()
    exp1_gain_train['vpnum']   = exp1_gain_data.vpnum
    exp1_gain_train['context'] = exp1_gain_data.context
    exp1_gain_train['safe']    = np.nanmean(exp1_gain_data.iloc[:,5:8],axis=1)
    exp1_gain_train['risky']   = np.nanmean(exp1_gain_data.iloc[:,8:11],axis=1)
    
    exp1_loss_train = pd.DataFrame()
    exp1_loss_train['vpnum']   = exp1_loss_data.vpnum
    exp1_loss_train['context'] = exp1_loss_data.context
    exp1_loss_train['safe']    = np.nanmean(exp1_loss_data.iloc[:,5:8],axis=1)
    exp1_loss_train['risky']   = np.nanmean(exp1_loss_data.iloc[:,8:11],axis=1)

    # gain_train means
    exp1_gain_train_y = np.array([exp1_gain_train.safe.mean(),
                             exp1_gain_train.risky.mean()])
    
    exp1_gain_train_err = np.array([exp1_gain_train.safe.sem(),
                               exp1_gain_train.risky.sem()])
    
    # loss_train means
    exp1_loss_train_y = np.array([exp1_loss_train.safe.mean(),
                             exp1_loss_train.risky.mean()])
    
    exp1_loss_train_err = np.array([exp1_loss_train.safe.sem(),
                               exp1_loss_train.risky.sem()])
    

    # collect the mean performance for the main block pure trials 
    exp1_gain = pd.DataFrame()
    exp1_gain['vpnum']   = exp1_gain_data.vpnum
    exp1_gain['context'] = exp1_gain_data.context
    exp1_gain['safe']    = np.nanmean(exp1_gain_data.iloc[:,11:14],axis=1)
    exp1_gain['risky']   = np.nanmean(exp1_gain_data.iloc[:,14:17],axis=1)
    
    exp1_loss = pd.DataFrame()
    exp1_loss['vpnum']   = exp1_loss_data.vpnum
    exp1_loss['context'] = exp1_loss_data.context
    exp1_loss['safe']    = np.nanmean(exp1_loss_data.iloc[:,11:14],axis=1)
    exp1_loss['risky']   = np.nanmean(exp1_loss_data.iloc[:,14:17],axis=1)

    # gain means
    exp1_gain_y = np.array([exp1_gain.safe.mean(),
                             exp1_gain.risky.mean()])
    
    exp1_gain_err = np.array([exp1_gain.safe.sem(),
                               exp1_gain.risky.sem()])
    
    # loss means
    exp1_loss_y = np.array([exp1_loss.safe.mean(),
                        exp1_loss.risky.mean()])
    
    exp1_loss_err = np.array([exp1_loss.safe.sem(),
                         exp1_loss.risky.sem()])

   
    # collect the mean performance for the main block UE trials 
    exp1_gain_UE = pd.DataFrame()
    exp1_gain_UE['vpnum']   = exp1_gain_data.vpnum
    exp1_gain_UE['context'] = exp1_gain_data.context
    exp1_gain_UE['safe']    = np.nanmean(exp1_gain_data.iloc[:,17:20],axis=1)
    exp1_gain_UE['risky']   = np.nanmean(exp1_gain_data.iloc[:,20:23],axis=1)
    
    exp1_loss_UE = pd.DataFrame()
    exp1_loss_UE['vpnum']   = exp1_loss_data.vpnum
    exp1_loss_UE['context'] = exp1_loss_data.context
    exp1_loss_UE['safe']    = np.nanmean(exp1_loss_data.iloc[:,17:20],axis=1)
    exp1_loss_UE['risky']   = np.nanmean(exp1_loss_data.iloc[:,20:23],axis=1)

    # gain means
    exp1_UE_gain_y = np.array([exp1_gain_UE.safe.mean(),
                          exp1_gain_UE.risky.mean()])
    
    exp1_UE_gain_err = np.array([exp1_gain_UE.safe.sem(),
                            exp1_gain_UE.risky.sem()])
    
    # loss means
    exp1_UE_loss_y = np.array([exp1_loss_UE.safe.mean(),
                          exp1_loss_UE.risky.mean()])
    
    exp1_UE_loss_err = np.array([exp1_loss_UE.safe.sem(),
                            exp1_loss_UE.risky.sem()])

    # now get the EQ Safe vs Risky trials
    exp1_gain_EQ_y = np.array([exp1_gain_data.EQ20.mean(),
                            exp1_gain_data.EQ50.mean(),
                            exp1_gain_data.EQ80.mean()])
    
    exp1_gain_EQ_yerr = np.array([exp1_gain_data.EQ20.sem(),
                             exp1_gain_data.EQ50.sem(),
                             exp1_gain_data.EQ80.sem()])
    
    exp1_loss_EQ_y = np.array([exp1_loss_data.EQ20.mean(),
                            exp1_loss_data.EQ50.mean(),
                            exp1_loss_data.EQ80.mean()])
    
    exp1_loss_EQ_yerr = np.array([exp1_loss_data.EQ20.sem(),
                                exp1_loss_data.EQ50.sem(),
                                exp1_loss_data.EQ80.sem()])


    #------------------
    # EXPERIMENT 2
    #------------------
    exp2_gain_train = pd.DataFrame()
    exp2_gain_train['vpnum']   = exp2_gain_data.vpnum
    exp2_gain_train['context'] = exp2_gain_data.context
    exp2_gain_train['safe']    = np.nanmean(exp2_gain_data.iloc[:,5:8],axis=1)
    exp2_gain_train['risky']   = np.nanmean(exp2_gain_data.iloc[:,8:10],axis=1)
    
    exp2_loss_train = pd.DataFrame()
    exp2_loss_train['vpnum']   = exp2_loss_data.vpnum
    exp2_loss_train['context'] = exp2_loss_data.context
    exp2_loss_train['safe']    = np.nanmean(exp2_loss_data.iloc[:,5:8],axis=1)
    exp2_loss_train['risky']   = np.nanmean(exp2_loss_data.iloc[:,8:10],axis=1)

    # gain_train means
    exp2_gain_train_y = np.array([exp2_gain_train.safe.mean(),
                             exp2_gain_train.risky.mean()])
    
    exp2_gain_train_err = np.array([exp2_gain_train.safe.sem(),
                               exp2_gain_train.risky.sem()])
    
    # loss_train means
    exp2_loss_train_y = np.array([exp2_loss_train.safe.mean(),
                             exp2_loss_train.risky.mean()])
    
    exp2_loss_train_err = np.array([exp2_loss_train.safe.sem(),
                               exp2_loss_train.risky.sem()])
    

    # collect the mean performance for the main block pure trials 
    exp2_gain = pd.DataFrame()
    exp2_gain['vpnum']   = exp2_gain_data.vpnum
    exp2_gain['context'] = exp2_gain_data.context
    exp2_gain['safe']    = np.nanmean(exp2_gain_data.iloc[:,11:14],axis=1)
    exp2_gain['risky']   = np.nanmean(exp2_gain_data.iloc[:,14:17],axis=1)
    
    exp2_loss = pd.DataFrame()
    exp2_loss['vpnum']   = exp2_loss_data.vpnum
    exp2_loss['context'] = exp2_loss_data.context
    exp2_loss['safe']    = np.nanmean(exp2_loss_data.iloc[:,11:14],axis=1)
    exp2_loss['risky']   = np.nanmean(exp2_loss_data.iloc[:,14:17],axis=1)

    # gain means
    exp2_gain_y = np.array([exp2_gain.safe.mean(),
                             exp2_gain.risky.mean()])
    
    exp2_gain_err = np.array([exp2_gain.safe.sem(),
                               exp2_gain.risky.sem()])
    
    # loss means
    exp2_loss_y = np.array([exp2_loss.safe.mean(),
                        exp2_loss.risky.mean()])
    
    exp2_loss_err = np.array([exp2_loss.safe.sem(),
                         exp2_loss.risky.sem()])
    
    # collect the mean performance for the main block UE trials 
    exp2_gain_UE = pd.DataFrame()
    exp2_gain_UE['vpnum']   = exp2_gain_data.vpnum
    exp2_gain_UE['context'] = exp2_gain_data.context
    exp2_gain_UE['safe']    = np.nanmean(exp2_gain_data.iloc[:,17:20],axis=1)
    exp2_gain_UE['risky']   = np.nanmean(exp2_gain_data.iloc[:,20:23],axis=1)
    
    exp2_loss_UE = pd.DataFrame()
    exp2_loss_UE['vpnum']   = exp2_loss_data.vpnum
    exp2_loss_UE['context'] = exp2_loss_data.context
    exp2_loss_UE['safe']    = np.nanmean(exp2_loss_data.iloc[:,17:20],axis=1)
    exp2_loss_UE['risky']   = np.nanmean(exp2_loss_data.iloc[:,20:23],axis=1)

    # gain means
    exp2_UE_gain_y = np.array([exp2_gain_UE.safe.mean(),
                          exp2_gain_UE.risky.mean()])
    
    exp2_UE_gain_err = np.array([exp2_gain_UE.safe.sem(),
                            exp2_gain_UE.risky.sem()])
    
    # loss means
    exp2_UE_loss_y = np.array([exp2_loss_UE.safe.mean(),
                          exp2_loss_UE.risky.mean()])
    
    exp2_UE_loss_err = np.array([exp2_loss_UE.safe.sem(),
                            exp2_loss_UE.risky.sem()])


    # now get the EQ Safe vs Risky trials
    exp2_gain_EQ_y = np.array([exp2_gain_data.EQ20.mean(),
                            exp2_gain_data.EQ50.mean(),
                            exp2_gain_data.EQ80.mean()])
    
    exp2_gain_EQ_yerr = np.array([exp2_gain_data.EQ20.sem(),
                             exp2_gain_data.EQ50.sem(),
                             exp2_gain_data.EQ80.sem()])
    
    exp2_loss_EQ_y = np.array([exp2_loss_data.EQ20.mean(),
                            exp2_loss_data.EQ50.mean(),
                            exp2_loss_data.EQ80.mean()])
    
    exp2_loss_EQ_yerr = np.array([exp2_loss_data.EQ20.sem(),
                                exp2_loss_data.EQ50.sem(),
                                exp2_loss_data.EQ80.sem()])

#----------------------------------------------------------------
#                         PLOTTING
#----------------------------------------------------------------


    # plot data
    # define the color map
    cmap = plt.cm.Paired(np.linspace(0, 1, 12))

    fig = plt.figure(figsize=(8, 4), dpi=300)

    gs = fig.add_gridspec(2,12)

    # experiment 1
    ax0 = fig.add_subplot(gs[0, 0: 2])
    ax1 = fig.add_subplot(gs[0, 2: 4])
    ax2 = fig.add_subplot(gs[0, 4: 6])
    ax3 = fig.add_subplot(gs[0, 8: 12])
    # experiment 2
    ax4 = fig.add_subplot(gs[1, 0: 2])
    ax5 = fig.add_subplot(gs[1, 2: 4])
    ax6 = fig.add_subplot(gs[1, 4: 6])
    ax7 = fig.add_subplot(gs[1, 8: 12])
    
    xlims = np.array([.8,2.2])
    
    if datatype == 'choice':
        ylims = np.array([0,1])
        eq_ylim = np.array([.2,.8])
        ylbl= 'p(Choose Best)'
        eq_ylbl = 'p(Choose Risky)'
        ytcks = np.array([.0,.5,1])
        eq_ytcks = np.array([.2, .5, .8])
        
    else:
        ylims = np.array([400,1000])
        ylbl = 'RT (ms)'
        eq_ylim = ylims
        eq_ylbl = ylbl
        ytcks = np.array([400, 700, 1000])
        eq_ytcks = ytcks

    #--------------------------
    #    EXP 1 - TRAINING
    #--------------------------
    ax0.errorbar(np.array([1,2]), exp1_gain_train_y, exp1_gain_train_err,
                   color = cmap[1,:],capsize=0, linewidth=2, marker='.')
    
    ax0.errorbar(np.array([1,2]), exp1_loss_train_y, exp1_loss_train_err,
                   color = cmap[5,:],capsize=0,linewidth=2, marker='.')
    
    ax0.set_ylim(ylims)
    ax0.set_yticks(ytcks)
    ax0.set_xticks([])
    #ax0.set_xticklabels(['S', 'R'])
    ax0.set_xlim(xlims)
    ax0.legend(['Gain', 'Loss'])
    ax0.set_ylabel('Exp 1 \n' + ylbl)
    ax0.set_title('Training')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)

    #--------------------------
    #  EXP 1 - PURE TRIALS 
    #--------------------------
    ax1.errorbar(np.array([1,2]), exp1_gain_y, exp1_gain_err,
                   color = cmap[1,:],capsize=0, linewidth=2, marker='.')
    
    ax1.errorbar(np.array([1,2]), exp1_loss_y, exp1_loss_err,
                   color = cmap[5,:],capsize=0,linewidth=2, marker='.')
    
    ax1.set_ylim(ylims)
    ax1.get_yaxis().set_visible(False)
    ax1.set_xticks([])
    ax1.set_xlim(xlims)
    #ax1.set_xticklabels(['S', 'R'])
    ax1.set_title('Pure')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    #--------------------------
    #  EXP 1 - UE TRIALS
    #--------------------------
    ax2.errorbar(np.array([1,2]), exp1_UE_gain_y, exp1_UE_gain_err,
                   color = cmap[1,:],capsize=0, linewidth=2, marker='.')
    
    ax2.errorbar(np.array([1,2]), exp1_UE_loss_y, exp1_UE_loss_err,
                   color = cmap[5,:],capsize=0,linewidth=2, marker='.')
    
    ax2.set_ylim(ylims)
    ax2.set_xticks([])
    #ax2.set_xticklabels(['S>R', 'R>S'])
    ax2.set_xlim(xlims)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title('Unequal S vs R')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    

    #--------------------------
    #  EXP 1 - EQ TRIALS
    #--------------------------
    ax3.errorbar(np.array([1,2,3]),exp1_gain_EQ_y, exp1_gain_EQ_yerr,
                   color=cmap[1,:], capsize=0, linewidth=2, marker='.')
    
    ax3.errorbar(np.array([1,2,3]),exp1_loss_EQ_y, exp1_loss_EQ_yerr,
                   color=cmap[5,:], capsize=0, linewidth=2, marker='.')
  
    ax3.set_ylim(eq_ylim)
    ax3.set_ylabel(eq_ylbl)
    ax3.set_xticks([])
    ax3.set_yticks(eq_ytcks)
    #ax3.set_xticklabels(['EQ20', 'EQ50', 'EQ80'])
    ax3.set_title('Equal S vs R')


    #---------------------------------------------
    #--------------------------
    #    EXP 2 - TRAINING
    #--------------------------
    ax4.errorbar(np.array([1,2]), exp2_gain_train_y, exp2_gain_train_err,
                   color = cmap[1,:],capsize=0, linewidth=2, marker='.')
    
    ax4.errorbar(np.array([1,2]), exp2_loss_train_y, exp2_loss_train_err,
                   color = cmap[5,:],capsize=0,linewidth=2, marker='.')
    
    ax4.set_ylim(ylims)
    ax4.set_yticks(ytcks)
    ax4.set_xticks([1,2])
    ax4.set_xlim(xlims)
    ax4.set_xticklabels(['S', 'R'])
    ax4.set_ylabel('Exp 2 \n' + ylbl)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    #--------------------------
    #  EXP 2 - PURE TRIALS 
    #--------------------------
    ax5.errorbar(np.array([1,2]), exp2_gain_y, exp2_gain_err,
                   color = cmap[1,:],capsize=0, linewidth=2, marker='.')
    
    ax5.errorbar(np.array([1,2]), exp2_loss_y, exp2_loss_err,
                   color = cmap[5,:],capsize=0,linewidth=2, marker='.')
    
    ax5.set_ylim(ylims)
    ax5.get_yaxis().set_visible(False)
    ax5.set_xticks([1,2])
    ax5.set_xlim(xlims)
    ax5.set_xticklabels(['S', 'R'])
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.spines['left'].set_visible(False)

    #--------------------------
    #  EXP 2 - UE TRIALS
    #--------------------------
    ax6.errorbar(np.array([1,2]), exp2_UE_gain_y, exp2_UE_gain_err,
                   color = cmap[1,:],capsize=0, linewidth=2, marker='.')
    
    ax6.errorbar(np.array([1,2]), exp2_UE_loss_y, exp2_UE_loss_err,
                   color = cmap[5,:],capsize=0,linewidth=2, marker='.')
    
    ax6.set_ylim(ylims)
    ax6.set_xticks([1,2])
    ax6.set_xlim(xlims)
    ax6.get_yaxis().set_visible(False)
    ax6.set_xticklabels(['S>R', 'R>S'])
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.spines['left'].set_visible(False)
    

    #--------------------------
    #  EXP 2 - EQ TRIALS
    #--------------------------
    ax7.errorbar(np.array([1,2,3]),exp2_gain_EQ_y, exp2_gain_EQ_yerr,
                   color=cmap[1,:], capsize=0, linewidth=2, marker='.')
    
    ax7.errorbar(np.array([1,2,3]),exp2_loss_EQ_y, exp2_loss_EQ_yerr,
                   color=cmap[5,:], capsize=0, linewidth=2, marker='.')
  
    ax7.set_ylim(eq_ylim)
    ax7.set_ylabel(eq_ylbl)
    ax7.set_xticks([1,2,3])
    ax7.set_yticks(eq_ytcks)
    ax7.set_xticklabels(['EQ20', 'EQ50', 'EQ80'])
 
    
# END of function

def compare_both_experiments_risk_preference(exp1_gain_data, exp1_loss_data,
                                             exp2_gain_data, exp2_loss_data):

    exp1_EQbias = np.concatenate([np.mean(exp1_gain_data.iloc[:,23:26], axis = 1),
                                  np.mean(exp1_loss_data.iloc[:,23:26], axis = 1)])

    exp2_EQbias = np.concatenate([np.mean(exp2_gain_data.iloc[:,23:26], axis = 1),
                                  np.mean(exp2_loss_data.iloc[:,23:26], axis = 1)])

    exp1_subj = np.concatenate([np.arange(len(exp1_gain_data)), 
                                np.arange(len(exp1_loss_data))])

    exp2_subj = np.concatenate([np.arange(len(exp2_gain_data))+100, 
                                np.arange(len(exp2_loss_data))+100])

    EQbias = np.concatenate([exp1_EQbias, exp2_EQbias])

    exp_factor = np.concatenate([np.ones(shape = len(exp1_EQbias),),
                                 np.ones(shape = len(exp2_EQbias),)*2])

    context_factor = np.concatenate([np.ones(shape = len(exp1_gain_data),),
                                     np.ones(shape = len(exp1_loss_data),)*2,
                                     np.ones(shape = len(exp2_gain_data),),
                                     np.ones(shape = len(exp2_loss_data),)*2])

    subject = np.concatenate([exp1_subj, exp2_subj])

    df = pd.DataFrame()
    df['EQbias'] = EQbias
    df['exp']   = exp_factor
    df['context'] = context_factor
    df['subject'] =subject

    results = pg.mixed_anova(data=df, 
                            dv = 'EQbias',
                            within = 'context',
                            between='exp',
                            subject= 'subject')

    
    exp1_gain_means = np.mean(exp1_gain_data.iloc[:,23:26], axis = 1)
    exp1_loss_means = np.mean(exp1_loss_data.iloc[:,23:26], axis = 1)
    exp2_gain_means = np.mean(exp2_gain_data.iloc[:,23:26], axis = 1)
    exp2_loss_means = np.mean(exp2_loss_data.iloc[:,23:26], axis = 1)

    exp1_gain_sem = exp1_gain_means.std()/np.sqrt(len(exp1_gain_means))
    exp1_loss_sem = exp1_loss_means.std()/np.sqrt(len(exp1_loss_means))
    exp2_gain_sem = exp2_gain_means.std()/np.sqrt(len(exp2_gain_means))
    exp2_loss_sem = exp2_loss_means.std()/np.sqrt(len(exp2_loss_means))

    
    # make a figure and plot
    cmap = plt.cm.Paired(np.linspace(0, 1, 12))

    fig, ax = plt.subplots(1,1, dpi = 300, figsize = (2,3))

    ax.errorbar([1,2], [np.mean(exp1_gain_means), np.mean(exp2_gain_means)],
                       [exp1_gain_sem, exp2_gain_sem],marker='o',color=cmap[1,:],label='Gain')
    ax.errorbar([1,2], [np.mean(exp1_loss_means), np.mean(exp2_loss_means)],
                       [exp1_loss_sem, exp2_loss_sem],marker='o',color=cmap[5,:],label='Loss')
    ax.set_xticks([1,2])
    ax.set_xlim([.7, 2.3])
    ax.set_xlabel('Experiment #')
    ax.set_ylabel('p(Choose Risky)')
    ax.legend()
# END of function


def plot_individual_subjectEQbiases(exp1_gain_data, exp1_loss_data,
                                    exp2_gain_data, exp2_loss_data):

    exp1_gain_EQbias = exp1_gain_data.iloc[:,23:26].T.to_numpy()
    exp1_loss_EQbias = exp1_loss_data.iloc[:,23:26].T.to_numpy()
    exp2_gain_EQbias = exp2_gain_data.iloc[:,23:26].T.to_numpy()
    exp2_loss_EQbias = exp2_loss_data.iloc[:,23:26].T.to_numpy()
    
    
    # make a figure and plot
    cmap = plt.cm.Paired(np.linspace(0, 1, 12))

    fig, ax = plt.subplots(1,2, dpi = 300, figsize = (6,3))
    fig.tight_layout(h_pad=4)


    ax[0].plot([1,2,3],exp1_gain_EQbias[:,0], color = cmap[1,:], linewidth = 1)
    ax[0].plot([1,2,3],exp1_loss_EQbias[:,0], color = cmap[5,:], linewidth = 1)
    ax[0].plot([1,2,3],exp1_gain_EQbias, color = cmap[1,:], linewidth = 1)
    ax[0].plot([1,2,3],exp1_loss_EQbias, color = cmap[5,:], linewidth = 1)
    ax[0].set_xticks([1,2,3])
    ax[0].set_xlim([.7, 3.3])
    ax[0].set_xticklabels({'EQ20','EQ50','EQ80'})
    ax[0].set_xlabel('Equivaluable Condition')
    ax[0].set_ylabel('p(Choose Risky)')
    ax[0].legend({'Gain','Loss'})
    ax[0].set_title('Exp. 1')

    ax[1].plot([1,2,3],exp2_gain_EQbias[:,0], color = cmap[1,:], linewidth = 1)
    ax[1].plot([1,2,3],exp2_loss_EQbias[:,0], color = cmap[5,:], linewidth = 1)
    ax[1].plot([1,2,3],exp2_gain_EQbias, color = cmap[1,:], linewidth = 1)
    ax[1].plot([1,2,3],exp2_loss_EQbias, color = cmap[5,:], linewidth = 1)
    ax[1].set_xticks([1,2,3])
    ax[1].set_xlim([.7, 3.3])
    ax[1].set_xticklabels({'EQ20','EQ50','EQ80'})
    ax[1].set_title('Exp. 2')


# END of function


def plotChoice_or_RT(gaindata,lossdata,datatype):
    '''
    This function plots and statistically assesses either the choice or RT
    data (as specified in the datatype argument).
    '''
    
    sns.set_theme(style="ticks")  
    palette = sns.color_palette("colorblind")
    
    gain_means = gaindata.mean()
    gain_sems  = gaindata.sem()
    loss_means = lossdata.mean()
    loss_sems  = lossdata.sem()
    
    choicex = [1, 2, 3]
    
    fig, ax = plt.subplots(2,2,figsize=(6, 6), dpi=300)
    fig.tight_layout(h_pad=4)
    
    cmap = plt.cm.Paired(np.linspace(0, 1, 12))

    # training means
    ax[0,0].errorbar(choicex,gain_means.iloc[3:6],gain_sems.iloc[3:6],marker='o',color=cmap[1,:],label='gain, safe')
    ax[0,0].errorbar(choicex,gain_means.iloc[6:9],gain_sems.iloc[6:9],marker='o',color=cmap[5,:],label='gain, risky')
    ax[0,0].errorbar(choicex,loss_means.iloc[3:6],loss_sems.iloc[3:6],marker='o',color=cmap[0,:],label='loss, safe')
    ax[0,0].errorbar(choicex,loss_means.iloc[6:9],loss_sems.iloc[6:9],marker='o',color=cmap[4,:],label='loss, risky')
    
    ax[0,0].set_xticks(ticks=[1,2,3])
    ax[0,0].set_xticklabels(("20vs50","50vs80","20vs80"))
    ax[0,0].set_xlabel('Condition')
    
    if datatype == 'choice':
        ax[0,0].set_ylim(0,1)
        ax[0,0].set_ylabel('p(Choose Best)')
    else:
        ax[0,0].set_ylabel('RT (ms)')
        ax[0,0].set_ylim(200,1000)

    
    ax[0,0].legend()
    ax[0,0].set_title('Training')
    


    ax[0,1].errorbar(choicex,gain_means.iloc[9:12],gain_sems.iloc[9:12],marker='o',color=cmap[1,:],label='gain, safe')
    ax[0,1].errorbar(choicex,gain_means.iloc[12:15],gain_sems.iloc[12:15],marker='o',color=cmap[5,:],label='gain, risky')
    ax[0,1].errorbar(choicex,loss_means.iloc[9:12],loss_sems.iloc[9:12],marker='o',color=cmap[0,:],label='gain, safe')
    ax[0,1].errorbar(choicex,loss_means.iloc[12:15],loss_sems.iloc[12:15],marker='o',color=cmap[4,:],label='gain, risky')
    ax[0,1].set_xticks(ticks=[1,2,3])
    ax[0,1].set_xticklabels(("20vs50","50vs80","20vs80"))
    ax[0,1].set_xlabel('Condition')
    
    if datatype == 'choice':
        ax[0,1].set_ylim(0,1)
        ax[0,1].set_ylabel('p(Choose Best)')
    else:
        ax[0,1].set_ylabel('RT (ms)')
        ax[0,1].set_ylim(200,1000)
    
    ax[0,1].set_title('Pure')
    
    

    ax[1,0].errorbar(choicex,gain_means.iloc[15:18],gain_sems.iloc[15:18],marker='o',color=cmap[1,:],label='gain, s > r')
    ax[1,0].errorbar(choicex,gain_means.iloc[18:21],gain_sems.iloc[18:21],marker='o',color=cmap[5,:],label='gain, r > s')
    ax[1,0].errorbar(choicex,loss_means.iloc[15:18],loss_sems.iloc[15:18],marker='o',color=cmap[0,:],label='loss, s > r')
    ax[1,0].errorbar(choicex,loss_means.iloc[18:21],loss_sems.iloc[18:21],marker='o',color=cmap[4,:],label='loss, r > s')
    ax[1,0].set_xticks(ticks=[1,2,3])
    ax[1,0].set_xticklabels(("20vs50","50vs80","20vs80"))
    ax[1,0].set_xlabel('Condition')
      
    if datatype == 'choice':
          ax[1,0].set_ylim(0,1)
          ax[1,0].set_ylabel('p(Choose Best)')
    else:
          ax[1,0].set_ylabel('RT (ms)')
          ax[1,0].set_ylim(200,1000)
     
    ax[1,0].legend()    
    ax[1,0].set_title('Unequal S vs R')
    
    
    ax[1,1].errorbar(choicex,gain_means.iloc[21:24],gain_sems.iloc[21:24],marker='o',color=cmap[9,:],label='gain')
    ax[1,1].errorbar(choicex,loss_means.iloc[21:24],loss_sems.iloc[21:24],marker='o',color=cmap[8,:],label='loss')
    ax[1,1].set_xticks(ticks=[1,2,3])
    ax[1,1].set_xticklabels(("20vs20","50vs50","80vs80"))
    ax[1,1].set_xlabel('Condition')
 
    if datatype == 'choice':
       ax[1,1].set_ylim(0,1)
       ax[1,1].set_ylabel('p(Choose Risky)')
       ax[1,1].plot(choicex,np.array([.5,.5,.5]),color='tab:gray')
    else:
       ax[1,1].set_ylabel('RT (ms)')
       ax[1,1].set_ylim(200,1000)
    
    ax[1,1].set_title('Equal S vs R')
    
    #----------------------------------
    #              stats 
    #----------------------------------
    if datatype == 'choice':
        loss_choicedataonly = lossdata.iloc[:,5:]
        gain_choicedataonly = gaindata.iloc[:,5:]
        
        # initialize results dataframes
        loss_choice_stats=pd.DataFrame()
        gain_choice_stats=pd.DataFrame()
        
        # do a t-test of every condition against chance (50%)
        for cond in range(loss_choicedataonly.shape[1]):
            
            loss_choice_stats = loss_choice_stats.append(pg.ttest(loss_choicedataonly.iloc[:,cond],.5))
            gain_choice_stats = gain_choice_stats.append(pg.ttest(gain_choicedataonly.iloc[:,cond],.5))
       
        # rename the rows so we can inspect later
        loss_choice_stats.index = loss_choicedataonly.columns
        gain_choice_stats.index = gain_choicedataonly.columns
    # END of t-testing choice data against chance 
    
    
    # do repeated measures anovas for the conditions 
    alldata = gaindata.append(lossdata)
    
    # TRAINING   
    train_responses = alldata.melt(id_vars = ['vpnum','context'],
                                   value_vars=['t_s_20v50','t_s_50v80','t_s_20v80',
                                               't_r_20v50','t_r_50v80','t_r_20v80'])
    
    train_responses['stimtype'] = train_responses['variable'].str.contains('r').astype(int)
    
    train_responses['cond'] = np.zeros(len(train_responses)).astype(int)
    train_responses['cond'].loc[train_responses['variable'].str.contains('20v50')] = 1
    train_responses['cond'].loc[train_responses['variable'].str.contains('50v80')] = 2
    train_responses['cond'].loc[train_responses['variable'].str.contains('20v80')] = 3
    
    train_mdl = AnovaRM(data = train_responses, depvar='value',subject = 'vpnum',
                            within=['cond','context','stimtype']).fit()
    
    train_results = train_mdl.anova_table
    
 
    # PURE
    pure_responses = alldata.melt(id_vars = ['vpnum','context'],
                                   value_vars=['s_20v50','s_50v80','s_20v80',
                                               'r_20v50','r_50v80','r_20v80'])
    
    pure_responses['stimtype'] = pure_responses['variable'].str.contains('r').astype(int)
    
    pure_responses['cond'] = np.zeros(len(pure_responses)).astype(int)
    pure_responses['cond'].loc[pure_responses['variable'].str.contains('20v50')] = 1
    pure_responses['cond'].loc[pure_responses['variable'].str.contains('50v80')] = 2
    pure_responses['cond'].loc[pure_responses['variable'].str.contains('20v80')] = 3
    
    pure_mdl = AnovaRM(data = pure_responses, depvar='value',subject = 'vpnum',
                            within=['cond','context','stimtype']).fit()
    
    pure_results = pure_mdl.anova_table
    
    # UNEQUAL Safe vs Risky
    UE_responses = alldata.melt(id_vars = ['vpnum','context'],
                                   value_vars=['UE_s_20v50','UE_s_50v80','UE_s_20v80',
                                               'UE_r_20v50','UE_r_50v80','UE_r_20v80'])
    
    UE_responses['stimtype'] = UE_responses['variable'].str.contains('r').astype(int)
    
    UE_responses['cond'] = np.zeros(len(UE_responses)).astype(int)
    UE_responses['cond'].loc[UE_responses['variable'].str.contains('20v50')] = 1
    UE_responses['cond'].loc[UE_responses['variable'].str.contains('50v80')] = 2
    UE_responses['cond'].loc[UE_responses['variable'].str.contains('20v80')] = 3
    
    UE_mdl = AnovaRM(data = UE_responses, depvar='value',subject = 'vpnum',
                            within=['cond','context','stimtype']).fit()
    
    UE_results = UE_mdl.anova_table
    
    # EQUAL Safe vs Risky
    EQ_responses = alldata.melt(id_vars = ['vpnum','context'],
                                   value_vars=['EQ20','EQ50','EQ80'])
    EQ_responses=EQ_responses.rename(columns={'variable':'prob'})
    
    EQ_mdl = AnovaRM(data = EQ_responses, depvar='value',subject = 'vpnum',
                            within=['context','prob']).fit()
    
    EQ_results = EQ_mdl.anova_table
    
    xx=[]   # set your breakpoint here to look at stats!    
# END of plotChoice_or_RT


def win_stay_analysis(alldata):
    '''
    This analysis looks for instances where a certain option was chosen 
    and yielded a non-zero outcome and asks how likely the person is to select
    that option the next time it's presented (i.e. p(Win-Stay)).
    '''

    # get subject IDs
    sIDs = alldata.vpNum.unique()
    img_IDs = alldata.imageNumberLeft.unique()
    
    # initialize the results array
    winstay = np.zeros(shape = (len(sIDs),len(img_IDs)))
    winstay[:] = np.nan
    
    # loop through each subject
    for s in range(len(sIDs)):
        
        sdata = alldata.loc[alldata.vpNum ==sIDs[s],:]
        
        # define trial stim
        stim = np.zeros(shape = (len(sdata),2))
        stim[:,0] = sdata.imageNumberLeft
        stim[:,1] = sdata.imageNumberRight
        stim = stim.astype(int)
        
        # what direction did the person pick each time?
        humanchoiceside = (sdata.responseSide == 'right').astype(int)
        
        # initialize an array of what the person picked
        humanchoice = np.zeros(shape = len(humanchoiceside))

        # figure out which image was chosen on each trial by the person
        for t in range(len(sdata)):
            
            humanchoice[t] = stim[t,humanchoiceside[t]]
         
            humanchoice = humanchoice.astype(int)
        
        for stimnum in range(6):
            
            stimintrial = (stim[:,0] == stimnum) | (stim[:,1] == stimnum)
            stim_tnums =  np.argwhere(stimintrial)

            
            # find all instances where this stimulus was chosen and yielded a hit         
            stimhits = (humanchoice == stimnum) & (sdata.rewardCode ==1)
          
            # get trial numbers
            hit_tnums = stimhits.index[stimhits]
            
            stim_ws = np.zeros(shape = len(hit_tnums))
            stim_ws[:] = np.nan
            
            # loop through each of these trials and find the next time this
            # stim was present           
            for i in range(len(hit_tnums)-1):
                
                trialsinrange = stim_tnums > hit_tnums[i]
                trialsinrange = np.argwhere(trialsinrange)  
                
                nexttrial_ix = trialsinrange[0,0]
                nexttrial    = stim_tnums[nexttrial_ix]
                stim_ws[i] = (humanchoice[nexttrial] == stimnum).astype(int)
            # END of looping over trials where a hit occurred
                        
            winstay[s,stimnum] = np.nanmean(stim_ws)  
        # END of looping over stimuli
    # END of looping over subjects
    
    
    # set nans to zeros
    winstay = np.nan_to_num(winstay)

    xx=[] # for setting a breakpoint
               
    return winstay
# END of WinStayAnalysis



def lose_stay_analysis(alldata):
    '''
    This analysis looks for instances where a certain option was chosen 
    and yielded a zero outcome and asks how likely the person is to select
    that option the next time it's presented (i.e. p(Lose-Stay)).
    '''

    # get subject IDs and image IDs
    sIDs = alldata.vpNum.unique()
    img_IDs = alldata.imageNumberLeft.unique()

    
    # initialize the results array
    losestay = np.zeros(shape = (len(sIDs),len(img_IDs)))
    losestay[:] = np.nan
    
    # loop through each subject
    for s in range(len(sIDs)):
        
        sdata = alldata.loc[alldata.vpNum ==sIDs[s],:]
        
        # define trial stim
        stim = np.zeros(shape = (len(sdata),2))
        stim[:,0] = sdata.imageNumberLeft
        stim[:,1] = sdata.imageNumberRight
        stim = stim.astype(int)
        
        # what direction did the person pick each time?
        humanchoiceside = (sdata.responseSide == 'right').astype(int)
        
        # initialize an array of what the person picked
        humanchoice = np.zeros(shape = len(humanchoiceside))

        # figure out which image was chosen on each trial by the person
        for t in range(len(sdata)):
            
            humanchoice[t] = stim[t,humanchoiceside[t]]
         
            humanchoice = humanchoice.astype(int)
                
        
        for stimnum in range(len(img_IDs)):
            
            stimintrial = (stim[:,0] == stimnum) | (stim[:,1] == stimnum)
            stim_tnums =  np.argwhere(stimintrial)

            
            # find all instances where this stimulus was chosen and yielded a miss         
            stimmiss = (humanchoice == stimnum) & (sdata.rewardCode ==0)

            
            # get trial numbers
            miss_tnums = stimmiss.index[stimmiss]
            
            stim_ls = np.zeros(shape = len(miss_tnums))
            stim_ls[:] = np.nan
            
            # loop through each of these trials and find the next time this
            # stim was present           
            for i in range(len(miss_tnums)-1):
                
                trialsinrange = stim_tnums > miss_tnums[i]
                trialsinrange = np.argwhere(trialsinrange)  
                
                nexttrial_ix = trialsinrange[0,0]
                nexttrial    = stim_tnums[nexttrial_ix]
                stim_ls[i] = (humanchoice[nexttrial] == stimnum).astype(int)

            # END of looping over trials where a hit occurred
            
            losestay[s,stimnum] = np.nanmean(stim_ls)  
        # END of looping over stimuli
    # END of looping over subjects.
    
    # set nan to zeros
    losestay = np.nan_to_num(losestay)

    xx=[] # for setting a breakpoint
               
    return losestay          
# END of WinStayAnalysis




def plotWinStay_LoseStay(gain_winstay,loss_winstay,gain_losestay,loss_losestay):
        
    
    # define colormap for plotting 
    cmap = plt.cm.Paired(np.linspace(0, 1, 12))
    
    # get means and sems
    gain_WS_mean = np.nanmean(gain_winstay,axis=0)
    gain_WS_sem  = np.nanstd(gain_winstay,axis=0) / np.sqrt(len(gain_winstay[:,1]))
    
    loss_WS_mean = np.nanmean(loss_winstay,axis=0)
    loss_WS_sem  = np.nanstd(loss_winstay,axis=0)/ np.sqrt(len(loss_winstay[:,1]))
    
    xvals = np.array([.2,.5,.8])
    
    fig, ax = plt.subplots(2,2,figsize=(6, 3), dpi=300)
    fig.tight_layout(h_pad=4)
    
    plt.subplot(1, 2, 1)
    plt.errorbar(xvals,gain_WS_mean[0:3],gain_WS_sem[0:3],label='gain, safe',
                 color=cmap[1,:], LineWidth = 2, marker = '.')
    plt.errorbar(xvals,gain_WS_mean[3:7],gain_WS_sem[3:7],label='gain, risky',
                 color=cmap[0,:], LineWidth = 2, marker = '.')
    plt.xlabel('p(Gain)')
    plt.ylabel('p(Hit-Stay)')
    plt.xticks(xvals)
    plt.ylim([0,1])
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.errorbar(xvals,loss_WS_mean[0:3],loss_WS_sem[0:3],label='loss, safe',
                 color=cmap[5,:], LineWidth = 2, marker = '.')
    plt.errorbar(xvals,loss_WS_mean[3:7],loss_WS_sem[3:7],label='loss, risky',
                 color=cmap[4,:], LineWidth = 2, marker = '.')
    plt.xlabel('p(Loss)')
    plt.ylabel('p(Miss-Stay)')
    plt.xticks(xvals)
    plt.ylim([0,1])
    plt.legend()
    

    # prepare data for linear mixed effects model
    
    # subject factor
    subj_ids = np.arange(len(gain_winstay))
    probs = np.array([1,2,3,1,2,3])
    stim_type = np.array([1,1,1,-1,-1,-1])

    #set NaN to zeros
     
    # initialize dataframe for lmemodel
    lmedata = pd.DataFrame()
    lmedata['gain_winstay'] = gain_winstay.reshape(-1,1).flatten()
    lmedata['loss_winstay'] = loss_winstay.reshape(-1,1).flatten()
    lmedata['stim_type']    = np.matlib.repmat(stim_type,int(len(lmedata.gain_winstay)/len(stim_type)),1).flatten()
    lmedata['prob']         = np.matlib.repmat(probs,int(len(lmedata.gain_winstay)/len(probs)),1).flatten()
    lmedata['subj'] = np.matlib.repmat(subj_ids,int(len(lmedata.gain_winstay)/len(subj_ids)),1).flatten()

    # now fit the linear mixed effects models
    gain_mdl = smf.mixedlm('gain_winstay ~ prob*stim_type', lmedata, groups=lmedata['subj']).fit()
    loss_mdl = smf.mixedlm('loss_winstay ~ prob*stim_type', lmedata, groups=lmedata['subj']).fit()
  
# END of plotWinStay


def distRLmodel_MLE(alldata):
     
    alphavals = np.linspace(.05,1,int(1/.05))
    #betas = np.linspace(1,40,20)
    #nparams = 3
    betas = np.array([1]) # this is for debugging
    nparams=2
    
    # get the combinations of alphas and betas
    #1st col = alphaplus, 2nd = alphaminus, 3rd = beta
    Qparams = np.array(np.meshgrid(alphavals, alphavals,betas)).T.reshape(-1,3)
    
    # get a subject's data
    subjIDs = alldata['vpNum'].unique()
    
    # initialize output
    bestparams = np.zeros(shape = (len(subjIDs),int(Qparams.shape[1])+2))
    bestAccOpt = np.zeros(shape = (len(subjIDs),2))
    best_Qtbl  = np.zeros(shape = (len(subjIDs),6))
    
    # iterate through each subject's data
    for s in range(len(subjIDs)):
 
        # pull out this subject's data        
        sdata = alldata.loc[alldata.vpNum == subjIDs[s],:]

        # accuracies for each parameter set
        paramAccs = np.zeros(shape = len(Qparams))
        paramOpt  = np.zeros(shape = len(Qparams))
        paramQtbl = np.zeros(shape = (len(Qparams),6))
        
        # log likelihoods for each param set
        paramLL = np.zeros(shape = len(Qparams))
        
        # define trial stim
        stim = np.column_stack([sdata.imageNumberLeft , sdata.imageNumberRight]).astype(int)
 
        # what direction did the person pick each time?
        humanchoiceside = (sdata.responseSide == 'right').astype(int)
        
        # figure out which image was chosen on each trial by the person
        humanchoice = np.diag(stim[:,humanchoiceside].astype(int))
           
        optimalside = (sdata.highProbSide == 'right').astype(int)
        
        if sdata.blkType[0] == 'Loss':
            optimalside = (optimalside == 0).astype(int)


        # now iterate over each combination of alphas and beta
        for a in range(len(Qparams)):
            
            alphaplus  = Qparams[a,0]
            alphaminus = Qparams[a,1]
            beta       = Qparams[a,2]
            
            # initialilze a set of log likelihoods for these trials
            this_param_LLs = np.zeros(len(sdata))

            print('\rSubject#: '+str(s) + ', Param#: '+ str(a)+' / '+str(len(Qparams))+'   ', end='')
                  
            # initialize a Qtable
            Qtbl = np.zeros(6) 
            xx=[] 
            
            # initialize an array for the learner's choices
            Qchoices = np.zeros(shape = len(humanchoiceside))
            
            Qoptimalchoice = np.zeros(shape = len(humanchoiceside))
            
            # loop through each trial 
            for t in range(len(sdata)):
                 
                # get some basic info about the trial            
                tOutcome = sdata.rewardPoints[t]

                # get likelihood that the Qlearner would pick the left option                            
                softmax = 1/(1+np.exp(beta*(Qtbl[stim[t,1]]-Qtbl[stim[t,0]])))
                
                if softmax == 1:
                    softmax = 0.99999999
                    
                if softmax ==0:
                    softmax = 0.00000001
                
                # get likelihood that Qlearner picks same as human
                if humanchoiceside[t] == 0: # if they went left
                    fit_likelihood = softmax
                else:
                    fit_likelihood = 1-softmax
                    
                this_param_LLs[t] = np.log(fit_likelihood)
                
                # does the TD learner pick left?
                wentleft = softmax > .5
                
                if wentleft:
                    side = 0
                else:
                    side = 1 # he went right
               
                
                # which stim is it?
                choice = stim[t,side] 
                
                # keep track of what the learner picked
                Qchoices[t] = choice
                
                # was this choice optimal?
                Qoptimalchoice[t] = (optimalside[t] == side).astype(int)   

                # what did the person pick?                                    
      
                # figure out which alpha value to use
                if (tOutcome - Qtbl[choice]) > 0:
                    tAlpha = alphaplus
                else:
                    tAlpha = alphaminus
                    
                # do the TD update based on what the persosn picked
                Qtbl[humanchoice[t]] = Qtbl[humanchoice[t]] + (tAlpha*(tOutcome -Qtbl[humanchoice[t]]))
            #END of looping over trials
        
            # what was the MLE for this param set?
            paramLL[a] = this_param_LLs.sum()
            
            # how accurate was this iteration in predicting a person's choices?
            paramAccs[a] = (Qchoices == humanchoice).astype(int).mean() 
            
            # how optimal did the learner perform?
            paramOpt[a] = Qoptimalchoice.mean() 
            
            paramQtbl[a,:] = Qtbl
            
            xx=[]             
        # END of looping over parameters 
        
        # assess which parameter set was best 
        bestix = np.nanargmax(paramLL)
        
        bestparams[s,0:int(Qparams.shape[1])]=Qparams[bestix,:]
        bestparams[s,int(Qparams.shape[1])]  =Qparams[bestix,0] / (Qparams[bestix,0:2].sum())
        
        # calculate the BIC score
        bic = np.log(len(sdata))*nparams -2*paramLL[bestix]
        
        # save bic score
        bestparams[s,int(Qparams.shape[1])+1] = bic
        
        # store accuracies
        bestAccOpt[s,0]  = paramAccs[bestix]
        bestAccOpt[s,1]  = paramOpt[bestix]
        
        # store best Qtbl
        best_Qtbl[s,:] = paramQtbl[bestix,:]
        
        
        
        """
        paramgrid = paramLL.reshape(10,10)
        plt.imshow(paramgrid)
        plt.colorbar()
        
        optgrid = paramOpt.reshape(10,10)
        plt.imshow(optgrid)
        plt.colorbar()
        """

    # END of looping through subjects  

    return bestparams, bestAccOpt, best_Qtbl           
# END of distRLmodel



def relate_distRL_to_EQbias(gain_bestparams, loss_bestparams,
                           gain_choice, loss_choice,
                           gain_bestAccOpt,loss_bestAccOpt):   


    # define colormap for plotting 
    cmap = plt.cm.Paired(np.linspace(0, 1, 12))
        
    gain_EQ_bias = np.nanmean(gain_choice.iloc[:,23:26], axis =1)
    
    loss_EQ_bias = np.nanmean(loss_choice.iloc[:,23:26], axis =1)
  
    
    # fit a linear mixed effect model
    lmedata = pd.DataFrame()
    lmedata['EQbias'] = np.concatenate([gain_EQ_bias,loss_EQ_bias])
    lmedata['context'] = np.concatenate([np.ones(len(gain_EQ_bias)),np.ones(len(gain_EQ_bias))*-1])
    lmedata['RLparam'] = np.concatenate([gain_bestparams[:,3],loss_bestparams[:,3]])
    lmedata['subject'] = np.concatenate([np.arange(len(gain_EQ_bias)),np.arange(len(gain_EQ_bias))])
    
    # fit the model
    RLxbias_mdl = smf.mixedlm('RLparam ~ EQbias', lmedata, groups=lmedata['subject']).fit()
    
    # extract intercept and beta for distRL quantile
    params = RLxbias_mdl.params
    intercept = params['Intercept']
    slope     = params['EQbias']
    
    
    # make figure and get plot    
    fig = plt.figure(figsize=(8, 2), dpi=300)

    gs = fig.add_gridspec(1,12)
    ax0 = fig.add_subplot(gs[0: 2])
    ax1 = fig.add_subplot(gs[3: 5])
    ax2 = fig.add_subplot(gs[6: 9])
 
    
    
    
    # plot model accuracy
    gain_acc_mean = gain_bestAccOpt[:,0].mean()
    gain_acc_sem = gain_bestAccOpt[:,0].std()/np.sqrt(len(gain_bestAccOpt[:,0]))
    
    loss_acc_mean = loss_bestAccOpt[:,0].mean()
    loss_acc_sem = loss_bestAccOpt[:,0].std()/np.sqrt(len(loss_bestAccOpt[:,0]))

    
    ax0.bar(1, gain_acc_mean, yerr=gain_acc_sem, align='center', alpha=1, 
              ecolor='black', capsize=0, color = cmap[1,:])
    
    ax0.bar(2, loss_acc_mean, yerr=loss_acc_sem, align='center', alpha=1, 
              ecolor='black', capsize=0, color = cmap[5,:])

    ax0.plot([0,3],[.5,.5],color = 'tab:gray', linestyle = '--')
    ax0.set_ylabel('Model Accuracy')
    ax0.set_xticks([1,2])
    ax0.set_ylim([0,1])
    ax0.set_yticks([0, .5, 1])
    ax0.set_xlim([.5,2.5])
    ax0.set_xticklabels(['Gain','Loss'])
    
    # plot best quantiles
    gain_q_mean = gain_bestparams[:,3].mean()
    gain_q_sem = gain_bestparams[:,3].std()/np.sqrt(len(gain_bestparams[:,3]))
    
    loss_q_mean = loss_bestparams[:,3].mean()
    loss_q_sem = loss_bestparams[:,3].std()/np.sqrt(len(loss_bestparams[:,3]))

    
    ax1.bar(1, gain_q_mean, yerr=gain_q_sem, align='center', alpha=1, 
              ecolor='black', capsize=0, color = cmap[1,:])
    
    ax1.bar(2, loss_q_mean, yerr=loss_q_sem, align='center', alpha=1, 
              ecolor='black', capsize=0, color = cmap[5,:])


    ax1.set_xticks([1,2])
    ax1.set_ylim([0,1])
    ax1.set_yticks([0, .5, 1])
    ax1.set_xlim([.5,2.5])
    ax1.set_xticklabels(['Gain','Loss'])
    ax1.set_ylabel('distRL Quantile')
    
    # predict model quantile from bias
    ax2.scatter(gain_EQ_bias,gain_bestparams[:,3],color=cmap[1,:], s =20)
    ax2.scatter(loss_EQ_bias,loss_bestparams[:,3],color=cmap[5,:], s =20)
    
    # plot line of best fit
    ax2.plot(np.array([0,1]),np.array([0,1])*slope + intercept, 
               color = 'black', linewidth=2)
    
    
    ax2.set_xlabel('EQ bias')
    ax2.set_ylabel('distRL Quantile')
    ax2.set_xticks(ticks = np.array([0, .5, 1]))
    ax2.set_yticks(ticks = np.array([0, .5, 1]))
    ax2.set_xlim([0,1])
    ax2.set_ylim([0,1])
  
    xx=[] 
# END of relate_distRL_to_EQbias


def both_exp_distRLxEQbias(exp1_gain_bestparams, exp1_loss_bestparams,
                           exp1_gain_choice, exp1_loss_choice,
                           exp1_gain_bestAccOpt,exp1_loss_bestAccOpt,
                           exp2_gain_bestparams, exp2_loss_bestparams,
                           exp2_gain_choice, exp2_loss_choice,
                           exp2_gain_bestAccOpt,exp2_loss_bestAccOpt):   


    # define colormap for plotting 
    cmap = plt.cm.Paired(np.linspace(0, 1, 12))
        
    exp1_gain_EQ_bias = np.nanmean(exp1_gain_choice.iloc[:,23:26], axis =1)
    exp1_loss_EQ_bias = np.nanmean(exp1_loss_choice.iloc[:,23:26], axis =1)

    exp2_gain_EQ_bias = np.nanmean(exp2_gain_choice.iloc[:,23:26], axis =1)
    exp2_loss_EQ_bias = np.nanmean(exp2_loss_choice.iloc[:,23:26], axis =1)

    n_subs1 = len(exp1_gain_EQ_bias)
    n_subs2 = len(exp2_gain_EQ_bias)

    
    # fit linear mixed effects models
    # EXPERIMENT 1
    exp1_lmedata = pd.DataFrame()
    exp1_lmedata['EQbias'] = np.concatenate([exp1_gain_EQ_bias,exp1_loss_EQ_bias])
    exp1_lmedata['context'] = np.concatenate([np.ones(shape=(n_subs1,)),np.ones(shape=(n_subs1,))*-1])
    exp1_lmedata['RLparam'] = np.concatenate([exp1_gain_bestparams[:,3],exp1_loss_bestparams[:,3]])
    exp1_lmedata['subject'] = np.concatenate([np.arange(n_subs1),np.arange(n_subs1)])
    
    # fit the model
    exp1_RLxbias_mdl = smf.mixedlm('RLparam ~ EQbias', exp1_lmedata, groups=exp1_lmedata['subject']).fit()
    
    # extract intercept and beta for distRL quantile
    exp1_params = exp1_RLxbias_mdl.params
    exp1_intercept = exp1_params['Intercept']
    exp1_slope     = exp1_params['EQbias']

    # EXPERIMENT 2
    exp2_lmedata = pd.DataFrame()
    exp2_lmedata['EQbias'] = np.concatenate([exp2_gain_EQ_bias,exp2_loss_EQ_bias])
    exp2_lmedata['context'] = np.concatenate([np.ones(shape=(n_subs2,)),np.ones(shape=(n_subs2,))*-1])
    exp2_lmedata['RLparam'] = np.concatenate([exp2_gain_bestparams[:,3],exp2_loss_bestparams[:,3]])
    exp2_lmedata['subject'] = np.concatenate([np.arange(n_subs2),np.arange(n_subs2)])
    
    # fit the model
    exp2_RLxbias_mdl = smf.mixedlm('RLparam ~ EQbias', exp2_lmedata, groups=exp2_lmedata['subject']).fit()
    
    # extract intercept and beta for distRL quantile
    exp2_params = exp2_RLxbias_mdl.params
    exp2_intercept = exp2_params['Intercept']
    exp2_slope     = exp2_params['EQbias']

    #-------
    # calculate distRL model accuracy
    #-------
    # EXPERIMENT 1
    exp1_gain_acc_mean = exp1_gain_bestAccOpt[:,0].mean()
    exp1_gain_acc_sem = exp1_gain_bestAccOpt[:,0].std()/np.sqrt(len(exp1_gain_bestAccOpt[:,0]))
    
    exp1_loss_acc_mean = exp1_loss_bestAccOpt[:,0].mean()
    exp1_loss_acc_sem = exp1_loss_bestAccOpt[:,0].std()/np.sqrt(len(exp1_loss_bestAccOpt[:,0]))

    # EXPERIMENT 2
    exp2_gain_acc_mean = exp2_gain_bestAccOpt[:,0].mean()
    exp2_gain_acc_sem = exp2_gain_bestAccOpt[:,0].std()/np.sqrt(len(exp2_gain_bestAccOpt[:,0]))
    
    exp2_loss_acc_mean = exp2_loss_bestAccOpt[:,0].mean()
    exp2_loss_acc_sem = exp2_loss_bestAccOpt[:,0].std()/np.sqrt(len(exp2_loss_bestAccOpt[:,0]))

    #-----
    # calculate best model quantiles 
    #-----
    # EXPERIMENT 1
    exp1_gain_q_mean = exp1_gain_bestparams[:,3].mean()
    exp1_gain_q_sem = exp1_gain_bestparams[:,3].std()/np.sqrt(len(exp1_gain_bestparams[:,3]))
    
    exp1_loss_q_mean = exp1_loss_bestparams[:,3].mean()
    exp1_loss_q_sem = exp1_loss_bestparams[:,3].std()/np.sqrt(len(exp1_loss_bestparams[:,3]))

    # EXPERIMENT 2
    exp2_gain_q_mean = exp2_gain_bestparams[:,3].mean()
    exp2_gain_q_sem = exp2_gain_bestparams[:,3].std()/np.sqrt(len(exp2_gain_bestparams[:,3]))
    
    exp2_loss_q_mean = exp2_loss_bestparams[:,3].mean()
    exp2_loss_q_sem = exp2_loss_bestparams[:,3].std()/np.sqrt(len(exp2_loss_bestparams[:,3]))

    #---------------------------------------------
    #                  PLOTTING
    #---------------------------------------------
    # make figure and axes  
    fig = plt.figure(figsize=(8, 4), dpi=300)

    gs = fig.add_gridspec(2,12)
    ax0 = fig.add_subplot(gs[0, 0: 2])
    ax1 = fig.add_subplot(gs[0, 3: 5])
    ax2 = fig.add_subplot(gs[0, 6: 9])
    ax3 = fig.add_subplot(gs[1, 0: 2])
    ax4 = fig.add_subplot(gs[1, 3: 5])
    ax5 = fig.add_subplot(gs[1, 6: 9])
 
    
    #-----
    # EXPERIMENT 1
    #-----

    # plot distRL model accuracy 
    ax0.bar(1, exp1_gain_acc_mean, yerr=exp1_gain_acc_sem, align='center', alpha=1, 
              ecolor='black', capsize=0, color = cmap[1,:])
    
    ax0.bar(2, exp1_loss_acc_mean, yerr=exp1_loss_acc_sem, align='center', alpha=1, 
              ecolor='black', capsize=0, color = cmap[5,:])

    ax0.plot([0,3],[.5,.5],color = 'tab:gray', linestyle = '--')
    ax0.set_ylabel('Exp 1 \n Model Accuracy')
    ax0.set_xticks([])
    ax0.set_ylim([0,1])
    ax0.set_yticks([0, .5, 1])
    ax0.set_xlim([.5,2.5])
    #ax0.set_xticklabels(['Gain','Loss'])
    

    # plot best fitting quantiles for distRL model
    ax1.bar(1, exp1_gain_q_mean, yerr=exp1_gain_q_sem, align='center', alpha=1, 
              ecolor='black', capsize=0, color = cmap[1,:])
    
    ax1.bar(2, exp1_loss_q_mean, yerr=exp1_loss_q_sem, align='center', alpha=1, 
              ecolor='black', capsize=0, color = cmap[5,:])

    ax1.set_xticks([])
    ax1.set_ylim([0,1])
    ax1.set_yticks([0, .5, 1])
    ax1.set_xlim([.5,2.5])
    #ax1.set_xticklabels(['Gain','Loss'])
    ax1.set_ylabel('distRL Quantile')
    
    # predict model quantile from bias
    ax2.scatter(exp1_gain_EQ_bias,exp1_gain_bestparams[:,3],color=cmap[1,:], s =20)
    ax2.scatter(exp1_loss_EQ_bias,exp1_loss_bestparams[:,3],color=cmap[5,:], s =20)
    
    # plot line of best fit
    ax2.plot(np.array([0,1]),np.array([0,1])*exp1_slope + exp1_intercept, 
               color = 'black', linewidth=2)
    
    ax2.set_xlabel('EQ bias')
    ax2.set_ylabel('distRL Quantile')
    ax2.set_xticks(ticks = np.array([]))
    ax2.set_yticks(ticks = np.array([0, .5, 1]))
    ax2.set_xlim([0,1])
    ax2.set_ylim([0,1])

    #-----
    # EXPERIMENT 2
    #-----

    # plot distRL model accuracy 
    ax3.bar(1, exp2_gain_acc_mean, yerr=exp2_gain_acc_sem, align='center', alpha=1, 
              ecolor='black', capsize=0, color = cmap[1,:])
    
    ax3.bar(2, exp2_loss_acc_mean, yerr=exp2_loss_acc_sem, align='center', alpha=1, 
              ecolor='black', capsize=0, color = cmap[5,:])

    ax3.plot([0,3],[.5,.5],color = 'tab:gray', linestyle = '--')
    ax3.set_ylabel('Exp 2 \n Model Accuracy')
    ax3.set_xticks([1,2])
    ax3.set_ylim([0,1])
    ax3.set_yticks([0, .5, 1])
    ax3.set_xlim([.5,2.5])
    ax3.set_xticklabels(['Gain','Loss'])
    

    # plot best fitting quantiles for distRL model
    ax4.bar(1, exp2_gain_q_mean, yerr=exp2_gain_q_sem, align='center', alpha=1, 
              ecolor='black', capsize=0, color = cmap[1,:])
    
    ax4.bar(2, exp2_loss_q_mean, yerr=exp2_loss_q_sem, align='center', alpha=1, 
              ecolor='black', capsize=0, color = cmap[5,:])

    ax4.set_xticks([1,2])
    ax4.set_ylim([0,1])
    ax4.set_yticks([0, .5, 1])
    ax4.set_xlim([.5,2.5])
    ax4.set_xticklabels(['Gain','Loss'])
    ax4.set_ylabel('distRL Quantile')
    
    # predict model quantile from bias
    ax5.scatter(exp2_gain_EQ_bias,exp2_gain_bestparams[:,3],color=cmap[1,:], s =20)
    ax5.scatter(exp2_loss_EQ_bias,exp2_loss_bestparams[:,3],color=cmap[5,:], s =20)
    
    # plot line of best fit
    ax5.plot(np.array([0,1]),np.array([0,1])*exp2_slope + exp2_intercept, 
               color = 'black', linewidth=2)
    
    ax5.set_xlabel('EQ bias')
    ax5.set_ylabel('distRL Quantile')
    ax5.set_xticks(ticks = np.array([0, .5, 1]))
    ax5.set_yticks(ticks = np.array([0, .5, 1]))
    ax5.set_xlim([0,1])
    ax5.set_ylim([0,1])

  
    xx=[] 
# END of relate_distRL_to_EQbias