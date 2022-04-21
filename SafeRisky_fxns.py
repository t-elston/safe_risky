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
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
import pingouin as pg
import itertools as it
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import math



def load_processData(datadir,context,debug_):
    
    # check if we want to debug
    if debug_:
        pdb.set_trace() 
    
        
    # get files names with path
    fnames = [os.path.join(datadir, _) for _ in os.listdir(datadir) if _.endswith('.csv')]
    
    # initialize a dataframe for ALL of the data
    alldata = pd.DataFrame()
    
    # initialize a dataframe for each participant's mean choice behavior
    pChoicedata = pd.DataFrame()
   
    # dataframe for RTs
    pRTdata = pd.DataFrame()
    
    # dataframe for subject rejection
    p_reject = pd.DataFrame()

    ctr = 0
    
    print('assessing file#:', end=' ')
    # load a csv file and assess each one
    
    for i in range(len(fnames)):
                
        print(str(i),end=' ')
        
        df = pd.read_csv(fnames[i],header=[0])
        
        # get overall training performance so we can reject subjects
        all_train_ix = df.phase == "training"
        all_safe_ix  = (df.imgLeftType == 'Safe') & (df.imgRightType == 'Safe')
        all_risky_ix = (df.imgLeftType == 'Risky') & (df.imgRightType == 'Risky')
        gain_ix = df.blkType == 'Gain'
        loss_ix = df.blkType == 'Loss'
        
        # define the best options for the two contexts (gain/loss)
        all_picked_best = np.zeros(len(gain_ix))
        
        if context == 'Loss':
            all_picked_best = df.highProbSelected ==0
        else:
            all_picked_best = df.highProbSelected ==1
            
        all_picked_best = all_picked_best.astype(int)
        
        crit = 0.05
        
        
        gain_safe_stats = pg.ttest(all_picked_best[all_train_ix & gain_ix & all_safe_ix],.5)
        gain_safe_p = gain_safe_stats['p-val'][0]
        gain_risky_stats = pg.ttest(all_picked_best[all_train_ix & gain_ix & all_risky_ix],.5)
        gain_risky_p = gain_risky_stats['p-val'][0]
        loss_safe_stats = pg.ttest(all_picked_best[all_train_ix & loss_ix & all_safe_ix],.5)
        loss_safe_p = loss_safe_stats['p-val'][0]
        loss_risky_stats = pg.ttest(all_picked_best[all_train_ix & loss_ix & all_risky_ix],.5)
        loss_risky_p = loss_risky_stats['p-val'][0]
        
        if (gain_safe_p < crit) & (gain_risky_p < crit) & (loss_safe_p < crit) & (loss_risky_p < crit):
          
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
            picked_risky = np.empty(len(EQ))
            picked_risky[(df.responseSide == 'left') & (df.imgLeftType == 'Risky')] = 1
            picked_risky[(df.responseSide == 'right') & (df.imgRightType == 'Risky')] = 1
            
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
            

        
            #-----------------------------
            #    summarize each subject
            #-----------------------------   
        
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
            

                        

            # get subject idnum, version, sex, and age
            pChoicedata.at[ctr,'vpnum']   = df.vpNum[0]
            pChoicedata.at[ctr,'version'] = df.version[0]
            pChoicedata.at[ctr,'context'] = context
            pChoicedata.at[ctr,'age']     = df.age[0]
            pChoicedata.at[ctr,'sex']     = df.gender[0]
            
            # look at training choice data
            pChoicedata.at[ctr,'t_s_20v50'] = picked_best[trainix & t20v50 & safe_ix].mean()
            pChoicedata.at[ctr,'t_s_50v80'] = picked_best[trainix & t50v80 & safe_ix].mean()
            pChoicedata.at[ctr,'t_s_20v80'] = picked_best[trainix & t20v80 & safe_ix].mean()
            
            pChoicedata.at[ctr,'t_r_20v50'] = picked_best[trainix & t20v50 & risky_ix].mean()
            pChoicedata.at[ctr,'t_r_50v80'] = picked_best[trainix & t50v80 & risky_ix].mean()
            pChoicedata.at[ctr,'t_r_20v80'] = picked_best[trainix & t20v80 & risky_ix].mean()
    
            # main block pure trials
            pChoicedata.at[ctr,'s_20v50'] = picked_best[testix & t20v50 & safe_ix].mean()
            pChoicedata.at[ctr,'s_50v80'] = picked_best[testix & t50v80 & safe_ix].mean()
            pChoicedata.at[ctr,'s_20v80'] = picked_best[testix & t20v80 & safe_ix].mean()
            
            pChoicedata.at[ctr,'r_20v50'] = picked_best[testix & t20v50 & risky_ix].mean()
            pChoicedata.at[ctr,'r_50v80'] = picked_best[testix & t50v80 & risky_ix].mean()
            pChoicedata.at[ctr,'r_20v80'] = picked_best[testix & t20v80 & risky_ix].mean()
            
            # main block unequal trials
            # safe is better
            pChoicedata.at[ctr,'UE_s_20v50'] = picked_best[UE & t20v50 & riskybest].mean()
            pChoicedata.at[ctr,'UE_s_50v80'] = picked_best[UE & t50v80 & riskybest].mean()
            pChoicedata.at[ctr,'UE_s_20v80'] = picked_best[UE & t20v80 & riskybest].mean()
            
            # risky is better
            pChoicedata.at[ctr,'UE_r_20v50'] = picked_best[UE & t20v50 & safebest].mean()
            pChoicedata.at[ctr,'UE_r_50v80'] = picked_best[UE & t50v80 & safebest].mean()
            pChoicedata.at[ctr,'UE_r_20v80'] = picked_best[UE & t20v80 & safebest].mean()
            
            # main block equivaluable trials
            pChoicedata.at[ctr,'EQ20'] = picked_risky[t20v20].mean()
            pChoicedata.at[ctr,'EQ50'] = picked_risky[t50v50].mean()
            pChoicedata.at[ctr,'EQ80'] = picked_risky[t80v80].mean()
            
            
            # do the same but with RTs
            pRTdata.at[ctr,'vpnum']   = df.vpNum[0]
            pRTdata.at[ctr,'version'] = df.version[0]
            pRTdata.at[ctr,'context'] = context
            pRTdata.at[ctr,'age']     = df.age[0]
            pRTdata.at[ctr,'sex']     = df.gender[0]
            
            # look at training choice data
            pRTdata.at[ctr,'t_s_20v50'] = rt[trainix & t20v50 & safe_ix].mean()
            pRTdata.at[ctr,'t_s_50v80'] = rt[trainix & t50v80 & safe_ix].mean()
            pRTdata.at[ctr,'t_s_20v80'] = rt[trainix & t20v80 & safe_ix].mean()
            
            pRTdata.at[ctr,'t_r_20v50'] = rt[trainix & t20v50 & risky_ix].mean()
            pRTdata.at[ctr,'t_r_50v80'] = rt[trainix & t50v80 & risky_ix].mean()
            pRTdata.at[ctr,'t_r_20v80'] = rt[trainix & t20v80 & risky_ix].mean()
    
            # main block pure trials
            pRTdata.at[ctr,'s_20v50'] = rt[testix & t20v50 & safe_ix].mean()
            pRTdata.at[ctr,'s_50v80'] = rt[testix & t50v80 & safe_ix].mean()
            pRTdata.at[ctr,'s_20v80'] = rt[testix & t20v80 & safe_ix].mean()
            
            pRTdata.at[ctr,'r_20v50'] = rt[testix & t20v50 & risky_ix].mean()
            pRTdata.at[ctr,'r_50v80'] = rt[testix & t50v80 & risky_ix].mean()
            pRTdata.at[ctr,'r_20v80'] = rt[testix & t20v80 & risky_ix].mean()
            
            # main block unequal trials
            # narrow is better
            pRTdata.at[ctr,'UE_s_20v50'] = rt[UE & t20v50 & safebest].mean()
            pRTdata.at[ctr,'UE_s_50v80'] = rt[UE & t50v80 & safebest].mean()
            pRTdata.at[ctr,'UE_s_20v80'] = rt[UE & t20v80 & safebest].mean()
            
            pRTdata.at[ctr,'UE_r_20v50'] = rt[UE & t20v50 & riskybest].mean()
            pRTdata.at[ctr,'UE_r_50v80'] = rt[UE & t50v80 & riskybest].mean()
            pRTdata.at[ctr,'UE_r_20v80'] = rt[UE & t20v80 & riskybest].mean()
            
            # main block equivaluable trials
            pRTdata.at[ctr,'EQ20'] = rt[t20v20].mean()
            pRTdata.at[ctr,'EQ50'] = rt[t50v50].mean()
            pRTdata.at[ctr,'EQ80'] = rt[t80v80].mean()
            
            ctr = ctr+1
            
            # add all data to the aggregate dataframe
            alldata = alldata.append(df)



        
        
                
    xx=[]
    return pChoicedata , pRTdata, alldata
# END of load_processData 




def plotChoice_or_RT(gaindata,lossdata,datatype,debug_):
    '''
    This function plots and statistically assesses either the choice or RT
    data (as specified in the datatype argument). To see the stats, set
    debug_ = True and set a breakpoint at the end of the function.
    '''
    
    # check if we want to debug
    if debug_:
        pdb.set_trace() 
    
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


def win_stay_analysis(alldata, debug_):
    '''
    This analysis looks for instances where a certain option was chosen 
    and yielded a non-zero outcome and asks how likely the person is to select
    that option the next time it's presented (i.e. p(Win-Stay)).
    '''
    
    # check if we want to debug
    if debug_:
        pdb.set_trace() 

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



def lose_stay_analysis(alldata, debug_):
    '''
    This analysis looks for instances where a certain option was chosen 
    and yielded a zero outcome and asks how likely the person is to select
    that option the next time it's presented (i.e. p(Lose-Stay)).
    '''
    
    # check if we want to debug
    if debug_:
        pdb.set_trace() 

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




def plotWinStay_LoseStay(gain_winstay,loss_winstay,gain_losestay,loss_losestay,debug_):
    
    # check if we want to debug
    if debug_:
        pdb.set_trace() 
        
    
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
    plt.errorbar(xvals,gain_WS_mean[0:3],gain_WS_sem[0:3],label='gain, safe',color=cmap[1,:])
    plt.errorbar(xvals,gain_WS_mean[3:7],gain_WS_sem[3:7],label='gain, risky',color=cmap[5,:])
    plt.xlabel('p(Gain)')
    plt.ylabel('p(Hit-Stay)')
    plt.xticks(xvals)
    plt.ylim([0,1])
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.errorbar(xvals,loss_WS_mean[0:3],loss_WS_sem[0:3],label='loss, safe',color=cmap[0,:])
    plt.errorbar(xvals,loss_WS_mean[3:7],loss_WS_sem[3:7],label='loss, risky',color=cmap[4,:])
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




def distRLmodel_MLE(alldata,debug_):
    
    # check if we want to debug
    if debug_:
        pdb.set_trace() 
     
    alphavals = np.linspace(.1,1,int(1/.1))
    betas = np.linspace(1,40,40)
    betas = np.array([1]) # this is for debugging
    
    # get the combinations of alphas and betas
    #1st col = alphaplus, 2nd = alphaminus, 3rd = beta
    Qparams = np.array(np.meshgrid(alphavals, alphavals,betas)).T.reshape(-1,3)
    
    # get a subject's data
    subjIDs = alldata['vpNum'].unique()
    
    # initiialize output
    bestparams = np.zeros(shape = (len(subjIDs),int(Qparams.shape[1])+1))
    bestAccOpt = np.zeros(shape = (len(subjIDs),2))
    
    # iterate through each subject's data
    for s in range(len(subjIDs)):
 
        # pull out this subject's data        
        sdata = alldata.loc[alldata.vpNum == subjIDs[s],:]

        # accuracies for each parameter set
        paramAccs = np.zeros(shape = len(Qparams))
        paramOpt  = np.zeros(shape = len(Qparams))
        
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

                
            print('Subject#: '+ str(s) + ', Param#: ' + str(a))
                  
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
                softmax = (np.exp(beta*Qtbl[stim[t,0]])) / (np.exp(beta*Qtbl[stim[t,0]]) + np.exp(beta*Qtbl[stim[t,1]]) )
                
                # get likelihood that Qlearner picks same as human
                fit_likelihood = (np.exp(beta*Qtbl[humanchoice[t]])) / (np.exp(beta*Qtbl[stim[t,0]]) + np.exp(beta*Qtbl[stim[t,1]]) )

                
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
            
             
            xx=[]             
        # END of looping over parameters 
        
        # assess which parameter set was best 
        bestix = paramAccs.argmax()
        bestix = paramLL.argmax()


        bestparams[s,0:int(Qparams.shape[1])]=Qparams[bestix,:]
        bestparams[s,int(Qparams.shape[1])]  =Qparams[bestix,0] / (Qparams[bestix,0:2].sum())
        
        bestAccOpt[s,0]  = paramAccs[bestix]
        bestAccOpt[s,1]  = paramOpt[bestix]
        
        """
        paramgrid = paramLL.reshape(10,10)
        plt.imshow(paramgrid)
        plt.colorbar()
        
        optgrid = paramOpt.reshape(10,10)
        plt.imshow(optgrid)
        plt.colorbar()
        """

    # END of looping through subjects  

    return bestparams, bestAccOpt           
# END of distRLmodel



def relate_distRL_to_EQbias(gain_bestparams, loss_bestparams,
                           gain_choice, loss_choice,
                           gain_bestAccOpt,loss_bestAccOpt,
                           debug_):

    # check if we want to debug
    if debug_:
        pdb.set_trace()     


    # define colormap for plotting 
    cmap = plt.cm.Paired(np.linspace(0, 1, 12))
        
    gain_EQ_bias = np.array([gain_choice.iloc[:,23], 
                             gain_choice.iloc[:,24], 
                             gain_choice.iloc[:,25] ]).mean(axis=0)
    
    loss_EQ_bias = np.array([loss_choice.iloc[:,23], 
                             loss_choice.iloc[:,24], 
                             loss_choice.iloc[:,25] ]).mean(axis=0)
  
    
    # fit a linear mixed effect model
    lmedata = pd.DataFrame()
    lmedata['EQbias'] = np.concatenate([gain_EQ_bias,loss_EQ_bias])
    lmedata['context'] = np.concatenate([np.ones(len(gain_EQ_bias)),np.ones(len(gain_EQ_bias))*-1])
    lmedata['RLparam'] = np.concatenate([gain_bestparams[:,3],loss_bestparams[:,3]])
    lmedata['subject'] = np.concatenate([np.arange(len(gain_EQ_bias)),np.arange(len(gain_EQ_bias))])
    
    # fit the model
    RLxbias_mdl = smf.mixedlm('RLparam ~ context*EQbias', lmedata, groups=lmedata['subject']).fit()
    
    # extract intercept and beta for distRL quantile
    params = RLxbias_mdl.params
    intercept = params['Intercept']
    slope     = params['EQbias']
    
    
    # make figure and get plot
    fig, ax = plt.subplots(1,3,figsize=(6, 2), dpi=300)
    fig.tight_layout(h_pad=4)
    
    # plot model accuracy
    gain_acc_mean = gain_bestAccOpt[:,0].mean()
    gain_acc_sem = gain_bestAccOpt[:,0].std()/np.sqrt(len(gain_bestAccOpt[:,0]))
    
    loss_acc_mean = loss_bestAccOpt[:,0].mean()
    loss_acc_sem = loss_bestAccOpt[:,0].std()/np.sqrt(len(loss_bestAccOpt[:,0]))

    
    ax[0].bar(1, gain_acc_mean, yerr=gain_acc_sem, align='center', alpha=1, 
              ecolor='black', capsize=0, color = cmap[9,:])
    
    ax[0].bar(2, loss_acc_mean, yerr=loss_acc_sem, align='center', alpha=1, 
              ecolor='black', capsize=0, color = cmap[8,:])

    ax[0].plot([0,3],[.5,.5],color = 'tab:gray', linestyle = '--')
    ax[0].set_ylabel('best model acc')
    ax[0].set_xticks([1,2])
    ax[0].set_ylim([.25,1])
    ax[0].set_xlim([.5,2.5])
    ax[0].set_xticklabels(['gain','loss'])
    
    # plot best quantiles
    gain_q_mean = gain_bestparams[:,3].mean()
    gain_q_sem = gain_bestparams[:,3].std()/np.sqrt(len(gain_bestparams[:,3]))
    
    loss_q_mean = loss_bestparams[:,3].mean()
    loss_q_sem = loss_bestparams[:,3].std()/np.sqrt(len(loss_bestparams[:,3]))

    
    ax[1].bar(1, gain_q_mean, yerr=gain_q_sem, align='center', alpha=1, 
              ecolor='black', capsize=0, color = cmap[9,:])
    
    ax[1].bar(2, loss_q_mean, yerr=loss_q_sem, align='center', alpha=1, 
              ecolor='black', capsize=0, color = cmap[8,:])


    ax[1].plot([0,3],[.5,.5],color = 'tab:gray', linestyle='--')

    ax[1].set_xticks([1,2])
    ax[1].set_ylim([0,1])
    ax[1].set_xlim([.5,2.5])
    ax[1].set_xticklabels(['gain','loss'])
    ax[1].set_ylabel('best RL quantile')
    
    
    # predict model quantile from bias
    ax[2].scatter(gain_EQ_bias,gain_bestparams[:,3],color=cmap[9,:])
    ax[2].scatter(loss_EQ_bias,loss_bestparams[:,3],color=cmap[8,:])
    
    # plot line of best fit
    ax[2].plot(np.array([0,1]),np.array([0,1])*slope + intercept, 
               color = 'tab:gray', linestyle = '--')
    
    
    ax[2].set_xlabel('EQ bias')
    ax[2].set_ylabel('dist RL quantile')
    ax[2].set_xticks(ticks = np.array([0, .5, 1]))
    ax[2].set_yticks(ticks = np.array([0, .5, 1]))
    ax[2].set_xlim([0,1])
    ax[2].set_ylim([0,1])

    
    
    
    
    
    xx=[] 


# END of relate_distRL_to_EQbias






