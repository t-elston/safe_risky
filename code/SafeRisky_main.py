# -*- coding: utf-8 -*-
"""
Top of stack for analysis of safe vs risky human data

@author: Thomas Elston
"""
#%% import functions
import SafeRisky_fxns as sr
import importlib

#%%
importlib.reload(sr)

#%% set data directory

# working from lab
#exp1_dir = 'C:/Users/Thomas Elston/Documents/PYTHON/SafeRisky/exp1_data/'
#exp2_dir = 'C:/Users/Thomas Elston/Documents/PYTHON/SafeRisky/exp2_data/'


# working from home
exp1_dir = '/Users/thomaselston/Documents/PYTHON/SafeRisky/exp1_data/'
exp2_dir = '/Users/thomaselston/Documents/PYTHON/SafeRisky/exp2_data/'



#%% load and wrangle data
exp1_gain_choice , exp1_gain_rt, exp1_gain_all, exp1_p_perf =sr.load_processData(exp1_dir,context='Gain')
exp1_loss_choice , exp1_loss_rt, exp1_loss_all, exp1_p_perf =sr.load_processData(exp1_dir, context='Loss')

exp2_gain_choice , exp2_gain_rt, exp2_gain_all, exp2_p_perf =sr.load_processData(exp2_dir,context='Gain')
exp2_loss_choice , exp2_loss_rt, exp2_loss_all, exp2_p_perf =sr.load_processData(exp2_dir, context='Loss')

#%% do stats and plot basic conditions
exp1_choice_data = sr.collect_data_for_stats(exp1_gain_choice, exp1_loss_choice)
exp2_choice_data = sr.collect_data_for_stats(exp2_gain_choice, exp2_loss_choice)
exp1_rt_data = sr.collect_data_for_stats(exp1_gain_rt, exp1_loss_rt)
exp2_rt_data = sr.collect_data_for_stats(exp2_gain_rt, exp2_loss_rt)

# do the stats for each experiment - you can print e.g. print(exp1_choice_stats['train'])
exp1_choice_stats, exp2_choice_stats = sr.do_stats(exp1_choice_data, exp2_choice_data)
exp1_rt_stats, exp2_rt_stats = sr.do_stats(exp1_rt_data, exp2_rt_data)

# plot results of BOTH EXPERIMENTS (look at ^^ for stats)
sr.plot_both_experiments_perf(exp1_gain_choice, exp1_loss_choice,
                              exp2_gain_choice, exp2_loss_choice,
                              datatype = 'choice')

sr.plot_both_experiments_perf(exp1_gain_rt, exp1_loss_rt,
                              exp2_gain_rt, exp2_loss_rt,
                              datatype = 'rt')

#%% do win-stay analysis for data from both contexts
gain_winstay = sr.win_stay_analysis(gain_all)
loss_winstay = sr.win_stay_analysis(loss_all)

# do lose-stay analysis (returns NaNs for safe options)
gain_losestay = sr.lose_stay_analysis(gain_all)
loss_losestay = sr.lose_stay_analysis(loss_all)

# plot/assess the win-stay analysis
sr.plotWinStay_LoseStay(gain_winstay,loss_winstay,gain_losestay,loss_losestay)


#%% do some distributional RL modelling
exp1_gain_bestparams, exp1_gain_bestAccOpt, exp1_gain_Qtbl = sr.distRLmodel_MLE(exp1_gain_all)
exp1_loss_bestparams, exp1_loss_bestAccOpt, exp1_loss_Qtbl = sr.distRLmodel_MLE(exp1_loss_all)
exp2_gain_bestparams, exp2_gain_bestAccOpt, exp1_gain_Qtbl = sr.distRLmodel_MLE(exp2_gain_all)
exp2_loss_bestparams, exp2_loss_bestAccOpt, exp1_loss_Qtbl = sr.distRLmodel_MLE(exp2_loss_all)

# plot and quantify distRL parameters
sr.both_exp_distRLxEQbias(exp1_gain_bestparams, exp1_loss_bestparams,
                           exp1_gain_choice, exp1_loss_choice,
                           exp1_gain_bestAccOpt,exp1_loss_bestAccOpt,
                           exp2_gain_bestparams, exp2_loss_bestparams,
                           exp2_gain_choice, exp2_loss_choice,
                           exp2_gain_bestAccOpt,exp2_loss_bestAccOpt)
