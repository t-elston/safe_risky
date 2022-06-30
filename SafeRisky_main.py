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

#%% point of entry

# working from lab
exp1_dir = 'C:/Users/Thomas Elston/Documents/PYTHON/SafeRisky/exp1_data/'
exp2_dir = 'C:/Users/Thomas Elston/Documents/PYTHON/SafeRisky/exp2_data/'


# working from home
#datadir = '/Users/thomaselston/Documents/PYTHON/SafeRisky/exp2_data/'


# load and process data
exp1_gain_choice , exp1_gain_rt, exp1_gain_all, exp1_p_perf =sr.load_processData(exp1_dir,context='Gain')
exp1_loss_choice , exp1_loss_rt, exp1_loss_all, exp1_p_perf =sr.load_processData(exp1_dir, context='Loss')

exp2_gain_choice , exp2_gain_rt, exp2_gain_all, exp2_p_perf =sr.load_processData(exp2_dir,context='Gain')
exp2_loss_choice , exp2_loss_rt, exp2_loss_all, exp2_p_perf =sr.load_processData(exp2_dir, context='Loss')

#%%
#  plot mean participant performance, show excluded subjects
#sr.show_excluded_subjects(p_perf)

# get stats and plot summary for ONE EXPERIMENT AT A TIME
# TODO write stats to dict in sr.plot_mean_perf()
sr.plot_mean_perf(exp1_gain_choice, exp1_loss_choice, datatype='choice')
sr.plot_mean_perf(exp1_gain_rt, exp1_loss_rt, datatype='rt')

sr.plot_mean_perf(exp2_gain_choice, exp2_loss_choice, datatype='choice')
sr.plot_mean_perf(exp2_gain_rt, exp2_loss_rt, datatype='rt')

# plot results of BOTH EXPERIMENTS (look at ^^ for stats)
sr.plot_both_experiments_perf(exp1_gain_choice, exp1_loss_choice,
                              exp2_gain_choice, exp2_loss_choice,
                              datatype = 'choice')

sr.plot_both_experiments_perf(exp1_gain_rt, exp1_loss_rt,
                              exp2_gain_rt, exp2_loss_rt,
                              datatype = 'rt')

#%%
# do win-stay analysis for data from both contexts
gain_winstay = sr.win_stay_analysis(gain_all)
loss_winstay = sr.win_stay_analysis(loss_all)

# do lose-stay analysis (returns NaNs for safe options)
gain_losestay = sr.lose_stay_analysis(gain_all)
loss_losestay = sr.lose_stay_analysis(loss_all)


# plot/assess the win-stay analysis
sr.plotWinStay_LoseStay(gain_winstay,loss_winstay,gain_losestay,loss_losestay)


#%%
# run the distRL model
exp1_gain_bestparams, exp1_gain_bestAccOpt, exp1_gain_Qtbl = sr.distRLmodel_MLE(exp1_gain_all)
exp1_loss_bestparams, exp1_loss_bestAccOpt, exp1_loss_Qtbl = sr.distRLmodel_MLE(exp1_loss_all)
exp2_gain_bestparams, exp2_gain_bestAccOpt, exp1_gain_Qtbl = sr.distRLmodel_MLE(exp2_gain_all)
exp2_loss_bestparams, exp2_loss_bestAccOpt, exp1_loss_Qtbl = sr.distRLmodel_MLE(exp2_loss_all)

# plot and quantify distRL parameters
sr.relate_distRL_to_EQbias(gain_bestparams, loss_bestparams,
                           gain_choice, loss_choice,
                           gain_bestAccOpt,loss_bestAccOpt)

#%%
# check how final learned values map onto choice biases
sr.q_learner_choice_bias(gain_bestparams, gain_Qtbl,
                         loss_bestparams, loss_Qtbl)

# %%
