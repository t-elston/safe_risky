"""
Top of stack for analysis of safe vs risky human data

@author: Thomas Elston
"""
# %% import functions
import SafeRisky_fxns2 as sr
import importlib
import pandas as pd
import numpy as np

# %%
importlib.reload(sr)

# %% set data directories
# working from lab
# datadir = 'C:/Users/Thomas Elston/Documents/PYTHON/SafeRisky/data/'

# working from home
datadir = '/Users/thomaselston/Documents/PYTHON/SafeRisky/data/'

# %% load and wrangle data
gain_choice, gain_rt, gain_all = sr.load_process_data(datadir, context='Gain')
loss_choice, loss_rt, loss_all = sr.load_process_data(datadir, context='Loss')

# %% plots + stats for main conditions of each experiment
sr.plot_choice_or_rt(gain_choice, loss_choice, data_type='choice')
sr.plot_choice_or_rt(gain_rt, loss_rt, data_type='rt')

# do t-tests against chance on conditions with a best option
gain_ttests, loss_ttests = sr.assess_conds_with_best_choice(gain_choice, loss_choice)

# do rm_anovas for each condition in each experiment
# see anova tables with e.g. print(exp1_rt_stats['train'])
exp1_choice_stats, exp2_choice_stats, exp3_choice_stats = sr.run_rmANOVAs(gain_choice, loss_choice)
exp1_rt_stats, exp2_rt_stats, exp3_rt_stats = sr.run_rmANOVAs(gain_rt, loss_rt)


# %% do win-stay analysis for data from both contexts
winstay_long, winstay_wide = sr.win_stay_analysis(gain_all, loss_all)

sr.plot_assess_win_stay(winstay_long, winstay_wide, gain_choice, loss_choice)




# %% do some distributional RL modelling

sr.plot_individual_subjectEQbiases(exp1_gain_choice, exp1_loss_choice,
                                   exp2_gain_choice, exp2_loss_choice)

exp1_gain_bestparams, exp1_gain_bestAccOpt, exp1_gain_Qtbl = sr.distRLmodel_MLE(exp1_gain_all)
exp1_loss_bestparams, exp1_loss_bestAccOpt, exp1_loss_Qtbl = sr.distRLmodel_MLE(exp1_loss_all)
exp2_gain_bestparams, exp2_gain_bestAccOpt, exp1_gain_Qtbl = sr.distRLmodel_MLE(exp2_gain_all)
exp2_loss_bestparams, exp2_loss_bestAccOpt, exp1_loss_Qtbl = sr.distRLmodel_MLE(exp2_loss_all)

# plot and quantify distRL parameters
sr.both_exp_distRLxEQbias(exp1_gain_bestparams, exp1_loss_bestparams,
                          exp1_gain_choice, exp1_loss_choice,
                          exp1_gain_bestAccOpt, exp1_loss_bestAccOpt,
                          exp2_gain_bestparams, exp2_loss_bestparams,
                          exp2_gain_choice, exp2_loss_choice,
                          exp2_gain_bestAccOpt, exp2_loss_bestAccOpt)

# %%
