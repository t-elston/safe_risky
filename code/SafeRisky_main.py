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

#%% set data directories
# working from lab
exp1_dir = 'C:/Users/Thomas Elston/Documents/PYTHON/SafeRisky/exp1_data/'
exp2_dir = 'C:/Users/Thomas Elston/Documents/PYTHON/SafeRisky/exp3_data/'

# working from home
#exp1_dir = '/Users/thomaselston/Documents/PYTHON/SafeRisky/exp1_data/'
#exp2_dir = '/Users/thomaselston/Documents/PYTHON/SafeRisky/exp2_data/'


#%% load and wrangle data
exp1_gain_choice , exp1_gain_rt, exp1_gain_all, exp1_p_perf =sr.load_processData(exp1_dir,context='Gain')
exp1_loss_choice , exp1_loss_rt, exp1_loss_all, exp1_p_perf =sr.load_processData(exp1_dir,context='Loss')

exp2_gain_choice , exp2_gain_rt, exp2_gain_all, exp2_p_perf =sr.load_processData(exp2_dir,context='Gain')
exp2_loss_choice , exp2_loss_rt, exp2_loss_all, exp2_p_perf =sr.load_processData(exp2_dir,context='Loss')

#%% do stats and plot basic conditions
# do t-tests on conditions with a best option
exp1_gain_ttests, exp1_loss_ttests = sr.assess_conds_with_best_choice(exp1_gain_choice, exp1_loss_choice)
exp2_gain_ttests, exp2_loss_ttests = sr.assess_conds_with_best_choice(exp2_gain_choice, exp2_loss_choice)


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

sr.compare_both_experiments_risk_preference(exp1_gain_choice, exp1_loss_choice,
                                            exp2_gain_choice, exp2_loss_choice)

sr.plot_individual_subjectEQbiases(exp1_gain_choice, exp1_loss_choice,
                                    exp2_gain_choice, exp2_loss_choice)

#%% do win-stay analysis for data from both contexts
exp1_gain_winstay = sr.win_stay_analysis(exp1_gain_all)
exp1_loss_winstay = sr.win_stay_analysis(exp1_loss_all)
exp2_gain_winstay = sr.win_stay_analysis(exp2_gain_all)
exp2_loss_winstay = sr.win_stay_analysis(exp2_loss_all)


# plot/assess the win-stay analysis
winstay_lme_results  = sr.plotWinStay(exp1_gain_winstay,exp1_loss_winstay,exp2_gain_winstay,exp2_loss_winstay,
                                      exp1_gain_choice, exp1_loss_choice, exp2_gain_choice, exp2_loss_choice)
# examine these results^^ with e.g.:
# print(winstay_lme_results['combined'])
# see the docstring in the function for more info




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

# %%
