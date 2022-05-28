# -*- coding: utf-8 -*-
"""
Top of stack for analysis of safe vs risky human data

@author: Thomas Elston
"""
#%% import functions
import SafeRisky_fxns as sr

#%% point of entry

# working from lab
#datadir = 'C:/Users/Thomas Elston/Documents/PYTHON/SafeRisky/data/'

# working from home
datadir = '/Users/thomaselston/Documents/PYTHON/SafeRisky/data/'


# load and process data
gain_choice , gain_rt, gain_all, p_perf =sr.load_processData(datadir, 
                                                             context='Gain',
                                                             debug_=False)

loss_choice , loss_rt, loss_all, p_perf =sr.load_processData(datadir, 
                                                             context='Loss',
                                                             debug_=False)

#%%
#  plot mean participant performance, show excluded subjects
sr.show_excluded_subjects(p_perf, debug_=False)

# plot high-level summary of all conditions 
sr.plot_mean_perf(gain_choice, loss_choice, datatype='choice', debug_=False)
sr.plot_mean_perf(gain_rt, loss_rt, datatype='rt', debug_=False)


# (DEPRECATED) break out choice and rt data by individual condition 
#sr.plotChoice_or_RT(gain_choice,loss_choice,datatype='choice',debug_ = False)
#sr.plotChoice_or_RT(gain_rt,loss_rt,datatype='rt',debug_ = False)

#%%
# do win-stay analysis for data from both contexts
gain_winstay = sr.win_stay_analysis(gain_all, debug_=False)
loss_winstay = sr.win_stay_analysis(loss_all, debug_=False)

# do lose-stay analysis (returns NaNs for safe options)
gain_losestay = sr.lose_stay_analysis(gain_all, debug_=False)
loss_losestay = sr.lose_stay_analysis(loss_all, debug_=False)


# plot/assess the win-stay analysis
sr.plotWinStay_LoseStay(gain_winstay,loss_winstay,gain_losestay,loss_losestay, debug_=False)


#%%
# run the distRL model
gain_bestparams, gain_bestAccOpt, gain_Qtbl = sr.distRLmodel_MLE(gain_all, debug_=False)
loss_bestparams, loss_bestAccOpt, loss_Qtbl = sr.distRLmodel_MLE(loss_all, debug_=False)

# plot and quantify distRL parameters
sr.relate_distRL_to_EQbias(gain_bestparams, loss_bestparams,
                           gain_choice, loss_choice,
                           gain_bestAccOpt,loss_bestAccOpt,
                           debug_=False)

#%%
# check how final learned values map onto choice biases
sr.q_learner_choice_bias(gain_bestparams, gain_Qtbl,
                         loss_bestparams, loss_Qtbl, debug_ = False)
