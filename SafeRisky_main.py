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
#datadir = 'C:/Users/Thomas Elston/Documents/PYTHON/SafeRisky/data/'

# working from home
datadir = '/Users/thomaselston/Documents/PYTHON/SafeRisky/exp2_data/'


# load and process data
gain_choice , gain_rt, gain_all, p_perf =sr.load_processData(datadir,context='Gain')

loss_choice , loss_rt, loss_all, p_perf =sr.load_processData(datadir, context='Loss')

#%%
#  plot mean participant performance, show excluded subjects
#sr.show_excluded_subjects(p_perf)

# plot high-level summary of all conditions 
sr.plot_mean_perf(gain_choice, loss_choice, datatype='choice')
sr.plot_mean_perf(gain_rt, loss_rt, datatype='rt')


# (DEPRECATED) break out choice and rt data by individual condition 
#sr.plotChoice_or_RT(gain_choice,loss_choice,datatype='choice')
#sr.plotChoice_or_RT(gain_rt,loss_rt,datatype='rt')

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
gain_bestparams, gain_bestAccOpt, gain_Qtbl = sr.distRLmodel_MLE(gain_all)
loss_bestparams, loss_bestAccOpt, loss_Qtbl = sr.distRLmodel_MLE(loss_all)

# plot and quantify distRL parameters
sr.relate_distRL_to_EQbias(gain_bestparams, loss_bestparams,
                           gain_choice, loss_choice,
                           gain_bestAccOpt,loss_bestAccOpt)

#%%
# check how final learned values map onto choice biases
sr.q_learner_choice_bias(gain_bestparams, gain_Qtbl,
                         loss_bestparams, loss_Qtbl)

# %%
