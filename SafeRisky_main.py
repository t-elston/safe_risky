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
gain_choice , gain_rt, gain_all  = sr.load_processData(datadir,context = 'Gain',debug_ = False)
loss_choice , loss_rt, loss_all  = sr.load_processData(datadir,context = 'Loss',debug_ = False)


# plot and analyze choice, rt, and survey data
sr.plotChoice_or_RT(gain_choice,loss_choice,datatype='choice',debug_ = True)
sr.plotChoice_or_RT(gain_rt,loss_rt,datatype='rt',debug_ = False)


# do win-stay analysis for data from both contexts
gain_winstay = sr.win_stay_analysis(gain_all, debug_=False)
loss_winstay = sr.win_stay_analysis(loss_all, debug_=False)

# do lose-stay analysis (returns NaNs for safe options)
gain_losestay = sr.lose_stay_analysis(gain_all, debug_=False)
loss_losestay = sr.lose_stay_analysis(loss_all, debug_=False)


# plot/assess the win-stay analysis
sr.plotWinStay_LoseStay(gain_winstay,loss_winstay,gain_losestay,loss_losestay, debug_=False)


# run the distRL model
gain_bestparams, gain_bestAccOpt = sr.distRLmodel_MLE(gain_all, debug_=False)
loss_bestparams, loss_bestAccOpt = sr.distRLmodel_MLE(loss_all, debug_=False)

# TODO: make function to plot and quantify distRL parameters
