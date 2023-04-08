"""
Top of stack for analysis of safe vs risky human data

@author: Thomas Elston
"""
# %% import functions
import SafeRisky_fxns2 as sr
import importlib

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

# assess cross-context differences in risk preference
prob_x_context_glms = sr.eq_bias_by_prob_posthoc(gain_choice, loss_choice)

# assess roles of information format and expected value across experiments
bias_mdl = sr.assess_prob_expval_infoformat_regression(gain_choice, loss_choice)

# do rm_anovas for each condition in each experiment (print(exp1_rt_stats['train'])
exp1_choice_stats, exp2_choice_stats, exp3_choice_stats = sr.run_rmANOVAs(gain_choice, loss_choice)
exp1_rt_stats, exp2_rt_stats, exp3_rt_stats = sr.run_rmANOVAs(gain_rt, loss_rt)


# %% do win-stay analysis for data from both contexts
winstay_long, winstay_wide = sr.win_stay_analysis(gain_all, loss_all)

hit_stay_GLMs = sr.plot_assess_hit_stay(winstay_long, winstay_wide, gain_choice, loss_choice)


# %% do some risk-sensitive RL modelling
gain_fits, loss_fits = sr.risk_sensitive_RL(gain_all, loss_all)

expectile_models, match_models = sr.plot_RSRL_results(gain_fits, loss_fits)
