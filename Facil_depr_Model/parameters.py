# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:40:58 2015

@author: alexandre
"""
from __future__ import division
import collections as coll
import numpy as np
import cPickle as pickle
from copy import deepcopy
import random

import pandas as pd

import scipy.stats
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import PyDSTool as dst

# -------------------------------------------------------------------- #
# ------------------------- Simulation parameters --------------------------- #
# -------------------------------------------------------------------- #

t_start = 0.0 # ms | Start of the simulation
t_end = 1000.0 # ms | End of the simulation

# -------------------------------------------------------------------- #
# ---------------- Inputs ------------------- #
# -------------------------------------------------------------------- #

# ------------ TEST 1 ----------- #

init_step = 0.010 # ms
t_stim_start = 30.9 # ms
max_step = 0.10 # ms | Maximal step-size

t_stim_dur = 10.0 # ms
freq_stim = 100.0 # Hz
ampl_stim = 1.0 # Dirac amplitude

# ------------ TEST 2 ----------- #

init_step = 0.010 # ms
t_stim_start = 40.9 # ms
max_step = 0.10 # ms | Maximal step-size

t_stim_dur = 10.0 # ms
freq_stim = 100.0 # Hz
ampl_stim = 1.0 # Dirac amplitude

# ------------ TEST 3 ----------- #

init_step = 0.010 # ms
t_stim_start = 30.9 # ms
max_step = 0.10 # ms | Maximal step-size

t_stim_dur = 100.0 # ms
freq_stim = 100.0 # Hz
ampl_stim = 1.0 # Dirac amplitude

# ------------ TEST 4 ----------- #

init_step = 0.010 # ms
t_stim_start = 40.9 # ms
max_step = 0.10 # ms | Maximal step-size

t_stim_dur = 100.0 # ms
freq_stim = 100.0 # Hz
ampl_stim = 1.0 # Dirac amplitude

# ---------------------------------------- #

ms_to_sec = 1000.0 # Factor

nb_stim = int(t_stim_dur/ms_to_sec *freq_stim) # Number of stimulation

nb_decim = len(str(init_step))-2 # Number of decimal to use with round

time_array_inputs = np.arange(t_start, t_end, init_step) # Time array for inputs
time_array_inputs = np.round_(time_array_inputs, nb_decim) # Round array to nb_decim
len_time_array_inputs = len(time_array_inputs) # Length time serie

f_stim_serie_facil_depr_model = pd.Series(np.zeros(len_time_array_inputs), index = time_array_inputs)

for i in xrange(nb_stim): # For the number of stim
    ind = t_stim_start + i*ms_to_sec/freq_stim

    if ind in f_stim_serie_facil_depr_model.index: # Test if it is in the index, meaning, it's in the time of stimulation
        f_stim_serie_facil_depr_model.loc[ind] = ampl_stim

array_times_to_compute = np.array(f_stim_serie_facil_depr_model[f_stim_serie_facil_depr_model.values == ampl_stim].index)


# -------------------------------------------------------------------- #
# ------------------------- Algorithm parameters --------------------------- #
# -------------------------------------------------------------------- #

max_pts = 100*int((t_end-t_start)/init_step) # No unit | Maximal number of point of the simulation | Must be an integer and positive

refine = 1 # Refine output by adding points interpolated using the RK4 polynomial (0, 1 or 2).

rtol = 1e-8 # Relative error tolerance
atol = 1e-8 # Absolute error tolerance

checklevel = 1 # integer >= 0: internal level of consistency checking when computing trajectories


# -------------------------------------------------------------------- #
# ------------------------- Plot parameters --------------------------- #
# -------------------------------------------------------------------- #

dt_plot = 0.001 # Step of sampling of trajectories

# -------------------------------------------------------------------- #
# ------------------------- Model parameters --------------------------- #
# -------------------------------------------------------------------- #

# ------------------------- Initial values --------------------------- #


init_dict_ord_facil_depr_model = coll.OrderedDict([
    ('recov_frac_syn', 1.0), # 0 - 1 | recovered | Fraction of syn ressource in recovered state
    ('effecti_frac_syn', 0.0) # 0 - 1 | effective | Fraction of syn ressource in effective state
    ])


# ----------- All dictionaries ordered in one ----------- #
init_dict_ord = coll.OrderedDict()

init_dict_ord.update(init_dict_ord_facil_depr_model)

init_dict = init_dict_ord

# ------------------------- Parameter values --------------------------- #

# Ordered dictionary #
params_dict_ord_facil_depr_model = coll.OrderedDict([
    ('tau_rec', 300.0), # ms | Recovery time constant
    ('tau_inac', 200.0), # ms | Inactivation time constant
    ('U_se', 0.8) # No unit | Utilization of synaptic efficacy
    ])

params_dict_ord_model_neuron_activ = coll.OrderedDict([
    ('A_se', 7.0) # No unit | Absolute synaptic strength
    ])

# ----------- All dictionaries ordered in one ----------- #
params_dict_ord = coll.OrderedDict()

params_dict_ord.update(params_dict_ord_facil_depr_model)
params_dict_ord.update(params_dict_ord_model_neuron_activ)

params_dict = params_dict_ord


# -------------------------------------------------------------------- #
# ------------------------- Parameters domain --------------------------- #
# -------------------------------------------------------------------- #


pdomain_dict_ord = coll.OrderedDict([
])

pdomain_dict = pdomain_dict_ord


# -------------------------------------------------------------------- #
# ------------------------- Variable domain --------------------------- #
# -------------------------------------------------------------------- #

xdomain_up = 1e30

xdomain_dict_ord_facil_depr_model = coll.OrderedDict([
    ('tau_rec', [0.0, xdomain_up]),
    ('tau_inac', [0.0, xdomain_up]),
    ('U_se', [0.0, xdomain_up])
    ])

xdomain_dict_ord_model_neuron_activ = coll.OrderedDict([
    ('A_se', [0.0, xdomain_up]) # No unit | Absolute synaptic strength
    ])



# ----------- All dictionaries ordered in one ----------- #
xdomain_dict_ord = coll.OrderedDict() # Initialisation

xdomain_dict_ord.update(xdomain_dict_ord_facil_depr_model)
xdomain_dict_ord.update(xdomain_dict_ord_model_neuron_activ)

xdomain_dict = xdomain_dict_ord


# ----------------------------------------------------------------------- #
# ---------------- Input dictionary ----------------- #
# ----------------------------------------------------------------------- #

inputs_Data = {'f_input' : f_stim_serie_facil_depr_model.values}

Input_interp_table = dst.InterpolateTable(
    {
    'tdata': time_array_inputs,
    'ics': inputs_Data,
    'name': 'Input_interp_table',
    'method': 'linear',
    'checklevel': 1,
    'abseps': 1e-6
    }).compute('interp')

inputs_dict = {'f_input': Input_interp_table.variables['f_input']}


# -------------------------------------------------------------------- #
# ------------------------- Parameters functions --------------------------- #
# -------------------------------------------------------------------- #

# auxiliary helper function(s) -- function name: ([func signature], definition)
parameters_functions = {
            }


# -------------------------------------------------------------------- #
# ------------------------- Auxiliaries variables -------------------- #
# -------------------------------------------------------------------- #

aux_var_dict = {

        # -------------------------------------------------------------------- #
        # ------------------- Facilitation-Depression model ------------------ #
        # -------------------------------------------------------------------- #

        'f_input_aux' : 'f_input', # Aux variable to plot input

        'inact_frac_syn' : '1 -recov_frac_syn -effecti_frac_syn', # 0 - 1 | Inactive | Fraction of synrce in inactive state

        # -------------------------------------------------------------------- #
        # ------------------- Modeling neuronal activity ------------------ #
        # -------------------------------------------------------------------- #

        'I_app' : 'A_se *effecti_frac_syn' # mA | Synaptic current

    }


# -------------------------------------------------------------------- #
# ------------------------- ODEs --------------------------- #
# -------------------------------------------------------------------- #

ode_dict = {

        # -------------------------------------------------------------------- #
        # ------------------- Facilitation-Depression model ------------------ #
        # -------------------------------------------------------------------- #

        'recov_frac_syn' : 'inact_frac_syn/tau_rec -U_se*recov_frac_syn*f_input_aux', # 0 - 1 | recovered | Fraction of syn ressource in recovered state
        'effecti_frac_syn' : '-effecti_frac_syn/tau_inac +U_se*recov_frac_syn*f_input_aux' # 0 - 1 | effective | Fraction of syn ressource in effective state

        }



# ------------- Merge dictionaries of ODE and auxiliary variables ------------- #
ode_and_aux_var_dict = ode_dict.copy()
ode_and_aux_var_dict.update(aux_var_dict)

# -------------------------------------------------------------------- #
# ------------------------- COMPUTATION --------------------------- #
# -------------------------------------------------------------------- #

fnspecs = parameters_functions
rhs = ode_and_aux_var_dict

