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
init_step = 0.010 # ms
max_step = 0.10 # ms | Maximal step-size

# -------------------------------------------------------------------- #
# ---------------- Inputs ------------------- #
# -------------------------------------------------------------------- #

t_stim_start = 50.0 # ms


t_stim_dur = 100.0 # ms
freq_stim = 100.0 # Hz

ms_to_sec = 1000.0 # Factor

nb_stim = int(t_stim_dur/ms_to_sec *freq_stim) +1 # Number of stimulation

input_val_Down = 0.0 # No unit | No stimulus value
input_val_Up = 1.0 # No unit | Amplitude of stimulus
input_time_step = 0.01 # ms | Time step around stimulus
input_dur = 0.1 # ms | Duration of stimulus

# -------------------------------------------------------------------- #
# ------------------------- Algorithm parameters --------------------------- #
# -------------------------------------------------------------------- #

max_pts = 100*int((t_end-t_start)/init_step) # No unit | Maximal number of point of the simulation | Must be an integer and positive

refine = 1 # Refine output by adding points interpolated using the RK4 polynomial (0, 1 or 2).

rtol = 1e-4 # Relative error tolerance
atol = 1e-4 # Absolute error tolerance

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
    ('A_se', -14.0) # No unit | Absolute synaptic strength
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
    ('A_se', [-xdomain_up, xdomain_up]) # No unit | Absolute synaptic strength
    ])



# ----------- All dictionaries ordered in one ----------- #
xdomain_dict_ord = coll.OrderedDict() # Initialisation

xdomain_dict_ord.update(xdomain_dict_ord_facil_depr_model)
xdomain_dict_ord.update(xdomain_dict_ord_model_neuron_activ)

xdomain_dict = xdomain_dict_ord

# ----------------------------------------------------------------------- #
# ---------------- Input dictionary ----------------- #
# ----------------------------------------------------------------------- #

xnames = ['f_input'] # Name of input

x1data = [input_val_Down] # Initialisation | list of stimulus value
timeData = [t_start] # Initialisation | list of special times


for i in xrange(nb_stim): # For each stimulus
    time_stim = t_stim_start + i*ms_to_sec/freq_stim

    if time_stim < t_end: # If it is before end of simulation

        # Just before stimulus #
        x1data += [input_val_Down]
        timeData += [time_stim -input_time_step]

        # Begining of stimulus #
        x1data += [input_val_Up]
        timeData += [time_stim]

        # End of stimulus #
        x1data += [input_val_Up]
        timeData += [time_stim +input_dur]

        # Juste after stimulus #
        x1data += [input_val_Down]
        timeData += [time_stim +input_dur +input_time_step]

# End of simulation #
x1data += [input_val_Down]
timeData += [t_end]

# Interpolation table #
xData = dict(list(zip(xnames, [x1data])))
interptable = dst.InterpolateTable({'tdata': timeData,
                              'ics': xData,
                              'name': 'interp'
                              })
itabletraj = interptable.compute('interp')

# Dictionary #
inputs_dict = {'f_input': itabletraj.variables['f_input']}

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

        'effecti_frac_syn_aux' : 'effecti_frac_syn +U_se*recov_frac_syn*f_input_aux', # 0 - 1 | effective | Fraction of syn ressource in effective state
        'recov_frac_syn_aux' : 'recov_frac_syn -U_se*recov_frac_syn*f_input_aux', # 0 - 1 | recovered | Fraction of syn ressource in recovered state

        'inact_frac_syn_aux' : '1 -recov_frac_syn_aux -effecti_frac_syn_aux', # 0 - 1 | Inactive | Fraction of synrce in inactive state

        # -------------------------------------------------------------------- #
        # ------------------- Modeling neuronal activity ------------------ #
        # -------------------------------------------------------------------- #

        'I_app' : 'A_se *effecti_frac_syn_aux' # mA | Synaptic current

    }


# -------------------------------------------------------------------- #
# ------------------------- ODEs --------------------------- #
# -------------------------------------------------------------------- #

ode_dict = {

        # -------------------------------------------------------------------- #
        # ------------------- Facilitation-Depression model ------------------ #
        # -------------------------------------------------------------------- #

        'recov_frac_syn' : 'inact_frac_syn_aux/tau_rec', # 0 - 1 | recovered | Fraction of syn ressource in recovered state
        'effecti_frac_syn' : '-effecti_frac_syn_aux/tau_inac' # 0 - 1 | effective | Fraction of syn ressource in effective state

        }



# ------------- Merge dictionaries of ODE and auxiliary variables ------------- #
ode_and_aux_var_dict = ode_dict.copy()
ode_and_aux_var_dict.update(aux_var_dict)

# -------------------------------------------------------------------- #
# ------------------------- COMPUTATION --------------------------- #
# -------------------------------------------------------------------- #

fnspecs = parameters_functions
rhs = ode_and_aux_var_dict

