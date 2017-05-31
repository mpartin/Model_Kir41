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
import string
import re
import pandas as pd

import scipy.stats
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import PyDSTool as dst

# Return dictionary | Function to convert string formulas with power #
def convert_equation_string_power(dictionary):

    for key, value in dictionary.iteritems():
        dictionary[key] = dst.convertPowers(value)
    return dictionary

# Return dictionary | Function to convert string formulas with power #
def convert_function_string_power(dictionary):

    for key, value in dictionary.iteritems():
        dictionary[key] = (value[0],dst.convertPowers(value[1]))
    return dictionary


# Return string with replaced terms | Recurrence function #
def replace_words_in_string(string_to_change, list_term_to_replace, replacing_term_dict):

    words_dict = {} # Initialisation
    for word in re.finditer(r"\w+", string_to_change): # For each term of the equation
        words_dict[word.group(0)] = (word.start(), word.end()) # Save position in a dictionary

    carac_list = re.findall(r".", string_to_change) # List of caracters of the equation

    for term_to_replace in list_term_to_replace: # For each term to replace
        if term_to_replace in words_dict.keys(): # If there is a term to replace

            for j in xrange(words_dict[term_to_replace][1] -words_dict[term_to_replace][0]): # For each caracter of the term to replace
                carac_list.pop(words_dict[term_to_replace][0]) # Remove caracter

            carac_list.insert(words_dict[term_to_replace][0], replacing_term_dict[term_to_replace]) # Insert new term

            carac_list = "".join(carac_list) # From list of caracter to string chain

            return replace_words_in_string(carac_list, list_term_to_replace, replacing_term_dict) # Recurrence

    carac_list = "".join(carac_list) # From list of caracter to string chain

    return carac_list

# -------------------------------------------------------------------- #
# ------------------------- Simulation parameters --------------------------- #
# -------------------------------------------------------------------- #

t_start = 0.0 # ms | Start of the simulation
t_end = 1000.0 # ms | End of the simulation

init_step = 0.10 # ms
max_step = 10.0 # ms | Maximal step-size

name = 'Sibille_model'
name_event = 'spike'

# -------------------------------------------------------------------- #
# ---------------- Inputs ------------------- #
# -------------------------------------------------------------------- #

t_stim_start = 30.9 # ms | Start of stimulation

t_stim_dur = 1000.0 # ms | Duration of stimulation
freq_stim = 10.0 # Hz | Frequency of stimulation

ms_to_sec = 1000.0 # Factor sec to msec
stim_interv = ms_to_sec/freq_stim # ms | Number of ms between each stimulation
t_stim_end = min(t_stim_start +t_stim_dur, t_end) # ms | End of stimulation (it can be because of end of simulation)

array_time_inputs = np.arange(t_stim_start, t_stim_end, stim_interv) # Array of timing of stimulation

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
    ('effecti_frac_syn', 0.0), # 0 - 1 | effective | Fraction of syn ressource in effective state
    ('f_input_aux', 0.0) # Auxiliary input
    ])


# ----------- All dictionaries ordered in one ----------- #
init_dict = coll.OrderedDict() # Initialisation

init_dict.update(init_dict_ord_facil_depr_model)


# ------------------------- Parameter values --------------------------- #

# Ordered dictionary #
params_dict_ord_facil_depr_model = coll.OrderedDict([
    ('tau_input', 0.1), # ms | Input time constant
    ('ampl_stim', 1.0), # Amplitude Dirac stimulation

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
    ('tau_input', [0.0, xdomain_up]),
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

        'inact_frac_syn_aux' : '(1 -recov_frac_syn -effecti_frac_syn)', # 0 - 1 | Inactive | Fraction of synrce in inactive state

        # -------------------------------------------------------------------- #
        # ------------------- Modeling neuronal activity ------------------ #
        # -------------------------------------------------------------------- #

        'I_app_aux' : '(A_se *effecti_frac_syn)' # mA | Synaptic current

    }


# -------------------------------------------------------------------- #
# ------------------------- ODEs --------------------------- #
# -------------------------------------------------------------------- #

ode_dict = {

        # -------------------------------------------------------------------- #
        # ------------------- Facilitation-Depression model ------------------ #
        # -------------------------------------------------------------------- #
        'f_input_aux' : '-f_input_aux/tau_input', # Auxiliary input
        'recov_frac_syn' : 'inact_frac_syn_aux/tau_rec -U_se*recov_frac_syn*f_input_aux', # 0 - 1 | recovered | Fraction of syn ressource in recovered state
        'effecti_frac_syn' : '-effecti_frac_syn/tau_inac +U_se*recov_frac_syn*f_input_aux' # 0 - 1 | effective | Fraction of syn ressource in effective state

        }



# ------------- Merge dictionaries of ODE and auxiliary variables ------------- #
ode_and_aux_var_dict = ode_dict.copy()
ode_and_aux_var_dict.update(aux_var_dict)

list_aux_var = aux_var_dict.keys()

# -------------------------------------------------------------------- #
# ------------------------- COMPUTATION --------------------------- #
# -------------------------------------------------------------------- #

fnspecs = convert_function_string_power(parameters_functions)
rhs = convert_equation_string_power(ode_and_aux_var_dict)
aux_var_equations = convert_equation_string_power(aux_var_dict)

