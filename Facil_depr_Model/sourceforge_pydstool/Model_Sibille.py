# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:34:10 2015

@author: alexandre
"""
from matplotlib import pyplot as plt
from random import random

from parameters import *


# -------------------------------------------------------------------- #
# ------------------------- FUNCTIONS --------------------------- #
# -------------------------------------------------------------------- #

def Hybrid_model(name, rhs, params_dict, list_stim, fnspecs, list_aux_var, name_event, evtol=None):


    sp_event_args = {'name': name_event,
                   'eventtol': 1e-3,
                   'eventdelay': 1e-5,
                   'starttime': 0,
                   'active': True,
                   'term': True,
                   'precise': True}

    # spike event triggers when global time of whole integration increases through
    # (next) spike time parameter value
    spike_ev = dst.Events.makeZeroCrossEvent('globalindepvar(t)-sp_arr', 1,
                                            sp_event_args, parnames=['sp_arr'])

    if len(array_time_inputs) > 1: # In case there is more than one stimulation
        spike_ev.createQ('sp_arrs', seq=array_time_inputs[1:])  # remaining spike times after first (see below)

    # -----------------  DST Object --------------------- #
    DST_object = dst.args()
    DST_object['name'] = name # Name of the object

    DST_object['algparams'] = {'init_step': init_step, 'refine': refine, 'max_step': max_step,
                               'max_pts' : max_pts
                             }
    DST_object['auxvars'] = list_aux_var
    DST_object['tdata'] = [t_start, t_end]
    DST_object['xdomain'] = xdomain_dict # Variables simulation bounds
    DST_object['pars'] = params_dict # Dictionary of parameters
    DST_object['fnspecs']  = fnspecs # functions
    DST_object['varspecs'] = rhs # ODE and auxiliaries variables
    DST_object['events'] = [spike_ev] # List of events
    DST_object['abseps'] = 1e-7
    DST_object['ics'] = init_dict


    # ------------------- Update RHS and params ---------------------- #
    DST_object['pars'].update({'sp_arr': array_time_inputs[0]}) # prep the integrator with the first spike time

    # Replace auxiliary variable by their expression in rhs of auxiliary variables #

    for variables in DST_object['varspecs'].keys():
        for aux_var in DST_object['auxvars']:
            DST_object['varspecs'][variables] = replace_words_in_string(DST_object['varspecs'][variables], [aux_var], rhs)

    return dst.embed(dst.Generator.Vode_ODEsystem(DST_object), icdict=init_dict, tdata=DST_object['tdata'])

# Return spike mapping depending on the number of time inputs #
def spike_mapping_func(DS_model, name_event):

    if len(array_time_inputs) == 0:
        spike_mapping = dst.EvMapping(defString="""
        xdict['f_input_aux'] = 0.0
        """, model=DS_model)

    elif len(array_time_inputs) == 1:
        spike_mapping = dst.EvMapping(defString="""
        xdict['f_input_aux'] = pdict['ampl_stim']
        """, model=DS_model)

    else:
        # update parameter for spike time from queue and "inhibit" V
        # replace name_event by its value (e.g. 'spike) because it's not known here
        spike_mapping = dst.EvMapping(defString="""
        xdict['f_input_aux'] = pdict['ampl_stim']
        try:
            pdict['sp_arr'] = estruct.events['spike'].popFromQ('sp_arrs')
        except IndexError:
            # no more spikes
            pdict['sp_arr'] = -1
        """, model=DS_model)

    return spike_mapping


def make_Hybrid_model(DS_model, event_mapping, name, name_event):

    DS_model_MI = dst.intModelInterface(DS_model)

    DS_model_info = dst.makeModelInfoEntry(DS_model_MI, [name],
                                  [(name_event, (name, event_mapping))])

    modelInfoDict = dst.makeModelInfo([DS_model_info])

    return dst.Model.HybridModel({'name': 'model_fit', 'modelInfo': modelInfoDict})



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# ----------- MAIN ----------- #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Not be executed if it's imported #
if __name__ == '__main__':


    DS_Sibille = Hybrid_model(name, rhs, params_dict, array_time_inputs, fnspecs, list_aux_var, name_event, evtol=None)

    spike_mapping = spike_mapping_func(DS_Sibille, name_event)

    Sibille_model = make_Hybrid_model(DS_Sibille, spike_mapping, name, name_event)



    Sibille_model.compute(trajname='test', tdata=[t_start, t_end], ics=init_dict, verboselevel=0,
                force=True)

    traj_sampled = Sibille_model.sample('test', dt=dt_plot) # Sub/sup-sampling of trajectories

    # -------------------------------------------------------------------- #
    # ------------------------- PLOT --------------------------- #
    # -------------------------------------------------------------------- #

    plt.plot(traj_sampled['t'],traj_sampled['effecti_frac_syn'])
    plt.plot(traj_sampled['t'],traj_sampled['f_input_aux'])
    plt.plot(traj_sampled['t'],traj_sampled['I_app_aux'])

#    plt.legend(['effecti_frac_syn', 'I_app_aux'])
