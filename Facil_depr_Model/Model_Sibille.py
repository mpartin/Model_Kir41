# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:34:10 2015

@author: alexandre
"""
from matplotlib import pyplot as plt

from parameters import *

# -------------------------------------------------------------------- #
# ------------------------- FUNCTIONS --------------------------- #
# -------------------------------------------------------------------- #
def main_glob_model(name, params_dict, rhs, fnspecs, icdict, inputs_dict):

    # -------------------------------------------------------------------- #
    # ------------------------- INITIALISATION --------------------------- #
    # -------------------------------------------------------------------- #

    DST_object = dst.args() # ODE system object
    DST_object['name'] = name # Name of the object

    DST_object['algparams'] = {'init_step': init_step, 'refine': refine, 'max_step': max_step,
                               'max_pts' : max_pts, 'rtol': rtol, 'atol': atol, 'use_special' : True,
                               'specialtimes' : array_times_to_compute
                             }

    DST_object['tdomain'] = [t_start, t_end] # Time simulation bounds
    DST_object['tdata'] = [t_start, t_end] # Time simulation bounds


    DST_object['xdomain'] = xdomain_dict # Variables simulation bounds
    DST_object['pdomain'] = pdomain_dict # Parameters simulation bounds

    DST_object['checklevel'] = checklevel

    DST_object['vars'] = icdict.keys() # List of ODE's variables

    DST_object['pars'] = params_dict # Dictionary of parameters
    DST_object['fnspecs']  = fnspecs
    DST_object['varspecs'] = rhs

    DST_object['inputs'] = inputs_dict

    DST_object['ics'] = icdict


    # -------------------------------------------------------------------- #
    # ------------------------- COMPUTATION --------------------------- #
    # -------------------------------------------------------------------- #

    generator_object = dst.Generator.Radau_ODEsystem(DST_object) # Generator object

    trajectories = generator_object.compute(DST_object.name) # Trajectories of each variable

    return trajectories, DST_object, generator_object



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# ----------- MAIN ----------- #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Not be executed if it's imported #
if __name__ == '__main__':
    trajectories, DST_object, generator_object = main_glob_model('Sibille_model', params_dict, rhs, fnspecs, init_dict, inputs_dict)

    traj_sampled = trajectories.sample(dt = dt_plot) # Sub/sup-sampling of trajectories
    # -------------------------------------------------------------------- #
    # ------------------------- PLOT --------------------------- #
    # -------------------------------------------------------------------- #

    plt.plot(traj_sampled['t'],traj_sampled['effecti_frac_syn'])
    plt.plot(traj_sampled['t'],traj_sampled['f_input_aux'])
    plt.plot(traj_sampled['t'],traj_sampled['I_app'])
    plt.legend(['effecti_frac_syn', 'f_input_aux', 'I_app'])
