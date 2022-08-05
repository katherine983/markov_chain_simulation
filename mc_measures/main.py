# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 10:59:49 2022

@author: Wuestney
"""

from mc_measures import gen_mc_transition as genmc
from mc_measures import mc_entropy


def gen_model(root_dir, order_i, states_temp):
    """
    Function to generate a MC matrix, calculate its entropy rate and save all
    of its data to a json file.

    Parameters
    ----------
    root_dir : PATH-LIKE OBJECT
        THE ROOT DIRECTORY FOR THE OUTPUT FILE.
    order_i : INTEGER
        ORDER OF THE MARKOV MATRIX TO BE GENERATED.
    states_temp : LIST OF UNICODE CHARACTERS
        lIST OF THE UNICODE CHARACTERS THAT WILL BE THE STATES OF THE MARKOV
        PROCESS.THE LENGTH OF states_temp SHOULD BE THE SAME AS THE SIZE OF THE
        ALPHABET.

    Returns
    -------
    MC_model : instance of GenMarkovTransitionProb.

    """
    MC_model = genmc.GenMarkovTransitionProb(states_temp, order_i)
    MC_model.entropy_rate()
    fout_file_path = create_output_file_path(root_dir, r'mc_matrices', f'Order{order_i}Alph{MC_model.m}ER{MC_model.ent_rate:.4f}.json')
    MC_model.dump(fout_file_path)
    return MC_model