# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 17:36:43 2022

@author: Wuestney

Package containing independent functions for calculating the true steady-state values
of various entropy measures from a ergodic Markov Chain stored in a
GenMarkovTransitionProb class instance from the mc_entropy/mc_measures/gen_mc_transition.py module
"""

import numpy as np

def entropy_rate(GMTP):
    """
    Identical to the entropy_rate() method of the GenMarkovTransitionProb class.
    See documentation in gen_mc_transition.py

    Parameters
    ----------
    GMTP : GENMARKOVTRANSITIONPROB CLASS FROM gen_mc_transition module
        CLASS INSTANCE WHICH CONTAINS THE TRANSITION MATRIX THAT THE ENTROPY RATE
        WILL BE CALCULATED FOR.

    Returns
    -------
    ent_rate : INTEGER
        ENTROPY RATE AS INTEGER VALUE.

    """
    q, P = GMTP.eig_steadystate()
    ent_rate = np.negative((q*P*np.log(P)).sum())
    return ent_rate

def markovApen(GMTP):
    """
    Approximate Entropy of a Markov Chain is equivalent to the entropy rate.

    Parameters
    ----------
    GMTP : GENMARKOVTRANSITIONPROB CLASS FROM gen_mc_transition module
        CLASS INSTANCE WHICH CONTAINS THE TRANSITION MATRIX THAT THE ENTROPY RATE
        WILL BE CALCULATED FOR.

    Returns
    -------
    apen : INTEGER
        APEN AS INTEGER VALUE.

    """
    apen = entropy_rate(GMTP)
    return apen

def markovSampen(GMTP):
    q, P = GMTP.eig_steadystate()
    A = -1 + np.sum(q*P)
    B = -1 + np.sum(q)
    sampen = np.negative(np.log((A/B)))
    return sampen