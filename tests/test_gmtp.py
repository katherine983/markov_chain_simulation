# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 11:10:30 2022

@author: Wuestney
"""
import pytest
from mc_measures import gen_mc_transition as genmc


def test_get_model():
    # root_dir = "/Users/BeiyuLin/Desktop/five_datasets/"
    root_dir = "./"
    # order_i is the given order of markov chain
    # order_i = int(sys.argv[1])
    order_i = 1
    # states_nums is the number of states in the process (i.e. the size of the alphabet)
    # states_nums = int(sys.argv[2])
    states_nums = 3
    #start_sample_size = int(sys.argv[3])
    #end_sample_size = int(sys.argv[4])
    # based on randomly generated transition matrix.
    # http://www.iaeng.org/publication/WCECS2014/WCECS2014_pp899-901.pdf
    states_temp = [chr(ord('a')+i) for i in range(states_nums)]
    # = ['a','b','c','d','e','f','g','h','u']
    # generate 1000 random transition matrix
    MC1 = genmc.gen_model(root_dir, order_i, states_temp)
    #print(MC1.tran)
    fpath = genmc.get_data_file_path(root_dir, r'mc_matrices', f'Order{MC1.k}Alph{MC1.m}ER{MC1.ent_rate:.4f}.json')
#%%
    MC2 = genmc.get_model(fpath)
    #MC1 == MC2
    #MC1_freq = MC1._asarray()
    estMC1P = MC1.prob_tran()
#%%
    estMC1q = MC1.stationary_dist()
#%%
    MC1q, MC1P = MC1.eig_steadystate()
    MC2q, MC2P = MC2.eig_steadystate()
#%%
    mcent = MC1.entropy_rate()
    #print('estP:', estMC1P)
    #print('P:', MC1P)
    #print('estq:', estMC1q)
    #print('q:', MC1q)
    #print('entropy rate:', MC1.ent_rate)
    MC1.entropy_rate_est()
    print('estimated entropy rate:', MC1.ent_rate_est)
    seq = MC1.gen_sample(tuple(states_temp[:order_i]), 100)
    print(seq)
    print(MC1.seed)