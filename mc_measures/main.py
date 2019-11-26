############################################################
################## Markov Model Order Selection ############
############ Beiyu Lin @ Washing State University ##########
#################### beiyu.lin@wsu.edu #####################
############################################################
#!/usr/bin/python
from collections import defaultdict  
import itertools
import csv
import pandas as pd
import os
import glob
import re
import numpy
import sys
from random import choice
from string import ascii_lowercase
from mc_class import MarkovModel
from gen_mc_transition import GenMarkovTransitionProb

# https://www.datasciencecentral.com/profiles/blogs/some-applications-of-markov-chain-in-python
# input_sequence = ['abc', 'abc', 'acd', 'ecs', 'adf', 'dafae', 'dafae', 'dafae', 'adf', 'abc', 'dafae', 'dafae', 'dafae', 'adf', 'abc']

def make_dir(path):
	try:
		os.stat(path)
	except:
		os.makedirs(path)

def create_output_file_path(root_dir, dir_name, sample_size):
	output_dir = root_dir + "measure_results/"+ dir_name + "/"
	make_dir(output_dir)
	fout_file_path = output_dir + "/syn_" + str(sample_size) + ".csv"
	return fout_file_path

def percentage_order_category(selected_order, order_i, percentage_order):

	under_estimate_count = {}
	over_estimate_count = {}
	for measure in selected_order.keys():
		# print("measure", measure)
		mean_error = float(selected_order[measure]) - float(order_i)
		if mean_error < 0.0
			# selected_order[measure] < order_i:
			percentage_order[measure]["under_estimate"] =  percentage_order.get(measure, 0.0).get("under_estimate", 0.0) + abs(mean_error)
			under_estimate_count[measure] = under_estimate_count.get(measure, 0) + 1
		elif mean == 0.0:
			# selected_order[measure] == order_i:
			percentage_order[measure]["equal"] =  percentage_order.get(measure, 0).get("equal", 0) + 1
		else: 
			percentage_order[measure]["over_estimate"] =  percentage_order.get(measure, 0).get("over_estimate", 0) + abs(mean_error)
			over_estimate_count[measure] = over_estimate_count.get(measure, 0) + 1

	for measure in selected_order.keys():
		for k in percentage_order[measure].keys():
			if k is "under_estimate": 
				percentage_order[measure]["under_estimate"] = float(percentage_order[measure]["under_estimate"])/under_estimate_count[measure]
			elif k is "equal":
				percentage_order[measure]["equal"] = percentage_order[measure]["equal"]/1000.0
			elif k is "over_estimate": 
				percentage_order[measure]["over_estimate"] = float(percentage_order[measure]["over_estimate"])/over_estimate_count[measure]

	return percentage_order

def calculate_measure_values(readin_data, order_i, percentage_order, measure_names):
	# header = ["order", "aic", "bic", "hqic", "edc", "new1", "new2", "new3"]
	mini_measure = defaultdict(float)
	selected_order = defaultdict(float)
	for i in range(0, 7):
		## readin_data: sample, kth order
		m = MarkovModel(readin_data, i)
		for measure in measure_names:
			measure_value = round(m.measure_values(measure), 2)
			if measure not in mini_measure.keys():
				mini_measure[measure] =	measure_value
				selected_order[measure] = i 
			else: 
				if measure_value < mini_measure[measure]:
					mini_measure[measure] = measure_value
					selected_order[measure] = i 
	percentage_order = percentage_order_category(selected_order, order_i, percentage_order)
	return percentage_order

def write_into_csv(percentage_order, fout_file_path):	
	header = ["measures", "under_estimate", "equal", "over_estimate"]
	
	# , "new2", "new3", "new4",
	# "new5", "new6", "new7", "new8", "new9", "new10", "new11", "new12", 
	# "new13", "new14", "new15", "new16"]	
	with open(fout_file_path, "wb") as f:
	    w = csv.DictWriter(f, header)
	    w.writeheader()
	    for measure in measure_names:
	    	temp_row = {}
	    	for h in header: 
	    		if h not in percentage_order[measure].keys():
	    			temp_row[h] = measure
	    		else:
	    			temp_row[h] = percentage_order[measure].get(h)
	    	w.writerow(temp_row)

def init_percent_order(measure_names):
	estimate = {"under_estimate": 0, "equal": 0, "over_estimate": 0}
	for m in measure_names: 
		percentage_order[m] = estimate
	return percentage_order

def syn2_paper(root_dir, states_temp, order_i, start_sample_size, end_sample_size, measure_names):
	percentage_order = init_percent_order(measure_names)
	for s in range(start_sample_size, end_sample_size, 500):
		for h  in range(1, 1000):
			print("generate sampel size %s with iteration %s"%(s-5000, h))
			gen_m = GenMarkovTransitionProb(states_temp, order_i)
			gen_sequence = gen_m.gen(tuple(states_temp[0:order_i]), s)
			percentage_order = calculate_measure_values(gen_sequence[5000:], order_i, percentage_order, measure_names)
	
		fout_file_path = create_output_file_path(root_dir, "syn_data/syn2_order_"+str(order_i) + "/"  , (s-5000))
		write_into_csv(percentage_order, fout_file_path)


if __name__ == '__main__':
	
	# root_dir = "/Users/BeiyuLin/Desktop/five_datasets/"
	root_dir = "./"
	order_i = int(sys.argv[1])
	states_nums = int(sys.argv[2])
	start_sample_size = int(sys.argv[3])
	end_sample_size = int(sys.argv[4])
	measure_names = ["aic", "bic", "hqic", "edc", "new1", "new2", "new3", "new4", "new5", "new6"]
	print("generate syn2 data")
	# based on randomly generated transition matrix.
	# http://www.iaeng.org/publication/WCECS2014/WCECS2014_pp899-901.pdf
	# each case is the order of the synthetic data
	states_temp = [chr(ord('a')+i) for i in xrange(states_nums)]
	 # = ['a','b','c','d','e','f','g','h','u']
	# generate 1000 random transition matrix 
	syn2_paper(root_dir, states_temp, order_i, start_sample_size, end_sample_size, measure_names)
	print("finished syn2")

