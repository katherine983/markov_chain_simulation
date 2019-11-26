import numpy as np 
import math
from collections import defaultdict  

class MarkovModel:      

	def __init__(self, sample, k):          
		'''create a Markov model of order k from given activities         
		Assume that sample has length at least k.'''
		# k is the order
		self.k = k
		# the transtion matrix (with count not probability)
		self.tran = defaultdict(float)  
		# a list of unique activities/states 
		self.alph = list(set(list(sample))) 
		# number of states
		self.m = len(self.alph)
		# kgrams shows the count of each k successive activities
		# e.g.: {('abc', 'abc', 'abc'): 1}
		self.kgrams = defaultdict(int)  
		# sample size
		self.n = len(sample)
		sample += sample[:k]
		for i in range(self.n):
			self.tran[tuple(sample[i:i+k]),sample[i+k]] += 1 
			self.kgrams[tuple(sample[i:i+k])] += 1    

	def order(self):
		# order k of Markov model
		return self.k

	def freq(self, kgram):
		# number of occurrences of kgram in sample
		assert len(kgram) == self.k    
		# (check if kgram is of length k)
		return self.kgrams[kgram]

	def freq2(self, kgram, c):
		# number of times that character c follows kgram
		assert len(kgram) == self.k
		# (check if kgram is of length k)
		return self.tran[kgram,c]

	def prob_tran(self):
		# transition matrix with probability
		prob_tran_matrix = defaultdict(float)
		for key, value in self.tran.items():
			# key[0]: k-successive activities
			# key[1]: k+1th activity
			# value: the count from k-successive activities to the k+1th activity
			prob_tran_matrix[key] = value/self.freq(key[0])
		return prob_tran_matrix

	def likelihood(self):
		likelihood_value = 0 
		for key, value in self.tran.items():
			# key[0]: k-successive activities
			# key[1]: k+1th activity
			# value: the count from k-successive activities to the k+1th activity
			likelihood_value += value*math.log(value/self.freq(key[0]))
		return likelihood_value

	def number_of_parameters(self, measure_name):
		# m: the number of states; k: the order; n: the sample size
		if measure_name == "aic":
			return 2.0*(self.m - 1)*(self.m**self.k)  ## overestmate
		elif measure_name == "bic":
			return math.log(self.n)*(self.m - 1.0)*(self.m**self.k)  ## underestimate
		elif measure_name == "hqic":
			return math.log(math.log(self.n))*(self.m - 1.0)*2*(self.m**self.k)
		elif measure_name == "edc":
			return math.log(math.log(self.n))*2*(self.m**(self.k+1.0))
		elif measure_name == "new1":
			return self.n**(1.0/self.n)*(math.log(self.n))*(5/6.0)*(self.m - 1)*(self.m**self.k)
		elif measure_name == "new2":
			return self.n**(1.0/self.n)*(math.log(self.n))*(1/2.0)*(self.m - 1)*(self.m**self.k)
		elif measure_name == "new3":
			return self.n**(1.0/self.n)*(math.log(math.log(self.n)))*(self.m - 1)*(self.m**self.k)
		elif measure_name == "new4":
			return 0.9*self.n**(1.0/self.n)*(math.log(math.log(self.n)))*(self.m - 1)*(self.m**self.k)
		elif measure_name == "new5":
			return 1.1*self.n**(1.0/self.n)*(math.log(math.log(self.n)))*(self.m - 1)*(self.m**self.k)
		elif measure_name == "new6":
			return 1.2*self.n**(1.0/self.n)*(math.log(math.log(self.n)))*(self.m - 1)*(self.m**self.k)
		# elif measure_name == "posterior":  
		# 	## modify
		# 	return self.n
		# elif measure_name == "new1":
		# 	return self.n**(1.0/self.n)*(self.m - 1)*(self.m**self.k)
		# elif measure_name == "new2":
		# 	return self.n**(1.0/self.n)*2*(self.m**(self.k+1))
		# elif measure_name == "new3":
		# 	return self.n**(1.0/self.n)*(self.m**(self.k+1))
		# elif measure_name == "new4":
		# 	return (math.log(self.n))*self.n**(1.0/self.n)*(self.m - 1)*(self.m**self.k)
		# elif measure_name == "new5":
		# 	return 0.7*(math.log(self.n, 4))
		# elif measure_name == "new6":
		# 	return 0.4*(math.log(self.n))*(self.n**(1/self.n))
		# elif measure_name == "new7":
		# 	return 1.5*(math.log(math.log(self.n)))
		# elif measure_name == "new8":
		# 	return 1.2*(math.log(math.log(self.n)))
		# elif measure_name == "new9":
		# 	return 0.7*(math.log(self.n, 5))
		# elif measure_name == "new10":
		# 	return 0.8*(math.log(self.n, 6))
		# elif measure_name == "new11":
		# 	return 0.9*(math.log(self.n, 7))
		# elif measure_name == "new12":
		# 	return (math.log(self.n, 8))
		# elif measure_name == "new13":
		# 	return 1.3*(math.log(math.log(self.n)))
		# elif measure_name == "new14":
		# 	return 1.4*(math.log(math.log(self.n)))
		# elif measure_name == "new15":
		# 	return (math.log(math.log(self.n)))
		# elif measure_name == "new16":
		# 	return 1.1*(math.log(math.log(self.n)))


	def measure_values(self, measure_name):
		return -2.0*self.likelihood() + self.number_of_parameters(measure_name)
		
	def rand(self, kgram):
		# random activity following given kgram
		assert len(kgram) == self.k
		# (check if kgram is of length k.
		Z = sum([self.tran[kgram, alph] for alph in self.alph])
		return np.random.choice(self.alph, 1, p=np.array([self.tran[kgram, alph] for alph in self.alph])/Z)

	def gen(self, kgram, N):
		# generate a list of length N activities
		assert len(kgram) == self.k
		# by simulating a trajectory through the corresponding
		activity_list = kgram
		for i in range(N):
			# generated list should be the argument kgram.
			# check if kgram is of length k.
			c =  self.rand(sample)[0]  	
			kgram = kgram[1:] + (c,)
			activity_list += c
		return activity_list
