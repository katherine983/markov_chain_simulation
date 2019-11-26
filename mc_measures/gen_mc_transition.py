import numpy as np 
import math
from collections import defaultdict  
from itertools import product
from string import ascii_lowercase

class GenMarkovTransitionProb:      

	def __init__(self, text, k):          
		'''create a Markov model of order k from given activities         
		Assume that text has length at least k.'''
		# k is the order
		# print("GenMarkovTransitionProb %s%s" %(text, k))
		self.k = k
		self.prob_tran_matrix = defaultdict(float)
		# the transtion matrix (with count not probability)
		self.tran = defaultdict(float)  
		# a list of unique activities/states 
		self.alph = list(set(list(text))) 
		# number of states
		self.m = len(self.alph)
		# kgrams shows the count of each k successive activities
		# e.g.: {('abc', 'abc', 'abc'): 1}
		self.kgrams = defaultdict(int)  
		# sample size
		self.n = len(text)
		# text += text[:k]
		
		# if self.k == 0:
		# 	k_successive = [() for a in product(text, repeat=0)]
		## use k_successive to generate the past states. 
		if self.k == 1:
			k_successive = [(a) for a in product(text, repeat=1)]
		elif self.k == 2:
			k_successive = [(a, b) for a,b in product(text, repeat=2)]
		elif self.k == 3:
			k_successive = [(a, b, c) for a,b, c in product(text, repeat=3)]
		elif self.k == 4:
			print("4th order.")
			k_successive = [(a, b, c, d) for a,b,c, d in product(text, repeat=4)]
		elif self.k == 5:
			print("5th order.")
			k_successive = [(a, b, c, d, e) for a,b,c,d,e in product(text, repeat=5)]
		elif self.k == 6:
			print("6th order.")
			k_successive = [(a, b, c, d, e, f) for a,b,c,d,e,f in product(text, repeat=6)]
		else: 
			print("chose the MC order between 0 to 6.")

		## assign a random number to each cell/element in the transition matrix. 
		for successive in k_successive:
			# print("in successive", successive)
			for j in range(self.n):
				if j-1+k < self.n:
					# successive is the past states, text[j-1+k] is the next state. 
					# tran is the transition matrix.
					# e.g.: {(('adfg', 'dafae', 'dafae'), 'dafae'): 1.0} print("m.tran", m.tran, len(m.tran))
					# m.kgrams is a dictionary that shows the count of each k successive activities
					# e.g.: {('abc', 'abc', 'abc'): 1} print("m.kgrams", m.kgrams)
					self.tran[successive,text[j-1+k]] = float(np.random.randint(1, 100, size=1)[0])
					if successive not in self.kgrams.keys():
						self.kgrams[successive] = self.tran[successive,text[j-1+k]]
					else:
						self.kgrams[successive] += self.tran[successive,text[j-1+k]]
		
	def order(self):
		# order k of Markov model
		return self.k

	def freq(self, kgram):
		# number of occurrences of kgram in text
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
		
		for key, value in self.tran.items():
			# key[0]: k-successive activities
			# key[1]: k+1th activity
			# value: the count from k-successive activities to the k+1th activity
			self.prob_tran_matrix[key] = float(value)/self.freq(key[0])
		return self.prob_tran_matrix

	def rand(self, kgram):
		# random character following given kgram
		assert len(kgram) == self.k
		# (check if kgram is of length k.
		Z = sum([self.tran[kgram, alph] for alph in self.alph])
		# print(np.array([self.tran[kgram, alph] for alph in self.alph])/Z)
		return np.random.choice(self.alph, 1, p=np.array([self.tran[kgram, alph] for alph in self.alph])/Z)

	def gen(self, kgram, T):
		# generate a list of length T activities
		# initial k activities/strings. 
		assert len(kgram) == self.k
		# by simulating a trajectory through the corresponding
		activity_list = kgram
		# Markov chain. The first k activities of the newly
		for i in range(T):
			# generated list should be the argument kgram.
			#print kgram, c
			# check if kgram is of length k.
			c =  self.rand(text)[0]
			# Assume that T is at least k.
			kgram = kgram[1:] + (c,)
			activity_list += c
		return activity_list
=