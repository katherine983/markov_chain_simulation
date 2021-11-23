import numpy as np 
import math
from collections import defaultdict  
from itertools import product
from string import ascii_lowercase
from pathlib import Path
import datetime
import json
from ast import literal_eval
import sys

def create_output_file_path(root_dir=None, out_dir='MC_matrices', out_name=None, overide=False):
    """ Create path object for the output filepath for the output file.
    
    Optional Keyword arguments:
    root_dir -- string literal or pathlike object refering to the target root directory
    out_dir -- string literal representing target directory to be appended to root if root is not target directory
    out_name -- string literal representing the filename for the output file

    Return : path object to save simulation data in

    """
    if not root_dir:
        root_dir = Path.cwd()
    dir = Path(root_dir, out_dir)
    if not dir.is_dir():
        dir.mkdir(parents=True)
    if not out_name:
        out_name = f"MC_model_{datetime.now().isoformat(timespec='seconds')}.txt"
    out_path = dir / out_name
    if out_path.exists():
        if overide == False:
            raise Exception("Output file already exists. Please enter unique filepath information or use overide==True.")
    else:
        return out_path

class GenMarkovTransitionProb:      

    def __init__(self, text, k):          
        '''create a Markov model of order k from given activities         
        Assume that text has length at least k.'''
        # k is the order, type - int
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
        # stationary probability distribution of the markov matrix
        self.stat_prob = defaultdict(int)
        # marginal frequency of each alph in the matrix
        self.stat_freq = defaultdict(int)
        self.ent_rate = None
        # if self.k == 0:
        # k_successive = [() for a in product(text, repeat=0)]
        # use k_successive to generate the past states. 
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
                    #self.tran[successive,text[j-1+k]]=1000
                    self.tran[successive,text[j-1+k]] = float(np.random.randint(1, 100, size=1)[0])
                    #add count to marginal frequency of kgram
                    if successive not in self.kgrams.keys():
                        self.kgrams[successive] = self.tran[successive,text[j-1+k]]
                    else:
                        self.kgrams[successive] += self.tran[successive,text[j-1+k]]
                    #add count to marginal frequency of the alph at text[j-1+k]
                    if text[j-1+k] not in self.stat_freq.keys():
                        self.stat_freq[text[j-1+k]]= self.tran[successive,text[j-1+k]]
                    else:
                        self.stat_freq[text[j-1+k]]=self.tran[successive,text[j-1+k]]
        
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

    def freq3(self, alph):
        assert alph in self.stat_freq.keys()
        return self.stat_freq[alph]

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
        activity_list = list(kgram)
        # Markov chain. The first k activities of the newly
        for i in range(T):
            # generated list should be the argument kgram.
            #print kgram, c
            # check if kgram is of length k.
            c =  self.rand(kgram)[0]
            # Assume that T is at least k.
            kgram = tuple(kgram[1:]) + (c,)
            activity_list += c
        return activity_list
        
    def stationary_dist(self):
        # added function for entropy_Monte_Carlo
        Z = sum(self.stat_freq.values())
        
        for alph in self.alph:
            self.stat_prob[alph] = self.freq3(alph)/Z
        return self.stat_prob

    def entropy_rate(self):
        # Added function for entropy_Monte_Carlo.
        # Calculates basic entropy rate of the Markov process
        # (equivalent to the Approximate Entropy) using the transition probabilities
        # stored in self.prob_tran_matrix and the stationary probabilities
        # Entropy rate formula as -Sum over all i,j Pr(i)*Pr(i,j)ln(Pr(i,j))
        # where Pr(i) is the stationary probability of state i and Pr(i, j) is the transition probability 
        # of i given j. 
        ent_rate = []
        self.prob_tran()
        self.stationary_dist()
        for key, value in self.prob_tran_matrix.items():
            # key[0] = kgram, a tuple
            # key[1] = alph succeeding kgram, single unicode character
            # value =  transition probability from kgram to alph
            ent_rate += (self.stat_prob[key[1]]*self.prob_tran_matrix[key]*math.log(self.prob_tran_matrix[key]))
        self.ent_rate = -sum(ent_rate)
        return self.ent_rate

        

    def gen_sample(self, kgram, T, seed=None, generator='default'):
        # Updated method for generating a random sample from the Markov transition matrix combining
        # the functionality of the self.gen() and self.rand() methods.
        # This method replaces the use of legacy np.random.choice (deprecated after v1.16)
        # with current best practice numpy random number generation. 
        # gen_sample() allows for user specified BitGenerator and seed for reproducibility.
        # Default BitGenerator is to use the default numpy generator via using np.random.default_rng().
        # This uses the current default bitgenerator, allowing it to adapt as default generator is updated
        # in numpy package. Otherwise generator kwarg expects a bit_generator func from np.random
        # Function also saves seed in self.seed attribute for access later.
        # activity_list is a list containing the randomly generated markov chain

        # Set seed sequence using value given to seed arg.
        # If value is an entropy value from a previous seed sequence, ss will be identical to
        # the previous seed sequence.
        ss = np.random.SeedSequence(seed)
        # save entropy so that seed sequence can be reproduced later
        self.seed = ss.entropy
        if generator == 'default':
            rng = np.random.default_rng(ss)
        else:
            rng = np.random.Generator(generator(ss))
        assert len(kgram) == self.k
        # initiate list to contain generated markov chain
        activity_list = list(kgram)
        for i in range(T):
            #get marginal frequency of kgram
            Z = sum([self.tran[kgram, alph] for alph in self.alph])
            c = rng.choice(self.alph, 1, p=np.array([self.tran[kgram, alph] for alph in self.alph])/Z)[0]
            activity_list += c
            kgram = tuple(kgram[1:]) + (c,)
        return activity_list
    
    def dump(self, outpath, **kwargs):
        """Writes MC matrix data to a json file. Opens the file, writes, and closes it.
        
        outpath -- takes a path object pointing to a file location for the output to be written to. 
        **kwargs -- any additional arguments to be fed to json.dump()
        """
        with open(outpath) as fouthand:
            data = {'MC_order' : self.k, 'Alphabet' : self.alph, 'Transition_Matrix_Frequencies' : {str(k) : v for k, v in self.tran.items()}, 
            'Stationary_Distribution' : self.stat_freq, 'Entropy_Rate' : self.ent_rate, 'Date_created' : datetime.now().isoformat(timespec='seconds')}
            json.dump(data, fouthand, **kwargs)




def genMCmodel(root_dir, order_i, states_temp):
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
    None.

    """
    MC_model = GenMarkovTransitionProb(states_temp, order_i)
    MC_model.entropy_rate()
    fout_file_path = create_output_file_path(root_dir, r'MC_matrices', f'Order{order_i}Alph{MC_model.m}ER{MC_model.ent_rate}.json')
    MC_model.dump(fout_file_path)


if __name__ == '__main__':
    # root_dir = "/Users/BeiyuLin/Desktop/five_datasets/"
    root_dir = "./"
    # order_i is the given order of markov chain
    order_i = int(sys.argv[1])
    # states_nums is the number of states in the process (i.e. the size of the alphabet)
    states_nums = int(sys.argv[2])
    #start_sample_size = int(sys.argv[3])
    #end_sample_size = int(sys.argv[4])
    # based on randomly generated transition matrix.
    # http://www.iaeng.org/publication/WCECS2014/WCECS2014_pp899-901.pdf
    # 
    states_temp = [chr(ord('a')+i) for i in range(states_nums)]
    # = ['a','b','c','d','e','f','g','h','u']
    # generate 1000 random transition matrix
    genMCmodel(root_dir, order_i, states_temp)
