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
        out_name = f"MC_model_{datetime.datetime.now().isoformat(timespec='seconds')}.txt"
    out_path = dir / out_name
    if out_path.exists():
        if overide == False:
            raise Exception("Output file already exists. Please enter unique filepath information or use overide==True.")
    else:
        return out_path

def get_data_file_path(root_dir=None, out_dir='MC_matrices', out_name=None):
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
        out_name = f"MC_model_{datetime.datetime.now().isoformat(timespec='seconds')}.txt"
    out_path = dir / out_name
    if not out_path.exists():
        raise Exception("File does not exist. Please enter existing filepath.")
    else:
        return out_path

class GenMarkovTransitionProb:

    def __init__(self, text, k, transition_freq_matrix=None):
        '''create a Markov model of order k from given activities in text.
        Assume that text has length at least k.
        Parameters
        ----------
        text : STRING OR ITERABLE
            ITERABLE CONTAINING THE STATES OF THE MARKOV MODEL.
        k : INT
            ORDER OF THE MARKOV MODEL.
        transition_freq_matrix : DICT, default None.
            DICT OBJECT CONTAINING THE TRANSITION MATRIX WITH COUNT NOT PROBABILITY.
        '''

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
        # functions as the stationary distribution frequencies of the markov matrix
        self.kgrams = defaultdict(int)
        # sample size
        self.n = len(text)
        # text += text[:k]
        # stationary probability distribution of the markov matrix
        self.stat_prob = defaultdict(int)
        # marginal frequency of each alph in the matrix
        self.alph_freq = defaultdict(int)
        self.ent_rate_est = None
        self.ent_rate = None
        # if self.k == 0:
        # k_successive = [() for a in product(text, repeat=0)]
        # use k_successive to generate the past states.
        if not transition_freq_matrix:
            self._new_transition_matrix(text)
        else:
            self._load_transition_matrix(transition_freq_matrix)

    def _load_transition_matrix(self, transition_freq_matrix):
        self.tran = transition_freq_matrix
        for key, value in self.tran.items():
            # key[0] = kgram
            # key[1] = alph
            # value = count of transition from kgram to alph
            if key[0] not in self.kgrams.keys():
                self.kgrams[key[0]] = value
            else:
                self.kgrams[key[0]] += value
            if key[0] not in self.alph_freq.keys():
                self.alph_freq[key[1]] = value
            else:
                self.alph_freq[key[1]] += value
        return self

    def _new_transition_matrix(self, text):
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
                # successive is the past states, text[j-1+k] is the next state.
                # tran is the transition matrix.
                # e.g.: {(('adfg', 'dafae', 'dafae'), 'dafae'): 1.0} print("m.tran", m.tran, len(m.tran))
                # m.kgrams is a dictionary that shows the count of each k successive activities
                # e.g.: {('abc', 'abc', 'abc'): 1} print("m.kgrams", m.kgrams)
                #self.tran[successive,text[j-1+k]]=1000
                self.tran[successive,text[j]] = float(np.random.randint(1, 10000000, size=1)[0])
                #add count to marginal frequency of kgram
                if successive not in self.kgrams.keys():
                    self.kgrams[successive] = self.tran[successive,text[j]]
                else:
                    self.kgrams[successive] += self.tran[successive,text[j]]
                #add count to marginal frequency of the alph at text[j-1+k]
                if text[j] not in self.alph_freq.keys():
                    self.alph_freq[text[j]]= self.tran[successive,text[j]]
                else:
                    self.alph_freq[text[j]]=self.tran[successive,text[j]]

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
        assert alph in self.alph_freq.keys()
        return self.alph_freq[alph]

    def prob_tran(self):
        # transition matrix with probability

        for key, value in self.tran.items():
            # key[0]: k-successive activities
            # key[1]: k+1th activity
            # value: the count from k-successive activities to the k+1th activity
            self.prob_tran_matrix[key] = float(value)/self.freq(key[0])
        return self.prob_tran_matrix

    def _asarray(self):
        """
        Added function for mc_entropy package.
        Takes the transition frequencies stored in self.tran and formats as 2d
        numpy array. Column indices correspond to index of the kgram in
        self.kgrams.keys()and row indices
        of the matrix correspond to the index of each state in self.alph.

        Returns
        -------
        2d numpy array. For each column, j, in the array, the entries in each
        row, i, are the transition probabilities from the state at index j in
        self.alph to the state at index i in self.alph.

        """
        #get list of all kgrams and sort list so that array is always in expected order
        k_successive = list(self.kgrams.keys())
        k_successive.sort()
        self.alph.sort()
        #arrayvecs is empty list to hold the row vectors assembled from the iteration below
        arrayvecs = []
        for a in self.alph:
            row = []
            for kgram in k_successive:
                row.append(self.tran[kgram,a])
            arrayvecs.append(row)
        array = np.array(arrayvecs)
        #print((self.m, self.m**self.k))
        #print(array.shape)
        assert array.shape == (self.m, self.m**self.k)
        return array

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
        """
        Added function for mc_entropy package.

        Returns
        -------
        self.stat_prob - DEFAULTDICT
            DICTIONARY WHERE EACH KEY, VALUE PAIR IS A KGRAM IN THE SET OF ALL
            POSSIBLE KGRAMS AND ITS CORRESPONDING STATIONARY PROBABILITY IN THE
            MARKOV PROCESS.

        """
        # added function for mc_entropy
        Z = sum(self.kgrams.values())
        for kgram, freq in self.kgrams.items():
            self.stat_prob[kgram] = freq/Z
        return self.stat_prob

    def eig_steadystate(self):
        """
        Added function for mc_entropy package.
        Calculates the steady-state probability vector of the first order MC matrix
        assuming the matrix is a regular stochastic matrix (square, irreducible and aperiodic).

        Method is based on the theory that a steady-state probability vector, q,
        of a stochastic matrix, P, can be calculated by solving the matrix equation
        Pq = q. Thus, q is a right eigenvector of P corresponding to the eigenvalue
        lambda=1.

        Returns
        -------
        q : ndarray containing the steady-state probabilities of each state in self.alph.

        """
        #Get the transition frequency matrix as a numpy array.
        mcarray = self._asarray()
        #Convert the transition frequency matrix to a transition probability
        #matrix (stochastic matrix) whose columns sum to 1.
        P = mcarray/mcarray.sum(axis=0, keepdims=1)
        #generate eigenvalues and eigenvectors corresponding to P
        evals, evecs = np.linalg.eig(P)
        #print(evals)
        #print(evecs)
        #find index in evals associated with the eigenvalue of 1
        eig_index = np.where(evals.real.round(1) >= 1.)
        #we use the column evecs[:,eig_index[0]] as a basis for the eigenspace associated with lambda=1
        qbasis = evecs[:,eig_index].flatten()
        #scale qbasis to be a probability vector with entries summing to 1
        q = qbasis/qbasis.sum()
        #print(q)
        #print(q.real)
        assert q.imag.all() == 0.
        return q.real, P

    def entropy_rate_est(self):
        """ Added function for mc_entropy.

        Estimates basic entropy rate (equivalent to the Approximate Entropy)
        of the Markov process using the transition probabilities stored in
        self.prob_tran_matrix and the stationary probabilities.
        Uses Entropy rate formula as -Sum over all i,j Pr(i)*Pr(i,j)ln(Pr(i,j))
        where Pr(i) is the stationary probability of state i, which corresponds
        to kgram in this case, and Pr(i, j) is the transition probability
        of i to j, which corresponds to the transition from kgram to alph.

        Can accommodate non-square transition matrices.

        Returns
        -------
        self.ent_rate_est.

        """
        # ent_rate is list to store Pr(i)*Pr(i,j)ln(Pr(i,j)) for each i, j.
        ent_rate = []
        self.prob_tran()
        self.stationary_dist()
        for key, value in self.prob_tran_matrix.items():
            # key[0] = kgram, a tuple
            # key[1] = alph succeeding kgram, single unicode character
            # value =  transition probability from kgram to alph
            #stat_prob = self.stat_prob[key[0]]
            #print(key[0])
            #print(key[1])
            ent_rate.append(self.stat_prob[key[0]]*value*math.log(value))
        self.ent_rate_est = -sum(ent_rate)
        return self.ent_rate_est

    def entropy_rate(self):
        """
        Added function for mc_entropy.

        Calculates entropy rate using the transition probability matrix and
        steady state vector given by the GenMarkovTransitionProb.eig_steadystate() method.
        Function takes advantage of efficient numpy array broadcasting.
        Entropy rate calculated as -Sum over all i,j of qj*Pij*lnPij where Pij
        is the transition probability of j to i, qj is the stationary probability
        of state i. Assumes a transition matrix in the form of a left stochastic
        matrix whose columns sum to 1.

        Cannot accommodate non-square transition matrices.

        Returns
        -------
        self.ent_rate - INTEGER
            entropy rate as integer value.

        """
        q, P = self.eig_steadystate()
        self.ent_rate = np.negative((q*P*np.log(P)).sum())
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

        outpath : PATH-LIKE OBJECT OR STRING LITERAL OF A FILE PATH
            takes a path object pointing to a file location for the output to be written to.
        **kwargs -- any additional arguments to be fed to json.dump()
        """
        with open(outpath, 'w+') as fouthand:
            data = {'MC_order' : self.k,
                    'Alphabet' : self.alph,
                    'Transition_Matrix_Frequencies' : {str(k) : v for k, v in self.tran.items()},
                    'Stationary_Distribution_Frequencies' : {str(k) : v for k, v in self.kgrams.items()},
                    'Entropy_Rate' : self.ent_rate,
                    'Date_created' : datetime.datetime.now().isoformat(timespec='seconds')}
            json.dump(data, fouthand, **kwargs)

    @classmethod
    def load(cls, filepath, **kwargs):
        """
        Contructor method to construct a GenMarkovTransitionProb object from
        json file created by the GenMarkovTransitionProb.dump method.

        Parameters
        ----------
        filepath : PATH-LIKE OBJECT OR STRING LITERAL OF A FILE PATH
            DESCRIPTION.
        **kwargs : DICT
            OPTIONAL DICT OF KEYWORD ARGUMENTS TO BE PASSED TO json.load().

        Returns
        -------
        GenMarkovTransitionProb object with same transition matrix as one
        saved in the json file.

        """
        with open(filepath, 'r') as fhand:
            data = json.load(fhand, **kwargs)
            transition_freq_matrix = {literal_eval(k) : v for k, v in data['Transition_Matrix_Frequencies'].items()}
            text = data['Alphabet']
            return cls(text, data['MC_order'], transition_freq_matrix)

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
    MC_model = GenMarkovTransitionProb(states_temp, order_i)
    MC_model.entropy_rate()
    fout_file_path = create_output_file_path(root_dir, r'MC_matrices', f'Order{order_i}Alph{MC_model.m}ER{MC_model.ent_rate:.4f}.json')
    MC_model.dump(fout_file_path)
    return MC_model

def get_model(file_path):
    MC_model_copy = GenMarkovTransitionProb.load(file_path)
    return MC_model_copy

def gen_sample(MC_model, kgram, T, generator='default', seed=None):
    # Standalone function that takes a GenMarkovTransitionProb instance and
    # generates a random sample from the Markov transition matrix.
    # Combines the functionality of the self.gen() and self.rand() methods.
    # Expects generator to be an instantiated BitGenerator object, otherwise expects no generator
    # to be provided and will instantiate one using the np.random.default_rng() function.
    # gen_sample() allows for user specified BitGenerator and seed for reproducibility.
    # Default BitGenerator is to use the default numpy generator via using np.random.default_rng().
    # This uses the current default bitgenerator, allowing it to adapt as default generator is updated
    # in numpy package. Otherwise generator kwarg expects a bit_generator func from np.random
    # Function also saves seed in self.seed attribute for access later.
    # activity_list is a list containing the randomly generated markov chain
    # Returns the randomly generated sequence with the first k symbols dropped
    # so as to remove kgram from the sequence.

    if generator == 'default':
        # Set seed sequence using value given to seed arg.
        # If value is an entropy value from a previous seed sequence, ss will be identical to
        # the previous seed sequence.
        ss = np.random.SeedSequence(seed)
        # save entropy so that seed sequence can be reproduced later
        MC_model.seed = ss.entropy
        rng = np.random.default_rng(ss)
    else:
        rng = np.random.default_rng(generator)
    assert len(kgram) == MC_model.k
    # initiate list to contain generated markov chain
    activity_list = list(kgram)
    for i in range(T):
        #get marginal frequency of kgram
        Z = sum([MC_model.tran[kgram, alph] for alph in MC_model.alph])
        c = rng.choice(MC_model.alph, 1, p=np.array([MC_model.tran[kgram, alph] for alph in MC_model.alph])/Z)[0]
        activity_list.append(c)
        kgram = tuple(kgram[1:]) + (c,)
    return activity_list[len(kgram):]

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
    MC1 = genMCmodel(root_dir, order_i, states_temp)
    #print(MC1.tran)
    #fpath = get_data_file_path(root_dir, r'MC_matrices', f'Order{MC1.k}Alph{MC1.m}ER{MC1.ent_rate:.4f}.json')
    #MC2 = getMCmodel(fpath)
    #MC1 == MC2
    #MC1_freq = MC1._asarray()
    estMC1P = MC1.prob_tran()
#%%
    estMC1q = MC1.stationary_dist()
#%%
    MC1q, MC1P = MC1.eig_steadystate()
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
