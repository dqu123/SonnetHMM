########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Andrew Kang, Avishek Dutta
# Description:  Set 5 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (i.e. run `python 2G.py`) to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import sys

# Constants for unsupervised learning
NUM_ITERATIONS = 1000

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O, parser):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.
            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.
            parser:     SonnetParser with HMM mapping and cmudict.

        Parameters:
            L:          Number of states.
            D:          Number of observations.
            A:          The transition matrix.
            O:          The observation matrix.
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''
        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for i in range(self.L)]

        self.parser = parser

    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    Output sequence corresponding to x with the highest
                        probability.
        '''
        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for i in range(self.L)] for j in range(M + 1)]
        seqs = [['' for i in range(self.L)] for j in range(M + 1)]

        # Base case
        for i in range(self.L):
            probs[1][i] = self.A_start[i] * self.O[i][x[0]]
            seqs[1][i] += str(i)

        # Recursive case
        for length in range(2, M + 1):
            # Search all possible best combinations.
            for new_end in range(self.L):
                max_prob = 0
                best_seq = None
                for old_end in range(self.L):
                    cur_prob = (probs[length - 1][old_end] * 
                                self.A[old_end][new_end] *
                                self.O[new_end][x[length - 1]])
                    if cur_prob > max_prob:
                        max_prob = cur_prob
                        best_seq = seqs[length - 1][old_end] + str(new_end)
                
                probs[length][new_end] = max_prob
                seqs[length][new_end] = best_seq

        # Select best sequence
        max_prob = 0
        max_seq = ''
        for i in range(self.L): 
            cur_probs = probs[M][i]
            if cur_probs > max_prob:
                max_prob = cur_probs
                max_seq = seqs[M][i]

        return max_seq

    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.
                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''
        M = len(x)      # Length of sequence.
        alphas = [[0. for i in range(self.L)] for j in range(M + 1)]

        # Base case
        for z in range(self.L):
            alphas[1][z] = self.O[z][x[0]] * self.A_start[z]

        # Recursive case 
        for length in range(2, M + 1):
            for z in range(self.L):
                for j in range(self.L):
                    alphas[length][z] += alphas[length - 1][j] * self.A[j][z]
                alphas[length][z] *= self.O[z][x[length - 1]]

            if normalize:
                normal = 0.
                for idx in range(self.L):
                    normal += alphas[length][idx]

                if normal != 0:
                    for idx in range(self.L):
                        alphas[length][idx] /= normal

        return alphas

    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.
                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.
        '''
        M = len(x)      # Length of sequence.
        betas = [[0. for i in range(self.L)] for j in range(M + 1)]

        # Base case
        for z in range(self.L):
            betas[M][z] = 1.0

        # Recursive case
        for length in range(M - 1, -1, -1):
            for z in range(self.L):
                for j in range(self.L):
                    betas[length][z] += (betas[length + 1][j] * 
                                         self.A[z][j] * 
                                         self.O[z][x[length]])
            if normalize:
                normal = 0.
                for idx in range(self.L):
                    normal += betas[length][idx]
                if normal != 0:
                    for idx in range(self.L):
                        betas[length][idx] /= normal
        return betas

    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.
            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.
                        Note that the elements in X line up with those in Y.
        '''
        N = len(X)
        assert N == len(Y)

        # Calculate each element of A using the M-step formulas.
        for a in range(self.L):
            for b in range(self.L):
                numerator = 0
                denominator = 0
                for j in range(N):
                    for i in range(len(X[j])):
                        if i != (len(X[j]) - 1) and Y[j][i] == b:
                            if Y[j][i + 1] == a:
                                numerator += 1
                            denominator += 1
                self.A[b][a] = numerator / float(denominator)

        # Calculate each element of O using the M-step formulas.
        for w in range(self.D):
            for z in range(self.L):
                numerator = 0
                denominator = 0
                for j in range(N):
                    for i in range(len(X[j])):
                        if X[j][i] == w and Y[j][i] == z:
                            numerator += 1
                        if Y[j][i] == z:
                            denominator += 1
                self.O[z][w] = numerator / float(denominator)


    def unsupervised_learning(self, X):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.
        '''

        N = len(X)

        def marginal(alphas, betas, length, a):
            """Helper function for computing P(y^i = b)"""
            numerator = alphas[length][a] * betas[length][a]
            denominator = 0

            for a_ in range(self.L):
                denominator += alphas[length][a_] * betas[length][a_]
            
            result = 0
            if numerator != 0 and denominator != 0:
                result = numerator / float(denominator)
            return result

        def marginal_ab(alphas, betas, length, a, b, x):
            """Helper function for computing P(y^i = a, y^{i+1} = b)"""
            numerator = (alphas[length][a] * self.O[b][x[length + 1]] *
                             self.A[a][b] * betas[length + 1][b])
            denominator = 0

            for a_ in range(self.L):
                for b_ in range(self.L):
                    denominator += (alphas[length][a_] * 
                                    self.O[b_][x[length + 1]] *
                                    self.A[a_][b_] *
                                    betas[length + 1][b_])
            result = 0
            if numerator != 0 and denominator != 0:
                result = numerator / float(denominator)
            return result

        # Repeat for NUM_ITERATIONS
        for iteration in range(NUM_ITERATIONS):
            sys.stderr.write('Iteration: {}\n'.format(iteration))
            temp_A = [[0. for i in range(self.L)] for j in range(self.L)]
            temp_O = [[0. for i in range(self.D)] for j in range(self.L)]

            margin_list = []
            alphas_list = []
            betas_list = []
            for idx, x in enumerate(X):
                margin_list.append([])
                alphas = self.forward(x, True)
                betas = self.backward(x, True)
                alphas_list.append(alphas)
                betas_list.append(betas)

                for length in range(len(x)):
                    margin_list[idx].append([])
                    for state in range(self.L):
                        margin_list[idx][length].append(marginal(alphas, betas, length, state))

            # Compute temp_A
            for a in range(self.L):
                for b in range(self.L):
                    numerator = 0
                    denominator = 0
                    for j in range(N):
                        for i in range(len(X[j]) - 1):
                            numerator += marginal_ab(alphas_list[j], betas_list[j],
                                                     i, b, a, X[j])
                            denominator += margin_list[j][i][b]
                    temp_A[b][a] = numerator / float(denominator)

            # Compute temp_O
            for w in range(self.D):
                for z in range(self.L):
                    numerator = 0
                    denominator = 0
                    for j in range(N):
                        for i in range(len(X[j])):
                            if X[j][i] == w:
                                numerator += margin_list[j][i][z]
                            denominator += margin_list[j][i][z]
                    temp_O[z][w] = numerator / float(denominator)

            # Write A.
            for a in range(self.L):
                for b in range(self.L):
                    self.A[b][a] = temp_A[b][a]

            # Write O
            for w in range(self.D):
                for z in range(self.L):
                    self.O[z][w] = temp_O[z][w] 

    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a string.
        '''
        emission = ''

        # Initialize list of states
        y = [0 for i in range(M + 1)]
        y[0] = random.randint(0, self.L - 1)

        for i in range(1, M + 1):
            cur_prob = 0
            seed = random.random()
            
            for new_y in range(self.L):
                cur_prob += self.A[y[i - 1]][new_y]
                if cur_prob > seed:
                    y[i] = new_y
                    break

            cur_prob = 0
            seed = random.random()
            for x in range(self.D):
                cur_prob += self.O[y[i]][x]
                if cur_prob > seed:
                    if i > 1: 
                        emission += ' '
                    emission += self.parser.num_to_word[x]
                    # TODO: Use syllables instead of word count
                    if i % 6 == 0:
                        emission += '\n'
                    break

        return emission

    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''
        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the output sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any output sequence, i.e. the
        # probability of x.
        prob = sum(alphas[-1])
        return prob

    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''
        betas = self.backward(x)

        # beta_j(0) gives the probability of the output sequence. Summing
        # this over all states and then normalizing gives the total
        # probability of x paired with any output sequence, i.e. the
        # probability of x.
        prob = sum(betas[0]) / self.L

        return prob

def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learing.

    Arguments:
        X:          A list of variable length emission sequences 
        Y:          A corresponding list of variable length state sequences
                    Note that the elements in X line up with those in Y
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrices A and O.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(parser, n_states):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.
        n_states:   Number of hidden states to use in training.
    '''
    X = parser.hmm_lines

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrices A and O.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O, parser)
    HMM.unsupervised_learning(X)

    return HMM
