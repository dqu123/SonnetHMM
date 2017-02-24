import argparse
import pickle

from sonetto.HMM import unsupervised_HMM
from sonetto.parser import SonnetParser

DATA_DIR = 'data/'
FILE = 'shakespeare'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Makes Shakespearean Sonnets')
    sp = SonnetParser()
    sp.parse('{}{}.txt'.format(DATA_DIR, FILE))

    hmm = unsupervised_HMM(sp, 10)
    print hmm.generate_emission(84)

    with open('models/hmm_{}.pickle'.format(FILE), 'wb') as f:
        pickle.dump(hmm, f)
