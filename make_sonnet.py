import argparse
import pickle

from sonetto.HMM2 import unsupervised_HMM
from sonetto.parser import SonnetParser

DATA_DIR = 'data/'
FILE = 'shakespeare'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Makes Shakespearean Sonnets')
    parser.add_argument('-s', '--states', type=int)
    parser.add_argument('-i', '--iters', type=int)
    parser.add_argument('-f', '--file')

    args = parser.parse_args()

    sp = SonnetParser()
    sp.parse('{}{}.txt'.format(DATA_DIR, FILE))

    hmm = unsupervised_HMM(sp, n_states=args.states, n_iters=args.iters)
    print hmm.generate_emission(14)

    with open('models/{}'.format(args.file), 'wb') as f:
        pickle.dump(hmm, f)
