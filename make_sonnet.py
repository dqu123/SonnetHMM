import argparse
from sonetto.HMM import unsupervised_HMM
from sonetto.parser import SonnetParser

DATA_DIR = 'data/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Makes Shakespearean Sonnets')
    sp = SonnetParser()
    sp.parse('{}{}.txt'.format(DATA_DIR, 'mini_shakespeare'))

    hmm = unsupervised_HMM(sp, 10)
    print hmm.generate_emission(84)
