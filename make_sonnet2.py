import argparse

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file

from sonetto.parser import SonnetParser

DATA_DIR = 'data/'
FILE = 'mini_shakespeare'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Makes Shakespearean Sonnets')
    parser.add_argument('-s', '--states', type=int)
    parser.add_argument('-i', '--iters', type=int)
    parser.add_argument('-f', '--file')

    args = parser.parse_args()

    sp = SonnetParser()
    sp.parse('{}{}.txt'.format(DATA_DIR, FILE))

    SEQ_LEN = 15
    flatten = ' '.join(sp.lines)
    for word in flatten.split():



    model = Sequential()
    model.add(LSTM(128, input_shape = (SEQ_LEN, len(char))))
    model.add(Dense(len(char)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer)






    # hmm = unsupervised_HMM(sp, n_states=args.states, n_iters=args.iters)
    # print hmm.generate_emission(84)

    # with open('models/{}'.format(args.file), 'wb') as f:
    #     pickle.dump(hmm, f)
