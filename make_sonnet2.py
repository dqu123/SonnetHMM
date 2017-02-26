import argparse
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file

from sonetto.parser import SonnetParser

DATA_DIR = 'data/'
FILE = 'behemoth'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Makes Shakespearean Sonnets')
    parser.add_argument('-s', '--states', type=int)
    parser.add_argument('-i', '--iters', type=int)
    parser.add_argument('-f', '--file')

    args = parser.parse_args()

    sp = SonnetParser()
    sp.parse('{}{}.txt'.format(DATA_DIR, FILE))

    seed = "from fairest creatures we desire increase that thereby beauty's rose might never die".split(' ')
    print seed
    SEQ_LEN = len(seed)
    data = [item for sublist in sp.lines for item in sublist]
    print data

    seq = []
    nxt = []
    step = 5
    for i in range(0, len(data) - SEQ_LEN, step):
        seq.append(data[i:i + SEQ_LEN])
        nxt.append(data[i + SEQ_LEN])



    inp = np.zeros(((len(seq), SEQ_LEN, sp.word_count)))
    out = np.zeros(((len(seq), sp.word_count)))

    for i in range(len(seq)):
        for j in range(len(seq[i])):
            inp[i][j][sp.word_to_num[seq[i][j]]] = 1

        out[i][sp.word_to_num[nxt[i]]] = 1


    model = Sequential()
    model.add(LSTM(128, input_shape = (SEQ_LEN, sp.word_count)))
    model.add(Dense(sp.word_count))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer =optimizer)

    model.fit(inp, out, batch_size=128, nb_epoch=20, verbose=2)
    newSeed = list(seed)
    generated = list(seed)
    for i in range(100):
        genInput = np.zeros((1, SEQ_LEN, sp.word_count))
        for j in range(len(newSeed)):
            genInput[0][j][sp.word_to_num[newSeed[j]]] = 1 

        dist = model.predict(genInput)[0]
        dist = np.asarray(dist).astype('float64')
        logs = np.log(dist)
        softmax = np.exp(logs) / np.sum(np.exp(logs))
        multi = np.random.multinomial(1, softmax)
        sample = np.argmax(multi)

        pred = sp.num_to_word[sample]

        generated.append(pred)

        newSeed = newSeed[1: ]
        newSeed.append(pred)


    print ' '.join(generated)





    # hmm = unsupervised_HMM(sp, n_states=args.states, n_iters=args.iters)
    # print hmm.generate_emission(84)

    # with open('models/{}'.format(args.file), 'wb') as f:
    #     pickle.dump(hmm, f)
