from wordcloud import WordCloud
from numpy.random import choice
import pickle

def generate_cloud(hmm, hidden_state_num):
    prob_dist = hmm.O[hidden_state_num]
    words = []
    for _ in range(500):
        c = choice(range(hmm.D), p=prob_dist)
        words.append(hmm.parser.num_to_word[c])
    text = ' '.join(words)
    cloud = WordCloud(background_color='white', max_words=70).generate(text)
    import matplotlib.pyplot as plt
    plt.imshow(cloud)
    plt.axis('off')
    plt.show()

def gen_from_pickle(pickle_filename):
    f = open(pickle_filename, 'r')
    hmm = pickle.load(f)
    generate_cloud(hmm, 0)

