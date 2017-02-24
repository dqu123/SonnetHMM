from enum import Enum
from nltk.corpus import cmudict

diction = cmudict.dict()

class Stress(Enum):
    NO_STRESS = 0
    STRESS = 1
    UNSTRESS = 2

def valid_meter(word, num_syl):
    cur_syl = num_syl + 1

def num_syllables(word):
    pass

def get_stress(word):
    """Converts a word into a list of stresses"""
    if word not in diction:
        return None
    pronuncs = diction[word]

    stresses = []
    for pronunc in pronuncs:
        stresses.append([])

        for syl in pronunc:
            if '1' in syl:
                stresses[-1].append(Stress.STRESS)
            elif '2' in syl:
                stresses[-1].append(Stress.UNSTRESS)
    return stresses
