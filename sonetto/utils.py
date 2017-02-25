from enum import Enum
from nltk.corpus import cmudict

diction = cmudict.dict()

class Stress(Enum):
    NO_STRESS = 0
    STRESS = 1
    UNSTRESS = 2

def valid_meter(word, num_syl):
    """Determine whether a word is valid based on meter."""
    pronuncs = get_stress(word)
    index = None
    valid = False
    if pronuncs is None:
        return valid, index

    for i, pronunc in enumerate(pronuncs):
        valid_pronunc = True
        for idx, stress in enumerate(pronunc):
            if (((idx + num_syl) % 2 == 0 and stress == Stress.STRESS) or
                    ((idx + num_syl) % 2 == 1 and stress == Stress.UNSTRESS)):
                valid_pronunc = False
        if valid_pronunc:
            valid = True
            index = i
            break
    return valid, index

def get_stress(word):
    """Converts a word into a list of stresses"""
    if word not in diction:
        return None
    pronuncs = diction[word]

    stresses = []
    for pronunc in pronuncs:
        stresses.append([])

        for syl in pronunc:
            if '0' in syl:
                stresses[-1].append(Stress.NO_STRESS)
            elif '1' in syl:
                stresses[-1].append(Stress.STRESS)
            elif '2' in syl:
                stresses[-1].append(Stress.UNSTRESS)
    return stresses
