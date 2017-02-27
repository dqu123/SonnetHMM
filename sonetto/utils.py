from enum import Enum
from nltk.corpus import cmudict
import constants

diction = cmudict.dict()

class Stress(Enum):
    """Different types of stresses."""
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

def rhymes(word1, word2):
    """Checks whether two words rhyme."""
    if (word1 not in diction) or (word2 not in diction):
        return False
    pronuncs1 = diction[word1]
    pronuncs2 = diction[word2]

    rhyme = False
    for p1 in pronuncs1:
        for p2 in pronuncs2:
            if strip_num(p1[-1]) == strip_num(p2[-1]):
                if (len(p1) == 1 or len(p2) == 1 or
                        strip_num(p1[-2]) == strip_num(p2[-2])):
                    rhyme = True
                    break
    return rhyme


def is_valid(word, meter_valid, new_syl, num_lines, line_end):
    """Determines if the word is valid."""
    rhyme_valid = True

    if (num_lines % 4) in [2, 3] and new_syl == constants.SYLLABLES_PER_LINE:
        rhyme_valid = rhymes(word, line_end[num_lines - 2])
    elif num_lines == constants.COUPLET_LINE and new_syl == constants.SYLLABLES_PER_LINE:
        rhyme_valid = rhymes(word, line_end[num_lines - 1])

    return (meter_valid and rhyme_valid and
            new_syl <= constants.SYLLABLES_PER_LINE)

def strip_num(string):
    """Remove numbers from a string."""
    return ''.join([char for char in string if not char.isdigit()])
