import re
from nltk.corpus import cmudict

from constants import (
    LINES_PER_STANZA,
    COUPLET_LINE,
    LINES_PER_SONNET,
)

class SonnetParser():
    """Parses collections of sonnets"""

    def __init__(self):
        """Constructs a SonnetParser."""
        self.sonnets = []
        self.stanzas = []
        self.couplets = []
        self.lines = []

        self.dict = cmudict.dict()

        self.word_count = 0
        self.word_to_num = {}
        self.num_to_word = []

    def parse(self, fname, headers=True):
        """Parse a collection of sonnets."""
        with open(fname) as sonnet_file:
            sonnet_num = 0
            line_num = 0
            sonnet = []
            for line in sonnet_file:
                # Ignore whitespace lines and sonnet headers
                if line.strip() == '':
                    if not appended:
                        self.sonnets.append(sonnet)
                        sonnet = []
                        appended = True
                    continue
                elif line.strip().isdigit() or is_roman_numeral(line.strip()):
                    appended = False
                    continue

                words = self.tokenize(line)
                sonnet.append(words)
                # if line_num == LINES_PER_SONNET:
                #     self.sonnets.append(sonnet)
                #     sonnet = []
                #     line_num = 0
                #     sonnet_num += 1

    def tokenize(self, line):
        """Converts a line into a list of tokens."""
        # Replace punctuation with spaces
        line = re.sub("[-?;:.,'!()]", " ", line)
        tokens = line.split()

        words = []
        for token in tokens:
            t = token.lower()
            if t in self.dict:
                words.append(t)
                if t not in self.word_to_num:
                    self.num_to_word.append(t)
                    self.word_to_num[t] = self.word_count
                    self.word_count += 1
        return words

def is_roman_numeral(numeral):
    for ch in numeral:
        if ch not in "MDCLXVI()":
            return False
    return True

