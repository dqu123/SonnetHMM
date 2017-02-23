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

        # HMM lists
        self.hmm_sonnets = []
        self.hmm_stanzas = []
        self.hmm_couplets = []
        self.hmm_lines = []

    def parse(self, fname, headers=True):
        """Parse a collection of sonnets."""
        with open(fname) as sonnet_file:
            sonnet_num = 0
            line_num = 0
            sonnet = []
            appended = True
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
                self.lines.append(words)
        self.compute_stanzas_and_couplets()
        self.compute_hmm_lists()

    def compute_stanzas_and_couplets(self):
        """Group words into sequences based on stanza and couplet."""
        for sonnet in self.sonnets:
            stanza = []
            for i, line in enumerate(sonnet):
                if i < len(sonnet) - 2:
                    stanza += line
                    if (i + 1) % 4 == 0:
                        self.stanzas.append(stanza)
                        stanza = []

            # Last two lines form the couplet
            self.couplets.append(sonnet[-2] + sonnet[-1])

    def compute_hmm_lists(self):
        """Convert lists to numbers for HMM"""
        self.hmm_lines = [[self.word_to_num[word] for word in line]
                          for line in self.lines]


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

