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

    def parse(self, fname, headers=True):
        """Parse a collection of sonnets."""
        with open(fname) as sonnet_file:
            sonnet_num = 0
            line_num = 0
            next_sonnet = True
            stanza = []
            couplet = []
            sonnet = []
            for line in sonnet_file:
                # Ignore whitespace lines.
                if line.strip() == '':
                    continue

                if next_sonnet and headers:
                    next_sonnet = False
                    continue
                line_num += 1

                words = self.tokenize(line)
                self.lines.append(words)
                if line_num < COUPLET_LINE:
                    stanza += words
                else:
                    couplet += words
                sonnet += words

                if line_num % LINES_PER_STANZA == 0:
                    self.stanzas.append(stanza)
                    stanza = []

                if line_num == LINES_PER_SONNET:
                    self.couplets.append(couplet)
                    self.sonnets.append(sonnet)
                    couplet = []
                    sonnet = []
                    next_sonnet = True
                    sonnet_num += 1

    def tokenize(self, line):
        """Converts a line into a list of tokens."""
        # Replace punctuation with spaces
        line = re.sub("[-?;:.,'!()]", " ", line)
        tokens = line.split()

        words = [t.lower() for t in tokens if t.lower() in self.dict]
        return words
