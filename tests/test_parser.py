from sonetto.parser import SonnetParser

def test_shakespeare():
    """Tests the parser on some lines of Shakespeare."""
    sp = SonnetParser()
    sp.parse('data/shakespeare.txt')

    line1 = ['from', 'fairest', 'creatures', 'we', 'desire', 'increase']
    assert sp.sonnets[0][0] == line1
    for i, word in enumerate(line1):
        assert sp.num_to_word[i] == word
        assert sp.word_to_num[word] == i

