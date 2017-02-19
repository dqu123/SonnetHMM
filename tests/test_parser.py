from sonetto.parser import SonnetParser

def test_shakespeare():
    """Tests the parser on some lines of Shakespeare."""
    sp = SonnetParser()
    sp.parse('data/shakespeare.txt')

    assert sp.lines[0] == ['from', 'fairest', 'creatures', 'we', 'desire', 'increase']
