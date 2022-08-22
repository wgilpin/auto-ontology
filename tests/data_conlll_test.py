import pytest

from src.data_conll import *


class Test_conll_import:

    def test_get_1_sample_conll (self, capsys):
        """
        Test that the sample conll file is read correctly.
        """
        lines = get_sample_conll(1)
        with capsys.disabled():
            print(lines)
        assert len(lines) == 1
        assert lines[0] == 'SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .'

    def test_get_100_sample_conll (self):
        """
        Test that the sample conll file is read correctly.
        """
        lines = get_sample_conll(100)
        assert len(lines) == 100
        
