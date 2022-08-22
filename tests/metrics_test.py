import pytest
import numpy as np

from src.metrics import acc


class Test_metrics:

    def test_get_accuracy(self):
        """
        Test that the sample conll file is read correctly.
        """
        res = acc(np.array([1,2,1,2]), np.array([1,2,1,0]))
        assert res == 0.75

        res = acc(np.array([1,2,1,2]), np.array([1,2,0,0]))
        assert res == 0.5
