import pytest

from src.data import load_data


class Test_data_import:

    def test_load_data(self, capsys):
        """
        Test that the sample conll file is read correctly.
        """
        x, y, mapping, _ = load_data(10)
        assert len(mapping.keys()) == 5
        assert x.shape[0] == 270
        assert y.shape[0] == 270

    def test_load_data_with_strings(self, capsys):
        _, _, _, strings = load_data(15, get_text=True)

        assert len(strings) == 365

    def test_load_data_by_entity(self):
        x, _, mapping, _ = load_data(
                        15, get_text=True, entity_filter=['PERSON', 'ORG'])

        assert mapping[0] == 'PERSON'
        assert mapping[1] == 'ORG'
        assert x.shape == (12, 768)

