import pytest

from src.spacyNER import TrainingDataSpacy


class Test_spacy_NER:

    def test_get_training_data(self, capsys):
        """
        Test that the sample conll file is read correctly.
        """
        ner = TrainingDataSpacy()
        sent = "Boris Johnson is currently Prime Minister of the United Kingdom"
        df, mapping = ner.get_training_data_spacy([sent], length=1)

        first = df.iloc[0]
        with capsys.disabled():
            print(first.keys())

        assert first['sentence'] == sent
        assert first['chunk'] == 'Boris Johnson'
        assert first['label'] == 1
        assert first.shape == (772,) # 768 embeddings plus other fields
        assert len(mapping.keys()) == 3
        assert mapping[2] == 'PERSON'