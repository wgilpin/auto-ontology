import pytest

from src.extract_bert_features import get_pipe, embed


class Test_get_embeddings:

    def test_embed (self):
        """
        """
        pipe = get_pipe()
        features = embed(pipe, 'SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .')
        assert features is not None
        assert features.shape == (14, 768)