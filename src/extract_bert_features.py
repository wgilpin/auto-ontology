# %%
import numpy as np
from numpy import ndarray
from sklearn.pipeline import Pipeline
from transformers import AutoTokenizer, pipeline, logging
from transformers import TFDistilBertModel

from timer import timer

logging.set_verbosity_error()

_PIPE = None

@timer
def get_pipe(
        name: str='distilbert-base-uncased',
        model_name: type=TFDistilBertModel) -> Pipeline:
    """
    Get or create the pipeline for extracting features from text.
    """
    global _PIPE
    if _PIPE:
        return _PIPE

    model = model_name.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    _PIPE = pipeline('feature-extraction', model=model,
                    tokenizer=tokenizer)
    return _PIPE


def transformer_embedding(pipe: Pipeline, inp: str) -> ndarray:
    features = pipe(inp)
    features = np.squeeze(features)
    return features


def embed(pipe: Pipeline, text: str) -> ndarray:
    return transformer_embedding(pipe, text)
