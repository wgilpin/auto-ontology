# %%
import numpy as np
from numpy import ndarray
from sklearn.pipeline import Pipeline
from transformers import AutoTokenizer, pipeline, TFDistilBertModel, logging
from scipy.spatial.distance import cosine
from timer import timer

logging.set_verbosity_error()

@timer
def make_pipe(name: str, model_name: type) -> Pipeline:

    model = model_name.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    return pipeline('feature-extraction', model=model,
                    tokenizer=tokenizer)


def transformer_embedding(pipe: Pipeline, inp: str) -> ndarray:
    features = pipe(inp)
    features = np.squeeze(features)
    return features


@timer
def embed(pipe: Pipeline, text: str) -> ndarray:
    return transformer_embedding(pipe, text)
