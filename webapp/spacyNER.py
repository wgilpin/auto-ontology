# %%
# Perform standard imports
from typing import Tuple
import pandas as pd
import spacy
import numpy as np
from extract_bert_features import embed, get_pipe


class TrainingDataSpacy():
    """
    Creates training data for the model.
    returns a list of dicts as per embed_chunks
    """
    def __init__(self,
            radius: int=0,
            embed_sentence_level: bool = False):
        self.nlp = spacy.load('en_core_web_sm')
        self.emb_pipe = get_pipe()
        self.radius = radius
        self.embed_sentence_level = embed_sentence_level
        self.mapping = {}
        self.skip_chunks = ["it"]

    def embed_text(self, sent: str, start: int=0, end: int=0, embeddings: list=None):
        """
        embeds a sentence and returns the embedding
        """
        if self.embed_sentence_level:
            # embed whole sentence, then return appropriate part
            if embeddings is None:
                embeddings = embed(self.emb_pipe, sent)
            return np.mean(embeddings[start+1:end+1], axis=0)
        else:
            # embed only chunk
            embeddings = embed(self.emb_pipe, sent)
            return embeddings[0]

    def embed_chunk(self,
            doc,
            span,
            is_entity: bool,
            embeddings: list) -> dict:
        """
        Returns a dict with the following keys:
            - sentence: str
            - chunk: str
            - label_id: str
            - label_n: int
            - embedding: np.array
        label_n is 0 for noun chunk, or spacy entity id number
        """

        start_tok = max(span.start-self.radius, 0)
        end_tok = min(span.end+self.radius, len(doc))
        short_sent = str(doc[start_tok:end_tok])

        if self.embed_sentence_level:
            # we have the embeddings for the whole sentence already
            embedding = self.embed_text(str(doc), start_tok, end_tok, embeddings)
        else:
            # will have to embed the chunk
            embedding = self.embed_text(short_sent, start_tok, end_tok)

        return{
            "sentence": str(doc),
            "chunk": short_sent,
            # label_n: 0 for noun chunk, or spacy entity id number
            "label_n": span.label if is_entity else 0,
            "label_id": span.label_ if is_entity else 'UNKNOWN',
            "embedding": embedding,
        }

    def embed_sentence(self, sent) -> list[dict]:
        """
        Returns a list of embeddings for each chunk or entity in the sentence
        """
        s = self.nlp(sent)
        res=[]
        embs = None
        for nc in s.noun_chunks:
            # add any noun chunks and their entities
            if str(nc).lower() in self.skip_chunks:
                continue
            if embs is None and self.embed_sentence_level:
                # calculate embeddings once for whole sentence
                embs = embed(self.emb_pipe, sent)
            item = self.embed_chunk(s, nc, is_entity=False, embeddings=embs)
            res.append(item)
            for ent in nc.ents:
                if ent.start == nc.start and ent.end == nc.end:
                    # same span
                    res.append(
                        self.embed_chunk(
                            s, ent, is_entity=True, embeddings=embs))
        return res

    def get_training_data_spacy(self,
                    sents: list[str],
                    length: int = 10) -> Tuple[pd.DataFrame, dict]:
        """
        Creates training data for the autoencoder given a list of sentences
        """

        # get embedding data
        res = []
        mapping={}
        for sent in sents:
            res.extend(self.embed_sentence(sent))

        emb_df = pd.DataFrame(np.stack([m["embedding"] for m in res]))
        rest_df = pd.DataFrame(res).drop('embedding', axis=1)
        df = pd.concat([emb_df, rest_df], axis=1)

        df['label'] = np.unique(df['label_n'], return_inverse=True)[1]+1
        for _, row in df.iterrows():
            mapping[row['label']] = row['label_id']
        df.drop(columns=['label_n'], inplace=True)
        return df, mapping
    # %%
