# %%
# Perform standard imports
import os
import pickle
from typing import Optional, Tuple
import pandas as pd
import spacy
import numpy as np
from extract_bert_features import embed, get_pipe
from transformers import TFDistilBertModel
from tqdm import tqdm
from data_conll import get_sample_conll


from timer import timer
from logging_helper import setup_logging

import logging

# %%
# Setup
if (not setup_logging(logfile_file="project_log.log")):
    logging.info("Failed to setup logging, aborting.")

# %%

# remove stop words


def remove_stop_words(chunk):
    """
    Removes stop words from a spacy chunk.
    """
    filtered_toks = []
    for tok in chunk:
        if not tok.is_stop:
            filtered_toks.append(tok.text)
    return " ".join(filtered_toks)

# %%


def merge_chunks_ents(document) -> list:
    """
    Merges chunks and entities into a single list of dicts.
    returns a list of dicts with the following keys:
    - chunk: str
    - entity: spacy Entity
    """
    ents = document.ents
    e_idx = 0
    out = []
    for chunk in document.noun_chunks:
        no_stops = remove_stop_words(chunk)
        while (e_idx < len(ents)) and (ents[e_idx].start_char >= chunk.start_char) and (ents[e_idx].end_char <= chunk.end_char):
            # this entity is in the noun chunk
            # get the token index of the entity
            out.append({
                "chunk": no_stops,
                "entity": ents[e_idx],
            })
            e_idx += 1
    return out


# %%
def get_spacy_NER_data(length: int = 10) -> list:
    """
    Returns a list of dicts with the following keys:
    - sentence: str
    - nlp: spacy NLP object
    - chunks: list of dicts with the following keys:
        - chunk: str
        - entity: spacy Entity
    """
    nlp = spacy.load('en_core_web_sm')

    sentences = []
    for l in get_sample_conll(length):
        doc = nlp(l)
        merged = merge_chunks_ents(doc)
        sentences.append({'sentence': l, 'nlp': doc, 'chunks': merged})
    return sentences


# %%
@timer
def training_data_per_entity_spacy(
        length: int = 10,
        radius: int = 7,
        entity_filter: Optional[list[str]] = None) -> list:
    """
    Creates training data for the autoencoder.
    """
    if entity_filter is None:
        entity_filter = []

    print("Making pipe")

    emb_pipe = get_pipe('distilbert-base-uncased', TFDistilBertModel)

    print(
        f"Create Training Data for {length if length > 0 else 'all'} items, radius {radius}")
    # get spacy data
    spacy_data = get_spacy_NER_data(length)
    print(f"Created NER Data for {len(spacy_data)} items")

    # get embedding data
    res = []
    for s in tqdm(spacy_data):

        for chunk in s["chunks"]:
            start_token = max(chunk["entity"].start-radius, 0)
            end_token = min(chunk["entity"].end+radius, len(s["nlp"]))
            short_sentence = s["nlp"][start_token:end_token]
            res.append({
                "sentence": short_sentence,
                "chunk": str(chunk["chunk"]),
                "label": chunk["entity"].label_,
                "embedding": embed(emb_pipe, str(short_sentence))
                })
    print(f"Created {len(res)} training items")
    return res

# %%
# Create training data from pre-embedded sentences

def training_data_from_embeds_spacy(
        embeds: pd.DataFrame,
        radius: int = 0,
        length: int = 10,
        ) -> list:
    """
    Creates training data for the autoencoder.
    """
    nlp = spacy.load('en_core_web_sm')

    print(
        f"Create Training Data for {length if length > 0 else len(embeds)}"
        f" items, radius {radius}")
    # get spacy data
    embeds["spacy_data"] = embeds["sentence"][0:length-1].apply(nlp)
    print(f"Created NER Data for {len(embeds)} items")

    # get embedding data
    res = []
    for _, row in tqdm(embeds.iterrows()):
        s = row["spacy_data"]
        words = str(s.doc).split()
        for ent in s.ents:
            start_token = max(ent.start-radius, 0)
            end_token = min(ent.end+radius, len(words))
            short_sentence = ' '.join(words[start_token:end_token+1])
            res.append({
                "sentence": short_sentence,
                "chunk": ' '.join(words[ent.start:ent.end+1]),
                "label": ent.label,
                "label_id": ent.label_,
                "embedding": np.mean(row["embeddings"][start_token+1:end_token+1], axis=0)
            })
    print(f"Created {len(res)} training items")
    return res

class TrainingDataSpacy():
    """
    Creates training data for the model.
    returns a list of dicts as per embed_chunks
    """
    def __init__(self,
            radius: int=0,
            embed_sentence_level: bool = False):
        self.nlp = spacy.load('en_core_web_sm')
        self.emb_pipe = None
        self.radius = radius
        self.embed_sentence_level = embed_sentence_level
        self.mapping = {}
        self.skip_chunks = ["it"]
        self.embeddings = None
        self.data_name = ""
        self.sents = None

    def embed_all(self):
        """
        Embeds all sentences in the data.
        """
        print("pre-embedding all")
        self.embeddings = []
        self.emb_pipe = get_pipe()
        for sent in tqdm(self.sents):
            self.embeddings.append(embed(self.emb_pipe, sent))
        with open(self.data_name, 'wb') as handle:
            pickle.dump(self.embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Emedded {len(self.embeddings)} sentences")
        print(f"Saved to {self.data_name}")

    def load_embs_cache(self):
        """
        Loads the embeddings cache.
        """
        if os.path.exists(self.data_name) :
            print(f"Loading {self.data_name}") 
            self.embeddings = pd.read_pickle(self.data_name)
        else:
            self.embed_all()


    def cached_sent_embed(self, sentence_no: int) -> np.ndarray:
        """
        Returns the cached sentence embedding.
        """
        if self.embeddings is None :
            self.load_embs_cache()
        return self.embeddings[sentence_no]
        

    def embed_text(self, sent: str, sent_no: int, start: int=0, end: int=0, embeddings: list=None):
        """
        embeds a sentence and returns the embedding
        """
        if self.embed_sentence_level:
            # embed whole sentence, then return appropriate part
            if embeddings is None:
                embeddings = self.cached_sent_embed(sent_no)
            return np.mean(embeddings[start+1:end+1], axis=0)
        else:
            # embed only chunk
            self.emb_pipe = get_pipe()
            embeddings = embed(self.emb_pipe, sent)
            return embeddings[0]

    def embed_chunk(self,
            sent_no: int,
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
            embedding = self.embed_text(str(doc), sent_no, start_tok, end_tok, embeddings)
        else:
            # will have to embed the chunk
            embedding = self.embed_text(short_sent, sent_no, start_tok, end_tok)

        return{
            "sentence": str(doc),
            "chunk": short_sent,
            # label_n: 0 for noun chunk, or spacy entity id number
            "label_n": span.label if is_entity else 0,
            "label_id": span.label_ if is_entity else 'UNKNOWN',
            "embedding": embedding,
        }

    def embed_sentence(self, sent, sent_no: int) -> list[dict]:
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
                embs = self.cached_sent_embed(sent_no)
            item = self.embed_chunk(sent_no, s, nc, is_entity=False, embeddings=embs)
            res.append(item)
            for ent in nc.ents:
                if ent.start >= nc.start and ent.end <= nc.end:
                    # same span
                    res.append(
                        self.embed_chunk(
                            sent_no, s, ent, is_entity=True, embeddings=embs))
        return res

    @timer
    def get_training_data_spacy(self,
                    sents: list[str],
                    length: int = 10,
                    name:str = None
                    ) -> Tuple[pd.DataFrame, dict]:
        """
        Creates training data for the autoencoder given a list of sentences
        """
        logging.info(
            "Create Training Data for %s items, radius %s",
            length if length > 0 else 'all',
            self.radius)

        self.data_name = os.path.join("./data", name)
        self.sents = sents
        if self.embed_sentence_level:
            self.load_embs_cache()

        # get embedding data
        res = []
        mapping={}
        print(f"Embedding sentences radius {self.radius}")
        for sent_no, sent in tqdm(enumerate(sents)):
            res.extend(self.embed_sentence(sent, sent_no))

        emb_df = pd.DataFrame(np.stack([m["embedding"] for m in res]))
        rest_df = pd.DataFrame(res).drop('embedding', axis=1)
        df = pd.concat([emb_df, rest_df], axis=1)
        
        df['label'] = np.unique(df['label_n'], return_inverse=True)[1]+1
        for _, row in df.iterrows():
            mapping[row['label']] = row['label_id']
        df.drop(columns=['label_n'], inplace=True)
        logging.info("Created %s training items", len(res))
        return df, mapping
    # %%
