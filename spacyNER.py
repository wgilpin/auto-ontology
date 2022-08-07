# %%
# Perform standard imports
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
        entity_filter: list[str] = None) -> list:
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
        entity_filter: list[str] = None,
        length: int = 10,
        ) -> list:
    """
    Creates training data for the autoencoder.
    """
    nlp = spacy.load('en_core_web_sm')

    if entity_filter is None:
        entity_filter = []

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
        # displacy.render(s, style="ent")
        words = str(s.doc).split()
        for ent in s.ents:
            if len(entity_filter) == 0 or ent.label_ in entity_filter:
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

@timer
def get_training_data_spacy(sents: list[str],
                            radius: int,
                            embed_sentence_level: bool=True,
                            length: int = 10,
                            entity_filter: list[str] = None) -> list:
    """
    Creates training data for the autoencoder given a list of sentences
    """
    if entity_filter is None:
        entity_filter = []

    logging.info(
        "Create Training Data for %s items, radius %s",
        length if length > 0 else 'all',
        radius)

    nlp = spacy.load('en_core_web_sm')

    emb_pipe = get_pipe()
    # get embedding data
    res = []
    for sent in tqdm(sents):
        embeddings = None
        s = nlp(sent)
        for ent in s.ents:
            if len(entity_filter) == 0 or ent.label_ in entity_filter:
                if embed_sentence_level and embeddings is None:
                    embeddings = embed(emb_pipe, sent)
                start_token = max(ent.start-radius, 0)
                end_token = min(ent.end+radius, len(s.doc))
                short_sentence = str(s.doc[start_token:end_token])
                if not embed_sentence_level:
                    embeddings = embed(emb_pipe, short_sentence)
                    embedding = embeddings[0]
                else:
                    embedding = np.mean(embeddings[start_token+1:end_token+1], axis=0)
                res.append({
                    "sentence": sent,
                    "chunk": short_sentence,
                    "label": ent.label,
                    "label_id": ent.label_,
                    "embedding": embedding,
                })
    emb_df = pd.DataFrame(np.stack([m["embedding"] for m in res]))
    rest_df = pd.DataFrame(res).drop('embedding', axis=1)
    df = pd.concat([emb_df, rest_df], axis=1)
    df['label'] = np.unique(df['label'], return_inverse=True)[1]
    logging.info("Created %s training items", len(res))
    return df