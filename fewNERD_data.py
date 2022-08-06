# %%
# Perform standard imports
import os
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import numpy as np
from extract_bert_features import get_pipe, embed

from timer import timer
from logging_helper import setup_logging

import logging

# %%
# Setup logging
if (not setup_logging(logfile_file="project_log.log")):
    logging.info("Failed to setup logging, aborting.")

# %%

def get_entity_types():
    return {1: 'ART',
            2: 'EST',
            3: 'EVENT',
            4: 'LOC',
            5: 'ORG',
            6: 'PROD',
            7: 'PER',
            8: 'MISC'}

def read_fewNERD(count: int) -> list[dict]:
    """
    Reads the fewNERD data and returns a list of dicts with the following keys:
    - sentence: str
    - entities: list of dicts with the following keys:
        - start: int
        - end: int
        - label: int
        - text: str
    """

    dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")
    logging.info('Dataset loaded')

    entity_types = get_entity_types()
    result = []
    if count < 1:
        count = len(dataset['train']['tokens'])
    tokens = dataset['train']['tokens'][0:count-1]
    ner_tags = dataset['train']['ner_tags'][0:count-1]
    with tqdm(total=len(tokens)) as pbar:
        for idx in range(count-1):
            zipped = zip(
                tokens[idx],
                ner_tags[idx])
            sentence = ""

            filtered = list(filter(lambda tup: tup[0].isalnum(), zipped))
            sentence = ' '.join([tup[0] for tup in filtered])
            filtered.append(('<EOS>', 'O'))
            
            logging.debug(sentence)

            entities = []
            ent_label_id = 0
            ent_start = 0
            ent_end = 0
            for tok_idx, tup in enumerate(filtered):
                if tup[1] != ent_label_id or tok_idx == len(filtered) - 1:
                    # change state
                    if ent_label_id > 0:
                        # leaving entity
                        ent_end = tok_idx-1
                        ent_text = ' '.join([item[0] for item in filtered[ent_start:ent_end+1]])
                        entities.append({
                            'start': ent_start,
                            'end': ent_end,
                            'label_id': ent_label_id,
                            'label': entity_types[ent_label_id],
                            'text': ent_text,
                        })
                        logging.debug("entity: %s (%s)",
                                     ent_text,
                                     entity_types[ent_label_id])

                    ent_label_id = tup[1]
                    ent_start = tok_idx
                else:
                    # no change of state
                    continue

            if len(entities) > 0:
                result.append({
                    'sentence': sentence,
                    'words': tokens[idx],
                    'entities': entities,
                })
            pbar.update(1)
            
    logging.info('Dataset processed')
    return result

def read_fewNERD_sentences(count: int=0, verbose=1) -> list[str]:
    """
    Reads the fewNERD data and returns a list of sentences.
    -count: number of sentences to read. <=0 means all.
    -verbose: 0=silent, 1=progress bar, 2=verbose
    """

    dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")
    if verbose>0:
        logging.info('Dataset feNERD loaded')

    result = []
    if count < 1:
        count = len(dataset['train']['tokens'])
    tokens = dataset['train']['tokens'][0:count-1]

    pbar = None
    if verbose>0:
        pbar = tqdm(total=len(tokens))

    for idx in range(count-1):
        sentence = ""

        filtered = list(filter(lambda tup: tup[0].isalnum(), tokens[idx]))
        sentence = ' '.join(filtered)
        result.append(sentence)

        if verbose==2:
            logging.debug(sentence)
        if pbar is not None:
            pbar.update(1)

    if verbose>0:
        logging.info('Dataset processed: %s sentences', len(result))

    return result

@timer
def training_data_per_entity_fewNERD(
        length: int = 10,
        radius: int = 7,
        entity_filter: list[str] = []) -> list:
    """
    Creates training data for the autoencoder.
    """
    from tqdm import tqdm
    logging.info("Making pipe")
    from extract_bert_features import embed, get_pipe
    from transformers import TFDistilBertModel
    emb_pipe = get_pipe('distilbert-base-uncased', TFDistilBertModel)

    logging.info(
        f"Create Training Data for {length if length > 0 else 'all'} items, radius {radius}")
    # get spacy data
    fewNERD_data = read_fewNERD(length)

    logging.info("Created NER Data for %s items", len(fewNERD_data))

    # get embedding data
    res = []
    with tqdm(total=len(fewNERD_data)) as pbar:
        for s in fewNERD_data:
            if radius == 0:
                embeddings = embed(emb_pipe, str(s['sentence']))
            for chunk in s["entities"]:
                if len(entity_filter) == 0 or chunk["label"] in entity_filter:
                    start_token = max(chunk["start"]-radius, 0)
                    end_token = min(chunk["end"]+radius+1, len(s["words"]))
                    short_sentence = ' '.join(s['words'][start_token:end_token+1])
                    if radius>0:
                        embeddings = embed(emb_pipe, str(short_sentence))
                        embedding = np.mean(embeddings[1:], axis=0)
                    else:
                        sentence = s["sentence"]
                        embedding = np.mean(embeddings[start_token+1:end_token+1], axis=0)
                        
                    res.append({
                        "sentence": sentence,
                        "chunk": short_sentence,
                        "label": chunk["label"],
                        "label_id": chunk["label_id"],
                        "embedding": embedding,
                    })
            pbar.update(1)
    logging.info("Created %s training items", len(res))
    return res

