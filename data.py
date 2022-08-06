# %%

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import TFDistilBertModel
from spacyNER import training_data_per_entity_spacy, training_data_from_embeds_spacy
from fewNERD_data import training_data_per_entity_fewNERD, read_fewNERD_sentences

from extract_bert_features import get_pipe, embed

from logging_helper import setup_logging

import logging

# %%
# Setup logging
if (not setup_logging(logfile_file="project_log.log")):
    logging.info("Failed to setup logging, aborting.")

# Log some messages
# logging.debug("Debug message")
# logging.info("Info message")
# logging.warning("Warning message")
# logging.error("Error message")
# logging.critical("Critical message")
# %%
def read_conll_dataset(filename: str) -> list:
    """
    Reads the CONLL dataset.
    """
    data = []
    with open(f"{filename}", 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(line.split())
    return data

def parse_conll(data: list)->list[str]:
    """
        returns list of sentences
    """
    sentences = []
    sentence = []
    for toks in data:
        if toks[0] == '-DOCSTART-':
            continue
        if toks[0] == '.':
            # end of sentence
            sentences.append({
                        'sentence': ' '.join(sentence),
                        'tokens': sentence})
            sentence = []
            continue
        if toks[0][0].isalnum():
            # a word
            sentence.append(toks[0])

    return sentences
    
# %%

def join_punctuation(seq, characters=".,;?!'')-"):
    # https://stackoverflow.com/a/15950837/2870929
    characters = set(characters)
    seq = iter(seq)
    current = next(seq)

    for nxt in seq:
        if nxt[0] in characters:
            current += nxt
        else:
            yield current
            current = nxt

    yield current


# %%
#
# CREATE TRAINING DATA

def test_train_split(df, frac=1.0):
    df = df.dropna()
    if frac < 1.0:
        training_data = df.sample(frac=frac, random_state=42)
        testing_data = df.drop(training_data.index)
        test_y_df = np.array(testing_data["label"])
        test_x_df = np.array(testing_data.drop(
            columns=['chunk', 'label', 'label_id', 'sentence']))
        train_y_df = np.array(training_data["label"])
        train_x_df = np.array(training_data.drop(
            columns=['chunk', 'label', 'label_id', 'sentence']))
    else:
        training_data = df
        test_x_df = None
        test_y_df = None
    train_y_df = np.array(training_data["label"])
    train_x_df = np.array(training_data.drop(
        columns=['chunk', 'label', 'label_id', 'sentence']))
    logging.info("Full df: %s", df.shape)
    logging.info("Train df: %s", train_x_df.shape)
    if test_x_df:
        logging.info("Test df: %s", test_x_df.shape)

    # oversample for balance
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    x_train_bal, y_train_bal = ros.fit_resample(train_x_df, train_y_df)
    return x_train_bal, test_x_df, y_train_bal, test_y_df

def get_sentences_from_conll():
    data = read_conll_dataset("data/conll/train.txt")
    data = parse_conll(data)
    return [line['sentence'] for line in data]

def pre_embed(dataset: str,
              save_dir: str="./data",
              length: int=0,
              force_recreate=False) -> pd.DataFrame:
    """
    give a dataset name
        -conll | fewNERD
    Returns a DataFrame
        """
    filename = os.path.join(save_dir,
        f"embeddings_{length}:{dataset}.pck")
    if not force_recreate and os.path.exists(filename):
        logging.info("Loading %s", dataset)
        df = pd.read_pickle(filename)
        logging.info("Loaded from file%s", filename)
        return df

    # need to build it now
    if dataset == "conll":
        logging.info("Reading CONLL dataset")
        sentences = get_sentences_from_conll()
    elif dataset == "fewNERD":
        logging.info("Reading fewNERD dataset")
        sentences = read_fewNERD_sentences(length)
    result = []
    logging.info("Making pipe")
    emb_pipe = get_pipe('distilbert-base-uncased', TFDistilBertModel)
    logging.info("Made pipe")
    save_every = 2000
    start = 43*save_every
    last = len(sentences)//save_every
    count = length if length else len(sentences)
    for i in tqdm(range(start, count), miniters=save_every):
        sentence = sentences[i]
        embeddings = embed(emb_pipe, str(sentence))
        result.append({'sentence': sentence, 'embeddings': embeddings})
        if (i+1) % save_every == 0:
            df = pd.DataFrame(result)
            intermediate_filename = os.path.join(save_dir,
                f"embeddings_{length}:{i//save_every}_{dataset}.pck")
            df.to_pickle(intermediate_filename)
            result = []
    logging.info("Embeddings complete")
    # read back
    dfs = []
    for i in range(last):
        intermediate_filename = os.path.join(save_dir,
            f"embeddings_{length}:{i}_{dataset}.pck")
        dfs.append(pd.read_pickle(intermediate_filename))

    # rebuild full df
    df = pd.concat(dfs)

    # save data to single pickle
    full_filename = os.path.join(save_dir,
        f"embeddings_{length}:{dataset}.pck")
    df.to_pickle(full_filename)

    logging.info("Saved %s", filename)
    print("DONE")
    return df

def get_training_data(
        save_dir: str,
        source: str = "fewNERD",
        radius: int = 10,
        fraction: float = 0.8,
        count: int = 0,
        force_recreate: bool = False,
        entity_filter: list[str] = []) -> list:
    filename = os.path.join(save_dir,
        f"training_data_radius_{source}_{count if count > 0 else 'all'}_{radius}.csv")

    # create data if not already done

    if not force_recreate and os.path.exists(filename):
        logging.info("Loading %s", source)
        df = pd.read_csv(filename)
        logging.info("Loaded from file%s", filename)

    else:
        if source == "conll":
            logging.info("Reading CONLL dataset")
            merged = training_data_per_entity_spacy(length=count,
                                                    radius=radius,
                                                    entity_filter=entity_filter)
        elif source == "fewNERD":
            logging.info("Reading fewNERD dataset")
            merged = training_data_per_entity_fewNERD(length=count,
                                                    radius=radius,
                                                    entity_filter=entity_filter)
        elif source == "fewNERD_spacy":
            logging.info("Reading fewNERD dataset for spacy")
            df = pre_embed("fewNERD")
            if count > 0:
                df = df[0:count]
            merged = training_data_from_embeds_spacy(df,
                                                    radius=radius,
                                                    entity_filter=entity_filter)
        logging.info("Created entries: %s", len(merged))  # create full DF

        x_train_np = np.stack([m["embedding"] for m in merged])

        merged_df = pd.DataFrame(merged)
        emb_df = pd.DataFrame(x_train_np)

        merged_df.drop(columns=['embedding'], inplace=True)
        df = pd.concat([merged_df, emb_df], axis=1)

        df['label'] = np.unique(df['label'], return_inverse=True)[1]

        logging.info(df.shape)
        # save data to csv
        df.to_csv(filename, index=False)
        logging.info("Saved %s", filename)

    return test_train_split(df, fraction)