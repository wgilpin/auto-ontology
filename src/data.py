# %%

import os
import pickle
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple, overload

from imblearn.over_sampling import RandomOverSampler
from spacyNER import (TrainingDataSpacy,
                      training_data_per_entity_spacy,
                      training_data_from_embeds_spacy)
from data_conll import get_sample_conll_hf


from fewNERD_data import training_data_per_entity_fewNERD, read_fewNERD_sentences

from extract_bert_features import get_pipe, embed

from logging_helper import setup_logging

# process all entities
entity_types = [] #'PER', 'PERSON', 'ORG', 'LOC', 'GPE', 'MISC']

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


def output(s:str, verbose:int=1):
    """
    Outputs a string to the console if verbose is set to 1 or higher
    """
    if verbose > 0:
        print(s)
    logging.info(s)

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


def parse_conll(data: list) -> list[str]:
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


def test_train_split(df, frac=1.0, oversample: bool=True, verbose:int=1):
    """
    Returns a balanced train and test split of the dataframe
    -X train
    -X test
    -X train
    -Y test
    """
    df = df.dropna()
    if frac < 1.0:
        training_data = df.sample(frac=frac, random_state=42)
        testing_data = df.drop(training_data.index)
        test_y_df = np.array(testing_data["label"])
        test_x_df = np.array(testing_data.drop(
            columns=['chunk', 'label', 'label_id', 'sentence']))
        train_y_df = np.array(training_data["label"])
        logging.info("Test df: %s", test_x_df.shape)
    else:
        training_data = df
        test_x_df = None
        test_y_df = None
    train_x_df = training_data
    train_y_df = np.array(train_x_df["label"])
    logging.info("Full df: %s", df.shape)


    # see balance
    unique, counts = np.unique(train_y_df, return_counts=True)
    uniques = np.column_stack((unique, counts)) 
    output(verbose=verbose, s="Data balance:")
    output(verbose=verbose, s=uniques)

    if oversample:
        # oversample for balance
        output(verbose=verbose, s="Balancing data")
        ros = RandomOverSampler(random_state=42)
        train_x_df, train_y_df = ros.fit_resample(train_x_df, train_y_df)
    train_x_strings = train_x_df['chunk']
    train_x_df = np.array(train_x_df.drop(
        columns=['chunk', 'label', 'label_id', 'sentence']))

    logging.info("Train df: %s", train_x_df.shape)
    return train_x_df, test_x_df, train_y_df, test_y_df, train_x_strings


def get_sentences_from_conll():
    data = read_conll_dataset("data/conll/train.txt")
    data = parse_conll(data)
    return [line['sentence'] for line in data]


def pre_embed(dataset: str,
              save_dir: str = "./data",
              length: int = 0,
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

    from transformers import TFDistilBertModel
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
            intermediate_filename = os.path.join(
                save_dir,
                f"embeddings_{length}:{i//save_every}_{dataset}.pck")
            df.to_pickle(intermediate_filename)
            result = []
    logging.info("Embeddings complete")
    # read back
    dfs = []
    for i in range(last):
        intermediate_filename = os.path.join(
            save_dir,
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
    filename = os.path.join(
        save_dir,
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


def load_data(
        size: int,
        entity_filter: list=None,
        get_text: bool=False,
        oversample: bool=True,
        verbose:int=1,
        radius: int=0,
        train:bool=True) -> Tuple[pd.DataFrame, pd.DataFrame, dict, list]:
    """
    Load data from disk
    Arguments:
        -size: number of rows to load
        -entity_filter: list of entities to filter on
        -get_text: whether to get the text as well
    Returns:
        -x: training data
        -y: training labels
        -mapping: mapping from label id to string
        -strings: list of noun chunks in the data, 1 per (x, y) pair
    """
    if entity_filter is None:
        entity_filter = []
    ds_name = ''if train else '_test'

    filename_data = f"./data/conll_spacy_n{size}_r{radius}{ds_name}.pkl"
    filename_mapping = f"./data/conll_spacy_n{size}_r{radius}{ds_name}_map.pkl"
    if os.path.exists(filename_data) and os.path.exists(filename_mapping):
        output(f"Loading {filename_data}", verbose)
        trg = pd.read_pickle(filename_data)
        with open(filename_mapping, 'rb') as handle:
            mapping = pickle.load(handle)
            output(verbose=verbose, s=f"LOADED {mapping}")
    else:
        print("Creating data")
        sample_conll = get_sample_conll_hf(size, train=train)
        sample_name = f"conll_n{size}_{'train' if train else 'test'}"
        
        embedder = TrainingDataSpacy(
            embed_sentence_level=True,
            radius=radius)
        trg, mapping = embedder.get_training_data_spacy(
                                                sents=sample_conll,
                                                name = sample_name)
        trg.to_pickle(filename_data)
        with open(filename_mapping, 'wb') as handle:
            pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"SAVED {filename_data}")
        output(f"Map {mapping}", verbose)

    if entity_filter:
        # entity_filter to id list
        mapping = {k:v for k,v in mapping.items() if v in entity_filter}
        allowed_y_list = [k for k,v in mapping.items() if v in entity_filter]
        trg = trg[trg['label'].isin(allowed_y_list)]


    output(f'Loaded file: {trg.shape[0]} samples', verbose)

    x, _, y, _, strings = test_train_split(trg, oversample=oversample, verbose=verbose)
    if oversample:
        output("Post Oversampling", verbose)
    output(f"x: {x.shape}, y: {y.shape}", verbose)

    filtered_map = {}
    y_unq = np.unique(y)
    for idx, y_val in enumerate(y_unq):
        filtered_map[idx] = mapping[y_val]
    y = np.unique(y, return_inverse = True)[1]
    output(filtered_map, verbose)

    return x, y, filtered_map, strings if get_text else None
