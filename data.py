# %%

import os
import pandas as pd
import numpy as np

from spacyNER import create_training_data_per_entity_spacy
from fewNERD_data import create_training_data_per_entity_fewNERD

# %%
def read_conll_dataset(filename: str) -> list:
    """
    Reads the CONLL dataset.
    """
    data = []
    with open(f"data/{filename}", 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(line.split())
    return data

def parse_conll(data):
    """
        returns {
            sentence: str,
            entities: {
                start, end: int,
                label: str,
                text: str,
            }
        }
    """
    sentences = []
    sentence = []
    for line in data:
        entities = []
        toks = line.split()
        if len(line) == 0:
            sentences.append({'sentence': sentence, 'entities': entities})
            sentence = []
            entities = []
            continue
        if toks[0] == '-DOCSTART-':
            continue
        if toks[0][0].isalnum():
            # skip punctuation
            continue
        # a word
        entities.append({
            'start': int(line[0]),
            'end': int(line[1]),
            'label': line[3],
            'text': line[4],
        })
    return {
        'sentence': ' '.join(sentence),
        'entities': entities,
    }
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

def test_train_split(df, frac=0.8):
    training_data = df.sample(frac=frac, random_state=42)
    testing_data = df.drop(training_data.index)
    train_y_df = np.array(training_data["label"])
    test_y_df = np.array(testing_data["label"])
    train_x_df = np.array(training_data.drop(
        columns=['chunk', 'label', 'label_id', 'sentence']))
    test_x_df = np.array(testing_data.drop(
        columns=['chunk', 'label', 'label_id', 'sentence']))
    print(f"Full df: {df.shape}")
    print(f"Train df: {train_x_df.shape}")
    print(f"Test df: {test_x_df.shape}")
    return train_x_df, test_x_df, train_y_df, test_y_df


def get_training_data(
        save_dir: str,
        source: str = "spacy",
        radius: int = 10,
        fraction: float = 0.8,
        count: int = -1,
        force_recreate: bool = False,
        entity_filter: list[str] = []) -> list:
    filename = os.path.join(save_dir, f"training_data_radius_{source}_{radius}.csv")

    # create data if not already done

    if not force_recreate and os.path.exists(filename):
        df = pd.read_csv(filename)
        print(f"Loaded from file {filename}")

    else:
        if source == "spacy":
            merged = create_training_data_per_entity_spacy(length=count,
                                                    radius=radius,
                                                    entity_filter=entity_filter)
        elif source == "fewNERD":
            merged = create_training_data_per_entity_fewNERD(length=count,
                                                    radius=radius,
                                                    entity_filter=entity_filter)
        print(f"Created entries: {len(merged)}")  # create full DF

        x_train_np = np.stack([m["embedding"] for m in merged])

        merged_df = pd.DataFrame(merged)
        emb_df = pd.DataFrame(x_train_np)

        merged_df.drop(columns=['embedding'], inplace=True)
        df = pd.concat([merged_df, emb_df], axis=1)

        df['label'] = np.unique(df['label'], return_inverse=True)[1]

        print(df.shape)
        # save data to csv
        df.to_csv(filename, index=False)
        print(f"Saved {filename}")

    return test_train_split(df, fraction)