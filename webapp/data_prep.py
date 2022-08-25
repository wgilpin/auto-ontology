# Data preparation for flask app

import logging
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from src.spacyNER import TrainingDataSpacy
from src.extract_bert_features import get_pipe, embed

def test_train_split(df, frac=1.0, oversample: bool=True):
    """
    Returns a balanced train and test split of the dataframe
    -X train
    -X test
    -X train
    -Y test
    -strings for each x
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
    print("Train data balance:")
    print(uniques)

    if oversample:
        # oversample for balance
        print("Balancing data")
        ros = RandomOverSampler(random_state=0)
        train_x_df, train_y_df = ros.fit_resample(train_x_df, train_y_df)
    train_x_strings = train_x_df['chunk']
    train_x_df = np.array(train_x_df.drop(
        columns=['chunk', 'label', 'label_id', 'sentence']))

    logging.info("Train df: %s", train_x_df.shape)
    return train_x_df, test_x_df, train_y_df, test_y_df, train_x_strings


def prepare_data(sents, entity_filter):
    """
    Prep the data for evaluation
    """
    print("Creating data")

    embedder = TrainingDataSpacy(
            embed_sentence_level=True)
    trg, mapping = embedder.get_training_data_spacy(sents)

    # entity_filter to id list
    mapping = {k:v for k,v in mapping.items() if v in entity_filter}
    allowed_y_list = [k for k,v in mapping.items() if v in entity_filter]
    trg = trg[trg['label'].isin(allowed_y_list)]


    print(f'Done: {trg.shape}')

    x, _, y, _, strings = test_train_split(trg, oversample=True)
    print(f"x: {x.shape}, y: {y.shape}")

    filtered_map = {}
    y_unq = np.unique(y)
    for idx, y_val in enumerate(y_unq):
        filtered_map[idx] = mapping[y_val]
    y = np.unique(y, return_inverse = True)[1]
    print(filtered_map)

    return x, y, strings, mapping
