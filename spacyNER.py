# %%
# Perform standard imports
import spacy
from os import path
import numpy as np 
from extract_bert_features import embed, make_pipe
from transformers import TFDistilBertModel

from timer import timer

nlp = spacy.load('en_core_web_sm')

# %%


def get_sample(length: int = 10) -> list:
    test_file = path.join("./data/conll/test.txt")
    with open(test_file, encoding="utf-8") as f:
        file_lines = f.readlines()

    lines = []
    words = []
    for fl in file_lines:
        if fl == "\n":
            # blank line is separator
            lines.append(" ".join(words))
            words = []
        else:
            words.append(fl.split(" ")[0])

    sample = list(filter(lambda s: s[0].isalpha(), lines))
    if length > 0:
        sample = sample[0:length]
    return sample

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
    sentences = []
    for l in get_sample(length):
        doc = nlp(l)
        merged = merge_chunks_ents(doc)
        sentences.append({'sentence': l, 'nlp': doc, 'chunks': merged})
    return sentences


# %%
@timer
def create_training_data_per_entity_spacy(
        length: int = 10,
        radius: int = 7,
        entity_filter: list[str] = []) -> list:
    """
    Creates training data for the autoencoder.
    """
    from tqdm import tqdm
    print("Making pipe")
    
    emb_pipe = make_pipe('distilbert-base-uncased', TFDistilBertModel)

    print(
        f"Create Training Data for {length if length > 0 else 'all'} items, radius {radius}")
    # get spacy data
    spacy_data = get_spacy_NER_data(length)
    print(f"Created NER Data for {len(spacy_data)} items")

    # get embedding data
    res = []
    with tqdm(total=len(spacy_data)) as pbar:
        for s in spacy_data:
            for chunk in s["chunks"]:
                if len(entity_filter) == 0 or chunk["entity"].label_ in entity_filter:
                    start_token = max(chunk["entity"].start-radius, 0)
                    end_token = min(chunk["entity"].end+radius, len(s["nlp"]))
                    short_sentence = s["nlp"][start_token:end_token]
                    res.append({
                        "sentence": short_sentence,
                        "chunk": str(chunk["chunk"]),
                        "label": chunk["entity"].label,
                        "label_id": chunk["entity"].label_,
                        "embedding": embed(emb_pipe, str(short_sentence))
                    })
                pbar.update(1)
    print(f"Created {len(res)} training items")
    # average all the embeddings in a sample, dropping 1st (CLS)
    for r in res:
        r["embedding"] = np.mean(r["embedding"][1:], axis=0)
    return res