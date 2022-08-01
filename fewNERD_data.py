# %%
# Perform standard imports
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

from timer import timer

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
    print('Dataset loaded')

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
            
            # print(f'"{sentence}"')

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
                        # print(f"entity: {ent_text} ({ent_label})")

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
            
    print('Dataset processed')
    return result

@timer
def create_training_data_per_entity_fewNERD(
        length: int = 10,
        radius: int = 7,
        entity_filter: list[str] = []) -> list:
    """
    Creates training data for the autoencoder.
    """
    from tqdm import tqdm
    print("Making pipe")
    from extract_bert_features import embed, make_pipe
    from transformers import TFDistilBertModel
    emb_pipe = make_pipe('distilbert-base-uncased', TFDistilBertModel)

    print(
        f"Create Training Data for {length if length > 0 else 'all'} items, radius {radius}")
    # get spacy data
    fewNERD_data = read_fewNERD(length)

    print(f"Created NER Data for {len(fewNERD_data)} items")

    # get embedding data
    res = []
    with tqdm(total=len(fewNERD_data)) as pbar:
        for s in fewNERD_data:
            for chunk in s["entities"]:
                if len(entity_filter) == 0 or chunk["label"] in entity_filter:
                    start_token = max(chunk["start"]-radius, 0)
                    end_token = min(chunk["end"]+radius, len(s["words"]))
                    short_sentence = ' '.join(s['words'][start_token:end_token])
                    res.append({
                        "sentence": short_sentence,
                        "chunk": short_sentence,
                        "label": chunk["label"],
                        "label_id": chunk["label_id"],
                        "embedding": embed(emb_pipe, str(short_sentence))
                    })
            pbar.update(1)
    print(f"Created {len(res)} training items")

    # average all the embeddings in a sample, dropping 1st (CLS)
    with tqdm(total=len(res)) as pbar:
        for r in res:
            r["embedding"] = np.mean(r["embedding"][1:], axis=0)
            pbar.update(1)
        return res
