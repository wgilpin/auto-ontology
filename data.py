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
        current_entity = ""
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

