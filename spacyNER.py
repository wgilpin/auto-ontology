# %%
# Perform standard imports
import spacy
from os import path

nlp = spacy.load('en_core_web_sm')

# %%
def get_sample(length:int=10)->list:
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



    sample = lines[0:length]
    sample = list(filter(lambda s: s[0].isalpha(), sample))
    return sample

# %%

# remove stop words

def remove_stop_words(chunk):
    filtered_toks = []
    for tok in chunk:
        if not tok.is_stop:
            filtered_toks.append(tok.text)
    return " ".join(filtered_toks)

# %%

def merge_chunks_ents(document) -> list:
    ents = document.ents
    e_idx = 0
    out = []
    for chunk in document.noun_chunks:
        no_stops = remove_stop_words(chunk)
        while (e_idx < len(ents)) and (ents[e_idx].start_char >= chunk.start_char) and (ents[e_idx].end_char <= chunk.end_char):
            # this entity is in the noun chunk
            out.append({
                "chunk": no_stops,
                "entity": ents[e_idx],
            })
            e_idx += 1
    return out


# %%
def get_data(length:int=10)->list:
    sentences = []
    for l in get_sample(length):
        doc = nlp(l)
        merged = merge_chunks_ents(nlp(l))
        sentences.append ({'sentence': l, 'nlp': doc, 'chunks': merged })
    return sentences

# %%
for  s in get_data():
    spacy.displacy.render(s['nlp'], style='ent')
    print(f"\"{s['sentence']}\"")
    for c in s['chunks']:
        print(f"\"{c['entity']}\"({c['entity'].label_}) in \"{c['chunk']}\"")
    print("--------------------------------------------------------------")
# %%
