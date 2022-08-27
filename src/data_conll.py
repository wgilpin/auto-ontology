from os import path
from datasets import load_dataset
from tqdm import tqdm


def get_sample_conll(length: int = 10) -> list[str]:
    """
    Returns a list of conll formatted sentences.
    """
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


def get_sample_conll_hf(length: int = 0, train: bool=True) -> list[str]:
    """
    Returns a list of conll formatted sentences from HF dataset
    """
    ds_name = 'train' if train else 'test'
    dataset = load_dataset("conll2003")
    if length <= 0:
        length = len(dataset[ds_name])
    length = min(length, len(dataset[ds_name]))

    lines = []
    print("Reading lines")
    for line in tqdm(dataset[ds_name].data['tokens'][0:length].to_pylist()):
        alpha_toks = filter(lambda t: t[0].isalnum(), line)
        sent = ' '.join(alpha_toks)
        lines.append(sent)

    return lines
