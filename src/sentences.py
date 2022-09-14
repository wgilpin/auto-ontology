# -*- coding: utf-8 -*-

# from https://stackoverflow.com/a/31505798/2870929

import re
ALPHABETS = "([A-Za-z])"
ALPHACAPS = "([A-Z])"
PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
STARTERS = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|" +\
           "However\s|That\s|This\s|Wherever)"
ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
WEBSITES = "[.](com|net|org|io|gov)"
DIGITS = "([0-9])"


def split(text):
    """
    Given a text, split it into sentences. handles many  edge cases
    e.g. "Mr. John Johnson Jr. was born in the U.S.A but earned his Ph.D.
            in Israel before joining Nike Inc. as an engineer. "

    :param text: The text to split.
    :return: A list of sentences.
    """
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = text.replace("- ", "")
    text = re.sub(PREFIXES, "\\1<prd>", text)
    text = re.sub(WEBSITES, "<prd>\\1", text)
    text = re.sub(DIGITS + "[.]" + DIGITS, "\\1<prd>\\2", text)
    if "..." in text:
        text = text.replace("...", "<prd><prd><prd>")
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + ALPHABETS + "[.] ", " \\1<prd> ", text)
    text = re.sub(ACRONYMS+" "+STARTERS, "\\1<stop> \\2", text)
    text = re.sub(ALPHABETS + "[.]" + ALPHABETS + "[.]" +
                  ALPHABETS + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(ALPHABETS + "[.]" + ALPHABETS +
                  "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(ALPHACAPS + ". " + ALPHACAPS +
                  ". ", "\\1<prd>\\2<prd>", text)
    text = re.sub(" "+SUFFIXES+"[.] "+STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" "+SUFFIXES+"[.]", " \\1<prd>", text)
    text = re.sub(" " + ALPHABETS + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if "\"" in text:
        text = text.replace(".\"", "\".")
    if "!" in text:
        text = text.replace("!\"", "\"!")
    if "?" in text:
        text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    # sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences
