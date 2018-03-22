def sort_suggestion(list_suggest, fonction):
    return list(sorted(list_suggest, key=fonction, reverse=False))


def to_int(s):
    try:
        return int(s)
    except ValueError:
        return None


def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    """Converts a text to a sequence of words (or tokens).
    # Arguments
        text: Input text (string).
        filters: Sequence of characters to filter out.
        lower: Whether to convert the input to lowercase.
        split: Sentence split marker (string).
    # Returns
        A list of words (or tokens).
    """
    if lower:
        text = text.lower()

    translate_map = str.maketrans(filters, split * len(filters))

    text = text.translate(translate_map)
    seq = text.split(split)
    return [i for i in seq if i]
