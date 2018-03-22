import collections
from ast import literal_eval
import spacy
from random import random, shuffle


vocab_file_path = '../../data_lang8/en_vocab'
vocab = set()
misspelling_correction_reversed_dict = dict()
nlp = spacy.load('en')
tmp_vocab_words_to_be_corrected = set()


def initialize():
    with open(vocab_file_path, 'r', encoding='utf-8') as vocab_file:
        for line in vocab_file:
            word = line.strip()
            try:
                assert word not in vocab
            except AssertionError:
                print(word)
                exit(6)
            vocab.add(word)
    print('vocab constructed')
    global misspelling_correction_reversed_dict
    with open('../../data_lang8/misspelling_info_reversed.dict', 'r', encoding='utf-8') as f:
        misspelling_correction_reversed_dict = literal_eval(f.readline().strip())
    print('misspelling corrector constructed')


def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def recover_word(word):
    if word.title() in vocab:
        return word.title()
    global misspelling_correction_reversed_dict
    if word not in misspelling_correction_reversed_dict:
        if word.lower() in misspelling_correction_reversed_dict:
            word = word.lower()
        else:
            return word
    if not misspelling_correction_reversed_dict[word]:
        tmp_vocab_words_to_be_corrected.add(word)
        return word
    r = random()
    cur_p = 0
    for (w, p) in misspelling_correction_reversed_dict[word].items():
        cur_p += p
        if cur_p > r:
            return w


def split_dataset(merged_fp, learner_fp, native_fp, only_corrected=False):
    with open(merged_fp, 'r', encoding='utf-8') as merged_file, \
            open(learner_fp, 'w+', encoding='utf-8') as learner_file, \
            open(native_fp, 'w+', encoding='utf-8') as native_file:
        for line in merged_file:
            line = line.strip()
            segs = line.split('\t')
            if len(segs) > 1 and segs[0].strip() == segs[1].strip():
                continue
            if only_corrected and len(segs) == 1:
                continue
            learner_line = segs[0].strip()
            native_line = segs[0 if len(segs) == 1 else 1].strip()

            learner_file.write(learner_line + '\n')
            native_file.write(native_line + '\n')


def preprocess_token_seq(tokens, need_tag=False):
    # correct misspellings (should be improved)
    tokens = [x.lower() if x not in vocab and x.lower() in vocab else x for x in tokens]
    tokens = [x if x in vocab else recover_word(x) for x in tokens]

    doc = nlp(' '.join(tokens))
    tags = []
    for t in doc:
        if t.pos_ in {'VERB', 'ADJ', 'ADV', 'NOUN', 'INTJ', 'PROPN'} and \
                t.tag_ not in {'AFX', 'EX', 'HVS', 'PRP$', 'WDT', 'WP', 'WP$', 'WRB'}:
            tags.append('TAG_' + t.tag_)
        else:
            tags.append(t.text)
    tokens = [x if x in vocab else ['<u>'] + list(x) + ['</u>'] for x in tokens]
    tokens = [x for x in flatten(tokens)]

    tags = [x if x in vocab else ['<u>'] + list(x) + ['</u>'] for x in tags]
    tags = [x for x in flatten(tags)]

    return tokens, tags


def preprocess_file(raw_fp, final_fp, need_reverse=False):
    with open(raw_fp, 'r', encoding='utf-8') as raw_file, \
            open(final_fp, 'w+', encoding='utf-8') as final_file:
        for line in raw_file:
            tokens = line.strip().split()
            if not tokens:
                continue
            tokens, tags = preprocess_token_seq(tokens)
            if need_reverse:
                tokens = tokens[::-1]
                tags = tags[::-1]
            final_file.write(' '.join(tokens) + '\n')
            final_file.write(' '.join(tags) + '\n')


def test():
    initialize()
    tokens = 'I \'m not sure how to tip at restaulant because I \'ve just moved in America and my mother country Japan does n\'t have such a custom .'.strip().split()
    print(recover_word('mother'))


def main():
    initialize()
    split_dataset('../../data_lang8/stripped.train', '../../data_lang8/learner.train.raw',
                  '../../data_lang8/native.train.raw', only_corrected=True)
    split_dataset('../../data_lang8/stripped.dev',
                  '../../data_lang8/learner.dev.raw', '../../data_lang8/native.dev.raw',
                  only_corrected=True)
    split_dataset('../../data_lang8/stripped.test',
                  '../../data_lang8/learner.test.raw', '../../data_lang8/native.test.raw',
                  only_corrected=True)
    preprocess_file('../../data_lang8/learner.train.raw', '../../data_lang8/train.learner', True)
    print('Training data (learner) is processed')
    preprocess_file('../../data_lang8/native.train.raw', '../../data_lang8/train.native')
    print('Training data (native) is processed')
    preprocess_file('../../data_lang8/learner.dev.raw', '../../data_lang8/dev.learner', True)
    print('Dev data (learner) is processed')
    preprocess_file('../../data_lang8/native.dev.raw', '../../data_lang8/dev.native')
    print('Dev data (native) is processed')
    preprocess_file('../../data_lang8/learner.test.raw', '../../data_lang8/test.learner', True)
    print('Test data (learner) is processed')
    preprocess_file('../../data_lang8/native.test.raw', '../../data_lang8/test.native')
    print('Test data (native) is processed')
    print(len(tmp_vocab_words_to_be_corrected))
    print(tmp_vocab_words_to_be_corrected)


def shuf_file(learner_file_path, native_file_path, learner_shuf_file_path, native_shuf_file_path):
    with open(learner_file_path, 'r', encoding='utf-8') as learner_file, \
            open(native_file_path, 'r', encoding='utf-8') as native_file, \
            open(learner_shuf_file_path, 'w+', encoding='utf-8') as l_shuf_file, \
            open(native_shuf_file_path, 'w+', encoding='utf-8') as n_shuf_file:
        l_lines = learner_file.readlines()
        n_lines = native_file.readlines()
        c = list(zip(l_lines, n_lines))
        for _ in range(10):
            shuffle(c)
        l_lines, n_lines = zip(*c)
        l_shuf_file.writelines(l_lines)
        n_shuf_file.writelines(n_lines)


if __name__ == '__main__':
    main()
    shuf_file('../../data_lang8/train.learner', '../../data_lang8/train.native',
              '../../data_lang8/train_shuf.learner', '../../data_lang8/train_shuf.native')
    shuf_file('../../data_lang8/dev.learner', '../../data_lang8/dev.native',
              '../../data_lang8/dev_shuf.learner', '../../data_lang8/dev_shuf.native')
    shuf_file('../../data_lang8/test.learner', '../../data_lang8/test.native',
              '../../data_lang8/test_shuf.learner', '../../data_lang8/test_shuf.native')
