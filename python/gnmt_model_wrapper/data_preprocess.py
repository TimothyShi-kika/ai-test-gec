import collections
from ast import literal_eval
import spacy
from random import random
import sys


class Preprocessor:
    def __init__(self, vocab_file_path, misspelling_info_reversed_file_path):
        self.vocab = set()
        self.misspelling_correction_reversed_dict = dict()

        with open(vocab_file_path, 'r', encoding='utf-8') as vocab_file:
            for line in vocab_file:
                word = line.strip()
                try:
                    assert word not in self.vocab
                except AssertionError:
                    print(word)
                    sys.exit('vocab file has duplicated items')
                self.vocab.add(word)
        print('vocab constructed')
        with open(misspelling_info_reversed_file_path, 'r', encoding='utf-8') as f:
            self.misspelling_correction_reversed_dict = literal_eval(f.readline().strip())
        print('misspelling corrector constructed')

    def recover_word(self, word):
        if word.title() in self.vocab:
            return word.title()
        if word not in self.misspelling_correction_reversed_dict:
            if word.lower() in self.misspelling_correction_reversed_dict:
                word = word.lower()
            else:
                return word
        if not self.misspelling_correction_reversed_dict[word]:
            print('Preprocessor::recover_word: ' + word)
            return word
        r = random()
        cur_p = 0
        for (w, p) in self.misspelling_correction_reversed_dict[word].items():
            cur_p += p
            if cur_p > r:
                return w

    def flatten(self, l):
        for el in l:
            if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
                yield from self.flatten(el)
            else:
                yield el

    def preprocess_token_seq(self, tokens, need_split_unk=True):
        # correct misspellings (should be improved)
        tokens = [x.lower() if x not in self.vocab and x.lower() in self.vocab else x for x in tokens]
        tokens = [x if x in self.vocab else self.recover_word(x) for x in tokens]
        nlp = spacy.load('en')
        doc = nlp(' '.join(tokens))
        tags = []
        for t in doc:
            if t.pos_ in {'VERB', 'ADJ', 'ADV', 'NOUN', 'PROPN'} and \
                    t.tag_ not in {'AFX', 'EX', 'HVS', 'PRP$', 'WDT', 'WP', 'WP$', 'WRB'}:
                if t.text in {'am', 'is', 'are', 'was', 'were'}:
                    tags.append(t.text)
                else:
                    tags.append('TAG_' + t.tag_)
            else:
                tags.append(t.text)

        if need_split_unk:
            tokens = [x if x in self.vocab else ['<u>'] + list(x) + ['</u>'] for x in tokens]
            tokens = [x for x in self.flatten(tokens)]

            tags = [x if x in self.vocab else ['<u>'] + list(x) + ['</u>'] for x in tags]
            tags = [x for x in self.flatten(tags)]

        lemmas = [t.lemma for t in doc]
        for i in range(len(lemmas)):
            if lemmas[i] == '-PRON-':
                lemmas[i] = tokens[i]

        return tokens, tags, lemmas

