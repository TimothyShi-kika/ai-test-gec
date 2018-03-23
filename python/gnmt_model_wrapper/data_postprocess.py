from pattern.text.en import *


def build_vocab_dict(vocab_file_path):
    vocab_dict = {}
    reversed_vocab_dict = {}
    with open(vocab_file_path, 'r', encoding='utf-8') as vocab_file:
        for (i, line) in enumerate(vocab_file):
            word = line.strip()
            vocab_dict[word] = i
            reversed_vocab_dict[i] = word
    return vocab_dict, reversed_vocab_dict


def split_unk(sentence, vocab):
    tokens = sentence.strip().split()
    output_tokens = []
    for (i, t) in enumerate(tokens):
        if t in vocab:
            output_tokens.append(t)
        else:
            output_tokens += (list(t))
        if i < len(tokens) - 1:
            output_tokens.append('<ss>')
    return output_tokens


def split_unk_new(sentence, vocab):
    tokens = sentence.strip().split()
    output_tokens = []
    for (i, t) in enumerate(tokens):
        if t in vocab:
            output_tokens.append(t)
        else:
            output_tokens += (['<u>'] + list(t) + ['</u>'])
    return output_tokens[::-1]


def capitalize(tokens):
    cap_flag = True
    for (i, _) in enumerate(tokens):
        if tokens[i] in {'<u>', '</u>'}:
            continue
        if cap_flag:
            tokens[i] = tokens[i].title()
            cap_flag = False
        else:
            if tokens[i] in {'.', '?', '!'}:
                cap_flag = True


def merge_unk_new(tokens):
    unk_buffer = []
    output_buffer = []
    i = 0
    while i < len(tokens):
        if tokens[i] == '<u>':
            i += 1
            while i < len(tokens) and tokens[i] != '</u>':
                unk_buffer.append(tokens[i])
                i += 1
            if i < len(tokens) and tokens[i] == '</u>':
                i += 1
            output_buffer.append(''.join(unk_buffer))
            unk_buffer = []
        else:
            output_buffer.append(tokens[i])
            i += 1
    return ' '.join(output_buffer)


def calibrate_correction_result(res_sent_tokens, res_sent_tags, res_sent_lemmas, pred_tags):
    print('Calibrate: res_sent_tokens = {}, res_sent_tags = {}, pred_tags = {}'.format(
        res_sent_tokens, res_sent_tags, pred_tags))
    if len(res_sent_tokens) != len(res_sent_tags):
        return ' '.join(res_sent_tokens)
    if len(res_sent_tags) < len(pred_tags):
        return ' '.join(res_sent_tokens)
    src_num_tags = sum([1 for x in res_sent_tags if x.startswith('TAG_')])
    dst_num_tags = sum([1 for x in pred_tags if x.startswith('TAG_')])
    if src_num_tags != dst_num_tags:
        return ' '.join(res_sent_tokens)
    # greedy calibration, needs further refinement
    # definitely not applicable to passive -> active transformation
    tag_pos_dict = {
        'TAG_BES': 'VERB',
        'TAG_CD': 'NUM',
        'TAG_EX': 'ADV',
        'TAG_JJ': 'ADJ',
        'TAG_JJR': 'ADJ',
        'TAG_JJS': 'ADJ',
        'TAG_MD': 'VERB',
        'TAG_NN': 'NOUN',
        'TAG_NNP': 'PROPN',
        'TAG_NNPS': 'PROPN',
        'TAG_NNS': 'NOUN',
        'TAG_PDT': 'ADJ',
        'TAG_RB': 'ADV',
        'TAG_RBR': 'ADV',
        'TAG_RBS': 'ADV',
        'TAG_RP': 'PART',
        'TAG_VB': 'VERB',
        'TAG_VBD': 'VERB',
        'TAG_VBG': 'VERB',
        'TAG_VBN': 'VERB',
        'TAG_VBP': 'VERB',
        'TAG_VBZ': 'VERB'
    }
    seen_indices = set()
    i = 0
    output_tokens = []
    while i < len(pred_tags):
        while i < len(pred_tags) and not pred_tags[i].startswith('TAG_'):
            output_tokens.append(pred_tags[i])
            i += 1
        if i < len(pred_tags):
            dst_pos = tag_pos_dict[pred_tags[i]]
            found_corresponding_tag = False
            for (j, (src_lemma, src_tag)) in enumerate(zip(res_sent_lemmas, res_sent_tags)):
                if j not in seen_indices and src_tag.startswith('TAG_') and tag_pos_dict[src_tag] == dst_pos:
                    found_corresponding_tag = True
                    break
            if not found_corresponding_tag:
                return ' '.join(res_sent_tokens)
            seen_indices.add(j)
            if src_tag != pred_tags[i]:
                if res_sent_tokens[j] in {'am', 'are', 'was', 'were'}:
                    output_tokens.append(res_sent_tokens[j])
                elif pred_tags[i] in {'NN', 'NNP', 'JJ', 'RB', 'VB', 'VBP'}:
                    output_tokens.append(src_lemma)
                elif pred_tags[i] in {'NNS', 'NNPS'}:
                    output_tokens.append(pluralize(src_lemma))
                elif pred_tags[i] in {'RBR', 'JJR'}:
                    output_tokens.append(comparative(src_lemma))
                elif pred_tags[i] in {'RBS', 'JJS'}:
                    output_tokens.append(superlative(src_lemma))
                elif pred_tags[i] == 'VBD':
                    output_tokens.append(conjugate(src_lemma, PAST))
                elif pred_tags[i] == 'VBG':
                    output_tokens.append(conjugate(src_lemma, PARTICIPLE))
                elif pred_tags[i] == 'VBN':
                    output_tokens.append(conjugate(src_lemma, PAST, aspect=PROGRESSIVE))
                elif pred_tags[i] == 'VBZ':
                    output_tokens.append(conjugate(src_lemma, PRESENT, 3, SINGULAR))
            else:
                output_tokens.append(res_sent_tokens[j])
            i += 1

    try:
        if len(output_tokens) == len(res_sent_tokens):
            capitalize(output_tokens)
            return ' '.join(output_tokens)
        else:
            capitalize(res_sent_tokens)
            return ' '.join(res_sent_tokens)
    except TypeError:
        return ''


def recover_unwanted_words(final_context):
    pass
