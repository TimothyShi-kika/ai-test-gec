import random


def build_new_test_file(ori_test_learner_fp, ori_test_native_fp, new_test_src_fp, new_test_tgt_fp):
    with open(ori_test_learner_fp, 'r', encoding='utf-8') as ori_test_learner_file, \
            open(ori_test_native_fp, 'r', encoding='utf-8') as ori_test_native_file, \
            open(new_test_src_fp, 'w+', encoding='utf-8') as new_test_src_file, \
            open(new_test_tgt_fp, 'w+', encoding='utf-8') as new_test_tgt_file:
        for (ori_test_learner_line, ori_test_native_line) in zip(ori_test_learner_file, ori_test_native_file):
            if random.random() < 0.5:
                new_test_src_file.write(ori_test_native_line)
            else:
                new_test_src_file.write(' '.join(ori_test_learner_line.strip().split()[::-1]) + '\n')
            new_test_tgt_file.write(ori_test_native_line)
            ori_test_learner_file.readline()
            ori_test_native_file.readline()


def merge_unk(line):
    tokens = line.strip().split()
    new_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i] != '<u>':
            new_tokens.append(tokens[i])
            i += 1
        else:
            new_word = ''
            i += 1
            while i < len(tokens) and tokens[i] != '</u>':
                new_word += tokens[i]
                i += 1
            i += 1
            new_tokens.append(new_word)
    return ' '.join(new_tokens) + '\n'


def evaluate(learner_fp, native_fp, pred_fp, false_positive_fp, ignore_case=True):
    with open(learner_fp, 'r', encoding='utf-8') as learner_file, \
            open(native_fp, 'r', encoding='utf-8') as native_file, \
            open(pred_fp, 'r', encoding='utf-8') as pred_file, \
            open(false_positive_fp, 'w+', encoding='utf-8') as false_positive_file:
        ntp = 0
        nfp = 0
        nfn = 0
        n_wrong2wrong = 0
        n_blank = 0
        n_pos = 0
        n_neg = 0
        for (learner_line, native_line, pred_line) in zip(learner_file, native_file, pred_file):
            if not pred_line.strip():
                n_blank += 1
            learner_line = merge_unk(learner_line)
            native_line = merge_unk(native_line)
            if ignore_case:
                learner_line = learner_line.lower()
                native_line = native_line.lower()
                pred_line = pred_line.lower()
            if learner_line == native_line:
                if pred_line != native_line:
                    nfp += 1
                    false_positive_file.write(native_line)
                    false_positive_file.write(pred_line)
                    false_positive_file.write('\n')
                # else:
                #     ntp += 1
                n_pos += 1

            else:
                if pred_line == native_line:
                    ntp += 1
                else:
                    nfn += 1
                    if pred_line != learner_line:
                        n_wrong2wrong += 1
                n_neg += 1
    print('n_pos = {}'.format(n_pos))
    print('n_neg = {}'.format(n_neg))
    print('ntp = {}'.format(ntp))
    print('nfp = {}'.format(nfp))
    print('nfn = {}'.format(nfn))
    print('precision = {}'.format(float(ntp) / (ntp + nfp)))
    print('recall = {}'.format(float(ntp) / (ntp + nfn)))
    print('wrong to wrong = {}'.format(n_wrong2wrong))
    print('blank line num = {}'.format(n_blank))


# build_new_test_file('data_lang8/test.learner', 'data_lang8/test.native',
#                     'data_lang8/test.tingxun.lang8.learner', 'data_lang8/test.tingxun.lang8.native')
evaluate('data_lang8/test.tingxun.lang8.learner', 'data_lang8/test.tingxun.lang8.native',
         'data_lang8/test.tingxun.lang8.pred.225000', 'data_lang8/test.tingxun.lang8.pred.225000.fp')
# print(merge_unk('it sounds really tough and hard & <u> n b s p </u> ; for me . . . . . .'))
