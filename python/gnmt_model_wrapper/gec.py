import tensorflow as tf
import numpy as np
import sys

from .nmt_model import NMTModel
from .config import Config
from .data_postprocess import build_vocab_dict, merge_unk_new, split_unk_new, \
    capitalize, calibrate_correction_result
from .data_preprocess import Preprocessor
from .symspellcompound.symspellcompound import SySpellCompound


def correct_sentence(raw_sentence, model, session, preprocessor,
                     learner_vocab, native_reversed_vocab):
    tokens, tags, _ = preprocessor.preprocess_token_seq(raw_sentence.strip().split())
    rev_tokens = tokens[::-1]
    rev_tags = tags[::-1]
    input_ids = [learner_vocab.get(k) if k in learner_vocab else learner_vocab.get('<unk>')
                 for k in rev_tokens]
    input_tag_ids = [learner_vocab.get(k) if k in learner_vocab else learner_vocab.get('<unk>')
                     for k in rev_tags]
    reshaped_input = np.reshape(np.array(input_ids), (1, len(input_ids)))
    reshaped_tag_input = np.reshape(np.array(input_tag_ids), (1, len(input_tag_ids)))
    output_ids = session.run(model.sample_id,
                             feed_dict={model.input_data: reshaped_input,
                                        model.source_sequence_length: np.array([len(input_ids)])})
    output_tag_ids = session.run(model.sample_id,
                                 feed_dict={model.input_data: reshaped_tag_input,
                                            model.source_sequence_length: np.array([len(input_tag_ids)])})
    if model.time_major:
        output_ids = output_ids.transpose()
        output_tag_ids = output_tag_ids.transpose()
    output_words = [native_reversed_vocab[i] for i in output_ids[0][0] if i >= 0]
    output_tags = [native_reversed_vocab[i] for i in output_tag_ids[0][0] if i >= 0]
    try:
        eos_index = output_words.index('</s>')
    except ValueError:
        eos_index = len(output_words)
    try:
        tag_eos_index = output_tags.index('</s>')
    except ValueError:
        tag_eos_index = len(output_tags)
    raw_output_tokens = output_words[:eos_index]
    raw_output_tags = output_tags[:tag_eos_index]
    capitalize(raw_output_tokens)
    output_sentence = merge_unk_new(raw_output_tokens).split()
    output_tags = merge_unk_new(raw_output_tags).split()
    output_sentence_str = ' '.join(output_sentence)
    # print('After merge_unk_new')
    # print(output_sentence)
    # print(output_tags)
    output_sentence_tokens, output_sentence_tags, output_sentence_lemmas = \
        preprocessor.preprocess_token_seq(output_sentence, need_split_unk=False)
    if output_tags != output_sentence_tags:
        output_sentence_str = calibrate_correction_result(output_sentence_tokens, output_sentence_tags,
                                                          output_sentence_lemmas, output_tags)
    return output_sentence_str


def file_trans(src_file_path, dst_file_path, model, session, learner_vocab, native_reversed_vocab):
    preprocessor = Preprocessor('data_lang8/en_vocab', 'data_lang8/misspelling_info_reversed.dict')
    with open(src_file_path, 'r', encoding='utf-8') as learner_file_in, \
            open(dst_file_path, 'w+', encoding='utf-8') as output_file:
        for line in learner_file_in:
            raw_sentence = line.strip()
            output_sentence = correct_sentence(raw_sentence, model, session, preprocessor,
                                               learner_vocab, native_reversed_vocab)
            output_file.write(output_sentence + '\n')


def inter_trans(model, session, learner_vocab, native_reversed_vocab):
    preprocessor = Preprocessor('data_lang8/en_vocab', 'data_lang8/misspelling_info_reversed.dict')
    print('Input sentence now')
    while True:
        raw_sentence = input()
        if raw_sentence == 'end' or not raw_sentence.strip():
            break
        print(correct_sentence(raw_sentence, model, session, preprocessor,
                               learner_vocab, native_reversed_vocab))


def main(argv):
    session = tf.InteractiveSession()
    config = Config()
    gnmt_model = NMTModel(config)
    # spell_checker = SySpellCompound()

    # ckpt = argv[1]
    ckpt_num = 225000
    ckpt = 'tmp/best_bleu/translate.ckpt-{}'.format(ckpt_num)
    gnmt_model.saver.restore(session, ckpt)
    learner_vocab, _ = build_vocab_dict('tmp/en_vocab')
    _, native_reversed_vocab = build_vocab_dict('tmp/en_vocab')
    print('Model restored')
    # src_file_path = '/Users/txshi/EngineProjects/m2scorer/conll2014_test'
    # dst_file_path = '/Users/txshi/EngineProjects/m2scorer/conll2014_test_pred.{}'.format(ckpt_num)
    src_file_path = 'data_lang8/test.tingxun.learner'
    dst_file_path = 'data_lang8/test.tingxun.pred.{}'.format(ckpt_num)
    file_trans(src_file_path, dst_file_path, gnmt_model, session, learner_vocab, native_reversed_vocab)
    # inter_trans(gnmt_model, session, learner_vocab, native_reversed_vocab)
    # gnmt_model.saver.save(session, '/Users/txshi/EngineProjects/GEC_Decoder_Wrapper/tmp/best_bleu/txshi.ckpt')


if __name__ == '__main__':
    tf.app.run(main=main, argv=sys.argv)
