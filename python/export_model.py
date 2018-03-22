import tensorflow as tf
import os
from tensorflow.python.ops import lookup_ops
from .gnmt_model import GNMTModel
from .utils import iterator_utils, vocab_utils

session = tf.InteractiveSession()
hparams = tf.contrib.training.HParams(
    # Data
    src="",
    tgt="",
    train_prefix="",
    dev_prefix="",
    test_prefix="",
    vocab_prefix="",
    embed_prefix="",
    out_dir=".",

    # Networks
    num_units=1024,
    num_layers=2,
    num_encoder_layers=2,
    num_decoder_layers=2,
    num_residual_layers=0,
    dropout=0.2,
    unit_type="lstm",
    encoder_type="gnmt",
    residual=False,
    time_major=True,
    num_embeddings_partitions=0,

    # Attention mechanisms
    attention="normed_bahdanau",
    attention_architecture="gnmt_v2",
    output_attention=True,
    pass_hidden_state=True,

    # Train
    optimizer="sgd",
    batch_size=128,
    init_op="uniform",
    init_weight=0.1,
    max_gradient_norm=5.0,
    learning_rate=1.0,
    warmup_steps=0,
    warmup_scheme="t2t",
    colocate_gradients_with_ops=True,
    start_decay_step=130000,
    decay_steps=17000,
    decay_factor=0.5,
    num_train_steps=340000,

    # Data constraints
    num_buckets=5,
    max_train=0,
    src_max_len=50,
    tgt_max_len=50,
    src_max_len_infer=0,
    tgt_max_len_infer=0,

    # Data format
    sos="<s>",
    eos="</s>",
    subword_option="",
    check_special_token=True,

    # Misc
    forget_bias=1.0,
    num_gpus=1,
    epoch_step=0,  # record where we were within an epoch.
    steps_per_stats=100,
    steps_per_external_eval=0,
    share_vocab=False,
    metrics=["bleu"],
    log_device_placement=False,
    random_seed=None,
    # only enable beam search during inference when beam_width > 0.
    beam_width=0,
    length_penalty_weight=0.0,
    override_loaded_hparams=True,
    num_keep_ckpts=5,
    avg_ckpts=False,

    # For inference
    inference_indices=None,
    infer_batch_size=1,
    sampling_temperature=0.0,
    num_translations_per_input=1,
)

hparams.tgt_vocab_file = '../tmp/vocab.word.native'
hparams.src_vocab_file = '../tmp/vocab.word.learner'
hparams.beam_width = 5
hparams.src_vocab_size = 20097
hparams.tgt_vocab_size = 20097
hparams.attention_architecture = 'gnmt_v2'
hparams.infer_batch_size = 1

src_vocab_file = hparams.src_vocab_file
tgt_vocab_file = hparams.tgt_vocab_file
src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
    src_vocab_file, tgt_vocab_file, False)
reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
    tgt_vocab_file, default_value=vocab_utils.UNK)
src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
src_dataset = tf.data.Dataset.from_tensor_slices(src_placeholder)
iterator = iterator_utils.get_infer_iterator(
    src_dataset,
    src_vocab_table,
    batch_size=batch_size_placeholder,
    eos=hparams.eos,
    src_max_len=hparams.src_max_len_infer,
    source_reverse=False
)

model_version = 4
export_dir = './export'
version_export_path = os.path.join(tf.compat.as_bytes(export_dir), tf.compat.as_bytes(str(model_version)))
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

model = GNMTModel(hparams,
                  tf.contrib.learn.ModeKeys.INFER,
                  iterator,
                  src_vocab_table,
                  tgt_vocab_table,
                  reverse_tgt_vocab_table)
tensor_input = tf.saved_model.utils.build_tensor_info(model.iterator)
tensor_output_values = tf.saved_model.utils.build_tensor_info(model.sample_words)

prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'iterator': tensor_input},
        outputs={'sample_words': tensor_output_values},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )
)
builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING],
                                     signature_def_map={'predict_words': prediction_signature})
builder.save()
