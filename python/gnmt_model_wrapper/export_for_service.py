import tensorflow as tf
import os
from .nmt_model import NMTModel
from .config import Config


export_path = '/Users/txshi/EngineProjects/GEC_Decoder_Wrapper/tmp/best_bleu/model_for_service/'


def save_serving_model(sess, model):
    # 版本号和目录结构
    model_version = 2
    version_export_path = os.path.join(
        tf.compat.as_bytes(export_path),
        tf.compat.as_bytes(str(model_version)))
    print('Exporting trained model to', version_export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(version_export_path)

    # Build the signature_def_map.
    # inputs & outputs
    tensor_input_content = tf.saved_model.utils.build_tensor_info(model.input_data)
    tensor_input_length = tf.saved_model.utils.build_tensor_info(model.source_sequence_length)
    tensor_output = tf.saved_model.utils.build_tensor_info(model.sample_id)
    # signature_def
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            # inputs name: input
            inputs={'input_data': tensor_input_content,
                    'source_sequence_length': tensor_input_length},
            # outputs name: output_values & output indices
            outputs={'sample_id': tensor_output},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    # signature name: 'predict_words'
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'gec_nmt': prediction_signature
        })
    builder.save()
    print('Done exporting!')
    return


# how to run, 以 load 特定的 model 为例，如果训练完之后直接导出，则直接调用:
with tf.Graph().as_default():
    config = Config()
    nmt_model = NMTModel(config)

    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session,
                      '/Users/txshi/EngineProjects/GEC_Decoder_Wrapper/tmp/best_bleu/' + 'txshi.ckpt')
        save_serving_model(session, nmt_model)
