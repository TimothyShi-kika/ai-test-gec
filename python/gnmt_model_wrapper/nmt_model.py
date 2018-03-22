import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from tensorflow.python.util import nest


class NMTModel:
    def __init__(self, hparams):
        self.num_encoder_layers = hparams.num_encoder_layers
        self.num_decoder_layers = hparams.num_decoder_layers
        self.encoder_vocab_size = hparams.encoder_vocab_size
        self.decoder_vocab_size = hparams.decoder_vocab_size
        self.num_units = hparams.num_units
        self.forget_bias = hparams.forget_bias
        self.embed_size = hparams.embed_size
        self.time_major = True
        self.sos_id = hparams.sos_id
        self.eos_id = hparams.eos_id
        self.beam_width = hparams.beam_width

        self.input_data = tf.placeholder(dtype=tf.int32, shape=[1, None], name='input_word_ids')
        self.source_sequence_length = tf.placeholder(dtype=tf.int32, shape=[1], name='input_length')

        with tf.variable_scope('embeddings'):
            with tf.variable_scope('encoder'):
                self.embedding_encoder = tf.get_variable('embedding_encoder',
                                                         [self.encoder_vocab_size, self.embed_size],
                                                         dtype=tf.float32)
            with tf.variable_scope('decoder'):
                self.embedding_decoder = tf.get_variable('embedding_decoder',
                                                         [self.decoder_vocab_size, self.embed_size],
                                                         dtype=tf.float32)

        with tf.variable_scope('build_network'):
            with tf.variable_scope('decoder/output_projection'):
                self.output_layer = layers_core.Dense(self.decoder_vocab_size, use_bias=False,
                                                      name='output_projection')

        with tf.variable_scope('dynamic_seq2seq'):
            encoder_outputs, encoder_state = self._build_encoder()
            self.logits, self.sample_id, self.final_context_state = self._build_decoder(encoder_outputs, encoder_state)
        self.saver = tf.train.Saver(tf.global_variables())

    def _create_rnn_cell(self, num_layers, only_need_list=False):
        cell_list = []
        for _ in range(num_layers):
            single_cell = tf.contrib.rnn.BasicLSTMCell(self.num_units, forget_bias=self.forget_bias)
            cell_list.append(single_cell)
        if only_need_list:
            return cell_list
        if num_layers == 1:
            return cell_list[0]
        else:
            return tf.contrib.rnn.MultiRNNCell(cell_list)
        # return tf.contrib.rnn.MultiRNNCell(cell_list)

    def _build_bidirectional_rnn(self, inputs, sequence_length, num_bi_layers):
        fw_cell = self._create_rnn_cell(num_bi_layers)
        bw_cell = self._create_rnn_cell(num_bi_layers)
        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, dtype=tf.float32,
                                                               sequence_length=sequence_length,
                                                               time_major=self.time_major)
        return tf.concat(bi_outputs, -1), bi_state

    # wrapper only implements encoder/decoder of GNMT architecture. Need to be extended for various architectures
    def _build_encoder(self):
        num_bi_layers = 1
        num_uni_layers = self.num_encoder_layers - num_bi_layers
        source = self.input_data
        if self.time_major:
            source = tf.transpose(source)

        with tf.variable_scope('encoder'):
            encoder_emb_inp = tf.nn.embedding_lookup(self.embedding_encoder, source)

            bi_encoder_outputs, bi_encoder_state = self._build_bidirectional_rnn(
                encoder_emb_inp, self.source_sequence_length, num_bi_layers)

            uni_cell = self._create_rnn_cell(num_uni_layers)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(uni_cell, bi_encoder_outputs, dtype=tf.float32,
                                                               sequence_length=self.source_sequence_length,
                                                               time_major=self.time_major)
            encoder_state = (bi_encoder_state[1], ) + (encoder_state, ) if num_uni_layers == 1 else encoder_state
        return encoder_outputs, encoder_state

    def _build_decoder(self, encoder_outputs, encoder_state):
        decoding_length_factor = 2.0
        max_encoder_length = tf.reduce_max(self.source_sequence_length)
        maximum_iterations = tf.to_int32(tf.round(tf.to_float(max_encoder_length) * decoding_length_factor))
        with tf.variable_scope('decoder') as decoder_scope:
            cell, decoder_initial_state = self._build_decoder_cell(encoder_outputs, encoder_state,
                                                                   self.source_sequence_length)
            # only inference here
            assert self.beam_width > 0
            length_penalty_weight = 0.0
            # batch_size is fixed to 1
            start_tokens = tf.fill([1], self.sos_id)
            end_token = self.eos_id

            my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=cell,
                embedding=self.embedding_decoder,
                start_tokens=start_tokens,
                end_token=end_token,
                initial_state=decoder_initial_state,
                beam_width=self.beam_width,
                output_layer=self.output_layer,
                length_penalty_weight=length_penalty_weight
            )
            outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                my_decoder,
                maximum_iterations=maximum_iterations,
                output_time_major=self.time_major,
                swap_memory=True,
                scope=decoder_scope
            )
            logits = tf.no_op()
            sample_id = outputs.predicted_ids

        return logits, sample_id, final_context_state

    def _build_decoder_cell(self, encoder_outputs, encoder_state, source_sequence_length):
        num_units = self.num_units
        beam_width = self.beam_width
        dtype = tf.float32

        memory = tf.transpose(encoder_outputs, [1, 0, 2])
        memory = tf.contrib.seq2seq.tile_batch(memory, multiplier=beam_width)
        source_sequence_length = tf.contrib.seq2seq.tile_batch(source_sequence_length, multiplier=beam_width)
        encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)
        batch_size = beam_width

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units, memory,
                                                                   memory_sequence_length=source_sequence_length,
                                                                   normalize=True)

        cell_list = self._create_rnn_cell(self.num_decoder_layers, only_need_list=True)
        attention_cell = cell_list.pop(0)
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(attention_cell, attention_mechanism,
                                                             attention_layer_size=None, output_attention=False,
                                                             alignment_history=False, name='attention')
        cell = GNMTAttentionMultiCell(attention_cell, cell_list)
        decoder_initial_state = tuple(
            zs.clone(cell_state=es) if isinstance(zs, tf.contrib.seq2seq.AttentionWrapperState) else es
            for zs, es in zip(cell.zero_state(batch_size, dtype), encoder_state)
        )
        return cell, decoder_initial_state


class GNMTAttentionMultiCell(tf.nn.rnn_cell.MultiRNNCell):
    def __init__(self, attention_cell, cells):
        cells = [attention_cell] + cells
        super(GNMTAttentionMultiCell, self).__init__(cells, state_is_tuple=True)

    def __call__(self, inputs, state, scope=None):
        if not nest.is_sequence(state):
            raise ValueError(
                "Expected state to be a tuple of length %d, but received: %s"
                % (len(self.state_size), state))

        with tf.variable_scope(scope or "multi_rnn_cell"):
            new_states = []

            with tf.variable_scope("cell_0_attention"):
                attention_cell = self._cells[0]
                attention_state = state[0]
                cur_inp, new_attention_state = attention_cell(inputs, attention_state)
                new_states.append(new_attention_state)

            for i in range(1, len(self._cells)):
                with tf.variable_scope("cell_%d" % i):
                    cell = self._cells[i]
                    cur_state = state[i]
                    cur_state = cur_state._replace(h=tf.concat([cur_state.h, new_attention_state.attention], 1))
                    cur_inp, new_state = cell(cur_inp, cur_state)
                    new_states.append(new_state)

        return cur_inp, tuple(new_states)
