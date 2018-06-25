import logging
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.python.estimator.export import export_output
from tensorflow.python.layers import core as layers_core

from utilities.constants import NUM_UNITS_HRE_UTTERANCE_CELL, NUM_UNITS_HRE_CONTEXT_CELL, NUM_HOPS, \
    WORD_VEC_DIM, ENCODER_VOCABUALRY_SIZE, LEARNING_RATE, OPTIMIZER, LOGITS, \
    WORD_PROBABILITIES, TOKEN_IDS, TARGET_SOS_ID, TARGET_EOS_ID, MAX_NUM_UTTER_TOKENS, BATCH_SIZE, \
    ENCODER_NUM_TRAINABLE_TOKENS, \
    DIALOGUES, DECODER_NUM_TRAINABLE_TOKENS, DECODER_VOCABUALRY_SIZE
from utilities.tensorflow_estimator_utils import get_estimator_specification, get_optimizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class CSQANetwork(object):

    def __init__(self, initial_encoder_embeddings=None, initial_decoder_embeddings=None):
        self.initial_encoder_embeddings = initial_encoder_embeddings
        self.initial_decoder_embeddings = initial_decoder_embeddings
        self.encoder_embeddings = None
        self.decoder_embeddings = None

    def initialize_embedding_layer(self, encoder_embeddings, enocoder_num_trainable_tokens, encoder_vocab_size,
                                   decoder_embeddings, decoder_num_trainable_tokens, decoder_vocab_size, word_vec_dim):

        with tf.variable_scope('embedding_layer'):
            # Initialize encoder embeddings
            if encoder_embeddings is not None:
                # Embeddings for unknown token, start of sentence token etc. are trainable
                trainable_encoder_embeddings = self.initial_encoder_embeddings[0:enocoder_num_trainable_tokens]
                encoder_embeddings = self.initial_encoder_embeddings[enocoder_num_trainable_tokens:]

                initializer = tf.constant_initializer(encoder_embeddings)

                trainable_encoder_embeddings = tf.get_variable(name="trainable_encoder_embeddings",
                                                               shape=[enocoder_num_trainable_tokens, word_vec_dim],
                                                               initializer=tf.constant_initializer(
                                                                   trainable_encoder_embeddings),
                                                               trainable=True)

                pretrained_encoders_embeddings = tf.get_variable(name="pretrained_encoder_embeddings",
                                                                 shape=[
                                                                     encoder_vocab_size - enocoder_num_trainable_tokens,
                                                                     word_vec_dim],
                                                                 initializer=initializer,
                                                                 trainable=False)

                encoder_embeddings = tf.concat([trainable_encoder_embeddings, pretrained_encoders_embeddings],
                                               name='encoder_embeddings', axis=0)

            else:
                encoder_embeddings = tf.get_variable(name="encoder_embeddings",
                                                     shape=[encoder_vocab_size, word_vec_dim],
                                                     initializer=tf.random_normal_initializer(),
                                                     trainable=True)

            # ------------Initialize decoder embeddings---------------
            if decoder_embeddings is not None:
                # Embeddings for unknown token, start of sentence token etc. are trainable
                trainable_decoder_embeddings = self.initial_decoder_embeddings[0:decoder_num_trainable_tokens]
                decoder_embeddings = self.initial_decoder_embeddings[decoder_num_trainable_tokens:]

                initializer = tf.constant_initializer(decoder_embeddings)

                trainable_decoder_embeddings = tf.get_variable(name="trainable_decoder_embeddings",
                                                               shape=[decoder_num_trainable_tokens, word_vec_dim],
                                                               initializer=tf.constant_initializer(
                                                                   trainable_decoder_embeddings),
                                                               trainable=True)

                pretrained_decoder_embeddings = tf.get_variable(name="pretrained_decoder_embeddings",
                                                                shape=[
                                                                    decoder_vocab_size - decoder_num_trainable_tokens,
                                                                    word_vec_dim],
                                                                initializer=initializer,
                                                                trainable=False)

                decoder_embeddings = tf.concat([trainable_decoder_embeddings, pretrained_decoder_embeddings],
                                               name='embeddings', axis=0)

            else:
                decoder_embeddings = tf.get_variable(name="embeddings",
                                                     shape=[encoder_vocab_size, word_vec_dim],
                                                     initializer=tf.random_normal_initializer(),
                                                     trainable=True)

            return encoder_embeddings, decoder_embeddings

    def model_fct(self, features, labels, mode, params):
        """
        This function defines the neural network called by tf.Estimator
        :param features: Features of instances
        :param labels: Responses to last question in each batch
        :param mode: Defines in which mode function is called (train, eval or test)
        :param params: Dictionary containing model parameters
        :return:
        """
        dialgoue_representations = []
        batch_dialogues = features[DIALOGUES]

        # TODO: Change until 30.05.2018
        embedded_keys = features['keys_embedded']
        embedded_values = features['values_embedded']

        if self.encoder_embeddings is None or self.decoder_embeddings is None:

            self.encoder_embeddings, self.decoder_embeddings = self.initialize_embedding_layer(
                encoder_embeddings = self.initial_encoder_embeddings,
                enocoder_num_trainable_tokens = params[ENCODER_NUM_TRAINABLE_TOKENS],
                encoder_vocab_size = params[ENCODER_VOCABUALRY_SIZE],
                decoder_embeddings = self.initial_decoder_embeddings,
                decoder_num_trainable_tokens = params[DECODER_NUM_TRAINABLE_TOKENS],
                decoder_vocab_size = params[DECODER_VOCABUALRY_SIZE],
                word_vec_dim=params[WORD_VEC_DIM])


        for i in range(params[BATCH_SIZE]):
            # Shape: [num_utterances, max_utter_length]
            dialogue = batch_dialogues[i]

            # Shape: (num_utterances, max_utter_length, vec_dimension)
            embedded_dialogue = tf.nn.embedding_lookup(self.encoder_embeddings, dialogue)
            # In training or eval mode the first embedded token represents <s>
            # In predict mode we will add on the fly the embedding for <s>
            embedded_decoder_input = embedded_dialogue[-1]
            embedded_decoder_input = tf.expand_dims(input=embedded_decoder_input, axis=0)
            # Remove last response since it is saved separately in embedded_decoder_input
            embedded_dialogue = embedded_dialogue[:-1]

            # In training and eval mode, last embedded token is </s>
            embedded_target = tf.nn.embedding_lookup(self.encoder_embeddings, labels[i])
            embedded_target = tf.expand_dims(input=embedded_target, axis=0)

            sequenece_lengths = self._compute_sequence_lengths(embedded_dialogue)
            sequenece_lengths_responses = self._compute_sequence_lengths(embedded_target)
            key_cells = embedded_keys
            value_cells = embedded_values

            # ----------------Hierarchical Encoder----------------
            with tf.variable_scope('utterance_level_encoder'):
                utterance_level_encoder = tf.nn.rnn_cell.LSTMCell(num_units=params[NUM_UNITS_HRE_UTTERANCE_CELL])

                # state_tuple is a tuple containing the last hidden state and the last activation
                _, lstm_state_tuple_utter_level = tf.nn.dynamic_rnn(
                    cell=utterance_level_encoder,
                    dtype=tf.float32,
                    sequence_length=sequenece_lengths,
                    inputs=embedded_dialogue)

                # For each sequence extract the last hidden state
                # Shape of last [num_utterances, NUM_UNITS_IN_LSTM_CELL]
                utterances_last_hidden_states = lstm_state_tuple_utter_level[0]
                shape_last_hidden_states = tf.shape(utterances_last_hidden_states)
                num_context_utterances = shape_last_hidden_states[0]
                dimension_hidden_utterance_state = shape_last_hidden_states[1]

                # Reshape [num_utterances, NUM_UNITS_IN_LSTM_CELL] to [num_utterances,
                # 1, NUM_UNITS_IN_LSTM_CELL]
                # since each hidden state represents a summary with NUM_UNITS_IN_LSTM_CELL features
                utterances_last_hidden_states = tf.reshape(utterances_last_hidden_states, shape=(
                    num_context_utterances, 1, params[NUM_UNITS_HRE_UTTERANCE_CELL]))

            with tf.variable_scope('context_level_encoder'):
                context_level_encoder = tf.nn.rnn_cell.LSTMCell(num_units=params[NUM_UNITS_HRE_CONTEXT_CELL])

                _, lstm_cell_tuple_context_level = tf.nn.dynamic_rnn(
                    cell=context_level_encoder,
                    dtype=tf.float32,
                    # Shape of each last hidden state is (1, NUM_UNITS_HRED_UTTERANCE_CELL)
                    sequence_length=tf.ones(num_context_utterances),
                    inputs=utterances_last_hidden_states)

                # Only hidden state of last input to context encoder is relevant
                # since it represents the last dialogue state
                lastest_dialogue_representation = lstm_cell_tuple_context_level[0][-1]
                dialgoue_representations.append(lastest_dialogue_representation)

        # [batch_size, NUM_UNITS_HRED_UTTERANCE_CELL]
        dialgoue_representations = tf.stack(dialgoue_representations)

        # ----------------Key-Value Memory Network----------------
        with tf.variable_scope('key_value_memory_network'):
            word_vec_dim = params[WORD_VEC_DIM]
            feature_size = params[NUM_UNITS_HRE_CONTEXT_CELL]
            num_hops = params[NUM_HOPS]
            initial_queries = dialgoue_representations

            # R_i: Matrix used to get query representation q_i+1
            self.R = [tf.Variable(
                tf.truncated_normal([feature_size, feature_size],
                                    stddev=0.1)) for R_i in range(num_hops)]
            self.A = tf.Variable(
                tf.truncated_normal([feature_size, word_vec_dim], stddev=0.1),
                name="A")

            # output after last iteration over memory adressing/reading
            # Shape: (batch_size, feature_size)
            memory_output = self._get_response_from_memory(num_hops=num_hops, initial_queries=initial_queries,
                                                           key_cells=key_cells, value_cells=value_cells)

        # ----------------Decoder----------------
        train_loss = None

        with tf.variable_scope('decoder'):
            batch_size = tf.shape(memory_output)[0]
            decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=feature_size)

            decoder_seq_lengths = np.array(np.repeat(a=params[MAX_NUM_UTTER_TOKENS], repeats=params[BATCH_SIZE]),
                                           dtype=np.int32)

            helper = self.get_helper(mode, decoder_embedded_input=embedded_decoder_input,
                                     sequenece_lengths=decoder_seq_lengths, params=params)

            projection_layer = layers_core.Dense(units=params[DECODER_VOCABUALRY_SIZE], use_bias=False)

            initial_state_tuple = tf.contrib.rnn.LSTMStateTuple(c=memory_output,
                                                                h=tf.zeros(shape=(batch_size, feature_size)))

            decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper,
                                                      initial_state=initial_state_tuple,
                                                      output_layer=projection_layer)

            if mode == tf.estimator.ModeKeys.PREDICT:
                # During inference maximum length of response is not known. Therefore, limit response length.
                outputs, final_context_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder, maximum_iterations=params[MAX_NUM_UTTER_TOKENS] * 2)
            else:
                outputs, final_context_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder)

            logits = outputs.rnn_output

            predicted_word_ids = outputs.sample_id

            if mode != tf.estimator.ModeKeys.PREDICT:
                # target's shape: [batch_size, max_seq_length] logit's shape: [batch_size, max_seq_length, word_vec_dim]
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits)

                # Mask padding vectors
                # target_weights shape: [batch_size, max_seq_len]
                target_weights = tf.sequence_mask(lengths=sequenece_lengths_responses,
                                                  maxlen=params[MAX_NUM_UTTER_TOKENS],
                                                  dtype=logits.dtype)

                # Normalize loss based on batch_size
                train_loss = (tf.reduce_sum(cross_entropy * target_weights) / tf.to_float(batch_size))

        # ----------------Prepare Output----------------

        # Dictionary containing the predictions
        predictions_dict = OrderedDict()
        predictions_dict[LOGITS] = logits
        predictions_dict[WORD_PROBABILITIES] = tf.nn.softmax(logits)
        predictions_dict[TOKEN_IDS] = predicted_word_ids

        # Needed by Java applications. Model can be called from Java
        classification_output = export_output.ClassificationOutput(
            scores=tf.nn.softmax(logits))

        # Check for prediction mode first, since in prediction mode there is no 'train_op'
        if mode == tf.estimator.ModeKeys.PREDICT:
            logging.info("In prediction mode")
            return get_estimator_specification(mode=mode, predictions_dict=predictions_dict,
                                               classifier_output=classification_output)

        # If no learning rate is specified then use the default value in the case specified optimizer
        #  has a default value
        if LEARNING_RATE in params:
            optimizer = get_optimizer(optimizer=params[OPTIMIZER], learning_rate=params[LEARNING_RATE])
        else:
            optimizer = get_optimizer(optimizer=params[OPTIMIZER])

        train_op = optimizer.minimize(train_loss, global_step=tf.train.get_global_step())

        if mode == tf.estimator.ModeKeys.TRAIN:
            logging.info("In training mode")
            return get_estimator_specification(mode=mode, predictions_dict=predictions_dict,
                                               classifier_output=classification_output, loss=train_loss,
                                               train_op=train_op)

        if mode == tf.estimator.ModeKeys.EVAL:
            logging.info("In evaluation mode")
            return get_estimator_specification(mode=mode, predictions_dict=predictions_dict,
                                               classifier_output=classification_output, loss=train_loss)

    def _get_response_from_memory(self, num_hops, initial_queries, key_cells, value_cells):
        """

        :param num_hops:
        :param initial_queries:
        :param key_cells: Embeddings for keys representing concatenations of relations and subjects
        :param value_cells: Embeddings for objects
        :return:
        """

        # Shape: (batch_size, feature_size)
        queries = initial_queries
        shape_qs = tf.shape(queries)
        batch_size = shape_qs[0]
        feature_size = shape_qs[1]

        # For each batch element the columns represent the keys
        shape_key_cells = tf.shape(key_cells)
        k_keys = shape_key_cells[0]
        m_keys = shape_key_cells[1]
        n_keys = shape_key_cells[2]

        shape_value_cells = tf.shape(value_cells)
        k_values = shape_value_cells[0]
        m_values = shape_value_cells[1]
        n_values = shape_value_cells[2]

        # shape = [k_keys,m_keys,n_keys] transpose to [m_keys,k_keys,n_keys]
        key_cells = tf.transpose(key_cells, [1, 0, 2])
        # [embedding_size, batch_size * memory_size]
        key_cells = tf.reshape(key_cells, shape=(m_keys, k_keys * n_keys))

        # For each batch element the columns represent the values
        value_cells = tf.transpose(value_cells, [1, 0, 2])
        value_cells = tf.reshape(value_cells, shape=(m_values, k_values * n_values))

        for hop in range(num_hops):
            log.info("----------Key Adressing----------")
            # Shape: (feature_size, batch_size * memory_size)
            A_keys = tf.matmul(self.A, key_cells)

            # Reshape to (batch_size, memory_size, feature_size)
            A_keys = tf.transpose(A_keys)
            A_keys = tf.reshape(A_keys, shape=(-1, n_keys, feature_size))

            # Reshape to (batch_size, 1, feature_size)
            queries_reshaped = tf.reshape(queries, shape=(batch_size, 1, feature_size))

            # Computed dot product between A_queries and A_keys
            # Multiply element wise and then add along columns: Compute dot product for each element of batch
            memory_confidence_scores = tf.reduce_sum(queries_reshaped * A_keys, 2)

            # Shape: (batch_size, memory_size)
            memory_probabilities = tf.nn.softmax(memory_confidence_scores)

            # Expand to (batch_size, memory_size, 1)
            memory_probabilities = tf.expand_dims(memory_probabilities, axis=-1)

            log.info("----------Value Reading----------")

            # Shape: (feature_size, batch_size * memory_size)
            A_values = tf.matmul(self.A, value_cells)
            # Reshape to (batch_size, memory_size, feature_size)
            A_values = tf.transpose(A_values)
            A_values = tf.reshape(A_values, shape=(-1, n_values, feature_size))

            # [batch_size, feature_size]
            o = tf.reduce_sum(memory_probabilities * A_values, 1)

            # Update queries
            # Shape: (feature_size, feature_size)
            R = self.R[hop]
            # Shape: (feature_size, batch_size)
            queries = tf.matmul(a=R, b=o, transpose_b=True)

            # Transpose to (batch_size, feature_size)
            queries = tf.transpose(queries)

        return queries

    def _compute_sequence_lengths(self, embedded_sequneces):
        """
        Compute sequnce length of sequences on the fly.
        :param embedded_sequeneces:
        :return: Tensorflow Tensor
        """
        # For a non-padding vector for each row a value of 1 is computed, and for padding-rows 0 is computed
        binary_flags = tf.sign(tf.reduce_max(tf.abs(embedded_sequneces), axis=2))
        # Sum of 1s indicate how many non padding vectors are contained in specific embedding
        lengths = tf.reduce_sum(binary_flags, axis=1)
        lengths = tf.cast(lengths, tf.int32)
        return lengths

    def get_helper(self, mode, decoder_embedded_input, sequenece_lengths, params):
        helper = None

        if mode != tf.estimator.ModeKeys.PREDICT:
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embedded_input,
                                                       sequence_length=sequenece_lengths,
                                                       time_major=False)
        else:
            batch_size = tf.shape(decoder_embedded_input)[0]
            start_tokens = tf.fill([batch_size], params[TARGET_SOS_ID])
            end_token_id = params[TARGET_EOS_ID]
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                decoder_embedded_input, start_tokens, end_token_id)

        return helper
