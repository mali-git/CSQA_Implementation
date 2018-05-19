import logging
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.python.estimator.export import export_output

from utilities.constants import EMBEDDED_SEQUENCES, NUM_UNITS_HRE_UTTERANCE_CELL, NUM_UNITS_HRE_CONTEXT_CELL, NUM_HOPS, \
    WORD_VEC_DIM, KEY_CELLS, VALUE_CELLS, VOCABUALRY_SIZE, EMBEDDED_RESPONSES, LEARNING_RATE, OPTIMIZER, LOGITS, \
    WORD_PROBABILITIES, WORD_IDS, TARGET_SOS_ID, TARGET_EOS_ID, MAX_NUM_UTTER_TOKENS
from utilities.tensorflow_estimator_utils import get_estimator_specification

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class CSQANetwork(object):

    def __init__(self, num_trainable_tokens, word_vec_dim, vocab_size, initial_embeddings=None):
        self.initial_embeddings = initial_embeddings
        self.embeddings = self.initialize_embedding_layer(embeddings=self.initial_embeddings,
                                                          num_trainable_tokens=num_trainable_tokens,
                                                          word_vec_dim=word_vec_dim, vocab_size=vocab_size)

    def initialize_embedding_layer(self, embeddings, num_trainable_tokens, word_vec_dim, vocab_size):
        with tf.variable_scope('embedding_layer'):
            if embeddings is not None:
                # Embeddings for unknown token, start of sentence token etc. are trainable
                trainable_embeddings = self.initial_embeddings[0:num_trainable_tokens]
                embeddings = self.initial_embeddings[num_trainable_tokens:]

                initializer = tf.constant_initializer(embeddings)

                trainable_embeddings = tf.get_variable(name="trainable_embeddings",
                                                 shape=[num_trainable_tokens, word_vec_dim],
                                                 initializer=tf.constant_initializer(trainable_embeddings),
                                                 trainable=True)

                pretrained_embeddings = tf.get_variable(name="pretrained_embeddings",
                                                        shape=[vocab_size - num_trainable_tokens, word_vec_dim],
                                                        initializer=initializer,
                                                        trainable=False)

                embeddings = tf.concat([trainable_embeddings, pretrained_embeddings], name='embeddings', axis=0)

            else:
                embeddings = tf.get_variable(name="embeddings",
                                             shape=[vocab_size, word_vec_dim],
                                             initializer=tf.random_normal_initializer(),
                                             trainable=True)

            return embeddings

    def model_fct(self, features, targets, mode, params):
        """
        This function defines the neural network called by tf.Estimator
        :param features: Features of instances
        :param targets: Responses to last question in each batch
        :param mode: Defines in which mode function is called (train, eval or test)
        :param params: Dictionary containing model parameters
        :return:
        """

        dialgoue_representations = []

        for feature in features:
            # Shape: (num_utterances, max_utter_length, vec_dimension)
            embedded_sequences = tf.nn.embedding_lookup(self.embeddings, feature[EMBEDDED_SEQUENCES])
            embedded_response = tf.nn.embedding_lookup(self.embeddings, feature[EMBEDDED_RESPONSES])

            exit(0)

            sequenece_lengths = self._compute_sequence_lengths(embedded_sequences)
            sequenece_lengths_responses = self._compute_sequence_lengths(embedded_response)
            key_cells = features[KEY_CELLS]
            value_cells = features[VALUE_CELLS]

            # ----------------Hierarchical Encoder----------------
            with tf.variable_scope('utterance_level_encoder'):
                utterance_level_encoder = tf.nn.rnn_cell.LSTMCell(num_units=params[NUM_UNITS_HRE_UTTERANCE_CELL])

                # state_tuple is a tuple containing the last hidden state and the last activation
                _, state_tuple_utter_level = tf.nn.dynamic_rnn(
                    cell=utterance_level_encoder,
                    dtype=tf.float32,
                    sequence_length=sequenece_lengths,
                    inputs=embedded_sequences)

                # For each sequence extract the last hidden state
                # Shape of last [num_utterances, NUM_UNITS_IN_LSTM_CELL]
                utterances_last_hidden_states = state_tuple_utter_level[0]
                shape_last_hidden_states = tf.shape(utterances_last_hidden_states)
                num_context_utterances = shape_last_hidden_states[0]
                dimension_hidden_utterance_state = shape_last_hidden_states[1]
                # Reshape [num_utterances, NUM_UNITS_IN_LSTM_CELL] to [num_utterances,
                # 1, NUM_UNITS_IN_LSTM_CELL]
                # since each hidden state represents a summary with NUM_UNITS_IN_LSTM_CELL features
                utterances_last_hidden_states = tf.reshape(utterances_last_hidden_states, shape=(
                    num_context_utterances, 1, dimension_hidden_utterance_state))

            with tf.variable_scope('context_level_encoder'):
                context_level_encoder = tf.nn.rnn_cell.LSTMCell(num_units=params[NUM_UNITS_HRE_CONTEXT_CELL])

                _, state_tuple_context_level = tf.nn.dynamic_rnn(
                    cell=context_level_encoder,
                    dtype=tf.float32,
                    # Shape of each last hidden state is (1, NUM_UNITS_HRED_UTTERANCE_CELL)
                    sequence_length=[1 for _ in range(num_context_utterances)],
                    inputs=utterances_last_hidden_states)

                # Only hidden state of last input to context encoder is relevant
                # since it represents the last dialogue state
                lastest_dialogue_representation = state_tuple_context_level[0][-1]
                dialgoue_representations.append(lastest_dialogue_representation)

        # [batch_size, NUM_UNITS_HRED_UTTERANCE_CELL]
        dialgoue_representations = np.array(dialgoue_representations, dtype=np.float32)

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
            batch_size, _ = tf.shape(memory_output)
            decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=params[WORD_VEC_DIM])

            helper = self.get_helper(mode, decoder_embedded_input=embedded_response,
                                     sequenece_lengths=sequenece_lengths_responses, params=params)

            projection_layer = tf.layers.dense(
                units=params[VOCABUALRY_SIZE], use_bias=False)

            decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper, initial_state=memory_output,
                                                      output_layer=projection_layer)

            outputs, final_context_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder)
            logits = outputs.rnn_output
            predicted_word_ids = outputs.sample_id

            if mode != tf.estimator.ModeKeys.PREDICT:
                # target's shape: [batch_size, max_seq_length] logit's shape: [batch_size, max_seq_length, word_vec_dim]
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=targets, logits=logits)

                # Mask padding vectors
                # target_weights shape: [batch_size, max_seq_len]
                target_weights = tf.sequence_mask(lengths=sequenece_lengths_responses,
                                                  maxlen=params[MAX_NUM_UTTER_TOKENS],
                                                  dtype=logits.dtype)
                # Normalize loss based on batch_size
                train_loss = (tf.reduce_sum(cross_entropy * target_weights) / batch_size)

        # ----------------Prepare Output----------------

        # Dictionary containing the predictions
        predictions_dict = OrderedDict()
        predictions_dict[LOGITS] = logits
        predictions_dict[WORD_PROBABILITIES] = tf.nn.softmax(logits)
        predictions_dict[WORD_IDS] = predicted_word_ids

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
            optimizer = self.get_optimizer(optimizer=params[OPTIMIZER], learning_rate=params[LEARNING_RATE])
        else:
            optimizer = self.get_optimizer(optimizer=params[OPTIMIZER])

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
        # For each batch element the columns represent the keys
        k, m, n = tf.shape(self.key_cells)

        for hop in range(num_hops):
            log.info("----------Key Adressing----------")
            # shape = [k,m,n] transpose to [m,k,n]
            key_cells = tf.transpose(key_cells, [1, 0, 2])
            # [embedding_size, batch_size * memory_size]
            key_cells = tf.reshape(key_cells, shape=(m, k * n))

            # Shape: (feature_size, batch_size * memory_size)
            A_keys = tf.matmul(self.A, key_cells)
            # Reshape to (batch_size, memory_size, feature_size)
            A_keys = tf.transpose(A_keys)
            A_keys = tf.reshape(A_keys, shape=(-1, n, m))

            # Reshape to (batch_size, 1, feature_size)
            queries_reshaped = tf.reshape(queries, shape=(shape_qs[0], 1, shape_qs[1]))
            # Computed dot product between A_queries and A_keys
            # Multiply element wise and then add along columns: Compute dot product for each element of batch
            memory_confidence_scores = tf.reduce_sum(queries_reshaped * A_keys, 2)
            # Shape: (batch_size, memory_size)
            memory_probabilities = tf.nn.softmax(memory_confidence_scores)
            # Expand to (batch_size, memory_size, 1)
            memory_probabilities = tf.expand_dims(memory_probabilities, axis=-1)

            log.info("----------Value Reading----------")
            # For each batch element the columns represent the values
            value_cells = tf.transpose(value_cells, [1, 0, 2])
            value_cells = tf.reshape(value_cells, shape=(m, k * n))
            # Shape: (feature_size, batch_size * memory_size)
            A_values = tf.matmul(self.A, value_cells)
            # Reshape to (batch_size, memory_size, feature_size)
            A_values = tf.transpose(A_values)
            A_values = tf.reshape(A_values, shape=(-1, n, m))

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
