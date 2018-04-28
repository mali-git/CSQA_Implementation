import logging

import tensorflow as tf

from utilities.constants import EMBEDDED_SEQUENCES, NUM_UNITS_HRED_UTTERANCE_CELL, NUM_UNITS_HRED_CONTEXT_CELL

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class CSQANetwork(object):

    def model_fct(self, features, responses, mode, params):
        """
        This function defines the neural network is called by tf.Estimator
        :param features: Features of instances
        :param responses: Responses to last question in each batch
        :param mode: Defines in which mode function is called (train, eval or test)
        :param params: Dictionary containing model parameters
        :return:
        """
        # Important: The utterances in a batch represent a specific dialogue context. Therefore, utterances
        # in batch are related to each other.
        # Shape: (batch_size, max_utter_length, vec_dimension)
        embedded_sequences = features[EMBEDDED_SEQUENCES]
        sequenece_lengths = self._compute_sequence_lengths(embedded_sequences)

        # ----------------Hierarchical Encoder----------------
        with tf.variable_scope('Utterance Level Encoder'):
            utterance_level_encoder = tf.nn.rnn_cell.LSTMCell(num_units=params[NUM_UNITS_HRED_UTTERANCE_CELL])

            # state_tuple is a tuple containing the last hidden state and the last activation
            _, state_tuple_utter_level = tf.nn.dynamic_rnn(
                cell=utterance_level_encoder,
                dtype=tf.float32,
                sequence_length=sequenece_lengths,
                inputs=embedded_sequences)

            # For each sequence extract the last hidden state
            # Shape of last [batch_size, NUM_UNITS_IN_LSTM_CELL]
            utterances_last_hidden_states = state_tuple_utter_level[0]
            shape_last_hidden_states = tf.shape(utterances_last_hidden_states)
            num_context_utterances = shape_last_hidden_states[0]
            dimension_hidden_utterance_state = shape_last_hidden_states[1]
            # Reshape [batch_size, NUM_UNITS_IN_LSTM_CELL] to [batch_size,
            # 1, NUM_UNITS_IN_LSTM_CELL]
            # since each hidden state represents a summary with NUM_UNITS_IN_LSTM_CELL features
            utterances_last_hidden_states = tf.reshape(utterances_last_hidden_states, shape=(
                num_context_utterances, 1, dimension_hidden_utterance_state))

        with tf.variable_scope('Context Level Encoder'):
            context_level_encoder = tf.nn.rnn_cell.LSTMCell(num_units=params[NUM_UNITS_HRED_CONTEXT_CELL])

            _, state_tuple_context_level = tf.nn.dynamic_rnn(
                cell=context_level_encoder,
                dtype=tf.float32,
                # Shape of each last hidden state is (1, NUM_UNITS_HRED_UTTERANCE_CELL)
                sequence_length=[1 for _ in range(num_context_utterances)],
                inputs=utterances_last_hidden_states)

            context_last_hidden_states = state_tuple_context_level[0]
            shape_context_last_hidden_states = tf.shape(context_last_hidden_states)
            # Reshape [batch_size, NUM_UNITS_IN_CONTEXT_CELL] to [batch_size, 1, NUM_UNITS_IN_CONTEXT_CELL]
            # context_last_hidden_states = tf.reshape(context_last_hidden_states, shape=(
            #     shape_context_last_hidden_states[0], 1, shape_context_last_hidden_states[1]))

        # ----------------Key-Value Memory Network----------------
        with tf.variable_scope('Key Value Memory Network'):
            pass

    # ----------------Decoder----------------

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
