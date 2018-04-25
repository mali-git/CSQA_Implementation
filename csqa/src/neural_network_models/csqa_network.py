import logging
import tensorflow as tf

from utilities.constants import EMBEDDED_SEQUENCES, NUM_UNITS_IN_LSTM_CELL

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class CSQANetwork(object):

    def model_fct(self, features, labels, mode, params):
        """
        :param features:
        :param labels:
        :param mode:
        :param params:
        :return:
        """
        embedded_sequences = features[EMBEDDED_SEQUENCES]
        sequnece_lengths = self._compute_sequence_lengths(embedded_sequences)

        # ----------------Encoder----------------
        cell = tf.nn.rnn_cell.LSTMCell(num_units=params[NUM_UNITS_IN_LSTM_CELL])
        outputs, _ = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            sequence_length=sequnece_lengths,
            inputs=embedded_sequences)

        # For each sequence extract the last hidden state
        # Shape of last [bacthSize, self.nmbrHiddenUnits]
        last_hidden_states = self._get_last_hidden_state(outputs, sequnece_lengths)

        # ----------------Key-Value Memory Network----------------

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
        lengths = tf.reduce_sum(binary_flags,axis=1)
        lengths = tf.cast(lengths,tf.int32)
        return lengths

    @staticmethod
    def _get_last_hidden_state(hidden_states, sequence_lengths):
        hidden_states_shape = tf.shape(hidden_states)
        batch_size = hidden_states_shape[0]
        max_seq_length = hidden_states_shape[1]
        hidden_state_dimension = hidden_states_shape[2]
        # Index is supported by tf only in first dimension,
        # so create own index = [0 * #rows + length-1, 1 * #rows + length-1, 2 * #rows + length-1, ...]
        # The index gives the position of the last time step of each sequence.
        index = tf.range(0, batch_size) * max_seq_length + (sequence_lengths - 1)
        # -1 in shape has special meaning: The size of that dimension is computed such that the
        # total size remains constant
        # flat has as many columns as hidden_states and the number of rows is inferred.
        # flat has max_seq_length * batch_size rows
        flat = tf.reshape(hidden_states, [-1, hidden_state_dimension])
        # Collect from every instance the output of the last time step
        last_hidden_state = tf.gather(flat, index)
        return last_hidden_state
