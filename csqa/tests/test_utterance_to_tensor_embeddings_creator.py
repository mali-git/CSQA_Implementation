import logging
import unittest

import numpy as np
from collections import OrderedDict

from utilities.constants import WORD_VEC_DIM
from utilities.corpus_preprocessing.text_manipulation_utils import compute_nlp_features
from utilities.tensor_embeddings_creation.feature_utils import get_feature_specification_dict
from utilities.tensor_embeddings_creation.utterance_to_tensor_embeddings_creator import Utterance2TensorCreator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class TestUtterance2TensorCreator(unittest.TestCase):
    def setUp(self):
        self.feature_specification_dict = get_feature_specification_dict(word_vec_dim=100, position_vec_dim=None,
                                                                         part_of_speech_vec_dim=None)

        path_to_entity_mapping_file = '../test_resources/filtered_entity_mapping.json'
        path_to_word_vec_model_one = '../test_resources/word2Vec_model_one'
        path_to_word_vec_model_two = '../test_resources/word2Vec_model_two'
        word_to_vec_model_dict = dict()
        word_to_vec_model_dict[path_to_word_vec_model_one] = [False, False]
        word_to_vec_model_dict[path_to_word_vec_model_two] = [False, False]

        self.embedding_creator = Utterance2TensorCreator(max_num_utter_tokens=15,
                                                         features_spec_dict=self.feature_specification_dict,
                                                         path_to_entity_id_to_label_mapping=path_to_entity_mapping_file,
                                                         word_to_vec_dict=word_to_vec_model_dict,
                                                         path_to_kb_embeddings=None)

    def test_compute_sequence_embedding(self):
        txt = 'The chancellor of Germany visited Paris in February this year.'
        offsets_info_dict = OrderedDict()
        offset_tuple_one = (0, 4)
        offset_tuple_two = (4, 25)
        offset_tuple_three = (25, 34)
        offset_tuple_four = (34, 39)
        offset_tuple_five = (39, len(txt))

        offsets_info_dict[offset_tuple_one] = False
        offsets_info_dict[offset_tuple_two] = True
        offsets_info_dict[offset_tuple_three] = False
        offsets_info_dict[offset_tuple_four] = True
        offsets_info_dict[offset_tuple_five] = False

        num_word_to_vec_models = len(self.embedding_creator.word_to_vec_models)
        word_vec_dim = self.feature_specification_dict[WORD_VEC_DIM]

        spans = compute_nlp_features(txt=txt, offsets_info_dict=offsets_info_dict)
        span_one = spans[0]
        span_two = spans[1]
        span_three = spans[2]
        span_four = spans[3]
        span_five = spans[4]

        # Case: Substring doesn't represent an entity: ' in February this year'
        embedded_seq = self.embedding_creator.compute_sequence_embedding(txt=txt, offset_tuple=offset_tuple_five,
                                                                         is_entity=False, nlp_span=span_five)

        embedded_seq = np.array(embedded_seq)
        num_tokens_in_seq = len(span_five)

        self.assertEqual(embedded_seq.shape, (num_tokens_in_seq, num_word_to_vec_models, word_vec_dim))

        # Case: Substring represents an entity: 'chancellor of Germany'
        # For an entity a KG embedding is returned, and not three word embeddings
        embedded_seq = self.embedding_creator.compute_sequence_embedding(txt=txt, offset_tuple=offset_tuple_two,
                                                                         is_entity=True, nlp_span=span_two)

        embedded_seq = np.array(embedded_seq)

        # For an entity embedding the number of tokens is irrelevant: First dimension is always 1
        self.assertEqual(embedded_seq.shape, (1, num_word_to_vec_models, word_vec_dim))

    def test_get_sequence_embedding_for_entity(self):
        txt = 'The chancellor of Germany visited Paris in February this year.'
        offsets_info_dict = OrderedDict()
        offset_tuple_one = (0, 4)
        offset_tuple_two = (4, 25)
        offset_tuple_three = (25, 34)
        offset_tuple_four = (34, 39)
        offset_tuple_five = (39, len(txt))

        offsets_info_dict[offset_tuple_one] = False
        offsets_info_dict[offset_tuple_two] = True
        offsets_info_dict[offset_tuple_three] = False
        offsets_info_dict[offset_tuple_four] = True
        offsets_info_dict[offset_tuple_five] = False

        spans = compute_nlp_features(txt=txt, offsets_info_dict=offsets_info_dict)
        span_two = spans[1]
        entity = 'chancellor of Germany'

        entity_embedding = self.embedding_creator.get_sequence_embedding_for_entity(entity=entity, nlp_span=span_two,
                                                                                    use_part_of_speech_embedding=False)

        entity_embedding = np.array(entity_embedding)
        # Entity is considered as single token word, but NLP feature extraction is done based on actual tokens
        num_words_of_entity = 1
        num_word_to_vec_models = len(self.embedding_creator.word_to_vec_models)
        word_vec_dim = self.feature_specification_dict[WORD_VEC_DIM]

        # embedding_creator.get_sequence_embedding_for_entity()
        shape_of_entity_embedding = (num_words_of_entity, num_word_to_vec_models, word_vec_dim)

        self.assertEqual(entity_embedding.shape, shape_of_entity_embedding)

    def test_add_padding_to_embedding(self):
        txt = 'The chancellor of Germany visited Paris in February this year.'
        offsets_info_dict = OrderedDict()
        offset_tuple_one = (0, 4)
        offset_tuple_two = (4, 25)
        offset_tuple_three = (25, 34)
        offset_tuple_four = (34, 39)
        offset_tuple_five = (39, len(txt))

        offsets_info_dict[offset_tuple_one] = False
        offsets_info_dict[offset_tuple_two] = True
        offsets_info_dict[offset_tuple_three] = False
        offsets_info_dict[offset_tuple_four] = True
        offsets_info_dict[offset_tuple_five] = False

        num_word_to_vec_models = len(self.embedding_creator.word_to_vec_models)
        word_vec_dim = self.feature_specification_dict[WORD_VEC_DIM]

        spans = compute_nlp_features(txt=txt, offsets_info_dict=offsets_info_dict)

        embedded_seqs = []
        counter = 0

        for offset_tuple, is_entity in offsets_info_dict.items():
            # [  [ [token-1_model-1 embedding],...,[token-1_model-n embedding] ],...,
            # [ [token-k_model-1 embedding],...,[token-k_model-n embedding] ]  ]

            nlp_span = spans[counter]
            embedded_seq = self.embedding_creator.compute_sequence_embedding(txt=txt, offset_tuple=offset_tuple,
                                                           is_entity=is_entity, nlp_span=nlp_span)
            embedded_seqs += embedded_seq
            counter += 1

        seq_padded = self.embedding_creator.add_padding_to_embedding(seq_embedding=embedded_seqs)
        num_total_embedded_tokens = len(seq_padded)

        self.assertEqual(self.embedding_creator.max_num_utter_tokens,num_total_embedded_tokens)

        seq_padded = np.array(seq_padded)

        num_left_padded_vecs = 2
        left_padding = np.zeros(shape=(num_left_padded_vecs,num_word_to_vec_models,word_vec_dim))
        right_padding = np.zeros(shape=(num_left_padded_vecs, num_word_to_vec_models, word_vec_dim))

        self.assertTrue(np.array_equal(seq_padded[0:2],left_padding))
        self.assertTrue(np.array_equal(seq_padded[-2:], left_padding))



if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUtterance2TensorCreator)
    unittest.TextTestRunner(verbosity=2).run(suite)
