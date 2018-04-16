import logging
import unittest

from utilities.tensor_embeddings_creation.feature_utils import get_feature_specification_dict
from utilities.tensor_embeddings_creation.utterance_to_tensor_embeddings_creator import Utterance2TensorCreator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class TestUtterance2TensorCreator(unittest.TestCase):
    def setUp(self):
        self.feature_specification_dict = get_feature_specification_dict(word_vec_dim=100, position_vec_dim=None,
                                                                         part_of_speech_vec_dim=None)

        path_to_entity_mapping_file = '../../../csqa/test_resources/filtered_entity_mapping.json'
        path_to_word_vec_model = '../../../csqa/test_resources/word2Vec_model'
        word_to_vec_model_dict = {path_to_word_vec_model: [False, False]}


        self.embedding_creator = Utterance2TensorCreator(max_num_utter_tokens=100,
                                    features_spec_dict=self.feature_specification_dict,
                                    path_to_entity_id_to_label_mapping=path_to_entity_mapping_file,
                                    word_to_vec_dict=word_to_vec_model_dict, path_to_kb_embeddings=None)

        def test_get_sequence_embedding_for_entity(self):
            pass
