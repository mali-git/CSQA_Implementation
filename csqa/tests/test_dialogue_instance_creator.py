import logging
import unittest
from collections import OrderedDict

from utilities.general_utils import load_dict_from_disk
from utilities.instance_creation_utils.dialogue_instance_creator import DialogueInstanceCreator
from utilities.test_utils.create_test_resources import create_test_dialogue_instance_creator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class TestDialogueInstanceCreator(unittest.TestCase):

    def setUp(self):
        create_test_dialogue_instance_creator()
        path_to_dict = '../test_resources/csqa_vocab_context_vocab_5.pkl'
        self.ctx_vocab_freq_dict = load_dict_from_disk(path_to_dict=path_to_dict)

        path_to_dict = '../test_resources/csqa_vocab_response_vocab_5.pkl'
        self.response_vocab_freq_dict = load_dict_from_disk(path_to_dict=path_to_dict)

        max_num_utter_tokens = 10
        max_dialogue_context_length = 2
        path_to_entity_id_to_label_mapping = '../test_resources/filtered_entity_mapping.json'
        path_to_word_vec_model = '../test_resources/test_soccer_word_to_vec'

        word_to_vec_model_dict = dict()
        word_to_vec_model_dict['Path'] = path_to_word_vec_model
        word_to_vec_model_dict['is_c_format'] = False
        word_to_vec_model_dict['is_binary'] = False
        path_to_kb_embeddings = '../test_resources/entity_to_embeddings.pkl'

        self.instance_creator = DialogueInstanceCreator(max_num_utter_tokens=max_num_utter_tokens,
                                                        max_dialogue_context_length=max_dialogue_context_length,
                                                        path_to_entity_id_to_label_mapping=path_to_entity_id_to_label_mapping,
                                                        ctx_vocab_freq_dict=self.ctx_vocab_freq_dict,
                                                        response_vocab_freq_dict=self.response_vocab_freq_dict,
                                                        word_to_vec_dict=word_to_vec_model_dict,
                                                        path_to_kb_embeddings=path_to_kb_embeddings)

    def test_initialize_token_mappings(self):
        token_to_embeddings, word_to_id_dict = self.instance_creator._initialize_token_mappings(
            vocab_freqs=self.ctx_vocab_freq_dict, is_ctx_vocab=True)

        self.assertEqual(type(token_to_embeddings), OrderedDict)
        assert (len(token_to_embeddings) == 10)
        assert (len(word_to_id_dict) == 10)

        expected_words = 'ronaldo final june soccer goal'.split() + ['messi']

        for w in expected_words:
            self.assertTrue(w in token_to_embeddings)

        token_to_embeddings, word_to_id_dict = self.instance_creator._initialize_token_mappings(
            vocab_freqs=self.response_vocab_freq_dict, is_ctx_vocab=True)

        assert (len(token_to_embeddings) == 10)
        assert (len(word_to_id_dict) == 10)

        expected_words = 'messi argentine aguero striker draw'.split() + ['ronaldo']

        for w in expected_words:
            self.assertTrue(w in token_to_embeddings)
