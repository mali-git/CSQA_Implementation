import logging
import unittest
from collections import OrderedDict

import spacy

from utilities.constants import CSQA_ENTITIES_IN_UTTERANCE, CSQA_UTTERANCE
from utilities.corpus_preprocessing_utils.text_manipulation_utils import compute_nlp_features
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

        nlp_parser = spacy.load('en')

    def test_initialize_token_mappings(self):
        token_to_embeddings, word_to_id_dict = self.instance_creator._initialize_token_mappings(
            vocab_freqs=self.ctx_vocab_freq_dict, is_ctx_vocab=True)

        self.assertEqual(type(token_to_embeddings), OrderedDict)
        assert (len(token_to_embeddings) == 12)
        assert (len(word_to_id_dict) == 12)

        expected_tokens = 'cristiano ronaldo final june soccer goal'.split() + ['Q615']

        for token in expected_tokens:
            self.assertTrue(token in token_to_embeddings)

        token_to_embeddings, word_to_id_dict = self.instance_creator._initialize_token_mappings(
            vocab_freqs=self.response_vocab_freq_dict, is_ctx_vocab=True)

        assert (len(token_to_embeddings) == 12)
        assert (len(word_to_id_dict) == 12)

        expected_tokens = 'lionel messi argentine aguero striker draw'.split() + ['Q11571']

        for token in expected_tokens:
            self.assertTrue(token in token_to_embeddings)

    def test_get_offsets_of_relevant_entities_in_utterance(self):
        utterance_txt = 'cristiano ronaldo final june soccer goal'
        utterance_dict = OrderedDict()
        utterance_dict[CSQA_UTTERANCE] = utterance_txt
        utterance_dict[CSQA_ENTITIES_IN_UTTERANCE] = ['Q11571']
        start_offsets, end_offsets = self.instance_creator._get_offsets_of_relevant_entities_in_utterance(
            utterance_dict=utterance_dict)

        self.assertEqual(len(start_offsets), 1)
        self.assertEqual(len(end_offsets), 1)

        start_offset = start_offsets[0]
        end_offset = end_offsets[0]

        self.assertEqual(start_offset, 0)
        self.assertEqual(end_offset, 17)

    def test_map_utter_toks_to_ids(self):
        utterance_txt = 'cristiano ronaldo final june soccer goal'
        utterance_dict = OrderedDict()
        utterance_dict[CSQA_UTTERANCE] = utterance_txt
        utterance_dict[CSQA_ENTITIES_IN_UTTERANCE] = ['Q11571']

        offsets_info_dict = OrderedDict()
        offsets_info_dict[(0, 17)] = True
        offsets_info_dict[(17, len(utterance_txt))] = False
        spans = compute_nlp_features(txt=utterance_txt, offsets_info_dict=offsets_info_dict)


        token_ids = self.instance_creator._map_utter_toks_to_ids(utterance_dict=utterance_dict,
                                                                 utterance_offsets_info_dict=offsets_info_dict,
                                                                 nlp_spans=spans,
                                                                 is_reponse_utter=False)

        expected_ids = [10,6,7,8,9]

        self.assertEqual(token_ids,expected_ids)



