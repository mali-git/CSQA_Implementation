import logging
import unittest
from collections import OrderedDict

from utilities.constants import CSQA_ENTITIES_IN_UTTERANCE, CSQA_UTTERANCE, TOKEN_IDS, SOS_TOKEN, EOS_TOKEN
from utilities.corpus_preprocessing_utils.load_dialogues import load_data_from_json_file
from utilities.corpus_preprocessing_utils.text_manipulation_utils import compute_nlp_features
from utilities.general_utils import load_dict_from_disk
from utilities.instance_creation_utils.dialogue_instance_creator import DialogueInstanceCreator
from utilities.test_utils.create_test_resources import create_test_resources_dialogue_instance_creator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class TestDialogueInstanceCreator(unittest.TestCase):

    def setUp(self):
        create_test_resources_dialogue_instance_creator()
        path_to_dict = '../test_resources/csqa_vocab_context_vocab_5.pkl'
        self.ctx_vocab_freq_dict = load_dict_from_disk(path_to_dict=path_to_dict)

        path_to_dict = '../test_resources/csqa_vocab_response_vocab_5.pkl'
        self.response_vocab_freq_dict = load_dict_from_disk(path_to_dict=path_to_dict)

        max_num_utter_tokens = 10
        max_dialogue_context_length = 2
        path_to_entity_id_to_label_mapping = '../test_resources/filtered_entity_mapping.json'
        path_to_property_id_to_label_mapping = '../test_resources/filtered_property_mapping.json'
        path_to_word_vec_model = '../test_resources/test_soccer_word_to_vec'

        word_to_vec_model_dict = dict()
        word_to_vec_model_dict['Path'] = path_to_word_vec_model
        word_to_vec_model_dict['is_c_format'] = False
        word_to_vec_model_dict['is_binary'] = False
        path_to_kb_embeddings = '../test_resources/entity_to_embeddings.pkl'

        path_to_wikidata_triples = '../test_resources/example_wikidata_triples.csv'

        self.instance_creator = DialogueInstanceCreator(max_num_utter_tokens=max_num_utter_tokens,
                                                        max_dialogue_context_length=max_dialogue_context_length,
                                                        path_to_entity_id_to_label_mapping=path_to_entity_id_to_label_mapping,
                                                        path_to_property_id_to_label_mapping=path_to_property_id_to_label_mapping,
                                                        ctx_vocab_freq_dict=self.ctx_vocab_freq_dict,
                                                        response_vocab_freq_dict=self.response_vocab_freq_dict,
                                                        word_to_vec_dict=word_to_vec_model_dict,
                                                        path_to_kb_entities_embeddings=path_to_kb_embeddings,
                                                        path_to_wikidata_triples=path_to_wikidata_triples,
                                                        min_count_n_gram_matching=5)

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

        expected_ids = [10, 6, 7, 8, 9]

        self.assertEqual(token_ids, expected_ids)

    def test_get_token_id_for_entity(self):
        entity_id = 'Q11571'
        is_reponse_utter = False

        tok_id = self.instance_creator._get_token_id_for_entity(entity_id=entity_id, is_reponse_utter=is_reponse_utter)

        self.assertEqual(tok_id, 10)

        is_reponse_utter = True
        tok_id = self.instance_creator._get_token_id_for_entity(entity_id=entity_id, is_reponse_utter=is_reponse_utter)

        self.assertEqual(tok_id, 4)

        entity_id = 'Q107365'
        is_reponse_utter = False

        tok_id = self.instance_creator._get_token_id_for_entity(entity_id=entity_id, is_reponse_utter=is_reponse_utter)

        self.assertEqual(tok_id, 2)

    def test_add_utter_padding(self):
        utter_tok_ids = [10, 6, 7, 8, 9]
        is_reponse_utter = False

        padded_ids = self.instance_creator.add_utter_padding(utter_tok_ids=utter_tok_ids,
                                                             is_reponse_utter=is_reponse_utter)

        expected_list = [10, 6, 7, 8, 9] + [3, 3, 3, 3, 3]
        self.assertEqual(padded_ids, expected_list)

        utter_tok_ids = [9 for i in range(10)]
        is_reponse_utter = False

        padded_ids = self.instance_creator.add_utter_padding(utter_tok_ids=utter_tok_ids,
                                                             is_reponse_utter=is_reponse_utter)

        expected_list = [9 for i in range(10)]
        self.assertEqual(padded_ids, expected_list)

    def test_extract_relevant_kg_triples(self):
        utterance_txt = 'cristiano ronaldo final june soccer goal'
        utterance_dict = OrderedDict()
        utterance_dict[CSQA_UTTERANCE] = utterance_txt
        utterance_dict[CSQA_ENTITIES_IN_UTTERANCE] = ['Q11571']

        relevant_triples = self.instance_creator._extract_relevant_kg_triples(utter_dict=utterance_dict)

        """Keep in mind:
            csqa_vocab_context_frq['cristiano'] = 3
            csqa_vocab_context_frq['ronaldo'] = 3
            csqa_vocab_context_frq['final'] = 10
            csqa_vocab_context_frq['june'] = 10
            csqa_vocab_context_frq['soccer'] = 1
            csqa_vocab_context_frq['goal'] = 10 
            min_count_n_gram_matching = 5 """

        num_triples = relevant_triples.shape[0]

        t1 = ['Q11571', 'P166', 'Q2291862']
        t2 = ['Q11571', 'P166', 'Q182529']
        t3 = ['Q11571', 'P166', 'Q794775']
        t4 = ['Q11571', 'P19', 'Q25444']
        expected_triples = [t1, t2, t3, t4]

        self.assertEqual(num_triples, 4)

        for t in expected_triples:
            self.assertTrue(t in relevant_triples)

    def test_create_training_instances(self):
        input_path = '../test_resources/example_dialogue.json'
        file_id = 'example_dialogue.json'
        dialogue = load_data_from_json_file(input_path=input_path)

        instances_of_dialogue, target_inst, relevant_kg_triples = self.instance_creator._create_instances(
            dialogue=dialogue,
            file_id=file_id,
            is_training_mode=True)

        self.assertEqual(len(instances_of_dialogue), 4)
        self.assertEqual(len(relevant_kg_triples), 4)

        expected_utter = 'cristiano ronaldo final june soccer goal'

        for utter_dict in instances_of_dialogue[:-1]:
            utter_txt = utter_dict[CSQA_UTTERANCE]
            self.assertEqual(utter_txt, expected_utter)

        expected_utter = 'lionel messi argentine aguero striker draw'
        utter_txt = instances_of_dialogue[-1][CSQA_UTTERANCE]
        self.assertEqual(utter_txt, expected_utter)

        # Next case
        self.instance_creator.max_dialogue_context_length = 1

        instances_of_dialogue, target_inst, relevant_kg_triples = self.instance_creator._create_instances(
            dialogue=dialogue,
            file_id=file_id,
            is_training_mode=True)

        self.assertEqual(len(instances_of_dialogue), 2)
        self.assertEqual(len(relevant_kg_triples), 4)

        decoder_out_tok_ids = target_inst[TOKEN_IDS]
        decoder_in_tok_ids = instances_of_dialogue[-1][TOKEN_IDS]

        sos_tok_id = self.instance_creator.response_word_to_id[SOS_TOKEN]
        eos_tok_id = self.instance_creator.response_word_to_id[EOS_TOKEN]

        self.assertEqual(decoder_in_tok_ids[0], sos_tok_id)
        self.assertEqual(decoder_out_tok_ids[-1], eos_tok_id)

        for i in range(1, len(decoder_in_tok_ids) - 1):
            self.assertEqual(decoder_in_tok_ids[i], decoder_out_tok_ids[i - 1])

        expected_utter = 'cristiano ronaldo final june soccer goal'
        utter_txt = instances_of_dialogue[0][CSQA_UTTERANCE]
        self.assertEqual(utter_txt, expected_utter)

        expected_utter = 'lionel messi argentine aguero striker draw'
        utter_txt = instances_of_dialogue[1][CSQA_UTTERANCE]
        self.assertEqual(utter_txt, expected_utter)
