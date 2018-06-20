import logging
import pickle
from collections import OrderedDict

import numpy as np
import spacy
from gensim.models import KeyedVectors

from utilities.constants import CSQA_UTTERANCE, CSQA_ENTITIES_IN_UTTERANCE, UNKNOWN_TOKEN, SOS_TOKEN, EOS_TOKEN, \
    KG_WORD, PADDING_TOKEN
from utilities.corpus_preprocessing_utils.load_dialogues import load_data_from_json_file
from utilities.corpus_preprocessing_utils.text_manipulation_utils import save_insertion_of_offsets, mark_parts_in_text, \
    compute_nlp_features

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class DialogueInstanceCreator(object):

    def __init__(self, max_num_utter_tokens, max_dialogue_context_length,
                 path_to_entity_id_to_label_mapping, ctx_vocab_freq_dict, response_vocab_freq_dict,
                 word_to_vec_dict, path_to_kb_embeddings):

        self.word_to_vec_model = self._load_word_to_vec_model(word_to_vec_dict=word_to_vec_dict)
        self.kg_embeddings_dict = self._load_kg_embeddings(path_to_kb_embeddings=path_to_kb_embeddings)
        self.entity_id_to_label_dict = self._load_entity_to_label_mapping(path_to_entity_id_to_label_mapping)
        self.label_to_id_dict = {v: k for k, v in self.entity_id_to_label_dict.items()}
        self.max_num_utter_tokens = max_num_utter_tokens
        # Context of length is defined by a user utterance followed by a system utterance
        self.max_dialogue_context_length = max_dialogue_context_length

        # Note: Flag is important
        self.ctx_token_to_embeddings, self.ctx_word_to_id = self._initialize_token_mappings(
            vocab_freqs=ctx_vocab_freq_dict, is_ctx_vocab=True)


        # Note: Flag is important
        self.response_token_to_embeddings, self.response_word_to_id = self._initialize_token_mappings(
            vocab_freqs=response_vocab_freq_dict, is_ctx_vocab=False)

        self.responses_vocab = OrderedDict()
        self.nlp_parser = spacy.load('en')
        self.current_file_in_progress = None

        self._initialize_token_mappings(vocab_freqs=response_vocab_freq_dict, is_ctx_vocab=False)

    def _load_kg_embeddings(self, path_to_kb_embeddings):
        """
        Load dict containing the entity id's as keys and the embeddings as values
        :param path_to_kb_embeddings: Path to the serialized dict
        :rtype: dict
        """
        with open(path_to_kb_embeddings, 'rb') as f:
            entity_to_embeddings_dict = pickle.load(f)
            return entity_to_embeddings_dict

    def _add_kg_embeddings_to_vocab(self, token_to_embeddings):
        """
        Add the entity id's and their embeddings to the passed dict
        :param token_to_embeddings:
        :param token_to_id_dict:
        :rtype: dict, dict
        """
        # Add KG embeddings
        # no assignment nedded for 'token_to_embeddings' since update() returns None
        token_to_embeddings.update(self.kg_embeddings_dict)

        return token_to_embeddings

    def _initialize_token_mappings(self, vocab_freqs, is_ctx_vocab):
        """
        Extract tokens from vocab_freqs, and the embeddings from the word2Vec model, and create corresponding mappings
        :param vocab_freqs: Dict containing tokens as keys and their frequencies in the corpus as values
        :param is_ctx_vocab: Flag indication whether the vocabulary is extracted from context utterances
        :rtype: dict, dict
        """
        token_to_embeddings = OrderedDict()
        vec_dim = self.word_to_vec_model.vector_size
        token_to_embeddings[SOS_TOKEN] = np.random.uniform(low=-0.1, high=0.1, size=(vec_dim,))
        token_to_embeddings[EOS_TOKEN] = np.random.uniform(low=-0.1, high=0.1, size=(vec_dim,))
        token_to_embeddings[UNKNOWN_TOKEN] = np.random.uniform(low=-0.1, high=0.1, size=(vec_dim,))
        token_to_embeddings[PADDING_TOKEN] = np.zeros(shape=(vec_dim,))

        if is_ctx_vocab == False:
            token_to_embeddings[KG_WORD] = np.random.uniform(low=-0.1, high=0.1, size=(vec_dim,))

        # TODO: Improove
        for word, _ in vocab_freqs.items():
            if word not in token_to_embeddings:
                if word in self.word_to_vec_model:
                    token_to_embeddings[word] = self.word_to_vec_model[word]

        if is_ctx_vocab:
            # KG entities are added only in context vocabulary
            # In response vocabulary the special token KG_WORD is used for entities
            token_to_embeddings = self._add_kg_embeddings_to_vocab(
                token_to_embeddings=token_to_embeddings)

        token_to_id_dict = {token: id for id, token in enumerate(token_to_embeddings.keys())}
        print(token_to_id_dict)
        return token_to_embeddings, token_to_id_dict

    def _load_entity_to_label_mapping(self, path_to_entity_id_to_label_mapping):
        """
        Load dict containing entity id's as keys and the corresponding labels as values
        :param path_to_entity_id_to_label_mapping:
        :rtype: dict
        """

        return load_data_from_json_file(path_to_entity_id_to_label_mapping)

    def _load_word_to_vec_model(self, word_to_vec_dict):
        """
        Load pretrained word2Vec model
        :param word_to_vec_dict:
        :return:
        """
        path = word_to_vec_dict['Path']
        is_c_format = word_to_vec_dict['is_c_format']
        is_binary = word_to_vec_dict['is_binary']

        if is_c_format:
            word_to_vec_model = KeyedVectors.load_word2vec_format(path, binary=is_binary)
        else:
            word_to_vec_model = KeyedVectors.load(fname_or_handle=path)

        return word_to_vec_model

    def _get_offsets_of_relevant_entities_in_utterance(self, utterance_dict):
        """
        Returns sorted (increasing) offsets of utterance-parts based on mentioned entities
        :param utterance_dict: Dictionary containing all information about passed utterance
        :rtype: list, list
        """
        start_offsets, end_offsets = [], []

        # No entities in utterance
        if CSQA_ENTITIES_IN_UTTERANCE not in utterance_dict:
            utterance_dict[CSQA_ENTITIES_IN_UTTERANCE] = []
            return start_offsets, end_offsets

        utterance = utterance_dict[CSQA_UTTERANCE]

        ids_of_entities_in_utterance = utterance_dict[CSQA_ENTITIES_IN_UTTERANCE]
        entities_in_utterance = [self.entity_id_to_label_dict[entity_id] for entity_id in ids_of_entities_in_utterance]

        for entity_in_utterance in entities_in_utterance:
            # Get offsets of entity
            start = utterance.find(entity_in_utterance)

            if start == -1:
                # Entity not found
                log.info("Entity '%s' not found in utterance '%s' contained in file '%s'" % (
                    entity_in_utterance, utterance, self.current_file_in_progress))
                continue

            end = start + len(entity_in_utterance)

            # Get updated list of ofsets. Function handles overlaps.
            start_offsets, end_offsets = save_insertion_of_offsets(start_offsets, end_offsets,
                                                                   start, end)

        return start_offsets, end_offsets

    def _map_utter_toks_to_ids(self, utterance_dict, utterance_offsets_info_dict, nlp_spans, is_reponse_utter):
        """

        :param utterance_dict:
        :param utterance_offsets_info_dict:
        :param nlp_spans:
        :param is_reponse_utter:
        :return:
        """

        utterance = utterance_dict[CSQA_UTTERANCE]
        counter = 0

        token_ids = []
        for offset_tuple, is_entity in utterance_offsets_info_dict.items():
            nlp_span = nlp_spans[counter]
            token_ids += self._determine_token_ids(txt=utterance, offset_tuple=offset_tuple, is_entity=is_entity,
                                                  nlp_span=nlp_span, is_reponse_utter=is_reponse_utter)

            counter += 1

        return token_ids

    def _apply_instance_creation_steps(self, utterance_dict, is_reponse_utter):
        utterance_txt = utterance_dict[CSQA_UTTERANCE]

        # Step 1: Get offsets of utterance-parts based on mentioned entites
        relevant_entity_start_offsets_in_utterance, relevant_entity_end_offsets_in_utterance = \
            self._get_offsets_of_relevant_entities_in_utterance(
                utterance_dict)

        utterance_offset_info_dict = mark_parts_in_text(
            start_offsets_entities=relevant_entity_start_offsets_in_utterance,
            end_offsets_entities=relevant_entity_end_offsets_in_utterance,
            text=utterance_txt)

        # Step 2: Compute NLP features
        utterance_nlp_spans = compute_nlp_features(txt=utterance_txt,
                                                   offsets_info_dict=utterance_offset_info_dict)

        utterance_tok_ids = self._map_utter_toks_to_ids(utterance_dict=utterance_dict,
                                                        utterance_offsets_info_dict=utterance_offset_info_dict,
                                                        nlp_spans=utterance_nlp_spans,
                                                        is_reponse_utter=is_reponse_utter)

        padded_utter_tok_id = self.add_utter_padding(utter_tok_ids=utterance_tok_ids, is_reponse_utter=is_reponse_utter)

        return padded_utter_tok_id

    def _determine_token_ids(self, txt, offset_tuple,
                             is_entity, nlp_span, is_reponse_utter):

        if is_reponse_utter:
            word_to_id = self.response_word_to_id
        else:
            word_to_id = self.ctx_word_to_id

        if is_entity:
            start = offset_tuple[0]
            end = offset_tuple[1]
            # Entity don't transform to lowercase
            entity = txt[start:end]
            entity_id = self.label_to_id_dict[entity]

            print("ID: ", self._get_token_id_for_entity(entity_id, is_reponse_utter))

            return [self._get_token_id_for_entity(entity_id, is_reponse_utter)]

        else:
            return [word_to_id[token.lower_] if token.lower_ in word_to_id else word_to_id[UNKNOWN_TOKEN] for token in
                    nlp_span]

    def _get_token_id_for_entity(self, entity_id, is_reponse_utter):

        id = None

        if is_reponse_utter:
            id = self.response_word_to_id[KG_WORD]
        else:
            print("In _get: ", self.ctx_word_to_id)
            if entity_id in self.ctx_word_to_id:
                id = self.ctx_word_to_id[entity_id]
            else:
                id = self.ctx_word_to_id[UNKNOWN_TOKEN]

        return id

    def create_training_instances(self, dialogue, file_id):

        # Set state
        self.current_file_in_progress = file_id

        if len(dialogue) > self.max_dialogue_context_length * 2:
            dialogue = dialogue[-(self.max_dialogue_context_length * 2):]

        context = dialogue[:-1]
        response = dialogue[-1]

        for utter_dict in context:
            utter_token_ids = self._apply_instance_creation_steps(utterance_dict=utter_dict, is_reponse_utter=False)

    def add_utter_padding(self, utter_tok_ids, is_reponse_utter):
        num_tokens = len(utter_tok_ids)
        right_padding_size = (self.max_num_utter_tokens - num_tokens)
        if is_reponse_utter:
            padding = [self.response_word_to_id[PADDING_TOKEN] for i in range(right_padding_size)]
        else:
            padding = [self.ctx_word_to_id[PADDING_TOKEN] for i in range(right_padding_size)]

        return utter_tok_ids + padding
